"""CTRL-051: fault-trace overflow, atomic write, retention and read-only export (no robot, no transport)."""
import errno
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from arm_control.multi_arm import diagnostics, trace_export


def digest_tree(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file()}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "faults"
        # 1. Overflow is explicit: 510 recorded, 500 kept, 10 evicted and said so.
        trace = diagnostics.PairFaultTrace("s1")
        for i in range(510):
            trace.record(dict(index=i))
        first = trace.save("overflow", {}, root=root)
        report = json.loads(first.read_text())
        assert report["samples_kept"] == 500 and report["dropped_samples"] == 10 and report["samples"][0]["index"] == 10
        assert report["retention"] == dict(keep_files=diagnostics.RETAIN_FILES, keep_bytes=diagnostics.RETAIN_BYTES)
        info = trace_export.verify(first)
        assert info["ok"] and info["dropped_samples"] == 10 and info["samples_kept"] == 500, info
        # 2. A write that fails part-way (disk full at fsync) leaves nothing that looks like a trace.
        before = digest_tree(root)
        with patch.object(diagnostics.os, "fsync", side_effect=OSError(errno.ENOSPC, "No space left on device")):
            try:
                trace.save("disk full", {}, root=root)
            except OSError as exc:
                assert exc.errno == errno.ENOSPC
            else:
                raise AssertionError("a failed write was reported as saved")
        assert digest_tree(root) == before, "a failed write left a file behind"
        # 3. Capacity: every save keeps the newest RETAIN_FILES (operator 2026-10-10), the new one always among them, sidecars
        #    with their traces; pruning further is the operator's explicit, dry-run-by-default action.
        with patch.object(diagnostics, "RETAIN_FILES", 4):
            for i in range(6):
                small = diagnostics.PairFaultTrace(f"r{i}")
                small.record(dict(index=i))
                last = small.save(f"r{i}", {}, root=root)
        names = sorted(p.name for p in root.glob("*.json"))
        assert len(names) == 4 and names[-1] == last.name and first.name not in names, names
        assert sorted(p.name for p in root.glob("*.sha256")) == [n + ".sha256" for n in names]
        assert diagnostics.capacity_warning(root, keep_files=3) and not diagnostics.capacity_warning(root)
        assert diagnostics.prune_traces(root, keep_files=3, dry_run=True) and len(list(root.glob("*.json"))) == 4
        assert trace_export.main(["--root", str(root), "--prune", "--keep-files", "3"]) == 0  # dry run
        assert len(list(root.glob("*.json"))) == 4
        removed = diagnostics.prune_traces(root, keep_files=3)
        names = sorted(p.name for p in root.glob("*.json"))
        assert len(names) == 3 and len(removed) == 1 and all(n.endswith(".json") for n in removed), (names, removed)
        # Over MAX_TRACE_BYTES the OLDEST samples go, counted; the trace still verifies.
        big = diagnostics.PairFaultTrace("big")
        for i in range(400):
            big.record(dict(index=i, pad="x" * 1000))
        with patch.object(diagnostics, "MAX_TRACE_BYTES", 100_000):
            capped = big.save("capped", {}, root=Path(tmp) / "capped")
        report = json.loads(capped.read_text())
        assert capped.stat().st_size <= 100_000 and report["truncated_samples"] > 0, capped.stat().st_size
        assert report["samples"][-1]["index"] == 399 and report["samples_kept"] + report["truncated_samples"] == 400
        assert trace_export.verify(capped)["ok"]
        assert sorted(p.name for p in root.glob("*.sha256")) == [n + ".sha256" for n in names]
        assert all(trace_export.verify(root / n)["ok"] for n in names)
        newest = root / names[-1]
        assert diagnostics.prune_traces(root, keep_files=0, keep_bytes=0, protect=newest) == names[:-1]
        assert [p.name for p in root.glob("*.json")] == [newest.name]  # the protected one survives any limit
        # 4. Export is read-only, verifies, refuses tampered traces and unsafe destinations, never overwrites.
        for i in range(3):
            t = diagnostics.PairFaultTrace(f"e{i}")
            t.record(dict(index=i))
            t.save(f"e{i}", {}, root=root)
        bad = sorted(root.glob("*.json"))[-1]
        with bad.open("ab") as stream:
            stream.write(b" ")  # tampered after the sidecar was written
        assert not trace_export.verify(bad)["ok"] and "sha256" in trace_export.verify(bad)["problem"]
        source_before = digest_tree(root)
        dest = Path(tmp) / "export"
        manifest = trace_export.export(root, dest)
        assert digest_tree(root) == source_before, "export modified the source"
        exported = sorted(p.name for p in dest.glob("*.json") if p.name != "EXPORT_MANIFEST.json")
        assert bad.name not in exported and len(exported) == len(list(root.glob("*.json"))) - 1
        flags = {t["name"]: t["exported"] for t in manifest["traces"]}
        assert flags[bad.name] is False and sum(flags.values()) == len(exported)
        assert all(trace_export.verify(dest / n)["ok"] for n in exported)
        for unsafe in (dest, root, root / "inside"):
            try:
                trace_export.export(root, unsafe)
            except ValueError:
                pass
            else:
                raise AssertionError(f"export accepted {unsafe}")
        assert trace_export.main(["--root", str(root), "--verify"]) == 1  # one bad trace: non-zero
        # CTRL-010: a bare --dest exports under the deployment's export root, never without one.
        import os
        with patch.dict(os.environ, {diagnostics.EXPORT_ROOT_ENV: ""}):
            assert trace_export.main(["--root", str(root), "--dest"]) == 2
        with patch.dict(os.environ, {diagnostics.EXPORT_ROOT_ENV: str(Path(tmp) / "exports")}):
            assert trace_export.main(["--root", str(root), "--dest", "--include-unverified"]) == 0
        auto = list((Path(tmp) / "exports").glob("*/faults-*/EXPORT_MANIFEST.json"))
        assert len(auto) == 1 and auto[0].parent.name.startswith(f"faults-{root.parent.name}-"), auto
        # 4b. Round-1 review hardening.
        sub = Path(tmp) / "hard"
        t = diagnostics.PairFaultTrace("h")
        t.record(dict(index=1))
        # crash between writing and renaming: the final name never exists, nothing is left that looks like a trace
        with patch.object(diagnostics.os, "replace", side_effect=OSError("simulated crash before rename")):
            try:
                t.save("crash", {}, root=sub)
            except OSError:
                pass
        assert not list(sub.glob("*.json")) and not list(sub.glob("*.partial")) and not list(sub.glob("*.sha256"))
        # ENOSPC while writing the sidecar: nothing left behind
        real_open = Path.open

        def failing_sidecar(self, mode="r", *a, **k):
            if self.name.endswith(".sha256") and "x" in mode:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_open(self, mode, *a, **k)
        with patch.object(Path, "open", failing_sidecar):
            try:
                t.save("sidecar full", {}, root=sub)
            except OSError:
                pass
        assert not any(sub.iterdir())
        # a name collision (same session, same microsecond) never damages the committed trace
        frozen = diagnostics.datetime.now(diagnostics.timezone.utc)
        with patch.object(diagnostics, "datetime") as fake_dt:
            fake_dt.now.return_value = frozen
            first_trace = t.save("first", {}, root=sub)
            good = digest_tree(sub)
            try:
                t.save("third", {}, root=sub)
            except FileExistsError:
                pass
            else:
                raise AssertionError("a colliding save replaced a committed trace")
        assert digest_tree(sub) == good and trace_export.verify(first_trace)["ok"]
        # legacy trace (no sidecar): parses, is reported as legacy, exported, never a "failed verification"
        legacy = sub / "20200101T000000000000Z-legacy.json"
        legacy.write_text(json.dumps(dict(schema="arm_control.dual_fault/v1", reason="old", samples=[{"a": 1}])))
        info = trace_export.verify(legacy)
        assert info["ok"] and info["legacy"] and "sidecar" in info["problem"]
        # a valid JSON that is not a trace, with a matching sidecar, is a reported problem, never an exception
        odd = sub / "20200102T000000000000Z-odd.json"
        odd.write_text("[1, 2]")
        (sub / (odd.name + ".sha256")).write_text(hashlib.sha256(odd.read_bytes()).hexdigest() + "  " + odd.name + "\n")
        assert not trace_export.verify(odd)["ok"]
        # samples_kept must equal the samples present
        lying = sub / "20200103T000000000000Z-lying.json"
        lying.write_text(json.dumps(dict(schema="arm_control.dual_fault/v2", samples_kept=5, samples=[1])))
        (sub / (lying.name + ".sha256")).write_text(hashlib.sha256(lying.read_bytes()).hexdigest() + "  " + lying.name + "\n")
        assert "samples_kept" in trace_export.verify(lying)["problem"]
        # a trace written by this code whose sidecar was deleted is BAD, not legacy
        stripped = sub / "20200104T000000000000Z-stripped.json"
        stripped.write_text(json.dumps(dict(schema="arm_control.dual_fault/v2", samples_kept=1, samples=[1])))
        assert not trace_export.verify(stripped)["ok"] and "sidecar" in trace_export.verify(stripped)["problem"]
        stripped.unlink()
        # prune deletes only traces with a byte-identical copy; the dry run predicts exactly that
        exp, cnt = Path(tmp) / "exp_prune", Path(tmp) / "cnt_prune"
        shutil.copytree(sub, exp)
        cnt.mkdir()
        for p in sub.glob("2*.json"):
            shutil.copy(p, cnt / p.name)
        victim = sorted(cnt.glob("2*.json"))[0]
        victim.write_text(victim.read_text() + " ")  # one confirmation copy differs
        base = ["--prune", "--keep-files", "0", "--root", str(sub), "--confirm-exported", str(cnt)]
        import io
        from contextlib import redirect_stdout
        def run(extra):
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = trace_export.main(base + extra)
            return rc, buf.getvalue()
        rc, dry = run([])
        assert victim.name not in dry and dry.startswith("would remove")
        before = {p.name for p in sub.glob("2*.json")}
        rc, real = run(["--yes"])
        after = {p.name for p in sub.glob("2*.json")}
        assert after == {victim.name} and before - after == set(before) - {victim.name}, (before, after)
        assert trace_export.main(["--verify", "--root", str(Path(tmp) / "nonexistent")]) == 2
        # an export directory used as a root: its manifest is not a trace; a crash leftover is listed as incomplete
        out = Path(tmp) / "out"
        trace_export.export(sub, out)
        assert all(i["name"] != "EXPORT_MANIFEST.json" for i in trace_export.list_traces(out))
        assert (out / "20200101T000000000000Z-legacy.json").is_file()  # the legacy trace was exported
        (sub / "20990101T000000000000Z-x.json.partial").write_bytes(b"{")
        assert any(i["problem"] == "incomplete write (crash leftover)" for i in trace_export.list_traces(sub))
        # export never overwrites (exclusive create)
        real_open2 = Path.open
        modes = []

        def spy(self, mode="r", *a, **k):
            modes.append(mode)
            return real_open2(self, mode, *a, **k)
        with patch.object(Path, "open", spy):
            trace_export.export(sub, Path(tmp) / "out2")
        assert "wb" not in modes and "xb" in modes
        # prune: refuses without confirmation, deletes only byte-identical exported copies, rejects a negative count
        assert trace_export.main(["--root", str(sub), "--prune", "--keep-files", "0", "--yes"]) == 2
        assert trace_export.main(["--root", str(sub), "--prune", "--keep-files", "-1"]) == 2
        before = sorted(p.name for p in sub.glob("2*.json"))
        assert trace_export.main(["--root", str(sub), "--prune", "--keep-files", "0", "--yes",
                                  "--confirm-exported", str(Path(tmp) / "empty-none")]) == 0
        assert sorted(p.name for p in sub.glob("2*.json")) == before, "deleted traces that were never exported"
        assert trace_export.main(["--root", str(sub), "--prune", "--keep-files", "0", "--yes",
                                  "--confirm-exported", str(out)]) == 0
        assert not [p for p in sub.glob("2*.json") if (out / p.name).is_file()], "an exported trace survived the prune"
        # 4b. A symlink named like a trace is not a trace: not counted, never pruned, never exported, and what it points to is
        # untouched. Orphan .partial files count toward capacity (and are named by the warning).
        sym_root, outside = Path(tmp) / "symroot", Path(tmp) / "outside.json"
        sym_root.mkdir()
        outside.write_text("{}")
        (sym_root / "20990101T000000000000Z-link.json").symlink_to(outside)
        assert diagnostics._traces(sym_root) == []
        assert diagnostics.prune_traces(sym_root, keep_files=0, keep_bytes=0) == [] and outside.read_text() == "{}"
        assert not trace_export.export(sym_root, Path(tmp) / "symexport")["traces"]
        (sym_root / "20990101T000000000001Z-x.json.partial").write_bytes(b"x" * 2048)
        assert diagnostics.capacity_warning(sym_root, keep_files=10, keep_bytes=1024), "a partial file did not count"
        assert not diagnostics.capacity_warning(sym_root, keep_files=10, keep_bytes=4096)
        assert "1 unfinished .partial" in diagnostics.capacity_warning(sym_root, keep_files=10, keep_bytes=1024)
        # a dangling .partial link or a vanished file does not hide the warning for the rest
        (sym_root / "dangling.partial").symlink_to(Path(tmp) / "nowhere")
        assert diagnostics.capacity_warning(sym_root, keep_files=10, keep_bytes=1024), "a dangling link hid the warning"
        assert diagnostics._size(Path(tmp) / "nowhere") == 0  # a vanished file is not countable and not an error
        big = Path(tmp) / "big.bin"
        big.write_bytes(b"x" * (2 << 20))
        (sym_root / "linked.partial").symlink_to(big)  # a link is not a partial file: what it points at is not ours to count
        assert not diagnostics.capacity_warning(sym_root, keep_files=10, keep_bytes=4096), "a .partial link was counted"
        # 4c. Durability: the trace and its sidecar are fsynced before the rename and the directory after it, in that order.
        order = []
        real_fsync, real_replace = diagnostics.os.fsync, diagnostics.os.replace
        with patch.object(diagnostics.os, "fsync", side_effect=lambda fd: order.append("fsync") or real_fsync(fd)), \
                patch.object(diagnostics.os, "replace", side_effect=lambda a, b: order.append("replace") or real_replace(a, b)):
            t = diagnostics.PairFaultTrace("durable")
            t.record(dict(index=0))
            t.save("durable", {}, root=Path(tmp) / "durable")
        assert order == ["fsync", "fsync", "replace", "fsync"], order  # trace, sidecar, rename, directory
        # 4d. An arm whose controller holds NO plan ('' or None: finished or not yet loaded) is idle, not running a foreign plan;
        # a different non-empty plan is foreign. The identity join is on plans, so an empty id never breaks or fakes a chain.
        def tick(life, plan_a, plan_b, i):
            arms = {a: dict(plan_id=p, tau_ff_pc=[0.], raw_pc=[0.], q_cmd_pc=[0.], last_cmd_seq=i, q_rt=[0.])
                    for a, p in (("a", plan_a), ("b", plan_b))}
            return dict(pair_id="p", lifecycle=life, pair_plans={"a": "pa", "b": "pb"}, arms=arms, pc_monotonic=float(i))
        done = [tick("ready", "pa", "pb", 0), tick("executing", "pa", "pb", 1), tick("executing", "pa", "", 2),
                tick("executing", "", None, 3), tick("complete", "", "", 4)]
        report = diagnostics.reconstruct_lineage(done, pair_id="p")
        assert report["complete"] and not report["foreign_arms"], report
        other = done[:2] + [tick("executing", "pa", "someone-else", 2)] + done[3:]
        report = diagnostics.reconstruct_lineage(other, pair_id="p")
        assert not report["complete"] and report["foreign_arms"] == ["b"], report
        # 5. The module opens no transport: importing it pulls in no plant, SDK or device module.
        code = ("import sys, arm_control.multi_arm.trace_export;"
                "bad=[m for m in sys.modules if m.startswith(('arm_control.plants','arm_control.sdk','arm_control.supervisor'))];"
                "assert not bad, bad")
        subprocess.run([sys.executable, "-c", code], check=True)
    print("fault traces: overflow counted, atomic write, retention, read-only verified export: PASS")


if __name__ == "__main__":
    main()
