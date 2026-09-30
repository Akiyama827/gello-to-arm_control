"""CTRL-051: fault-trace overflow, atomic write, retention and read-only export (no robot, no transport)."""
import errno
import hashlib
import json
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
        # 3. Capacity: a save never deletes; over the policy it warns; pruning is an explicit, dry-run-by-default action.
        for i in range(6):
            small = diagnostics.PairFaultTrace(f"r{i}")
            small.record(dict(index=i))
            small.save(f"r{i}", {}, root=root)
        assert len(list(root.glob("*.json"))) == 7, "a save deleted something"
        assert diagnostics.capacity_warning(root, keep_files=3) and not diagnostics.capacity_warning(root)
        assert diagnostics.prune_traces(root, keep_files=3, dry_run=True) and len(list(root.glob("*.json"))) == 7
        assert trace_export.main(["--root", str(root), "--prune", "--keep-files", "3"]) == 0  # dry run
        assert len(list(root.glob("*.json"))) == 7
        removed = diagnostics.prune_traces(root, keep_files=3)
        names = sorted(p.name for p in root.glob("*.json"))
        assert len(names) == 3 and len(removed) == 4 and all(n.endswith(".json") for n in removed), (names, removed)
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
        # 5. The module opens no transport: importing it pulls in no plant, SDK or device module.
        code = ("import sys, arm_control.multi_arm.trace_export;"
                "bad=[m for m in sys.modules if m.startswith(('arm_control.plants','arm_control.sdk','arm_control.supervisor'))];"
                "assert not bad, bad")
        subprocess.run([sys.executable, "-c", code], check=True)
    print("fault traces: overflow counted, atomic write, retention, read-only verified export: PASS")


if __name__ == "__main__":
    main()
