"""Bounded PC observation history, persisted only after a fault requests stop.

RT timestamps label published server states, not synchronized physical capture
times. Separate RT hosts must not have their timestamps directly subtracted.
"""

from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from arm_control import CONTROL_ROOT
from arm_control.multi_arm.closure import ASSUMPTIONS as CLOSURE_ASSUMPTIONS


def _json_values(value):
    if isinstance(value, np.ndarray):
        return _json_values(value.tolist())
    if isinstance(value, np.generic):
        return _json_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_values(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, deque)):
        return [_json_values(v) for v in value]
    return value


#: Capacity policy for persisted fault traces (CTRL-051, operator 2026-10-09/10): each save keeps the newest ``RETAIN_FILES``
#: traces of its directory and deletes older ones (and their sidecars); a trace is at most ``MAX_TRACE_BYTES`` (the oldest
#: samples go first, counted), so a directory holds at most ``RETAIN_BYTES``. Export before a trace ages out
#: (``trace_export --dest``). A console trace is ~5 MB (500 samples), a supervisor trace ~2.5 MB (12 s of ticks).
RETAIN_FILES = 20
MAX_TRACE_BYTES = 50 * 2**20
RETAIN_BYTES = RETAIN_FILES * MAX_TRACE_BYTES
FAULT_DIR = ("dual_arm", "faults")
#: The trace schemas this module writes (trace_export verifies these).
TRACE_SCHEMAS = ("arm_control.dual_fault/", "arm_control.supervisor_fault/")
#: A supervisor's fault trace (CTRL-051, operator 2026-10-09): the ticks from FAULT_PRE_S before the first fault to
#: FAULT_POST_S after it; faults inside that window join the same trace.
FAULT_PRE_S = 10.0
FAULT_POST_S = 2.0


#: Checks point every fault directory (console and supervisors, including their subprocesses) here, so a check never writes
#: into, or prunes, a real arm's traces.
FAULT_ROOT_ENV = "ARM_CONTROL_FAULT_ROOT"


def _fault_base():
    override = os.environ.get(FAULT_ROOT_ENV)
    if override and not Path(override).is_absolute():
        raise ValueError(f"{FAULT_ROOT_ENV} must be an absolute path, not {override!r}")
    return Path(override) if override else CONTROL_ROOT / ".cache"


def _leaf(sim):
    # A simulation's traces never share a directory, or its newest-20 retention, with the hardware's (the simulated bench
    # pair runs under the real instance ids).
    return "sim_faults" if sim else "faults"


def fault_root(sim=False):
    return _fault_base().joinpath(FAULT_DIR[0], _leaf(sim))


#: An operator export's root (CTRL-010): set by the deployment, outside any repository (this project:
#: /mnt/data/experiment_data/modular_arm/control_exports, operator 2026-10-09). Unset: no default, a destination is required.
EXPORT_ROOT_ENV = "ARM_CONTROL_EXPORT_ROOT"


def export_dir(kind):
    """``$ARM_CONTROL_EXPORT_ROOT/<local date>/<kind>-<local time>``, a fresh directory name for one export, or None when
    the variable is unset."""
    root = os.environ.get(EXPORT_ROOT_ENV)
    if not root:
        return None
    if not Path(root).is_absolute():
        raise ValueError(f"{EXPORT_ROOT_ENV} must be an absolute path (keep it outside any repository), not {root!r}")
    now = datetime.now()
    return Path(root) / f"{now:%Y-%m-%d}" / f"{kind}-{now:%H%M%S-%f}"


def supervisor_fault_root(instance_id, sim=False):
    return _fault_base().joinpath("supervisor", str(instance_id) or "unnamed", _leaf(sim))


TRACE_GLOB = "2*.json"  # timestamped traces only: an EXPORT_MANIFEST.json in the directory is not a trace


def _traces(root):
    return sorted(p for p in root.glob(TRACE_GLOB) if p.is_file() and not p.is_symlink())


def prune_traces(root, *, keep_files=RETAIN_FILES, keep_bytes=RETAIN_BYTES, protect=None, dry_run=False, only=None):
    """Remove the oldest ``*.json`` traces (and their ``.sha256`` sidecars) beyond the limits. Called by every save (the
    newest ``RETAIN_FILES`` kept, the new trace protected) and by the operator's ``trace_export --prune``. Never raises, never removes ``protect`` and, when ``only`` is given, nothing outside that
    set of names; returns the names (that would be) removed.
    Names begin with a UTC timestamp, so name order is age."""
    removed = []
    try:
        traces = _traces(root)
        sizes = {p: p.stat().st_size for p in traces}
        total = sum(sizes.values())
        for p in traces:
            if len(traces) - len(removed) <= keep_files and total <= keep_bytes:
                break
            if (protect is not None and p == protect) or (only is not None and p.name not in only):
                continue
            try:
                if not dry_run:
                    p.unlink()
                total -= sizes[p]  # the trace is gone from here on, whatever happens to its sidecar
                removed.append(p.name)
                if not dry_run:
                    sidecar = p.with_name(p.name + ".sha256")
                    if sidecar.exists():
                        sidecar.unlink()
            except OSError:
                continue
    except OSError:
        pass
    return removed


def _fsync_dir(root):
    """Make the rename durable (best effort: a failure here does not undo a completed write)."""
    try:
        fd = os.open(root, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def write_trace(root, name, header, samples):
    """Persist one fault trace atomically (temporary file, fsync, rename) with a SHA-256 sidecar, then keep the newest
    ``RETAIN_FILES`` traces of ``root``. A full disk or a crash never leaves a partial file that looks complete, and an
    existing trace is never replaced. Over ``MAX_TRACE_BYTES`` the oldest samples are dropped (``truncated_samples``).
    Samples are encoded one at a time, so a writer thread never holds the interpreter for a whole trace."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    encoded, slice_start = [], time.monotonic()
    for s in samples:
        encoded.append(json.dumps(_json_values(s), allow_nan=False))
        if time.monotonic() - slice_start > 0.002:  # every ~2 ms of encoding, let a servo loop sharing this interpreter run
            time.sleep(0.001)
            slice_start = time.monotonic()
    head = json.dumps(_json_values(dict(header, samples_kept=0, truncated_samples=0)), allow_nan=False)
    budget = MAX_TRACE_BYTES - len(head) - 64
    size, keep = 0, len(encoded)
    for k in range(len(encoded) - 1, -1, -1):  # newest first: the fault end of the window is kept
        if size + len(encoded[k]) + 1 > budget:
            keep = len(encoded) - 1 - k
            break
        size += len(encoded[k]) + 1
    truncated = len(encoded) - keep
    encoded = encoded[truncated:]
    head = json.dumps(_json_values(dict(header, samples_kept=len(encoded), truncated_samples=truncated)), allow_nan=False)
    payload = (head[:-1] + ', "samples": [' + ",".join(encoded) + "]}\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    partial = path.with_name(path.name + ".partial")
    sidecar = path.with_name(path.name + ".sha256")
    created = []  # only what THIS call created is ever cleaned up (a name collision must not destroy another trace)
    try:
        with partial.open("xb") as stream:
            created.append(partial)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        with sidecar.open("x") as stream:  # exclusive: never replaces another trace's sidecar
            created.append(sidecar)
            stream.write(f"{digest}  {path.name}\n")
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError(str(path))  # never replace an existing trace
        os.replace(partial, path)  # the trace appears whole or not at all
        created.remove(partial)
        _fsync_dir(root)
    except BaseException:
        for leftover in created:
            try:
                leftover.unlink()
            except OSError:
                pass
        raise
    removed = prune_traces(root, keep_files=RETAIN_FILES, keep_bytes=RETAIN_BYTES, protect=path)
    if removed:
        print(f"[fault trace] kept the newest {RETAIN_FILES} in {root}; removed {removed}", flush=True)
    return path


def _stamp():
    now = datetime.now(timezone.utc)
    return now, f"{now:%Y%m%dT%H%M%S%fZ}"


class PairFaultTrace:
    """Last 500 console ticks; no disk I/O or JSON encoding in record().

    Overflow is explicit: the number of samples evicted from the bounded history is counted and written into the
    persisted trace (``dropped_samples``), never silent. Written by :func:`write_trace` (atomic, sidecar, newest
    ``RETAIN_FILES`` kept).
    """

    def __init__(self, session):
        self.session = session
        self.samples = deque(maxlen=500)
        self.recorded = 0  # every sample ever recorded; recorded - len(samples) = evicted

    def record(self, sample):
        self.samples.append(sample)
        self.recorded += 1

    def save(self, reason, context, *, samples=None, root=None, recorded=None, sim=False):
        now, stamp = _stamp()
        kept = list(self.samples) if samples is None else samples
        seen = self.recorded if recorded is None else recorded
        header = dict(
            schema="arm_control.dual_fault/v2",
            schema_version=2,
            closure_schema="arm_control.closure/v1",  # samples carry `closure_v1` beside the older `closure`
            closure_assumptions=list(CLOSURE_ASSUMPTIONS),
            observed_at_utc=now.isoformat(),
            reason=reason,
            session=self.session,
            timestamp_note=(
                "PC receipt/tick times are not physical capture times. RT state "
                "timestamps share a clock only when their attested clock_domain matches "
                "(rt_host string equality is the fallback for unattested servers); "
                "publication timestamps do not establish synchronized robot measurements."
            ),
            dropped_samples=max(0, seen - len(kept)),
            retention=dict(keep_files=RETAIN_FILES, keep_bytes=RETAIN_BYTES),
            context=context,
        )
        return write_trace(fault_root(sim) if root is None else root, f"{stamp}-{self.session}.json", header, kept)


class SupervisorFaultTrace:
    """A supervisor's own fault record (CTRL-051): its trace/1 ticks from ``pre_s`` before the first fault to ``post_s``
    after, written by one background thread so the servo loop never waits for encoding or the disk.

    The loop calls ``record`` every tick, ``trigger`` on a fault and ``poll`` every tick. A fault opens a window; faults
    inside it join it (``faults``) and get what remains of the FIRST fault's ``post_s``; then the window's samples are
    handed to the writer. ``trigger``/``poll`` never raise (a failure is counted in ``errors``). A writer
    still busy with the previous trace makes the new one SKIPPED (counted in ``skipped`` and in the next trace), never
    queued without bound; a failed write (disk full) is counted in ``errors`` and printed, never raised into the loop.
    It does not depend on any trace/1 reader: a console that died still leaves the supervisor's record."""

    MAX_FAULTS = 50  # per window; more are counted, not listed

    def __init__(self, instance_id, period, *, root=None, pre_s=FAULT_PRE_S, post_s=FAULT_POST_S, context=None, sim=False):
        self.instance_id, self.root, self.sim = str(instance_id), root, bool(sim)
        if root is None and os.environ.get(FAULT_ROOT_ENV):
            print(f"[fault trace] {self.instance_id}: traces go to {supervisor_fault_root(self.instance_id, self.sim)} "
                  f"({FAULT_ROOT_ENV} is set)", flush=True)
        self.pre_s, self.post_s = float(pre_s), float(post_s)
        self.context = dict(context or {})
        self._ring = deque(maxlen=int(math.ceil((self.pre_s + self.post_s) / float(period))) + 2)
        self._window = None
        self._writer = None
        self._busy = None
        self.skipped = self.errors = 0
        self.written = []

    def record(self, sample):
        self._ring.append(sample)

    def trigger(self, reason, now):
        try:
            self._trigger(reason, now)
        except Exception as exc:  # noqa: BLE001 -- the record must never stop the servo loop
            self._failed(exc)

    def poll(self, now, **extra):
        try:
            if self._window is not None and now >= self._window["start"] + self.post_s:
                self._flush(complete=True, **extra)
        except Exception as exc:  # noqa: BLE001
            self._failed(exc)

    def _failed(self, exc):
        self._window = None
        self.errors += 1
        print(f"[fault trace] {self.instance_id}: trace failed ({exc!r}); {self.errors} so far", flush=True)

    def _trigger(self, reason, now):
        w = self._window
        if w is None:
            self._window = dict(start=float(now), faults=[dict(at=float(now), reason=str(reason))], more=0)
        elif len(w["faults"]) < self.MAX_FAULTS:
            w["faults"].append(dict(at=float(now), reason=str(reason)))
        else:
            w["more"] += 1

    def _flush(self, *, complete, **extra):
        w, self._window = self._window, None
        if self._busy is not None and not self._busy.done():
            self.skipped += 1
            print(f"[fault trace] {self.instance_id}: writer still busy; the trace of {w['faults'][0]['reason']!r} is "
                  f"SKIPPED ({self.skipped} so far)", flush=True)
            return
        start = w["start"]
        samples = [s for s in self._ring if s.get("pc_monotonic", start) >= start - self.pre_s]
        now, stamp = _stamp()
        header = dict(
            schema="arm_control.supervisor_fault/v1", schema_version=1, observed_at_utc=now.isoformat(),
            instance_id=self.instance_id, reason=w["faults"][0]["reason"], faults=w["faults"], more_faults=w["more"],
            fault_at_pc_monotonic=start, pre_s=self.pre_s, post_s=self.post_s, post_window_complete=complete,
            window_pc_monotonic=[samples[0].get("pc_monotonic"), samples[-1].get("pc_monotonic")] if samples else None,
            dropped_samples=0, skipped_traces_before=self.skipped, write_errors_before=self.errors,
            timestamp_note=("pc_monotonic is this supervisor's tick time on the PC; state_t_mono_ns is the RT host's "
                            "publication stamp (clock domain per the HELLO), not a physical capture time."),
            retention=dict(keep_files=RETAIN_FILES, max_trace_bytes=MAX_TRACE_BYTES),
            context=dict(self.context, **extra),
        )
        root = supervisor_fault_root(self.instance_id, self.sim) if self.root is None else self.root
        name = f"{stamp}-{self.instance_id or 'supervisor'}.json"
        if self._writer is None:
            self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fault-trace")
        self._busy = self._writer.submit(self._write, root, name, header, samples)

    def _write(self, root, name, header, samples):
        try:
            path = write_trace(root, name, header, samples)
        except Exception as exc:  # noqa: BLE001 -- a failed record is reported, never raised into the servo
            self.errors += 1
            print(f"[fault trace] {self.instance_id}: write failed ({exc!r}); {self.errors} so far", flush=True)
            return None
        self.written.append(path)
        print(f"[fault trace] {self.instance_id}: {header['reason']} -> {path}", flush=True)
        return path

    def close(self, timeout_s=5.0, **extra):
        """Shutdown, after the arm is disarmed: a window still open is written with what it has
        (``post_window_complete`` false); waits at most ``timeout_s`` per write. Never raises."""
        try:
            for flush in (self._window is not None, False):
                if self._busy is not None:
                    self._busy.result(timeout=timeout_s)
                if flush:
                    self._flush(complete=False, **extra)
        except Exception as exc:  # noqa: BLE001 -- shutdown continues
            self.errors += 1
            print(f"[fault trace] {self.instance_id}: close: {exc!r}", flush=True)
        if self._writer is not None:
            self._writer.shutdown(wait=False)



# The execution-chain links the dual_fault sample schema (v1 and v2) carries, in chain
# order (CTRL-051). A lineage report is complete exactly when every link is
# JOINED by identity, not when each field merely appears somewhere.
LINE_LINKS = (
    "requested",    # pair_id: the plan identity the policy requested
    "admitted",     # lifecycle: the admitted phase the pair reached
    "reference",    # observed per-arm plan_id == the pair's declared member plan
    "raw",          # tau_ff_pc / integral_bias_pc / raw_pc: controller terms
    "sent",         # q_cmd_pc: the intent actually sent
    "echo",         # last_cmd_seq: the freshest command the servo consumed
    "measured",     # q_rt / dq_rt: measured state
)

#: Pair phases that mean the pair was actually admitted. STAGING is a request
#: waiting for member ACKs, not an admission; only READY (both ACKed) and
#: EXECUTING (released) prove the pair was taken up.
ADMITTED_PHASES = ("ready", "executing")

#: (link, per-arm sample field) for the links that must belong to the SAME
#: selected reference as the joined reference link.
CHAIN_FIELDS = (("raw", "tau_ff_pc"), ("sent", "q_cmd_pc"),
                 ("echo", "last_cmd_seq"), ("measured", "q_rt"))


def reconstruct_lineage(samples, *, pair_id=None):
    """Correlate one execution chain from the bounded samples a trace kept.

    A chain is a JOIN on identities, never field presence, and it belongs to
    the whole pair. Inside the admitted span (``ADMITTED_PHASES``; STAGING is a
    request, not an admission):

    * the reference link exists only once EVERY declared member has been seen
      on its own declared ``pair_plans`` entry — one member joining is not the
      pair's selected reference;
    * every command link comes from a JOINED arm only, so an arm still running
      a foreign plan can never supply a link the joined arm cannot;
    * command links fall at or after that reference tick;
    * a foreign-plan arm anywhere in the span, or a pair that was never
      admitted, makes the report incomplete rather than a fabricated chain.

    Read-only detached analysis: this opens no transport and takes only the
    recorded samples. A returned chain is a report, never a safe executable
    trajectory — a command log is not automatically executable, and any
    approved physical replay goes through fresh scene/model/start/limits/
    operator admission as a new request. The bounded trace keeps only the
    newest ticks, so an evicted window reports fewer ticks, never a
    fabricated success.
    """
    kept = [s for s in samples if s.get("pair_id") is not None]
    if pair_id is None:
        pair_id = kept[-1]["pair_id"] if kept else None
    window = [s for s in kept if s.get("pair_id") == pair_id] if pair_id else []
    # The pair's declared member plans are the identity the reference joins on.
    declared = {}
    for sample in window:
        for arm, plan in (sample.get("pair_plans") or {}).items():
            if plan is not None:
                declared.setdefault(arm, plan)
    # The chain lives in the pair's ADMITTED span: before staging and after a
    # terminal phase the controllers may legitimately hold another plan, and
    # those ticks are not this pair's lineage.
    admitted = [i for i, s in enumerate(window)
                if s.get("lifecycle") in ADMITTED_PHASES]
    links, foreign, joined = {}, set(), set()
    if window:
        links["requested"] = 0
    if admitted:
        first, last = admitted[0], admitted[-1]
        links["admitted"] = first
        for index in range(first, last + 1):
            arms = window[index].get("arms") or {}
            joined_here = {arm for arm, view in arms.items()
                           if view.get("plan_id")
                           and declared.get(arm) == view.get("plan_id")}
            foreign.update(arm for arm, view in arms.items()
                           if view.get("plan_id")
                           and declared.get(arm) != view.get("plan_id"))
            if not joined_here:
                continue
            joined |= joined_here
            # The reference link is the pair's: it exists only once EVERY
            # declared member has been seen on its own declared plan, so a
            # pair with one member still (or forever) elsewhere cannot claim
            # a selected reference.
            if joined == set(declared) and "reference" not in links:
                links["reference"] = index
            if "reference" not in links:
                continue
            # Command links come from a JOINED arm only. An arm still running a
            # foreign plan contributes nothing to this chain — its data must
            # never satisfy a link the joined arm cannot.
            for name, field in CHAIN_FIELDS:
                if name not in links and any(arms[arm].get(field) is not None
                                             for arm in joined_here):
                    links[name] = index
    missing = [name for name in LINE_LINKS if name not in links]
    # A foreign arm inside the admitted span means the pair/controller binding
    # was not clean for this window; report incomplete rather than a chain that
    # silently ignored it.
    complete = bool(window) and not missing and not foreign
    return dict(
        pair_id=pair_id,
        session=window[0].get("pc_monotonic") if window else None,
        ticks=len(window),
        admitted_ticks=len(admitted),
        declared_plans=dict(declared),
        joined_arms=sorted(joined),
        foreign_arms=sorted(foreign),
        complete=complete,
        links=links,
        missing=missing,
    )
