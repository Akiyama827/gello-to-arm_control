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


#: Capacity policy for persisted fault traces (CTRL-051): nothing is ever deleted automatically (a trace can be the only
#: record of a real fault); when the directory holds more than ``RETAIN_FILES`` traces or ``RETAIN_BYTES`` bytes every save
#: prints a visible warning, and the operator exports and prunes explicitly (``trace_export --prune``, dry-run by default).
#: A trace is ~5 MB (500 samples). Adjustable defaults, not safety constants.
RETAIN_FILES = 20
RETAIN_BYTES = 256 * 2**20
FAULT_DIR = ("dual_arm", "faults")


def fault_root():
    return CONTROL_ROOT.joinpath(".cache", *FAULT_DIR)


TRACE_GLOB = "2*.json"  # timestamped traces only: an EXPORT_MANIFEST.json in the directory is not a trace


def _traces(root):
    return sorted(p for p in root.glob(TRACE_GLOB) if p.is_file() and not p.is_symlink())


def capacity_warning(root, *, keep_files=RETAIN_FILES, keep_bytes=RETAIN_BYTES):
    """A message when the trace directory is over its capacity policy, else None. Never raises."""
    try:
        traces = _traces(root)
        total = sum(p.stat().st_size for p in traces) + sum(p.stat().st_size for p in root.glob("*.partial"))
    except OSError:
        return None
    if len(traces) > keep_files or total > keep_bytes:
        return (f"{len(traces)} fault trace(s), {total / 2**20:.1f} MiB in {root} exceed the capacity policy "
                f"({keep_files} files / {keep_bytes / 2**20:.0f} MiB): export and prune explicitly "
                f"(export first: python -m arm_control.multi_arm.trace_export --dest DIR; then --prune --yes --confirm-exported DIR); "
                f"nothing is deleted automatically")
    return None


def prune_traces(root, *, keep_files=RETAIN_FILES, keep_bytes=RETAIN_BYTES, protect=None, dry_run=False, only=None):
    """Remove the oldest ``*.json`` traces (and their ``.sha256`` sidecars) beyond the limits. Called only by an
    explicit operator action. Never raises, never removes ``protect`` and, when ``only`` is given, nothing outside that
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


class PairFaultTrace:
    """Last 500 console ticks; no disk I/O or JSON encoding in record().

    Overflow is explicit: the number of samples evicted from the bounded history is counted and written into the
    persisted trace (``dropped_samples``), never silent. A trace is written atomically (temporary file, fsync, rename)
    with a SHA-256 sidecar, so a full disk or a crash never leaves a partial file that looks complete, and old traces
    are never removed by a save: exceeding the capacity policy prints a warning (``RETAIN_FILES`` / ``RETAIN_BYTES``).
    """

    def __init__(self, session):
        self.session = session
        self.samples = deque(maxlen=500)
        self.recorded = 0  # every sample ever recorded; recorded - len(samples) = evicted

    def record(self, sample):
        self.samples.append(sample)
        self.recorded += 1

    def save(self, reason, context, *, samples=None, root=None, recorded=None):
        now = datetime.now(timezone.utc)
        root = fault_root() if root is None else root
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{now:%Y%m%dT%H%M%S%fZ}-{self.session}.json"
        kept = list(self.samples) if samples is None else samples
        seen = self.recorded if recorded is None else recorded
        report = dict(
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
            samples_kept=len(kept),
            dropped_samples=max(0, seen - len(kept)),
            retention=dict(keep_files=RETAIN_FILES, keep_bytes=RETAIN_BYTES),
            context=context,
            samples=kept,
        )
        payload = (json.dumps(_json_values(report), allow_nan=False) + "\n").encode()
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
        warning = capacity_warning(root)
        if warning:
            print(f"[fault trace] WARNING: {warning}", flush=True)
        return path


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
                           if view.get("plan_id") is not None
                           and declared.get(arm) == view.get("plan_id")}
            foreign.update(arm for arm, view in arms.items()
                           if view.get("plan_id") is not None
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
