"""Bounded PC observation history, persisted only after a fault requests stop.

RT timestamps label published server states, not synchronized physical capture
times. Separate RT hosts must not have their timestamps directly subtracted.
"""

from collections import deque
from datetime import datetime, timezone
import json
import math

import numpy as np

from arm_control import CONTROL_ROOT


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


class PairFaultTrace:
    """Last 500 console ticks; no disk I/O or JSON encoding in record()."""

    def __init__(self, session):
        self.session = session
        self.samples = deque(maxlen=500)

    def record(self, sample):
        self.samples.append(sample)

    def save(self, reason, context, *, samples=None):
        now = datetime.now(timezone.utc)
        root = CONTROL_ROOT / ".cache" / "dual_arm" / "faults"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{now:%Y%m%dT%H%M%S%fZ}-{self.session}.json"
        report = dict(
            schema="arm_control.dual_fault/v2",
            schema_version=2,
            closure_schema="arm_control.closure/v1",  # samples carry `closure_v1` beside the older `closure`
            observed_at_utc=now.isoformat(),
            reason=reason,
            session=self.session,
            timestamp_note=(
                "PC receipt/tick times are not physical capture times. RT state "
                "timestamps share a clock only when their attested clock_domain matches "
                "(rt_host string equality is the fallback for unattested servers); "
                "publication timestamps do not establish synchronized robot measurements."
            ),
            context=context,
            samples=list(self.samples) if samples is None else samples,
        )
        with path.open("x") as stream:
            json.dump(_json_values(report), stream, allow_nan=False)
            stream.write("\n")
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
