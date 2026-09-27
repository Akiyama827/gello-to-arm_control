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
            schema="arm_control.dual_fault/v1",
            observed_at_utc=now.isoformat(),
            reason=reason,
            session=self.session,
            timestamp_note=(
                "PC receipt/tick times are not physical capture times. RT state "
                "timestamps share a clock only when rt_host matches; publication "
                "timestamps do not establish synchronized robot measurements."
            ),
            context=context,
            samples=list(self.samples) if samples is None else samples,
        )
        with path.open("x") as stream:
            json.dump(_json_values(report), stream, allow_nan=False)
            stream.write("\n")
        return path


# The execution-chain links the dual_fault/v1 sample schema carries, in chain
# order (CTRL-051). A lineage report is complete exactly when every link is
# observed in the retained window.
LINE_LINKS = (
    "requested",    # pair_id: the plan identity the policy requested
    "admitted",     # lifecycle: the admitted phase the pair reached
    "reference",    # per-arm plan_id + elapsed: the selected reference
    "raw",          # tau_ff_pc / integral_bias_pc / raw_pc: controller terms
    "sent",         # q_cmd_pc: the intent actually sent
    "echo",         # last_cmd_seq: the freshest command the servo consumed
    "measured",     # q_rt / dq_rt: measured state
)


def reconstruct_lineage(samples, *, pair_id=None):
    """Correlate one execution chain from the bounded samples a trace kept.

    Read-only detached analysis (CTRL-051): this opens no transport and takes
    only the recorded samples. A returned chain is a report, never a safe
    executable trajectory — a command log is not automatically executable,
    and any approved physical replay goes through fresh scene/model/start/
    limits/operator admission as a new request. The bounded trace keeps only
    the newest ticks, so an evicted window reports fewer ticks, never a
    fabricated success.
    """
    kept = [s for s in samples if s.get("pair_id") is not None]
    if pair_id is None:
        pair_id = kept[-1]["pair_id"] if kept else None
    window = [s for s in kept if s.get("pair_id") == pair_id] if pair_id else []
    links = {}
    for name in LINE_LINKS:
        for index, sample in enumerate(window):
            if name == "requested":
                hit = sample.get("pair_id") is not None
            elif name == "admitted":
                hit = sample.get("lifecycle") is not None
            elif name == "reference":
                arms = sample.get("arms") or {}
                hit = any(v.get("plan_id") is not None for v in arms.values())
            elif name == "raw":
                arms = sample.get("arms") or {}
                hit = any(v.get("tau_ff_pc") is not None for v in arms.values())
            elif name == "sent":
                arms = sample.get("arms") or {}
                hit = any(v.get("q_cmd_pc") is not None for v in arms.values())
            elif name == "echo":
                arms = sample.get("arms") or {}
                hit = any(v.get("last_cmd_seq") is not None for v in arms.values())
            else:  # measured
                arms = sample.get("arms") or {}
                hit = any(
                    v.get("q_rt") is not None and v.get("dq_rt") is not None
                    for v in arms.values()
                )
            if hit:
                links[name] = index
                break
    return dict(
        pair_id=pair_id,
        session=window[0].get("pc_monotonic") if window else None,
        ticks=len(window),
        complete=bool(window) and len(links) == len(LINE_LINKS),
        links=links,
        missing=[name for name in LINE_LINKS if name not in links],
    )
