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
