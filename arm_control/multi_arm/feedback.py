"""Bracketed joint estimates at a common RT publication timestamp.

This is used only for the relative TCP monitor, never for controller feedback.
RT publication time is not robot capture time. Different RT hosts need clock
synchronization before this comparison is meaningful and are refused here.
"""

import numpy as np


def align_feedback(links, histories, *, now, max_age_s=.05, max_interval_s=.02):
    """Return common-time positions and the raw brackets that produced them.

    No extrapolation, timestamp rewriting, filtering or threshold adjustment.
    Latest packets must be fresh and finite. Brackets must belong to the same
    authority epoch as the latest packet and span at most 20 ms by default.
    """
    if len(links) != 2 or set(links) != set(histories):
        raise ValueError("paired feedback requires both arms")
    if len({link.config.host for link in links.values()}) != 1:
        raise ValueError("paired feedback requires one RT host clock")

    def epoch(state):
        return state.session_id, state.arm_epoch, state.stop_epoch

    def position(a, state):
        motor = links[a].motor_state(state=state)
        q, dq = (np.asarray(motor[key], dtype=float) for key in ("position", "velocity"))
        if any(v.shape != (state.n,) or not np.isfinite(v).all() for v in (q, dq)):
            raise ValueError(f"{a}: paired feedback contains invalid joints")
        return q

    latest = {}
    for a, history in histories.items():
        if not history:
            raise ValueError(f"{a}: paired feedback history missing")
        state, received = history[-1]
        authority = links[a].motor_health()
        if (state.session_id != int(authority["session_id"], 16)
                or (state.armed and state.arm_epoch != authority["arm_epoch"])):
            raise ValueError(f"{a}: paired feedback authority changed")
        if not np.isfinite(received) or not 0 <= now - received <= max_age_s:
            raise ValueError(f"{a}: paired feedback stale")
        if state.t_mono_ns <= 0 or state.faulted:
            raise ValueError(f"{a}: paired feedback timestamp invalid or plant faulted")
        if len(history) > 1 and history[-2][0].t_mono_ns >= state.t_mono_ns:
            raise ValueError(f"{a}: paired feedback timestamps are not increasing")
        position(a, state)  # Do not hide a rejected latest sample behind history.
        latest[a] = state
    stamps = [s.t_mono_ns for s in latest.values()]
    target = min(stamps)
    if (max(stamps) - target) * 1e-9 > max_interval_s:
        raise ValueError("paired feedback publication skew exceeds 20 ms")
    positions, brackets = {}, {}
    for a, history in histories.items():
        current = latest[a]
        before = after = None
        for state, received in reversed(history):
            if epoch(state) != epoch(current) or state.faulted:
                break
            if not 0 <= now - received <= max_age_s:
                break
            stamp = state.t_mono_ns
            if after is not None and stamp >= after.t_mono_ns:
                raise ValueError(f"{a}: paired feedback timestamps are not increasing")
            if stamp >= target:
                after = state
            if stamp <= target:
                before = state
                break
        if before is None or after is None or before.t_mono_ns <= 0:
            raise ValueError(f"{a}: paired feedback cannot bracket common time")
        interval = (after.t_mono_ns - before.t_mono_ns) * 1e-9
        if interval > max_interval_s:
            raise ValueError(f"{a}: paired feedback interpolation gap exceeds 20 ms")
        alpha = (0. if interval == 0 else
                 (target - before.t_mono_ns) / (after.t_mono_ns - before.t_mono_ns))
        q0, q1 = position(a, before), position(a, after)
        positions[a] = (1 - alpha) * q0 + alpha * q1
        brackets[a] = dict(
            before_t_mono_ns=before.t_mono_ns, after_t_mono_ns=after.t_mono_ns,
            before_seq=before.state_seq, after_seq=after.state_seq, fraction=alpha,
            q_before=q0.tolist(), q_after=q1.tolist(), q=positions[a].tolist(),
        )
    return positions, dict(
        rt_host=next(iter(links.values())).config.host,
        t_mono_ns=target, latest_skew_s=(max(stamps) - target) * 1e-9,
        brackets=brackets,
    )
