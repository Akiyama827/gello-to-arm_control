"""Bracketed joint estimates at a common RT publication timestamp.

This is used only for the relative TCP monitor, never for controller feedback.
RT publication time is not robot capture time. Different RT hosts need clock
synchronization before this comparison is meaningful and are refused here.
"""

import numpy as np


class FeedbackGap(ValueError):
    """Fresh packets exist, but their spacing cannot support interpolation."""

    def __init__(self, message, **details):
        super().__init__(message)
        self.details = details


def align_feedback(links, histories, *, now, max_age_s=.05, max_interval_s=.02):
    """Return common-time positions and the raw brackets that produced them.

    No extrapolation, timestamp rewriting, filtering or threshold adjustment.
    Latest packets must be fresh and finite. Brackets must belong to the same
    authority epoch as the latest packet and span at most 20 ms by default.
    """
    if len(links) != 2 or set(links) != set(histories):
        raise ValueError("paired feedback requires both arms")
    # CTRL-041: an attested clock domain (host + boot identity from the RT HELLO) is the
    # proof of one CLOCK_MONOTONIC; a host string only names where we connected. Cross-host
    # mapping is not attempted: different or half-attested domains are refused.
    domains = {a: getattr(link, "clock_domain", None) for a, link in links.items()}
    if any(d is not None for d in domains.values()):
        if any(d is None for d in domains.values()):
            raise ValueError("paired feedback: RT clock domain is attested on one arm only")
        if len(set(domains.values())) != 1:
            raise ValueError("paired feedback requires one RT clock domain (host and boot)")
    elif len({link.config.host for link in links.values()}) != 1:
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
        skew = (max(stamps) - target) * 1e-9
        raise FeedbackGap(
            f"paired feedback publication skew {skew * 1000:.3f} ms "
            f"exceeds {max_interval_s * 1000:g} ms",
            latest_skew_s=skew, limit_s=max_interval_s, target_t_mono_ns=target,
        )
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
            raise FeedbackGap(
                f"{a}: paired feedback interpolation gap {interval * 1000:.3f} ms "
                f"exceeds {max_interval_s * 1000:g} ms",
                arm_id=a, interval_s=interval, limit_s=max_interval_s,
                target_t_mono_ns=target,
                before_t_mono_ns=before.t_mono_ns, after_t_mono_ns=after.t_mono_ns,
                before_seq=before.state_seq, after_seq=after.state_seq,
            )
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
        clock_domain=("unattested" if domains[next(iter(domains))] is None
                      else ":".join(domains[next(iter(domains))])),
        t_mono_ns=target, latest_skew_s=(max(stamps) - target) * 1e-9,
        brackets=brackets,
    )
