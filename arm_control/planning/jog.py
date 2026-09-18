"""The safety envelope around a jog: what a held direction button may do.

A jog is the one motion in this package that is NOT planned. Plan+Execute goes
through OMPL against a collision oracle and an operator reviews the result
before anything moves; a jog moves while a finger is down, one small step at a
time, with nothing between the operator and the arm. So the envelope has to be
checked per step instead of once per plan.

Modelled on a materials-test machine's crosshead limits, which is the closest
well-worn analogue: you say "down at 1 cm/s", and the machine also wants to
know how far it may travel before it stops itself. Five gates, cheapest first,
so a refused step costs almost nothing:

1. **Stroke limit** -- how far from where this jog STARTED, per axis. The
   anchor latches when the button goes down and clears when it comes up, so
   "down 20 cm" means twenty centimetres from where the operator began, not an
   unbounded descent made of individually-legal 1 mm steps.
2. **Workspace box** -- absolute bounds, of which the floor is the one every
   cell has. Nothing else about the environment is assumed here.
3. **Joint limits**, with a margin -- but only against steps that make a
   violation WORSE. An arm parked at (or past) a limit must always be able to
   jog away from it, or the escape hatch is locked from the inside; this was
   found on a real spawn pose whose third joint sits exactly at its hard stop,
   where a naive check refused every direction including the useful one.
4. **Singularity** -- sigma_min of the Jacobian. This is the failure a velocity
   jog walks into that a planner never sees: the operator asks for 1 cm/s and a
   near-singular wrist answers with an unbounded joint rate.
5. **Collision** -- self-collision, via the caller's oracle. Last because it is
   by far the most expensive, and a step that already failed a cheaper gate
   never needs it.

Pure: no MuJoCo, no Pinocchio, no Dora. The two expensive facts (sigma_min and
collision) arrive as callables, so this whole envelope is testable -- and IS
tested below -- without a robot, a model, or a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = ["JogLimits", "JogVerdict", "check_step"]


@dataclass(frozen=True)
class JogLimits:
    """One arm's jog envelope. Every field is a number an operator can defend."""

    #: Stroke limit: metres from the jog anchor, per world axis.
    max_travel_m: float = 0.20
    #: Translation speed while a direction is held (m/s). 1 cm/s is a sane
    #: default for a first bench session; the config raises it deliberately.
    speed_m_s: float = 0.01
    #: Rotation speed while a rotation direction is held (rad/s).
    rot_speed_rad_s: float = 0.10
    #: World z of the surface under the arm, and how close the EE may get.
    #: None disables the floor gate -- for an arm that legitimately works below
    #: its own base frame, and it must be a decision, not an oversight.
    floor_z: float | None = 0.0
    floor_clearance_m: float = 0.02
    #: Absolute workspace box, or None per side to leave that bound open.
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    #: Stop this far short of a hard joint limit.
    joint_margin_rad: float = 0.05
    #: Refuse below this smallest singular value of the translational Jacobian.
    sigma_min: float = 0.02

    def __post_init__(self) -> None:
        for name in ("max_travel_m", "speed_m_s", "rot_speed_rad_s"):
            if not (float(getattr(self, name)) > 0.0):
                raise ValueError(f"jog {name} must be positive")
        for name in ("floor_clearance_m", "joint_margin_rad", "sigma_min"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"jog {name} must not be negative")


@dataclass(frozen=True)
class JogVerdict:
    ok: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


def check_step(
    p_next: Sequence[float],
    q_next: Sequence[float],
    *,
    anchor: Sequence[float],
    limits: JogLimits,
    q_lower: Sequence[float],
    q_upper: Sequence[float],
    q_now: Sequence[float] | None = None,
    sigma_min: Callable[[np.ndarray], float] | None = None,
    collides: Callable[[np.ndarray], bool] | None = None,
) -> JogVerdict:
    """Approve one jog step, or say in one sentence why not.

    The reason text goes straight to the operator's page, so it names the gate,
    the number that failed and the limit it failed against -- "refused" alone
    leaves someone holding a button that does nothing.
    """
    p_next = np.asarray(p_next, dtype=float).ravel()
    anchor = np.asarray(anchor, dtype=float).ravel()
    q_next = np.asarray(q_next, dtype=float).ravel()

    # 1. stroke limit, from where THIS jog started
    travel = np.abs(p_next - anchor)
    if float(travel.max()) > limits.max_travel_m:
        axis = "xyz"[int(np.argmax(travel))]
        return JogVerdict(
            False,
            f"stroke limit: {travel.max() * 100:.1f} cm from the jog origin on "
            f"{axis}, limit {limits.max_travel_m * 100:.0f} cm — release and "
            f"press again to re-anchor",
        )

    # 2. workspace box; the floor is its z-minimum and the only bound assumed
    if limits.floor_z is not None:
        floor = limits.floor_z + limits.floor_clearance_m
        if p_next[2] < floor:
            return JogVerdict(
                False,
                f"floor: z {p_next[2] * 100:.1f} cm is below the {floor * 100:.1f} cm "
                f"limit (floor {limits.floor_z * 100:.1f} cm + "
                f"{limits.floor_clearance_m * 100:.1f} cm clearance)",
            )
    for bound, sense in ((limits.workspace_min, -1), (limits.workspace_max, 1)):
        if bound is None:
            continue
        b = np.asarray(bound, dtype=float).ravel()
        bad = np.where((p_next - b) * sense < 0.0)[0]
        if bad.size:
            i = int(bad[0])
            side = "min" if sense < 0 else "max"
            return JogVerdict(
                False,
                f"workspace {side}: {'xyz'[i]} {p_next[i] * 100:.1f} cm past "
                f"{b[i] * 100:.1f} cm",
            )

    # 3. joint limits, with room left to jog back out. A joint ALREADY outside
    #    the usable band only blocks a step that pushes it further out -- an
    #    arm resting on a hard stop (some spawn poses do) must still be able to
    #    jog off it, and refusing every direction there strands the operator.
    lo = np.asarray(q_lower, dtype=float).ravel() + limits.joint_margin_rad
    hi = np.asarray(q_upper, dtype=float).ravel() - limits.joint_margin_rad
    now = None if q_now is None else np.asarray(q_now, dtype=float).ravel()
    for i in np.where((q_next < lo) | (q_next > hi))[0]:
        i = int(i)
        if now is not None:
            over_next = max(lo[i] - q_next[i], q_next[i] - hi[i], 0.0)
            over_now = max(lo[i] - now[i], now[i] - hi[i], 0.0)
            if over_next <= over_now:
                continue  # moving back toward the band, or no worse: allow it
        return JogVerdict(
            False,
            f"joint limit: joint {i} at {q_next[i]:.3f} rad, usable range "
            f"[{lo[i]:.3f}, {hi[i]:.3f}] (hard limit minus a "
            f"{limits.joint_margin_rad:.2f} rad margin)",
        )

    # 4. singularity -- the gate a planned move never needs
    if sigma_min is not None:
        sigma = float(sigma_min(q_next))
        if sigma < limits.sigma_min:
            return JogVerdict(
                False,
                f"singularity: sigma_min {sigma:.4f} below {limits.sigma_min:.4f} "
                f"— jog a joint directly to back out of it",
            )

    # 5. collision, last because it is the expensive one
    if collides is not None and collides(q_next):
        return JogVerdict(False, "collision: the arm would hit itself")

    return JogVerdict(True)


def _self_check() -> None:
    """Each gate fires on its own case, and a legal step passes all five."""
    limits = JogLimits(max_travel_m=0.20, floor_z=0.0, floor_clearance_m=0.02)
    lo, hi = np.full(6, -3.0), np.full(6, 3.0)
    anchor = np.array([0.4, 0.0, 0.5])
    q = np.zeros(6)

    def ok(p, qn=q, **kw):
        return check_step(p, qn, anchor=anchor, limits=limits,
                          q_lower=lo, q_upper=hi, **kw)

    # a small step inside every bound passes
    good = ok([0.42, 0.0, 0.5])
    assert good.ok, good.reason

    # 1. stroke: 25 cm from the anchor, limit 20
    v = ok([0.65, 0.0, 0.5])
    assert not v.ok and "stroke limit" in v.reason, v.reason
    # ...and it is measured from the ANCHOR, not from the previous step: a
    # descent made of legal 1 mm steps must still stop at the limit.
    p = anchor.copy()
    for _ in range(400):
        cand = p + np.array([0.0, 0.0, -0.001])
        if not ok(cand).ok:
            break
        p = cand
    assert abs(p[2] - anchor[2]) <= limits.max_travel_m + 1e-9, p
    assert abs(p[2] - anchor[2]) > 0.19, "the walk should reach the stroke limit"

    # 2. floor: below floor_z + clearance
    v = check_step([0.4, 0.0, 0.01], q, anchor=[0.4, 0.0, 0.05], limits=limits,
                   q_lower=lo, q_upper=hi)
    assert not v.ok and "floor" in v.reason, v.reason
    # the clearance is what makes it fire: 0.03 m is above floor+0.02
    v = check_step([0.4, 0.0, 0.03], q, anchor=[0.4, 0.0, 0.05], limits=limits,
                   q_lower=lo, q_upper=hi)
    assert v.ok, v.reason
    # and it can be disabled, but only deliberately
    open_floor = JogLimits(floor_z=None)
    v = check_step([0.4, 0.0, -5.0], q, anchor=[0.4, 0.0, -5.0], limits=open_floor,
                   q_lower=lo, q_upper=hi)
    assert v.ok, v.reason

    # 3. joint limit, with the margin doing the work
    near = np.zeros(6)
    near[2] = 2.97          # inside the hard limit, inside the 0.05 margin
    v = ok([0.42, 0.0, 0.5], near)
    assert not v.ok and "joint limit" in v.reason, v.reason

    # ...but a joint already ON its stop may still move AWAY from it. This is
    #    the case that locked a real spawn pose out of every jog direction.
    pinned = np.zeros(6)
    pinned[2] = 3.0                       # exactly at the hard limit
    worse = pinned.copy()
    worse[2] = 3.001
    better = pinned.copy()
    better[2] = 2.99
    assert not ok([0.42, 0.0, 0.5], worse, q_now=pinned).ok, "outward step allowed"
    v = ok([0.42, 0.0, 0.5], better, q_now=pinned)
    assert v.ok, f"an arm on its stop could not jog off it: {v.reason}"
    # without q_now there is nothing to compare against, so it stays strict
    assert not ok([0.42, 0.0, 0.5], better).ok

    # 4. singularity
    v = ok([0.42, 0.0, 0.5], sigma_min=lambda _q: 0.001)
    assert not v.ok and "singularity" in v.reason, v.reason
    assert ok([0.42, 0.0, 0.5], sigma_min=lambda _q: 0.5).ok

    # 5. collision, and the ordering: a step that fails a CHEAP gate must never
    #    reach the expensive one.
    reached = []
    v = ok([0.42, 0.0, 0.5], collides=lambda _q: True)
    assert not v.ok and "collision" in v.reason, v.reason
    ok([0.65, 0.0, 0.5], collides=lambda _q: reached.append(1) or True)
    assert not reached, "a stroke-limited step still ran the collision oracle"

    # a nonsense envelope is refused at construction, not at the arm
    for bad_kw in ({"max_travel_m": 0.0}, {"speed_m_s": -1.0}, {"sigma_min": -0.1}):
        try:
            JogLimits(**bad_kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"JogLimits accepted {bad_kw}")

    print("planning.jog self-check ok")


if __name__ == "__main__":
    _self_check()
