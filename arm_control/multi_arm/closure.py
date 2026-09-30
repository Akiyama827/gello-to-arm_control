"""The five closure quantities of a coordinated pair, logged separately (CTRL-041).

A monitor number that says "closure is small" hides which of five different things it measured. This
module builds ONE versioned record per tick with each quantity in its own slot, every present value carrying
its frame/model/time basis and every missing one carrying the reason it is missing (never a zero or a stale
value):

  planned_reference     closure/path error of the PLAN itself (geometric, from the planner's validation)
  rt_selected_reference closure of the reference the RT actually selected and runs (native reference only)
  tracking              per-arm tracking error: measured against the command/reference, per joint
  common_time_inferred  TCP agreement of the two arms inferred at a COMMON time from bracketed RT states
  physical              independent physical accuracy - present only when independently measured

Motion and stationary windows are labelled (a caller reports them separately; ``stationary_for_s`` lets an
analysis split settle from hold without this module inventing a threshold).

This is a record, not a monitor: it changes no threshold and gates nothing (offline / shadow, per the design).
Calibration correction is never inferred from a controller making its own modeled closure small, and captured
grasp transforms stay fixed during an operation. RT timestamps label published states, not capture times.
"""
from __future__ import annotations

import math

CLOSURE_SCHEMA = "arm_control.closure/v1"
QUANTITIES = ("planned_reference", "rt_selected_reference", "tracking", "common_time_inferred", "physical")
ASSUMPTIONS = (
    "RT state timestamps are publication times on the RT host clock, not robot capture times",
    "common-time values are interpolated between two published states (no extrapolation) and only when "
    "both arms share one attested RT clock domain",
    "no calibration is inferred from a controller's own modeled closure; grasp transforms stay fixed",
    "this record changes no threshold and gates nothing",
)


def absent(reason: str) -> dict:
    return {"value": None, "absent": reason}


def present(value, **basis) -> dict:
    return {"value": value, "basis": basis}


def _finite(x) -> bool:
    return x is not None and math.isfinite(float(x))


def planned_reference(metrics) -> dict:
    if not metrics:
        return absent("no cooperative plan metrics for this motion")
    keys = ("closure_m", "closure_rad", "path_m", "path_rad")
    if not all(_finite(metrics.get(k)) for k in keys):
        return absent("plan metrics are missing or not finite")
    return present({k: float(metrics[k]) for k in keys}, frame="object/TCP, planning kinematics",
                   model="planner world model", time="geometric maxima over the whole plan (no time base)")


def rt_selected_reference(reference) -> dict:
    """``reference``: None (streamed mode / no native reference) or dict(m=, rad=, reference_ids=, time_ns=)."""
    if reference is None:
        return absent("no RT-selected reference (streamed mode or no native reference selected)")
    if not (_finite(reference.get("m")) and _finite(reference.get("rad"))):
        return absent("RT-selected reference closure is not finite")
    return present({"m": float(reference["m"]), "rad": float(reference["rad"]),
                    "reference_ids": dict(reference.get("reference_ids") or {})},
                   frame="object/TCP, kinematics of the selected references", model="RT-selected references",
                   time=f"RT reference phase, common time {reference.get('time_ns')} ns (RT host clock)")


def tracking(arms) -> dict:
    """``arms``: per arm dict with q_rt, q_cmd_rt, q_cmd_pc (lists or None), as in ``diagnostic_sample``."""
    out = {}
    for name, arm in (arms or {}).items():
        entry = {}
        for label, key in (("vs_rt_command_rad", "q_cmd_rt"), ("vs_pc_command_rad", "q_cmd_pc")):
            q, ref = arm.get("q_rt"), arm.get(key)
            if q is None or ref is None or len(q) != len(ref):
                entry[label] = None
            else:
                err = [abs(float(a) - float(b)) for a, b in zip(q, ref)]
                entry[label] = max(err) if err and all(math.isfinite(e) for e in err) else None
        entry["state_seq"] = arm.get("state_seq")
        entry["state_t_mono_ns"] = arm.get("state_t_mono_ns")
        out[name] = entry
    if not out or all(all(v is None for k, v in e.items() if k.startswith("vs_")) for e in out.values()):
        return absent("no measured state and command for any arm")
    return present(out, frame="joint space, per arm", model="none (raw joint values)",
                   time="each arm's own latest published state (not time-aligned across arms)")


def common_time_inferred(errors, alignment) -> dict:
    """``errors``: (distance_m, angle_rad) TCP agreement or None; ``alignment``: align_feedback report or None."""
    if errors is None:
        return absent("not inferred (independent mode, pair not aligned, or arms not armed)")
    if not isinstance(alignment, dict) or "brackets" not in alignment:
        return absent("no alignment report for this estimate: " + str((alignment or {}).get("error", "missing")))
    d, a = errors
    if not (_finite(d) and _finite(a)):
        return absent("inferred agreement is not finite")
    brackets = {arm: {k: b[k] for k in ("before_seq", "after_seq", "before_t_mono_ns", "after_t_mono_ns", "fraction")}
                for arm, b in alignment["brackets"].items()}
    return present({"distance_m": float(d), "angle_rad": float(a)},
                   frame="TCP agreement in the pair's shared object frame",
                   model="kinematic model + fixed grasp transforms",
                   time=f"common time {alignment.get('t_mono_ns')} ns on RT clock domain "
                        f"{alignment.get('clock_domain', 'unattested')}",
                   latest_skew_s=alignment.get("latest_skew_s"), brackets=brackets)


def physical(measurement) -> dict:
    if not measurement:
        return absent("no independent measurement (a mocap/camera measurement is required)")
    return present(measurement.get("value"), source=measurement.get("source"), frame=measurement.get("frame"),
                   time=measurement.get("time"))


def closure_record(*, moving: bool, stationary_for_s=None, planned=None, rt_reference=None, arms=None,
                   common_time=None, alignment=None, independent=None) -> dict:
    """One tick's record. ``moving`` labels the window (motion vs stationary)."""
    return {
        "schema": CLOSURE_SCHEMA,
        "window": "motion" if moving else "stationary",
        "stationary_for_s": None if moving or not _finite(stationary_for_s) else float(stationary_for_s),
        "planned_reference": planned_reference(planned),
        "rt_selected_reference": rt_selected_reference(rt_reference),
        "tracking": tracking(arms),
        "common_time_inferred": common_time_inferred(common_time, alignment),
        "physical": physical(independent),
        "assumptions": list(ASSUMPTIONS),
    }
