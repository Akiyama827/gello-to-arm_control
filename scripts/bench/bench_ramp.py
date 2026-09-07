#!/usr/bin/env python3
"""Gain-ramp helper for the hardware bring-up ladder (rung 4).

The DM motors have never been servoed in MIT mode, so the gain table is found by
climbing: start soft (kp=20, kd=0.5 -- the user-tested per-motor anchor, ~30x
below the sim 600/60 the DM wire format can't carry), gravity-hold, take small
kp steps, and stop where tracking is crisp but the grasp stays compliant. This
script computes and PRINTS that ramp schedule (dry-run, offline, the default),
and can WRITE a chosen rung's gains back into the calibration table.

Commanding the motors at a rung is NOT done here -- that needs the live hardware
graph (``python scripts/view.py real grasp``, armed, operator on the kill
switch). ``--command`` only prints the guidance; the ramp-schedule computation
and the gain write are pure and offline-testable.

Examples
--------
  python scripts/bench_ramp.py                         # print the ramp schedule
  python scripts/bench_ramp.py --kp-target 160 --steps 6
  python scripts/bench_ramp.py --write-rung 3          # commit rung 3 to arm.kp/arm.kd
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml

CONTROL_ROOT = Path(__file__).resolve().parents[1]
# The gain table lives in the MEASURED half of the split config — this script is
# one of its writers, and write_gains() edits the file that literally contains
# the `arm.kp` lines (not the include stub, which has no such lines).
DEFAULT_CALIBRATION = CONTROL_ROOT / "configs" / "real" / "assembler" / "calibration.yaml"
# DM MIT encode ceiling (shared kp/kd range across DM4310/4340); see gains.py.
# Wire-format facts, so they stay with this DM-only writer.
KP_MAX = 500.0
KD_MAX = 5.0


def _fmt(v: float) -> str:
    return f"{float(v):g}"


def _flow(values) -> str:
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def replace_yaml_list(text: str, key: str, values, under: str | None = None) -> str | None:
    """Replace the value of an existing ``key: ...`` line with a flow list.

    With ``under`` set, only lines inside that top-level block mapping are
    candidates (the block ends at the next non-blank, non-comment line at
    indent 0) — an unscoped search would take the FIRST ``kp:`` in the file at
    any indent, e.g. a mode config's controller table. Preserves every other
    line (and its comment); only the matched line is rewritten, dropping its
    own trailing comment (a filled value is no longer a placeholder). Returns
    the new text, or ``None`` if the block or key line is absent.
    """
    start, end = 0, len(text)
    indent = r"[ \t]*"
    if under is not None:
        block = re.search(rf"^{re.escape(under)}:{indent}(#.*)?$", text, re.MULTILINE)
        if block is None:
            return None
        start = block.end()
        nxt = re.compile(r"^[^\s#]", re.MULTILINE).search(text, pos=start)
        end = nxt.start() if nxt else len(text)
        indent = r"[ \t]+"  # inside a block the key is necessarily indented
    seg = text[start:end]
    pat = re.compile(rf"^(?P<indent>{indent}){re.escape(key)}:.*$", re.MULTILINE)
    m = pat.search(seg)
    if m is None:
        return None
    seg = seg[: m.start()] + f"{m.group('indent')}{key}: {_flow(values)}" + seg[m.end():]
    return text[:start] + seg + text[end:]


def ramp_schedule(
    kp_start: float, kd_start: float, kp_target: float, kd_target: float, steps: int = 5
) -> list[tuple[float, float]]:
    """Monotonic linear (kp, kd) ramp from start to target, inclusive.

    ``steps`` rungs; rung 0 is exactly the start, rung ``steps-1`` the target.
    Requires ``steps >= 2`` and ``target >= start`` (a ramp climbs).
    """
    if steps < 2:
        raise ValueError("steps must be >= 2")
    if kp_target < kp_start or kd_target < kd_start:
        raise ValueError("ramp targets must be >= start (the ramp climbs)")
    if not (kp_target <= KP_MAX and kd_target <= KD_MAX):
        raise ValueError(f"target out of DM encode range (kp<={KP_MAX:g}, kd<={KD_MAX:g})")
    out = []
    for i in range(steps):
        frac = i / (steps - 1)
        out.append((kp_start + (kp_target - kp_start) * frac, kd_start + (kd_target - kd_start) * frac))
    return out


def _arm_gains(calibration_path: Path) -> tuple[list[float], list[float]]:
    """The calibration file's current ``arm.kp`` / ``arm.kd`` lists.

    Their length IS the arm's joint count for the writer — the table being
    replaced is the shape authority, so a 7-joint arm's file never gets a
    6-vector. Raises if either list is absent (this file is the measured gain
    table; an empty one is a setup error, not a default).
    """
    cfg = yaml.safe_load(calibration_path.read_text()) or {}
    arm = cfg.get("arm") or {}
    kp, kd = arm.get("kp"), arm.get("kd")
    if not kp or not kd:
        raise KeyError(f"arm.kp / arm.kd not found in {calibration_path}")
    return list(kp), list(kd)


def write_gains(calibration_path, kp: float, kd: float) -> None:
    """Set ``arm.kp``/``arm.kd`` to ``[kp]*n`` / ``[kd]*n``, preserving comments."""
    if not (0.0 <= kp <= KP_MAX and 0.0 <= kd <= KD_MAX):
        raise ValueError(f"kp={kp:g}/kd={kd:g} out of DM encode range")
    path = Path(calibration_path)
    n = len(_arm_gains(path)[0])
    text = path.read_text()
    for key, value in (("kp", [kp] * n), ("kd", [kd] * n)):
        updated = replace_yaml_list(text, key, value, under="arm")
        if updated is None:
            raise KeyError(f"arm.{key} not found in {path}")
        text = updated
    path.write_text(text)


def _start_gains(calibration_path: Path) -> tuple[float, float]:
    try:
        kp, kd = _arm_gains(calibration_path)
    except KeyError:
        return 20.0, 0.5
    return float(kp[0]), float(kd[0])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION,
                    help="calibration YAML holding the arm.kp/arm.kd table")
    ap.add_argument("--kp-start", type=float, help="ramp start kp (default: arm.kp[0] or 20)")
    ap.add_argument("--kd-start", type=float, help="ramp start kd (default: arm.kd[0] or 0.5)")
    ap.add_argument("--kp-target", type=float, default=120.0, help="ramp end kp")
    ap.add_argument("--kd-target", type=float, default=3.0, help="ramp end kd")
    ap.add_argument("--steps", type=int, default=5, help="number of rungs (>=2)")
    ap.add_argument("--write-rung", type=int, help="commit this rung's gains (1-indexed) to the calibration file")
    ap.add_argument("--command", action="store_true", help="print live-commanding guidance (requires the hardware graph)")
    args = ap.parse_args(argv)

    start = _start_gains(args.calibration) if args.calibration.exists() else (20.0, 0.5)
    kp0 = args.kp_start if args.kp_start is not None else start[0]
    kd0 = args.kd_start if args.kd_start is not None else start[1]
    schedule = ramp_schedule(kp0, kd0, args.kp_target, args.kd_target, args.steps)

    print(f"gain ramp  kp {kp0:g}->{args.kp_target:g}  kd {kd0:g}->{args.kd_target:g}  ({args.steps} rungs)")
    for i, (kp, kd) in enumerate(schedule, start=1):
        print(f"  rung {i}: kp={kp:7.2f}  kd={kd:5.2f}")

    if args.command:
        print(
            "\n-- commanding a rung is NOT automated: launch `python scripts/view.py real grasp`,\n"
            "   arm with an operator on the kill switch, and watch wrist droop / tracking.\n"
            "   Commit the rung that holds with `--write-rung N`, then re-run the sim shadow gate (rung 5)."
        )
        return 0

    if args.write_rung is not None:
        if not (1 <= args.write_rung <= len(schedule)):
            ap.error(f"--write-rung must be 1..{len(schedule)}")
        kp, kd = schedule[args.write_rung - 1]
        write_gains(args.calibration, kp, kd)
        print(f"\nwrote rung {args.write_rung} (kp={kp:g}, kd={kd:g}) -> arm.kp/arm.kd in {args.calibration}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
