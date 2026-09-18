"""Identify the FR3's bench extrinsic and its joint reference offsets.

The rig is repeatable but several millimetres off in ABSOLUTE terms against
the bench and the workcell model built from it. Two causes look identical
from a single pose:

* the bench <-> arm-base extrinsic (``world_T_arm``, the bench.yaml number)
  is wrong -- a rigid error, the same for every configuration;
* the joint references are off by a fraction of a degree -- an error that
  MOVES as the arm changes shape.

Separating them is the whole point, so the models are fitted in STAGES and
all of them are reported. A mount-only fit that removes most of the error
means one diagnosis; a mount fit that stalls and a joint fit that finishes
the job means the other.

    q_corrected = q_measured + delta_q          <- the one sign convention

Ground truth comes from the bench and the probe drawing only. No robot
Cartesian report and no FK enters as a target; see assembly/model/
bench_holes.py. That is why the targets are full SE(3): the two-pin probe,
seated, is fully constrained.

WHAT IS NOT IDENTIFIABLE, stated before any number is read: joint 1's axis
IS world z, because the nominal mount has no roll or pitch. A dq1 and a
mount yaw are the same rotation about the same axis and NO dataset can
separate them. The same is true of the probe's tool length and the mount
height, because a seated probe always points straight down. Those parameters
are therefore PINNED rather than fitted (see :class:`Model`) -- but the
degeneracy report is computed from the UNPINNED Jacobian, so it is named
rather than hidden, and any other near-degenerate direction the dataset
happens to have is named next to it.

Link geometry is deliberately NOT fitted. tools/bench/compare_flange_fk.py
answers that question separately and far more cheaply, and giving this fit
per-link freedom would let it absorb the very thing it is measuring.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
NJ = len(JOINTS)


# ----------------------------------------------------------------- kinematics

def ee_fk(urdf_path, ee_frame: str, joints: Sequence[str] = JOINTS):
    """``q (7,) -> 4x4`` pose of ``ee_frame`` in the ARM BASE frame.

    The one Pinocchio FK primitive for calibration work. It is deliberately
    lighter than ``planning.ik.PinocchioIK`` (no limits, no solver, no
    collision world) because a fit evaluates it tens of thousands of times,
    but it resolves joint slots the same way, so the two agree on which
    number drives which joint.
    """
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf_path))
    data = model.createData()
    for name in joints:
        if not model.existJointName(name):
            raise ValueError(f"{urdf_path}: no joint named {name!r}")
    slots = [model.joints[model.getJointId(n)].idx_q for n in joints]
    fid = model.getFrameId(ee_frame)
    if fid >= model.nframes:
        raise KeyError(f"{ee_frame!r} absent from {urdf_path}")
    neutral = pin.neutral(model)

    def fk(q):
        qq = neutral.copy()
        qq[slots] = np.asarray(q, float)
        pin.forwardKinematics(model, data, qq)
        pin.updateFramePlacements(model, data)
        return np.asarray(data.oMf[fid].homogeneous)

    return fk


def exp6(xi) -> np.ndarray:
    """SE(3) exponential of ``[v(3), omega(3)]``."""
    import pinocchio as pin

    xi = np.asarray(xi, float)
    return np.asarray(pin.exp6(pin.Motion(xi[:3], xi[3:])).homogeneous)


def log3(R) -> np.ndarray:
    """SO(3) logarithm -- a rotation vector, never an Euler difference."""
    import pinocchio as pin

    return np.asarray(pin.log3(np.asarray(R, float)))


def rot_angle(R) -> float:
    return float(np.linalg.norm(log3(R)))


# --------------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Sample:
    """One seated capture: measured q against a bench-imposed target pose."""

    q: np.ndarray                    # (7,) measured joint positions
    target: np.ndarray               # (4,4) world_T_tcp from the bench
    group: str                       # the hole pair -- repeated on purpose
    split: str = "calibration"       # or "validation"
    sample_id: str = ""


MOUNT_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw")
DQ_NAMES = tuple(f"dq{i}" for i in range(1, NJ + 1))
TOOL_NAMES = ("tool_z",)
PROBE_NAMES = ("probe_yaw",)


@dataclass(frozen=True)
class Model:
    """Which parameter blocks a stage may move, and which are pinned to 0.

    Pinning is how this module refuses to pretend: two families of pinned
    parameters are not small, they are NOT IDENTIFIABLE from seated-probe
    data, and leaving them free would produce a confident split of a sum the
    data only constrains jointly.

    * ``dq1`` vs ``dyaw`` -- joint 1 turns about world z because the nominal
      mount has no roll or pitch. The same rotation about the same axis.
    * ``tool_z`` vs ``dz`` -- the seated probe's approach axis is always
      VERTICAL, so lengthening the tool and raising the arm base are the
      same displacement. Nothing in this artifact can tell them apart.
    * ``probe_yaw`` vs ``dq7`` -- joint 7 turns about the approach axis, and
      so does a probe clocked in the fingers. EXACTLY the same rotation. The
      pair is bracketed rather than pinned once: ``mount_dq`` gives it all to
      the robot, ``mount_dq_tool`` gives it all to the tool, and the truth
      decides whether the number may be written into a robot calibration.

    Both are reported from the unpinned Jacobian regardless, so the reader
    sees the degeneracy rather than inferring it from a suspicious number.
    """

    name: str
    mount: bool = False
    dq: bool = False
    tool: bool = False
    probe: bool = False
    pinned: tuple[str, ...] = ("dq1",)

    @property
    def all_names(self) -> list[str]:
        """Every parameter this model's blocks contain, pinned included."""
        out = []
        if self.mount:
            out += list(MOUNT_NAMES)
        if self.dq:
            out += list(DQ_NAMES)
        if self.tool:
            out += list(TOOL_NAMES)
        if self.probe:
            out += list(PROBE_NAMES)
        return out

    @property
    def names(self) -> list[str]:
        """The FREE parameters, in theta order."""
        return [n for n in self.all_names if n not in self.pinned]

    @property
    def size(self) -> int:
        return len(self.names)

    def unpack(self, theta):
        """-> (xi (6,), dq (7,), tool, probe_yaw). Zeros where inactive."""
        theta = np.asarray(theta, float)
        full = dict.fromkeys(self.all_names, 0.0)
        for name, value in zip(self.names, theta):
            full[name] = float(value)
        xi = np.array([full.get(n, 0.0) for n in MOUNT_NAMES])
        dq = np.array([full.get(n, 0.0) for n in DQ_NAMES])
        return xi, dq, full.get("tool_z", 0.0), full.get("probe_yaw", 0.0)

    def pack(self, xi, dq, tool, probe_yaw=0.0) -> np.ndarray:
        """Inverse of :meth:`unpack` for the free entries only."""
        source = dict(zip(MOUNT_NAMES, np.asarray(xi, float)))
        source.update(zip(DQ_NAMES, np.asarray(dq, float)))
        source["tool_z"] = float(tool)
        source["probe_yaw"] = float(probe_yaw)
        return np.array([source[n] for n in self.names])


MODELS = (
    Model("nominal", pinned=()),
    Model("mount_only", mount=True, pinned=()),
    Model("mount_dq", mount=True, dq=True, pinned=("dq1",)),
    # dz is pinned HERE and only here: with the tool length free the two are
    # one parameter, and the riser plate gives the mount height an
    # independent 12 mm check (check_workcell_geometry.riser_view_check) that
    # the probe's TCP convention has no equivalent of. So the vertical offset
    # is attributed to the TOOL in this stage and to the MOUNT in the stage
    # above, and the pair brackets it.
    # The TOOL-owns-it counterpart to mount_dq: the vertical offset and the
    # clocking about the approach axis are attributed to the probe instead of
    # to the arm. Same fit quality by construction -- the point is which
    # numbers you are then entitled to write into a robot calibration.
    Model("mount_dq_tool", mount=True, dq=True, tool=True, probe=True,
          pinned=("dq1", "dz", "dq7")),
)


# ------------------------------------------------------------------ prediction

def predict(theta, samples, fk, world_T_arm_nom, model: Model) -> np.ndarray:
    """(N,4,4) predicted ``world_T_tcp``.

    ``exp(xi)`` is composed on the LEFT of the nominal mount, so xi is a
    correction expressed in the world/bench frame and reads directly as
    "the arm base is this far from where bench.yaml says it is".
    """
    xi, dq, tool, yaw = model.unpack(theta)
    world_T_arm = exp6(xi) @ np.asarray(world_T_arm_nom, float)
    # Clocking about the approach axis first, then the offset along it.
    c, sn = np.cos(yaw), np.sin(yaw)
    tool_T = np.array([[c, -sn, 0.0, 0.0], [sn, c, 0.0, 0.0],
                       [0.0, 0.0, 1.0, tool], [0.0, 0.0, 0.0, 1.0]])
    out = np.empty((len(samples), 4, 4))
    for i, s in enumerate(samples):
        out[i] = world_T_arm @ fk(np.asarray(s.q, float) + dq) @ tool_T
    return out


def errors(theta, samples, fk, world_T_arm_nom, model: Model):
    """Per-sample (position vector (N,3) m, rotation vector (N,3) rad)."""
    pred = predict(theta, samples, fk, world_T_arm_nom, model)
    dp = np.array([pred[i][:3, 3] - s.target[:3, 3] for i, s in enumerate(samples)])
    dr = np.array([log3(s.target[:3, :3].T @ pred[i][:3, :3])
                   for i, s in enumerate(samples)])
    return dp, dr


def residual(theta, samples, fk, world_T_arm_nom, model: Model, rot_weight: float):
    """Stacked [dp ; rot_weight * dr], flattened. Metres throughout."""
    dp, dr = errors(theta, samples, fk, world_T_arm_nom, model)
    return np.concatenate([dp, rot_weight * dr], axis=1).ravel()


# ---------------------------------------------------------------------- bounds

@dataclass(frozen=True)
class Bounds:
    dq_bound_deg: float = 1.0
    mount_xyz_bound_m: float = 0.020
    mount_rot_bound_deg: float = 2.0
    tool_bound_m: float = 0.030
    clocking_bound_deg: float = 6.0

    def limit(self, name: str) -> float:
        if name in ("dx", "dy", "dz"):
            return self.mount_xyz_bound_m
        if name in ("droll", "dpitch", "dyaw"):
            return np.deg2rad(self.mount_rot_bound_deg)
        if name == "tool_z":
            return self.tool_bound_m
        if name in ("probe_yaw", "dq7"):
            # dq7 carries the probe's clocking whenever probe_yaw is not free,
            # and a grasp is off by degrees where a joint zero is off by
            # tenths -- a 1 deg bound here would clip and report a wall.
            return np.deg2rad(self.clocking_bound_deg)
        return np.deg2rad(self.dq_bound_deg)

    def vectors(self, model: Model):
        lim = np.array([self.limit(n) for n in model.names])
        return -lim, lim


# ------------------------------------------------------------------------- fit

@dataclass
class FitResult:
    model: Model
    theta: np.ndarray
    at_bound: list[str] = field(default_factory=list)
    singular_values: np.ndarray | None = None
    param_sigma: np.ndarray | None = None
    correlation: np.ndarray | None = None
    degenerate: list[tuple[float, np.ndarray]] = field(default_factory=list)
    success: bool = True
    message: str = ""

    @property
    def dq(self) -> np.ndarray:
        return self.model.unpack(self.theta)[1]

    @property
    def xi(self) -> np.ndarray:
        return self.model.unpack(self.theta)[0]

    @property
    def tool(self) -> float:
        return self.model.unpack(self.theta)[2]

    @property
    def probe_yaw(self) -> float:
        return self.model.unpack(self.theta)[3]

    def mount(self, world_T_arm_nom) -> np.ndarray:
        """The COMPOSED mount: correction applied to the nominal."""
        return exp6(self.xi) @ np.asarray(world_T_arm_nom, float)


def jacobian(theta, samples, fk, world_T_arm_nom, model, rot_weight, *, eps=1e-7):
    """Central-difference Jacobian of the UNWEIGHTED-loss residual.

    Deliberately not ``sol.jac``: with a robust loss scipy returns a
    Jacobian scaled by the loss derivative, and observability is a property
    of the data and the model, not of the loss we chose to be tolerant with.
    """
    theta = np.asarray(theta, float)
    cols = []
    for k in range(len(theta)):
        step = np.zeros_like(theta)
        step[k] = eps
        plus = residual(theta + step, samples, fk, world_T_arm_nom, model, rot_weight)
        minus = residual(theta - step, samples, fk, world_T_arm_nom, model, rot_weight)
        cols.append((plus - minus) / (2 * eps))
    return np.column_stack(cols) if cols else np.zeros((6 * len(samples), 0))


def fit(samples, fk, world_T_arm_nom, model: Model, *, rot_weight=0.05,
        bounds=Bounds(), loss="soft_l1", f_scale=0.002) -> FitResult:
    """Robust least squares for one stage. ``samples`` are calibration only."""
    from scipy.optimize import least_squares

    if model.size == 0:
        return FitResult(model, np.zeros(0), message="no free parameters")
    lo, hi = bounds.vectors(model)
    sol = least_squares(
        residual, np.zeros(model.size), bounds=(lo, hi),
        args=(samples, fk, world_T_arm_nom, model, rot_weight),
        loss=loss, f_scale=f_scale, xtol=1e-14, ftol=1e-14, gtol=1e-14)

    at_bound = [n for n, t, a, b in zip(model.names, sol.x, lo, hi)
                if min(abs(t - a), abs(t - b)) < 1e-9 * max(1.0, abs(b - a))]

    res = FitResult(model, sol.x, at_bound=at_bound,
                    success=bool(sol.success), message=str(sol.message))
    _observability(res, samples, fk, world_T_arm_nom, rot_weight)
    return res


def _observability(res, samples, fk, world_T_arm_nom, rot_weight):
    """Singular values, parameter sigma, correlations, and null directions.

    The singular values come from the FULL parameter set even when the fit
    pinned dq1, because "dq1 is unobservable" is a fact about the dataset
    that the reader needs whether or not this run chose to fit it.
    """
    model = res.model
    J = jacobian(res.theta, samples, fk, world_T_arm_nom, model, rot_weight)
    r = residual(res.theta, samples, fk, world_T_arm_nom, model, rot_weight)
    dof = max(1, len(r) - model.size)
    s2 = float(r @ r) / dof
    JTJ_inv = np.linalg.pinv(J.T @ J, rcond=1e-12)
    cov = JTJ_inv * s2
    sig = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    outer = np.outer(sig, sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(outer > 0, cov / outer, 0.0)
    res.param_sigma, res.correlation = sig, corr

    full = replace(model, name=model.name + "+unpinned", pinned=())
    if full.size:
        theta_full = full.pack(*model.unpack(res.theta))
        Jf = jacobian(theta_full, samples, fk, world_T_arm_nom, full, rot_weight)
        sv = np.linalg.svd(Jf, compute_uv=False)
        _, _, vt = np.linalg.svd(Jf, full_matrices=False)
        res.singular_values = sv
        res.degenerate = [(float(s), vt[i], full.names)
                          for i, s in enumerate(sv) if s < 1e-6 * sv[0]]


# ------------------------------------------------------------------- reporting

def metrics(theta, samples, fk, world_T_arm_nom, model) -> dict:
    """Position in mm, orientation in deg -- the units a human reads."""
    if not samples:
        return {}
    dp, dr = errors(theta, samples, fk, world_T_arm_nom, model)
    pos = np.linalg.norm(dp, axis=1) * 1e3
    ang = np.rad2deg(np.linalg.norm(dr, axis=1))
    return {
        "n": len(samples),
        "pos_rms_mm": float(np.sqrt((pos ** 2).mean())),
        "pos_median_mm": float(np.median(pos)),
        "pos_p95_mm": float(np.percentile(pos, 95)),
        "pos_max_mm": float(pos.max()),
        "bias_xyz_mm": (dp.mean(0) * 1e3).tolist(),
        "rms_xyz_mm": (np.sqrt((dp ** 2).mean(0)) * 1e3).tolist(),
        "rot_rms_deg": float(np.sqrt((ang ** 2).mean())),
        "rot_median_deg": float(np.median(ang)),
        "rot_p95_deg": float(np.percentile(ang, 95)),
        "rot_max_deg": float(ang.max()),
    }


def group_spread(theta, samples, fk, world_T_arm_nom, model) -> dict:
    """Predicted-TCP spread per hole pair -- the decisive diagnostic.

    The physical target is IDENTICAL across a group and only q changes, so a
    rigid mount error cannot show up here at all: whatever spread remains is
    configuration-dependent by construction. If delta_q collapses these
    groups, the joint references are the story. If it does not, they are not.
    """
    out = {}
    for name in sorted({s.group for s in samples}):
        rows = [s for s in samples if s.group == name]
        if len(rows) < 2:
            continue
        pts = predict(theta, rows, fk, world_T_arm_nom, model)[:, :3, 3]
        d = np.linalg.norm(pts - pts.mean(0), axis=1) * 1e3
        out[name] = {
            "n": len(rows),
            "spread_rms_mm": float(np.sqrt((d ** 2).mean())),
            "spread_max_mm": float(d.max()),
        }
    return out


def bootstrap(samples, fk, world_T_arm_nom, model, *, draws=200, seed=0, **kw):
    """Resample HOLE PAIRS, not samples.

    Configurations of one seated pose are not independent observations of the
    parameters -- they share one physical target -- so resampling individual
    captures would report a confidence the dataset does not have.
    """
    groups = sorted({s.group for s in samples})
    by_group = {g: [s for s in samples if s.group == g] for g in groups}
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(int(draws)):
        pick = rng.choice(len(groups), size=len(groups), replace=True)
        drawn = [s for i in pick for s in by_group[groups[i]]]
        try:
            rows.append(fit(drawn, fk, world_T_arm_nom, model, **kw).theta)
        except Exception:                       # a degenerate resample is data
            continue                            # about the dataset, not a bug
    if not rows:
        return {}
    arr = np.array(rows)
    return {
        "draws": len(rows),
        "names": model.names,
        "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(),
        "min": arr.min(0).tolist(),
        "max": arr.max(0).tolist(),
    }


def format_parameters(res: FitResult, world_T_arm_nom) -> str:
    xi, dq, tool, yaw = res.xi, res.dq, res.tool, res.probe_yaw
    out = []
    if res.model.dq:
        out.append("Joint reference correction   (q_corrected = q_measured + delta_q)")
        out.append("-" * 66)
        for i in range(NJ):
            note = ("   (PINNED: it IS the mount yaw)"
                    if DQ_NAMES[i] in res.model.pinned else "")
            out.append(f"  J{i+1}   {np.rad2deg(dq[i]):+8.4f} deg   "
                       f"{dq[i]:+.6f} rad{note}")
        out.append("")
    if res.model.mount:
        out.append("Mount correction   world_T_arm = exp(xi) @ world_T_arm_nominal")
        out.append("-" * 66)
        for label, value in zip(MOUNT_NAMES[:3], xi[:3]):
            pin = "   (PINNED)" if label in res.model.pinned else ""
            out.append(f"  {label:6s} {value * 1e3:+8.3f} mm{pin}")
        for label, value in zip(MOUNT_NAMES[3:], xi[3:]):
            pin = "   (PINNED)" if label in res.model.pinned else ""
            out.append(f"  {label:6s} {np.rad2deg(value):+8.4f} deg{pin}")
        composed = res.mount(world_T_arm_nom)
        nom = np.asarray(world_T_arm_nom, float)
        out.append(f"  nominal  xyz {np.round(nom[:3, 3], 6).tolist()}")
        out.append(f"  composed xyz {np.round(composed[:3, 3], 6).tolist()}")
        out.append(f"  composed rpy {np.round(rpy_from_matrix(composed[:3, :3]), 6).tolist()}")
        out.append("")
    if res.model.tool:
        out.append(f"Probe TCP offset along the hand approach axis: "
                   f"{tool * 1e3:+.3f} mm")
        out.append("  0 means the probe origin really is at fr3_hand_tcp, which is")
        out.append("  what the probe drawing assumes; a few mm means the fingers")
        out.append("  index it elsewhere. READ THIS ONLY WITH dz PINNED: a vertical")
        out.append("  tool offset and a taller mount are the same displacement for")
        out.append("  a probe that always points down.")
        out.append("")
    if res.model.probe:
        out.append(f"Probe clocking about the approach axis: "
                   f"{np.rad2deg(yaw):+.3f} deg")
        out.append("  This is the SAME rotation as dq7, which is pinned here. The")
        out.append("  mount_dq fit above gives the identical number to joint 7. Only")
        out.append("  a physical check -- re-grasp the probe and see if it moves --")
        out.append("  says which one owns it, and writing a grasp artifact into a")
        out.append("  robot calibration would be wrong for every other tool.")
        out.append("")
    if res.at_bound:
        out.append(f"AT BOUND: {', '.join(res.at_bound)} -- the fit was clipped, so "
                   "these are not estimates. Widen the bound or distrust the run.")
    return "\n".join(out)


def format_observability(res: FitResult) -> str:
    out = []
    sv = res.singular_values
    if sv is not None and len(sv):
        rank = int((sv > 1e-6 * sv[0]).sum())
        out.append(f"singular values: {np.array2string(sv, precision=3)}")
        out.append(f"condition number {sv[0] / max(sv[-1], 1e-300):.3e}   "
                   f"effective rank {rank}/{len(sv)}")
    if res.param_sigma is not None and len(res.param_sigma):
        out.append("")
        out.append("parameter sigma (1 sigma, from the residual covariance):")
        for name, s in zip(res.model.names, res.param_sigma):
            unit = (f"{s * 1e3:9.4f} mm" if name in ("dx", "dy", "dz", "tool_z")
                    else f"{np.rad2deg(s):9.4f} deg")
            out.append(f"   {name:8s} {unit}")
        corr = res.correlation
        pairs = [(abs(corr[i, j]), res.model.names[i], res.model.names[j])
                 for i in range(len(corr)) for j in range(i + 1, len(corr))]
        strong = sorted((p for p in pairs if p[0] > 0.9), reverse=True)[:6]
        if strong:
            out.append("")
            out.append("STRONGLY CORRELATED parameters -- they trade against each "
                       "other, so do not read them singly:")
            for value, a, b in strong:
                out.append(f"   {a:8s} <-> {b:8s}  r = {value:+.3f}")
    for entry in res.degenerate:
        s, v, names = entry
        top = sorted(range(len(v)), key=lambda k: -abs(v[k]))[:3]
        terms = "  ".join(f"{v[k]:+.3f}*{names[k]}" for k in top)
        out.append("")
        out.append(f"UNOBSERVABLE direction  s={s:.2e}   {terms}")
        out.append("   Not determined by this data. Expected for dq1 vs the mount "
                   "yaw: same axis, same rotation.")
    return "\n".join(out)


def rpy_from_matrix(R) -> np.ndarray:
    """URDF extrinsic-xyz roll/pitch/yaw, matching frames.rpy_to_quat."""
    R = np.asarray(R, float)
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) > 1 - 1e-9:                    # gimbal: fold into yaw
        return np.array([0.0, pitch, np.arctan2(-R[0, 1], R[1, 1])])
    return np.array([np.arctan2(R[2, 1], R[2, 2]), pitch,
                     np.arctan2(R[1, 0], R[0, 0])])


# ------------------------------------------------------------------ self-check

def _legal_target(rng, yaw_index: int) -> np.ndarray:
    """A pose the seated probe can actually occupy.

    Tool straight DOWN, yaw one of four (the pin line runs along a bench row
    or column), origin 30 mm over the top face somewhere in front of the arm.
    A synthetic test drawn from free random poses would report a conditioning
    this artifact can never deliver, so the test is built from the same
    four-orientation set the bench imposes.
    """
    yaw = (np.pi / 2) * (yaw_index % 4)
    cz, sz = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    # z down; x and y span the horizontal plane, right-handed with z = -e_z.
    T[:3, :3] = np.array([[cz, sz, 0.0], [sz, -cz, 0.0], [0.0, 0.0, -1.0]])
    T[:3, 3] = [rng.uniform(-0.60, -0.10), rng.uniform(0.05, 0.60), 0.030]
    return T


def _synthesise(urdf, ee_frame, fk, world_T_arm_nom, truth_dq, truth_xi,
                truth_tool, *, noise_m=2e-5, seed=3, n_groups=9, per_group=5,
                validation_groups=2):
    """Build a dataset from a KNOWN arm the way the bench builds one.

    For each legal target, IK on the TRUE arm gives several configurations
    that reach the SAME physical point -- which is exactly the null-space
    variation the real session collects, and the only thing that separates a
    rigid mount error from a moving joint error. The measured q handed to the
    fitter is then ``q_true - truth_dq``, so that ``q_measured + delta_q``
    recovers the truth.
    """
    from arm_control.planning.ik import PinocchioIK

    # Tight tolerances on purpose: the IK error would otherwise become the
    # synthetic dataset's noise floor and mask the parameters being recovered.
    ik = PinocchioIK(urdf, ee_frame=ee_frame, joint_names=JOINTS, restarts=1,
                     max_iters=4000, tol_pos=1e-10, tol_rot=1e-10)
    rng = np.random.default_rng(seed)
    world_T_arm_true = exp6(truth_xi) @ np.asarray(world_T_arm_nom, float)
    arm_T_world_true = np.linalg.inv(world_T_arm_true)
    tool_inv = np.eye(4)
    tool_inv[2, 3] = -truth_tool

    samples, group = [], 0
    attempts = 0
    while group < n_groups and attempts < 40 * n_groups:
        attempts += 1
        target = _legal_target(rng, group)
        wanted = arm_T_world_true @ target @ tool_inv
        found = []
        for _ in range(6 * per_group):
            if len(found) >= per_group:
                break
            q = ik.solve(wanted, rng.uniform(-1.6, 1.6, NJ))
            if q is None:
                continue
            if any(np.linalg.norm(q - other) < 0.25 for other in found):
                continue                       # want DIFFERENT configurations
            reached = fk(q)
            if (np.linalg.norm(reached[:3, 3] - wanted[:3, 3]) > 1e-9
                    or rot_angle(wanted[:3, :3].T @ reached[:3, :3]) > 1e-8):
                continue                       # IK error must not become noise
            found.append(q)
        if len(found) < per_group:
            continue
        name = f"pair{group:02d}"
        split = "validation" if group >= n_groups - validation_groups else "calibration"
        for k, q in enumerate(found):
            noisy = target.copy()
            noisy[:3, 3] = noisy[:3, 3] + rng.normal(0.0, noise_m, 3)
            samples.append(Sample(np.asarray(q, float) - np.asarray(truth_dq, float),
                                  noisy, group=name, split=split,
                                  sample_id=f"{name}_{k}"))
        group += 1
    if group < n_groups:
        raise RuntimeError(f"only {group}/{n_groups} synthetic targets were reachable")
    return samples


def _self_check() -> None:
    from arm_control import CONTROL_ROOT, frames

    urdf = CONTROL_ROOT / "descriptions/franka/fr3_ft_gemini305_parallel/urdf/fr3.urdf"
    fk = ee_fk(urdf, "fr3_hand_tcp")

    def synth(dq, xi, tool, **kw):
        return _synthesise(urdf, "fr3_hand_tcp", fk, nom, dq, xi, tool, **kw)
    nom = frames.world_T_arm({"arm": {"world_origin": [-0.0875, 0.1585, 0.012],
                                      "world_rpy": [0.0, 0.0, np.pi / 2]}})

    # --- the model's own algebra, before any optimiser is trusted.
    xi = np.array([0.003, -0.002, 0.001, 0.004, -0.003, 0.002])
    assert np.allclose(exp6(np.zeros(6)), np.eye(4))
    assert np.allclose(log3(exp6(xi)[:3, :3]), xi[3:], atol=1e-9)
    m = Model("t", mount=True, dq=True, tool=True, probe=True,
              pinned=("dq1", "dz", "dq7"))
    a, b, c, d = m.unpack(np.arange(m.size, dtype=float))
    assert b[0] == 0.0 and b[6] == 0.0 and a[2] == 0.0
    assert len(a) == 6 and len(b) == NJ
    assert m.size == (6 - 1) + (NJ - 2) + 1 + 1 == len(m.names)
    # pack/unpack must be inverses on the free entries.
    theta = np.arange(m.size, dtype=float) / 100.0
    assert np.allclose(m.pack(*m.unpack(theta)), theta)
    assert Bounds().vectors(m)[1].shape == (m.size,)

    # --- a clean arm must fit to zero and stay there.
    truth_dq = np.deg2rad([0.0, 0.20, -0.35, 0.10, 0.45, -0.15, 0.25])
    clean = synth(np.zeros(NJ), np.zeros(6), 0.0, noise_m=0.0, n_groups=3,
                  per_group=2, validation_groups=0)
    cal = [s for s in clean if s.split == "calibration"]
    base = metrics(np.zeros(0), clean, fk, nom, MODELS[0])
    assert base["pos_rms_mm"] < 1e-4 and base["rot_rms_deg"] < 1e-4, base

    # --- injected joint offsets + mount error must come back.
    truth_xi = np.array([0.004, -0.003, 0.0015, 0.002, -0.0015, 0.003])
    data = synth(truth_dq, truth_xi, 0.0)
    cal = [s for s in data if s.split == "calibration"]
    val = [s for s in data if s.split == "validation"]
    assert val and len(cal) >= 30, (len(cal), len(val))

    nominal = metrics(np.zeros(0), cal, fk, nom, MODELS[0])
    mount_only = fit(cal, fk, nom, MODELS[1])
    with_dq = fit(cal, fk, nom, MODELS[2])
    m_mount = metrics(mount_only.theta, cal, fk, nom, MODELS[1])
    m_dq = metrics(with_dq.theta, cal, fk, nom, MODELS[2])

    assert nominal["pos_rms_mm"] > 1.0, f"synthetic fault too small: {nominal}"
    # The POINT of the staging: a rigid mount cannot fix a moving error.
    assert m_mount["pos_rms_mm"] > 5 * m_dq["pos_rms_mm"], (m_mount, m_dq)
    assert m_dq["pos_rms_mm"] < 0.05, m_dq
    got = np.rad2deg(with_dq.dq)
    want = np.rad2deg(truth_dq)
    assert np.abs(got[1:] - want[1:]).max() < 0.02, list(zip(got, want))
    # Held-out poses must improve too, not just the fitted ones.
    assert metrics(with_dq.theta, val, fk, nom, MODELS[2])["pos_rms_mm"] < 0.1

    # --- the group spread must collapse; that is the decisive diagnostic.
    s_nom = group_spread(np.zeros(0), cal, fk, nom, MODELS[0])
    s_dq = group_spread(with_dq.theta, cal, fk, nom, MODELS[2])
    worst_nom = max(v["spread_rms_mm"] for v in s_nom.values())
    worst_dq = max(v["spread_rms_mm"] for v in s_dq.values())
    assert worst_dq < 0.05 and worst_nom > 10 * max(worst_dq, 1e-6), (worst_nom, worst_dq)

    # --- NEGATIVE CONTROL: a pure mount error must NOT invent joint offsets.
    pure = synth(np.zeros(NJ), truth_xi, 0.0, seed=11)
    pure_cal = [s for s in pure if s.split == "calibration"]
    only_mount = fit(pure_cal, fk, nom, MODELS[1])
    also_dq = fit(pure_cal, fk, nom, MODELS[2])
    assert metrics(only_mount.theta, pure_cal, fk, nom, MODELS[1])["pos_rms_mm"] < 0.05
    assert np.abs(np.rad2deg(also_dq.dq)).max() < 0.05, np.rad2deg(also_dq.dq)

    # --- a probe clocked in the fingers must come back as probe_yaw, and it
    # must be reported as indistinguishable from dq7 rather than split.
    clocked = synth(np.deg2rad([0, .2, -.35, .1, .45, -.15, 2.6]), truth_xi, 0.0, seed=9)
    clocked_cal = [s for s in clocked if s.split == "calibration"]
    as_tool = fit(clocked_cal, fk, nom, MODELS[3])
    assert abs(np.rad2deg(as_tool.probe_yaw) - 2.6) < 0.05, np.rad2deg(as_tool.probe_yaw)
    as_joint = fit(clocked_cal, fk, nom, MODELS[2])
    assert abs(np.rad2deg(as_joint.dq[6]) - 2.6) < 0.05, np.rad2deg(as_joint.dq[6])
    # Same data, same quality, two attributions -- that is the whole point.
    m_tool = metrics(as_tool.theta, clocked_cal, fk, nom, MODELS[3])
    m_joint = metrics(as_joint.theta, clocked_cal, fk, nom, MODELS[2])
    assert abs(m_tool["pos_rms_mm"] - m_joint["pos_rms_mm"]) < 0.02, (m_tool, m_joint)
    assert "probe clocking" in format_parameters(as_tool, nom).lower()

    # --- the tool scalar, WITH dz pinned, must be recovered when it is
    # really there. The truth has no mount dz for the same reason the model
    # pins it: the two are one parameter for a down-pointing probe.
    flat_xi = truth_xi.copy()
    flat_xi[2] = 0.0
    tooled = synth(truth_dq, flat_xi, -0.009, seed=5)
    tooled_cal = [s for s in tooled if s.split == "calibration"]
    with_tool = fit(tooled_cal, fk, nom, MODELS[3])
    assert abs(with_tool.tool - (-0.009)) < 3e-4, with_tool.tool

    # --- and with dz FREE the pair must be reported as indistinguishable,
    # not silently split. This is the reason dz is pinned in MODELS[3].
    loose = fit(tooled_cal, fk, nom,
                replace(MODELS[3], name="tool_dz_free", pinned=("dq1",)))
    text_loose = format_observability(loose)
    assert "tool_z" in text_loose and ("dz" in text_loose), text_loose
    i, j = loose.model.names.index("tool_z"), loose.model.names.index("dz")
    assert abs(loose.correlation[i, j]) > 0.99, loose.correlation[i, j]

    # --- the degeneracy must be NAMED, from the unpinned Jacobian.
    text = format_observability(with_dq)
    assert "UNOBSERVABLE" in text or "STRONGLY CORRELATED" in text, text
    named = {n for _, v, names in with_dq.degenerate
             for n in [names[k] for k in sorted(range(len(v)),
                                                key=lambda k: -abs(v[k]))[:2]]}
    if with_dq.degenerate:
        assert "dq1" in named and "dyaw" in named, named

    # --- a bound that clips must be reported as clipped, not as an estimate.
    tight = fit(cal, fk, nom, MODELS[2], bounds=Bounds(dq_bound_deg=0.05))
    assert tight.at_bound, "a 0.05 deg bound against a 0.45 deg offset must clip"

    # --- bootstrap resamples hole pairs and reports a spread per parameter.
    boot = bootstrap(cal, fk, nom, MODELS[2], draws=12, seed=1)
    assert boot["draws"] > 0 and len(boot["mean"]) == MODELS[2].size
    assert max(abs(v) for v in boot["std"]) < np.deg2rad(0.5), boot["std"]

    # --- the parameter report must print, in degrees as well as radians.
    txt = format_parameters(with_dq, nom)
    assert "deg" in txt and "rad" in txt and "PINNED" in txt

    print("kinematic_fit self-check: recovers 6 joint offsets to <0.02 deg and a "
          "mount error from probe-legal poses; a rigid mount alone cannot "
          f"({m_mount['pos_rms_mm']:.3f} vs {m_dq['pos_rms_mm']:.3f} mm RMS); "
          "group spread collapses; a pure mount error invents no joint offsets; "
          "the dq1/yaw degeneracy and clipped bounds are reported")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        _self_check()
    else:
        print(__doc__)
