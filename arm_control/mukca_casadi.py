"""OFFLINE MUKCa fitter: CasADi/IPOPT over URDF joint-origin parameters.

Import only from offline calibration tools, in an environment that has
``casadi`` and ``forwardkinematics`` (maxspahn/forwardKinematics, branch
``ft-symbolic-fk``, as the upstream project declares). Nothing in control
startup or any RT path imports this module.

Adapted from ``calibrate_fk/parameter_optimizer.py`` of
platonics-delft/kinematics_calibration, MIT License, Copyright (c) 2024
platonics (Giovanni Franzese, Max Spahn, Mariano Ramirez). The symbolic FK,
the xyz/rpy-per-selected-joint parameterisation, the dynamic-mean objective,
the P-scaled regulariser excluding the tool/ball joint, the nominal start
point and the IPOPT solve are theirs. Changes, all visible in the result:

* headless: no tkinter, no dialogs, no plots, no URDF written by the fitter;
* joint columns mapped by NAME to the symbolic model's q order;
* ``mode="reference"`` insists on one shared spacing (upstream); ``mode=
  "per_placement"`` uses each placement's measured spacing (named adaptation);
* the solver's return status, iteration count and objective terms are
  returned; success is IPOPT's verdict, never "the budget ran out";
* ``noise_sigma`` perturbs the regularisation PRIOR only, exactly like the
  upstream ``variance_noise`` (which is passed to ``np.random.normal`` as a
  standard deviation despite its name), while the solve still starts from
  the un-noised nominal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arm_control.mukca import PARAM_FIELDS, Placement, read_origins


@dataclass
class FitSpec:
    urdf_xml: str
    root_link: str
    ball_link: str
    selected_joints: list[str]
    tool_joints: tuple[str, ...] = ("ball_joint",)
    lam: float = 1e-4
    max_iter: int = 30
    noise_sigma: float = 0.0
    seed: int | None = None
    mode: str = "reference"          # reference | per_placement
    ipopt: dict = field(default_factory=dict)


def fit(spec: FitSpec, placements: list[Placement]) -> dict:
    import casadi as ca
    from forwardkinematics import GenericURDFFk

    if spec.mode not in ("reference", "per_placement"):
        raise ValueError(f"unknown mode {spec.mode!r}")
    if not placements:
        raise ValueError("no placements")
    spacings = {p.spacing_m for p in placements}
    if spec.mode == "reference" and len(spacings) != 1:
        raise ValueError(f"reference mode uses ONE shared spacing, placements have {spacings}")
    missing_tool = [j for j in spec.tool_joints if j not in spec.selected_joints]
    if missing_tool:
        raise ValueError(f"tool joints {missing_tool} are not selected")

    fk = GenericURDFFk(spec.urdf_xml, root_link=spec.root_link, end_links=[spec.ball_link])
    joint_map = dict(fk.robot._joint_map)          # name -> q index of the symbolic model
    order = sorted(joint_map, key=joint_map.get)
    if [joint_map[j] for j in order] != list(range(fk.n())):
        raise ValueError(f"symbolic model q layout is not contiguous: {joint_map}")
    for p in placements:
        absent = [j for j in order if j not in p.joints]
        if absent:
            raise ValueError(f"{p.key}: data lacks chain joints {absent}")
    nominal = read_origins(spec.urdf_xml, spec.selected_joints)   # refuses absent names
    active = set(fk.robot.active_joints())
    not_in_chain = [j for j in spec.selected_joints if j not in active]
    if not_in_chain:
        raise ValueError(f"selected joints not on the {spec.root_link}->{spec.ball_link} "
                         f"chain: {not_in_chain}")

    rng = np.random.default_rng(spec.seed)
    params = {j: {f: ca.SX.sym(f"{j}_{f}") for f in PARAM_FIELDS} for j in spec.selected_joints}
    theta = ca.vertcat(*[params[j][f] for j in spec.selected_joints for f in PARAM_FIELDS])
    x0 = np.concatenate([nominal[j] for j in spec.selected_joints])
    prior = x0 + (rng.normal(0.0, spec.noise_sigma, x0.shape) if spec.noise_sigma else 0.0)
    reg_mask = np.array([j not in spec.tool_joints
                         for j in spec.selected_joints for _ in PARAM_FIELDS])

    q = ca.SX.sym("q", fk.n())
    ball = ca.Function("ball", [q, theta], [fk.casadi(q, spec.ball_link, position_only=True,
                                                      symbolic_parameters=params)])

    consistency = 0
    distortion = 0
    for p in placements:
        cols = [p.joints.index(j) for j in order]
        d = next(iter(spacings)) if spec.mode == "reference" else p.spacing_m
        sockets = []
        for rows in (p.q0, p.q1):
            xs = [ball(ca.DM(row[cols]), theta) for row in rows]
            mean = sum(xs) / len(xs)
            sq = sum(ca.sumsqr(x - mean) for x in xs)
            sockets.append((mean, sq, len(xs)))
        (m0, s0, n0), (m1, s1, n1) = sockets
        consistency += (s0 + s1) / (n0 + n1)
        distortion += (ca.norm_2(m0 - m1) - d) ** 2
    reg = len(placements) * spec.lam * ca.sumsqr((theta - ca.DM(prior))[np.flatnonzero(reg_mask).tolist()])
    objective = consistency + distortion + reg
    terms = ca.Function("terms", [theta], [objective, consistency, distortion, reg])

    options = {"ipopt": {"print_level": 0, "max_iter": int(spec.max_iter), **spec.ipopt},
               "print_time": False}
    solver = ca.nlpsol("mukca", "ipopt", {"x": theta, "f": objective}, options)
    sol = solver(x0=x0)
    stats = solver.stats()
    theta_star = np.asarray(sol["x"]).ravel()
    before = [float(v) for v in terms(x0)]
    after = [float(v) for v in terms(theta_star)]
    return {
        "mode": spec.mode,
        "names": [f"{j}:{f}" for j in spec.selected_joints for f in PARAM_FIELDS],
        "theta0": x0.tolist(), "prior": np.asarray(prior).tolist(), "theta": theta_star.tolist(),
        "regularized": reg_mask.tolist(),
        "J_before": before[0], "J_after": after[0],
        "terms_before": dict(zip(("J", "consistency", "distortion", "regularization"), before)),
        "terms_after": dict(zip(("J", "consistency", "distortion", "regularization"), after)),
        "solver_success": bool(stats.get("success")),
        "return_status": str(stats.get("return_status")),
        "iter_count": int(stats.get("iter_count", -1)),
        "max_iter": int(spec.max_iter), "lam": spec.lam, "noise_sigma": spec.noise_sigma,
        "q_order": order, "placements": [p.key for p in placements],
        "spacing_m": {p.key: (next(iter(spacings)) if spec.mode == "reference" else p.spacing_m)
                      for p in placements},
    }


def origins(result: dict) -> dict[str, np.ndarray]:
    """Fitted origins per joint, for :func:`arm_control.mukca.rewrite_origins`."""
    out: dict[str, list[float]] = {}
    for name, value in zip(result["names"], result["theta"]):
        joint, _ = name.rsplit(":", 1)
        out.setdefault(joint, []).append(value)
    return {j: np.asarray(v, float) for j, v in out.items()}
