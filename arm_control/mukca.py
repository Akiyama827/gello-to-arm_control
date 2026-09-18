"""MUKCa two-socket kinematic calibration: data, objective and evaluation.

Headless and runtime-safe: numpy, PyYAML and (for FK) Pinocchio only. The
CasADi/IPOPT fitter lives in :mod:`arm_control.mukca_casadi` and is imported
only by offline tools.

Method and code adapted from platonics-delft/kinematics_calibration
(``calibrate_fk/parameter_optimizer.py``, ``calibrate_fk/utils.py``),
MIT License, Copyright (c) 2024 platonics (Giovanni Franzese, Max Spahn,
Mariano Ramirez). Paper: Franzese et al., "MUKCa: Accurate and Affordable
Cobot Calibration Without External Measurement Devices", arXiv:2503.12584.

Three sources are kept apart on purpose:

* **Paper** Eq. (5): socket-consistency + distance-distortion +
  regularisation.
* **Upstream code (reference)**, reproduced by :func:`reference_objective`::

      mu[p,s] = mean_i x[p,s,i]
      C[p]    = (sum_i |x[p,0,i]-mu[p,0]|^2 + sum_i |x[p,1,i]-mu[p,1]|^2) / (N[p,0]+N[p,1])
      D[p]    = (|mu[p,1]-mu[p,0]| - d)^2
      J       = sum_p (C[p] + D[p]) + P * lam * sum_{params not of the tool joint} (theta-prior)^2

  population normalisation, socket terms weighted by their counts, ONE
  shared spacing ``d``, ``lam`` default 1e-4, the ball/tool joint excluded
  from the regulariser, prior = the (optionally noised) initial parameters
  while the solver STARTS from the un-noised nominal.
* **This adaptation**: per-placement spacing (upstream uses one), explicit
  frame names and spacing threaded through every evaluator, joint columns
  mapped by NAME, CSV header handling that never drops a row, and explicit
  refusal of empty/one-socket/misshapen data. Anything else (robust losses,
  outlier filtering, different freezes) must be named as a separate mode.

Upstream behaviours deliberately NOT inherited, each recorded in the fixture
provenance note: import-time ``tk.Tk()``; ``shutil.rmtree`` of the output
directory; success assumed from reaching ``max_iter``; ``evaluate_model``'s
hard-coded 0.05 m spacing and ``ball_link``; ``remove_outliers`` computing
masks it never applies; ``read_each``'s ``skiprows=1`` silently dropping the
first row of a headerless recorder CSV; data keyed by folder basename so two
``front`` placements collide.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

PARAM_FIELDS = ("x", "y", "z", "roll", "pitch", "yaw")


# ------------------------------------------------------------------ data I/O
def read_socket_csv(path: Path | str, joints: list[str] | tuple[str, ...]) -> np.ndarray:
    """``(N, len(joints))`` measured joint rows from one socket CSV.

    A first row that is not all-numeric is a HEADER and must name exactly
    ``joints`` in order; columns are then taken by name. A numeric first row
    is DATA (the upstream recorder writes headerless files) and is kept.
    Empty files, ragged rows and non-finite values are refused.
    """
    lines = [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"{path}: empty socket file")
    first = [c.strip() for c in lines[0].split(",")]
    try:
        [float(c) for c in first]
        header, rows = None, lines
    except ValueError:
        header, rows = first, lines[1:]
    n = len(joints)
    if header is not None:
        missing = [j for j in joints if j not in header]
        if missing:
            raise ValueError(f"{path}: header lacks joints {missing}")
        cols = [header.index(j) for j in joints]
    else:
        cols = list(range(n))
    if not rows:
        raise ValueError(f"{path}: header but no samples")
    out = np.empty((len(rows), n))
    for i, row in enumerate(rows):
        cells = [c.strip() for c in row.split(",")]
        width = len(header) if header is not None else n
        if len(cells) != width:
            raise ValueError(f"{path}: row {i + 1} has {len(cells)} columns, expected {width}")
        out[i] = [float(cells[c]) for c in cols]
    if not np.isfinite(out).all():
        raise ValueError(f"{path}: non-finite joint value")
    return out


def upstream_read_each(path: Path | str) -> np.ndarray:
    """The upstream loader's behaviour, for REFERENCE reproduction only.

    ``np.loadtxt(skiprows=1)`` first, raw load only if that raises: a
    headerless file therefore loses its first sample without any error.
    """
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    except (ValueError, IndexError):
        data = np.loadtxt(path, delimiter=",")
    return np.atleast_2d(data) if data.size else data


def write_socket_csv(path: Path | str, q: np.ndarray, joints) -> None:
    """Header-bearing export: readable by :func:`read_socket_csv` AND by the
    upstream loader without losing a sample (its skipped row is the header)."""
    q = np.atleast_2d(np.asarray(q, float))
    if q.shape[1] != len(joints) or not len(q):
        raise ValueError(f"{path}: {q.shape} does not match {len(joints)} joints")
    lines = [",".join(joints)] + [",".join(repr(float(v)) for v in row) for row in q]
    Path(path).write_text("\n".join(lines) + "\n")


@dataclass(frozen=True)
class Placement:
    """One physical fixture placement in one session on one arm."""

    key: str                        # "<arm_id>/<session>/<placement>", unique
    joints: tuple[str, ...]
    q0: np.ndarray                  # socket 0 rows (N0, n)
    q1: np.ndarray                  # socket 1 rows (N1, n)
    spacing_m: float                # MEASURED socket spacing
    spacing_sigma_m: float | None = None

    def __post_init__(self):
        for name, q in (("socket 0", self.q0), ("socket 1", self.q1)):
            if q.ndim != 2 or q.shape[1] != len(self.joints) or len(q) < 2:
                raise ValueError(f"{self.key}: {name} needs >= 2 rows of {len(self.joints)} "
                                 f"joints, got {q.shape}")
        if not (math.isfinite(self.spacing_m) and 0.0 < self.spacing_m < 1.0):
            raise ValueError(f"{self.key}: spacing {self.spacing_m} m is not a plausible measurement")


def load_placement(folder: Path | str) -> Placement:
    """An exported placement folder: ``manifest.yaml`` + ``hole_0.csv`` + ``hole_1.csv``."""
    folder = Path(folder)
    man = yaml.safe_load((folder / "manifest.yaml").read_text())
    joints = tuple(man["joints"])
    key = f"{man['arm_id']}/{man['session']}/{man['placement']}"
    return Placement(key, joints, read_socket_csv(folder / "hole_0.csv", joints),
                     read_socket_csv(folder / "hole_1.csv", joints),
                     float(man["fixture"]["spacing_m"]),
                     man["fixture"].get("spacing_sigma_m"))


def unique_placements(placements) -> list[Placement]:
    keys = [p.key for p in placements]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    if dupes:
        raise ValueError(f"placements must be unique per arm/session/placement: {dupes}")
    return list(placements)


# --------------------------------------------------------- URDF parameters
def _parse(urdf_xml: str) -> ET.Element:
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.fromstring(urdf_xml, parser=parser)


def read_origins(urdf_xml: str, joints) -> dict[str, np.ndarray]:
    root = _parse(urdf_xml)
    found = {j.get("name"): j for j in root.findall("joint")}
    missing = [j for j in joints if j not in found]
    if missing:
        raise ValueError(f"selected joints absent from the URDF: {missing}")
    out = {}
    for name in joints:
        origin = found[name].find("origin")
        xyz = (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()
        rpy = (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()
        out[name] = np.array([float(v) for v in xyz + rpy])
    return out


def rewrite_origins(urdf_xml: str, params: dict[str, np.ndarray]) -> str:
    """Only the named joints' ``<origin>`` changes; limits, axes, mimics,
    meshes, other branches and comments are carried through untouched."""
    root = _parse(urdf_xml)
    found = {j.get("name"): j for j in root.findall("joint")}
    missing = [j for j in params if j not in found]
    if missing:
        raise ValueError(f"cannot rewrite absent joints {missing}")
    for name, values in params.items():
        values = np.asarray(values, float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError(f"{name}: origin must be 6 finite numbers")
        origin = found[name].find("origin")
        if origin is None:
            origin = ET.SubElement(found[name], "origin")
        origin.set("xyz", " ".join(repr(float(v)) for v in values[:3]))
        origin.set("rpy", " ".join(repr(float(v)) for v in values[3:]))
    return ET.tostring(root, encoding="unicode")


def add_ball_joint(urdf_xml: str, parent_link: str, xyz, *, joint: str = "mukca_ball_joint",
                   link: str = "mukca_ball_link") -> str:
    """Calibration-only variant: a fixed ball-centre frame under ``parent_link``.

    Never written into a runtime description; the fit input is generated.
    """
    root = _parse(urdf_xml)
    links = {el.get("name") for el in root.findall("link")}
    if parent_link not in links:
        raise ValueError(f"parent link {parent_link!r} absent")
    if link in links:
        raise ValueError(f"{link!r} already present")
    ET.SubElement(root, "link", name=link)
    j = ET.SubElement(root, "joint", name=joint, type="fixed")
    ET.SubElement(j, "origin", xyz=" ".join(repr(float(v)) for v in xyz), rpy="0 0 0")
    ET.SubElement(j, "parent", link=parent_link)
    ET.SubElement(j, "child", link=link)
    return ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------- FK
class BallFk:
    """Ball-centre positions in ``root_link`` from a URDF string, joints by NAME."""

    def __init__(self, urdf_xml: str, root_link: str, ball_link: str, joints) -> None:
        import pinocchio as pin

        self._pin = pin
        self.model = pin.buildModelFromXML(urdf_xml)
        self.data = self.model.createData()
        self.joints = tuple(joints)
        for name in self.joints:
            if not self.model.existJointName(name):
                raise ValueError(f"model has no joint {name!r}")
        self.slots = [self.model.joints[self.model.getJointId(n)].idx_q for n in self.joints]
        for frame in (root_link, ball_link):
            if not self.model.existFrame(frame):
                raise ValueError(f"model has no frame {frame!r}")
        self.root = self.model.getFrameId(root_link)
        self.ball = self.model.getFrameId(ball_link)
        self.q0 = pin.neutral(self.model)

    def __call__(self, q_rows: np.ndarray) -> np.ndarray:
        pin = self._pin
        out = np.empty((len(q_rows), 3))
        for i, row in enumerate(np.atleast_2d(q_rows)):
            q = self.q0.copy()
            q[self.slots] = row
            pin.framesForwardKinematics(self.model, self.data, q)
            root = self.data.oMf[self.root]
            out[i] = (root.actInv(self.data.oMf[self.ball])).translation
        return out


# ---------------------------------------------------------------- objective
def socket_terms(x0: np.ndarray, x1: np.ndarray, spacing_m: float) -> dict:
    mu0, mu1 = x0.mean(0), x1.mean(0)
    n0, n1 = len(x0), len(x1)
    sq0 = float(((x0 - mu0) ** 2).sum())
    sq1 = float(((x1 - mu1) ** 2).sum())
    dist = float(np.linalg.norm(mu1 - mu0))
    return {"C": (sq0 + sq1) / (n0 + n1), "D": (dist - spacing_m) ** 2, "distance": dist,
            "mu0": mu0, "mu1": mu1, "n0": n0, "n1": n1}


def reference_objective(predictions, theta, prior, regularized_mask, lam: float = 1e-4) -> dict:
    """Upstream J. ``predictions`` = [(x0, x1, spacing_m)] per placement."""
    terms = [socket_terms(x0, x1, d) for x0, x1, d in predictions]
    theta, prior = np.asarray(theta, float), np.asarray(prior, float)
    mask = np.asarray(regularized_mask, bool)
    reg = len(predictions) * lam * float(((theta - prior)[mask] ** 2).sum())
    consistency = sum(t["C"] for t in terms)
    distortion = sum(t["D"] for t in terms)
    return {"J": consistency + distortion + reg, "consistency": consistency,
            "distortion": distortion, "regularization": reg, "per_placement": terms}


def evaluate(x0: np.ndarray, x1: np.ndarray, spacing_m: float) -> dict:
    """Upstream ``evaluate_model`` KPIs with the spacing passed explicitly.

    ``mean_absolute_error`` is distance from each socket's OWN predicted mean:
    self-consistency, not error against any independently measured point.
    """
    t = socket_terms(x0, x1, spacing_m)
    var0 = float(np.var(x0, axis=0).sum())
    var1 = float(np.var(x1, axis=0).sum())
    mae0 = float(np.linalg.norm(np.abs(x0 - t["mu0"]), axis=1).mean())
    mae1 = float(np.linalg.norm(np.abs(x1 - t["mu1"]), axis=1).mean())
    n0, n1 = t["n0"], t["n1"]
    return {"mean_1": t["mu0"].tolist(), "mean_2": t["mu1"].tolist(),
            "std_dev_1": math.sqrt(var0), "std_dev_2": math.sqrt(var1),
            "distance_error": abs(t["distance"] - spacing_m), "spacing_m": spacing_m,
            "mean_absolute_error": (n0 * mae0 + n1 * mae1) / (n0 + n1),
            "std_dev_fk": math.sqrt((var0 * n0 + var1 * n1) / (n0 + n1)),
            "n_1": n0, "n_2": n1}


def parameter_names(joints, fields=PARAM_FIELDS) -> list[str]:
    return [f"{j}:{f}" for j in joints for f in fields]


def admission_errors(result: dict, *, held_out_before: float | None = None,
                     held_out_after: float | None = None) -> list[str]:
    """Why a fit result is NOT a candidate. Empty list = admissible candidate.

    Reaching an iteration budget or writing a URDF is not success.
    """
    errs = []
    if not result.get("solver_success"):
        errs.append(f"solver status {result.get('return_status')!r} is not success")
    theta = np.asarray(result.get("theta", []), float)
    if theta.size == 0 or not np.isfinite(theta).all():
        errs.append("parameters are empty or non-finite")
    before, after = result.get("J_before"), result.get("J_after")
    if before is None or after is None or not (math.isfinite(before) and math.isfinite(after)):
        errs.append("objective not finite")
    elif after > before:
        errs.append(f"objective increased {before:.3e} -> {after:.3e}")
    if held_out_before is not None and held_out_after is not None and held_out_after > held_out_before:
        errs.append(f"held-out consistency worsened {held_out_before:.3e} -> {held_out_after:.3e}")
    return errs


def _self_check() -> None:
    import tempfile

    joints = ("j1", "j2", "j3")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "headerless.csv").write_text("0.1,0.2,0.3\n0.4,0.5,0.6\n")
        q = read_socket_csv(tmp / "headerless.csv", joints)
        assert q.shape == (2, 3) and q[0, 0] == 0.1, "headerless first row must be kept"
        assert upstream_read_each(tmp / "headerless.csv").shape == (1, 3), \
            "the upstream loader is expected to drop that row"
        (tmp / "one.csv").write_text("0.1,0.2,0.3\n")
        assert read_socket_csv(tmp / "one.csv", joints).shape == (1, 3)
        (tmp / "named.csv").write_text("j3,j1,j2\n3,1,2\n6,4,5\n")
        assert np.array_equal(read_socket_csv(tmp / "named.csv", joints), [[1, 2, 3], [4, 5, 6]])
        for body, why in (("", "empty"), ("j1,j2,j3\n", "no samples"), ("1,2\n", "columns"),
                          ("j1,j2\n1,2\n", "lacks"), ("1,nan,3\n", "non-finite")):
            (tmp / "bad.csv").write_text(body)
            try:
                read_socket_csv(tmp / "bad.csv", joints)
            except ValueError as exc:
                assert why in str(exc), (why, exc)
            else:
                raise AssertionError(f"accepted {why}")
        write_socket_csv(tmp / "rt.csv", q, joints)
        assert np.array_equal(read_socket_csv(tmp / "rt.csv", joints), q)
        assert np.array_equal(upstream_read_each(tmp / "rt.csv"), q), \
            "header-bearing export must survive the upstream loader row-for-row"

    # Objective: dynamic means, population normalisation, unequal counts.
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1e-3, (5, 3))
    x1 = rng.normal(0, 1e-3, (9, 3)) + [0.05, 0, 0]
    t = socket_terms(x0, x1, 0.05)
    manual_c = (((x0 - x0.mean(0)) ** 2).sum() + ((x1 - x1.mean(0)) ** 2).sum()) / 14
    assert abs(t["C"] - manual_c) < 1e-18
    assert abs(t["D"] - (np.linalg.norm(x1.mean(0) - x0.mean(0)) - 0.05) ** 2) < 1e-18
    theta, prior = np.arange(4.0), np.zeros(4)
    obj = reference_objective([(x0, x1, 0.05), (x0, x1, 0.06)], theta, prior,
                              [True, True, False, True], lam=1e-4)
    assert abs(obj["regularization"] - 2 * 1e-4 * (0 + 1 + 9)) < 1e-15, obj["regularization"]
    ev = evaluate(x0, x1, 0.0501)
    assert abs(ev["distance_error"] - abs(t["distance"] - 0.0501)) < 1e-15
    assert admission_errors({"solver_success": False, "theta": [1.0], "J_before": 1,
                             "J_after": 0.5})[0].startswith("solver status")
    assert admission_errors({"solver_success": True, "theta": [1.0], "J_before": 1,
                             "J_after": 0.5}) == []
    try:
        unique_placements([Placement("a/s/front", joints, x0, x0, 0.05),
                           Placement("a/s/front", joints, x1, x1, 0.05)])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate placement keys admitted")
    # Distinct sessions with the same placement NAME are distinct placements.
    unique_placements([Placement("a/s1/front", joints, x0, x0, 0.05),
                       Placement("a/s2/front", joints, x1, x1, 0.05)])

    # URDF rewriting keeps everything but the selected origins.
    urdf = """<robot name="r"><!-- keep me -->
  <link name="base"/><link name="l1"/><link name="tip"/>
  <joint name="j1" type="revolute"><origin xyz="0 0 0.1" rpy="0 0 0"/>
    <parent link="base"/><child link="l1"/><axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" effort="1" velocity="1"/></joint>
  <joint name="fixed_tip" type="fixed"><origin xyz="0 0 0.2" rpy="0 0 0"/>
    <parent link="l1"/><child link="tip"/></joint></robot>"""
    new = rewrite_origins(urdf, {"fixed_tip": [0.001, 0, 0.2, 0, 0.01, 0]})
    assert "keep me" in new and 'lower="-1"' in new and 'axis xyz="0 0 1"' in new
    assert np.allclose(read_origins(new, ["fixed_tip"])["fixed_tip"], [0.001, 0, 0.2, 0, 0.01, 0])
    assert np.allclose(read_origins(new, ["j1"])["j1"], [0, 0, 0.1, 0, 0, 0])
    try:
        rewrite_origins(urdf, {"nope": np.zeros(6)})
    except ValueError:
        pass
    else:
        raise AssertionError("absent selected joint silently skipped")
    ball = add_ball_joint(urdf, "tip", [0, 0, 0.04])
    fk = BallFk(ball, "base", "mukca_ball_link", ["j1"])
    assert np.allclose(fk(np.array([[0.3]])), [[0, 0, 0.34]])
    print("mukca: CSV header/row preservation and refusals, upstream-loader row loss, "
          "objective terms (dynamic means, population normalisation, unequal counts, "
          "P-scaled regulariser with exclusions), explicit spacing, admission, placement "
          "keys, origin rewriting and ball FK: PASS")


if __name__ == "__main__":
    _self_check()
