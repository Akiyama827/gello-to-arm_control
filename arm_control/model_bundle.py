"""Versioned kinematic model bundle: which geometry an arm instance runs, and why.

A bundle is a small YAML the consuming project owns. It names one robot
INSTANCE and the URDF its runtime loads, with enough provenance to reproduce
and to refuse the wrong one:

.. code-block:: yaml

    schema: arm_control.model_bundle/1
    bundle_id: fr3_1-nominal          # unique per instance
    revision: 1                       # strictly increases per instance
    status: nominal                   # nominal|candidate|approved|deployed|retired
    instance_id: fr3_1
    robot_kind: franka_fr3
    joints: [fr3_joint1, ...]         # order of the MEASURED joint vector
    units: {length: m, angle: rad}
    frames: {base: fr3_link0, flange: fr3_link8, tcp: fr3_hand_tcp}
    nominal: {urdf: <path>, sha256: <hex>}   # never modified
    model:   {urdf: <path>, sha256: <hex>}   # == nominal for a nominal bundle
    tool: {id: <text>, root_link: <link>}
    base: {source: <workcell yaml>, xyz: [..], rpy: [..]}
    parameterization: none | urdf_joint_origins
    fitted_joints: []                 # joints whose <origin> the model changed
    joint_offsets: null               # q_model = q_measured + offsets, if ever used
    datasets: []                      # [{path, sha256, kind}]
    validity: {}                      # free-form region/payload notes
    observability: {}                 # rank, conditioning, unobservable directions
    rt_task_space: {model: libfranka-vendor, calibrated: false}
    approval: {by: null, date: null}

The one transformation chain every consumer uses::

    bench_T_tcp[a]    = bench_T_base[a] @ FK_a(q_model; geometry_a) @ flange_T_tcp[a]
    baseA_T_baseB     = inv(bench_T_base[A]) @ bench_T_base[B]
    q_model           = q_measured + joint_offsets     (offsets default to zero)

A joint-zero correction is applied EXACTLY ONCE: a bundle whose URDF already
embeds it must carry ``joint_offsets: null``. This module provides the
raw<->model conversion and its round-trip check; a deployment that has not
wired the inverse mapping into commands and limits must refuse to activate a
bundle with nonzero offsets (:func:`activation_error` does).

Nothing here fits anything, loads CasADi, or belongs in an RT loop.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

SCHEMA = "arm_control.model_bundle/1"
STATUSES = ("nominal", "candidate", "approved", "deployed", "retired")
ACTIVE_STATUSES = ("nominal", "approved", "deployed")


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ModelBundle:
    path: Path
    bundle_id: str
    revision: int
    status: str
    instance_id: str
    robot_kind: str
    joints: tuple[str, ...]
    frames: dict
    nominal_urdf: Path
    nominal_sha256: str
    model_urdf: Path
    model_sha256: str
    tool: dict
    base: dict
    parameterization: str
    fitted_joints: tuple[str, ...]
    joint_offsets: np.ndarray | None
    datasets: tuple[dict, ...] = ()
    rt_task_space: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def calibrated(self) -> bool:
        """True once geometry differs from the vendor nominal in any way."""
        return (self.model_sha256 != self.nominal_sha256
                or self.joint_offsets is not None and bool(np.any(self.joint_offsets)))

    @property
    def pose_provenance(self) -> str:
        return "calibrated" if self.calibrated else "nominal"

    def token(self) -> str:
        """Plan/state binding token: instance, bundle, revision, geometry hash."""
        return (f"{self.instance_id}:{self.bundle_id}:r{self.revision}:"
                f"{self.model_sha256[:12]}")

    # -- joint coordinates --------------------------------------------------
    def to_model_q(self, q_measured) -> np.ndarray:
        q = np.asarray(q_measured, float)
        return q if self.joint_offsets is None else q + self.joint_offsets

    def to_raw_q(self, q_model) -> np.ndarray:
        q = np.asarray(q_model, float)
        return q if self.joint_offsets is None else q - self.joint_offsets


def load_bundle(path: Path | str, *, root: Path | str | None = None) -> ModelBundle:
    """Parse and VALIDATE a bundle. Relative paths resolve against ``root``."""
    path = Path(path)
    root = Path(root) if root is not None else path.parent
    raw = yaml.safe_load(path.read_text())
    problems = []
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} bundle")
    for key in ("bundle_id", "revision", "status", "instance_id", "robot_kind", "joints",
                "frames", "nominal", "model", "tool", "base", "parameterization"):
        if key not in raw:
            problems.append(f"missing {key}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    if raw["status"] not in STATUSES:
        problems.append(f"status {raw['status']!r} not in {STATUSES}")
    if raw.get("units", {"length": "m", "angle": "rad"}) != {"length": "m", "angle": "rad"}:
        problems.append("units must be {length: m, angle: rad}")
    if not isinstance(raw["revision"], int) or raw["revision"] < 1:
        problems.append("revision must be a positive integer")
    joints = tuple(raw["joints"])
    if len(set(joints)) != len(joints) or not joints:
        problems.append("joints must be a non-empty unique ordered list")
    urdfs = {}
    for key in ("nominal", "model"):
        entry = raw[key]
        file = (root / entry["urdf"]).resolve()
        urdfs[key] = file
        if not file.is_file():
            problems.append(f"{key}.urdf {entry['urdf']} does not exist")
        elif sha256_file(file) != entry.get("sha256"):
            problems.append(f"{key}.urdf sha256 does not match the file on disk")
    offsets = raw.get("joint_offsets")
    if offsets is not None:
        offsets = np.asarray(offsets, float)
        if offsets.shape != (len(joints),) or not np.isfinite(offsets).all():
            problems.append("joint_offsets must be finite, one per joint")
    if raw["parameterization"] not in ("none", "urdf_joint_origins"):
        problems.append(f"unknown parameterization {raw['parameterization']!r}")
    if raw["parameterization"] == "none" and raw.get("fitted_joints"):
        problems.append("fitted_joints given for parameterization none")
    if not problems and urdfs["model"].is_file():
        problems += _urdf_problems(urdfs["model"], joints, raw["frames"],
                                   raw.get("fitted_joints") or ())
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return ModelBundle(
        path=path, bundle_id=str(raw["bundle_id"]), revision=int(raw["revision"]),
        status=raw["status"], instance_id=str(raw["instance_id"]),
        robot_kind=str(raw["robot_kind"]), joints=joints, frames=dict(raw["frames"]),
        nominal_urdf=urdfs["nominal"], nominal_sha256=raw["nominal"]["sha256"],
        model_urdf=urdfs["model"], model_sha256=raw["model"]["sha256"],
        tool=dict(raw["tool"]), base=dict(raw["base"]),
        parameterization=raw["parameterization"],
        fitted_joints=tuple(raw.get("fitted_joints") or ()), joint_offsets=offsets,
        datasets=tuple(raw.get("datasets") or ()),
        rt_task_space=dict(raw.get("rt_task_space") or {}), raw=raw)


def _urdf_problems(urdf: Path, joints, frames: dict, fitted) -> list[str]:
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(urdf))
    out = []
    for name in joints:
        if not model.existJointName(name):
            out.append(f"model URDF has no joint {name!r}")
    slots = [model.joints[model.getJointId(n)].idx_q for n in joints if model.existJointName(n)]
    if slots != sorted(slots):
        out.append("bundle joint order differs from the URDF's kinematic order")
    for role, frame in frames.items():
        if not model.existFrame(frame):
            out.append(f"frames.{role} {frame!r} absent from the model URDF")
    import xml.etree.ElementTree as ET

    names = {j.get("name") for j in ET.parse(urdf).getroot().findall("joint")}
    for name in fitted:
        if name not in names:
            out.append(f"fitted joint {name!r} absent from the model URDF")
    return out


def compatibility_error(bundle: ModelBundle, *, instance_id: str, joints,
                        urdf: Path | str | None = None,
                        tool_root_link: str | None = None,
                        tcp_frame: str | None = None) -> str | None:
    """Why this bundle must not serve this instance/config, or None.

    A caller selecting an EE frame supplies it as ``tcp_frame``. Tool root,
    flange and TCP are distinct frames; matching the root does not bind the TCP.
    """
    if bundle.instance_id != instance_id:
        return f"bundle is for {bundle.instance_id!r}, not {instance_id!r}"
    if tuple(joints) != bundle.joints:
        return f"joint order {list(joints)} != bundle {list(bundle.joints)}"
    if urdf is not None and Path(urdf).resolve() != bundle.model_urdf:
        return f"runtime URDF {urdf} is not the bundle model {bundle.model_urdf}"
    if tool_root_link is not None and bundle.tool.get("root_link") not in (None, tool_root_link):
        return (f"tool root link {tool_root_link!r} != bundle tool "
                f"{bundle.tool.get('root_link')!r}")
    if tcp_frame is not None and tcp_frame != bundle.frames.get("tcp"):
        return f"selected TCP/EE frame {tcp_frame!r} != bundle TCP {bundle.frames.get('tcp')!r}"
    return None


def activation_error(bundle: ModelBundle) -> str | None:
    """Why this bundle may not be ACTIVATED for runtime use, or None."""
    if bundle.status not in ACTIVE_STATUSES:
        return f"status {bundle.status!r} is not activatable ({ACTIVE_STATUSES})"
    if bundle.joint_offsets is not None and np.any(bundle.joint_offsets):
        return ("nonzero joint_offsets: the raw<->model mapping is not wired into "
                "commands and limits; embed the correction in the URDF instead")
    return None


def chain(world_T_base, fk, flange_T_tcp=None):
    """``q_model -> world_T_tcp``; the single composition every consumer uses."""
    world_T_base = np.asarray(world_T_base, float)
    tail = np.eye(4) if flange_T_tcp is None else np.asarray(flange_T_tcp, float)

    def evaluate(q_model):
        return world_T_base @ fk(q_model) @ tail

    return evaluate


def base_to_base(world_T_base_a, world_T_base_b) -> np.ndarray:
    """``baseA_T_baseB`` -- the derived inter-arm transform, never stored."""
    return np.linalg.inv(np.asarray(world_T_base_a, float)) @ np.asarray(world_T_base_b, float)


def model_token(cfg) -> str | None:
    """The staged config's model binding token, or None when unbound."""
    raw = getattr(cfg, "raw", cfg)
    model = raw.get("model") if isinstance(raw, dict) else None
    return (model or {}).get("token")


def pose_provenance(cfg) -> str:
    raw = getattr(cfg, "raw", cfg)
    model = raw.get("model") if isinstance(raw, dict) else None
    return (model or {}).get("pose_provenance", "unbound")


def _self_check() -> None:
    import tempfile

    from arm_control import CONTROL_ROOT, frames
    from arm_control.kinematic_fit import ee_fk

    urdf = CONTROL_ROOT / "descriptions/franka/fr3_ft_gemini305_parallel/urdf/fr3.urdf"
    if not urdf.is_file():
        print(f"model_bundle: SKIPPED ({urdf} not present in this checkout)")
        return
    joints = [f"fr3_joint{i}" for i in range(1, 8)]
    sha = sha256_file(urdf)
    body = {
        "schema": SCHEMA, "bundle_id": "t-nominal", "revision": 1, "status": "nominal",
        "instance_id": "fr3_1", "robot_kind": "franka_fr3", "joints": joints,
        "frames": {"base": "fr3_link0", "flange": "fr3_link8", "tcp": "fr3_hand_tcp"},
        "nominal": {"urdf": str(urdf), "sha256": sha}, "model": {"urdf": str(urdf), "sha256": sha},
        "tool": {"id": "ft_gemini", "root_link": "fr3_ft_flange"},
        "base": {"source": "bench.yaml", "xyz": [-0.0875, 0.1585, 0.012], "rpy": [0, 0, 1.5708]},
        "parameterization": "none", "joint_offsets": None,
    }
    with tempfile.TemporaryDirectory() as tmp:
        def write(**changes):
            p = Path(tmp) / f"b{len(list(Path(tmp).iterdir()))}.yaml"
            p.write_text(yaml.safe_dump({**body, **changes}))
            return p

        b = load_bundle(write())
        assert not b.calibrated and b.pose_provenance == "nominal"
        assert b.token().startswith("fr3_1:t-nominal:r1:")
        assert compatibility_error(b, instance_id="fr3_1", joints=joints, urdf=urdf) is None
        assert "not 'fr3_2'" in compatibility_error(b, instance_id="fr3_2", joints=joints)
        assert "joint order" in compatibility_error(b, instance_id="fr3_1",
                                                    joints=joints[::-1])
        assert "tool root" in compatibility_error(b, instance_id="fr3_1", joints=joints,
                                                  tool_root_link="fr3_hand")
        assert compatibility_error(b, instance_id="fr3_1", joints=joints,
                                   tcp_frame="fr3_hand_tcp") is None
        assert "selected TCP/EE" in compatibility_error(b, instance_id="fr3_1", joints=joints,
                                                        tcp_frame="fr3_link8")
        assert "runtime URDF" in compatibility_error(b, instance_id="fr3_1", joints=joints,
                                                     urdf=Path(tmp) / "other.urdf")
        assert activation_error(b) is None
        for bad, why in (({"nominal": {"urdf": str(urdf), "sha256": "0" * 64}}, "sha256"),
                         ({"joints": joints[:6] + ["nope"]}, "no joint"),
                         ({"frames": {"tcp": "missing_frame"}}, "absent"),
                         ({"joints": [joints[1], joints[0], *joints[2:]]}, "kinematic order"),
                         ({"status": "blessed"}, "status"),
                         ({"units": {"length": "mm", "angle": "rad"}}, "units")):
            try:
                load_bundle(write(**bad))
            except ValueError as exc:
                assert why in str(exc), (why, exc)
            else:
                raise AssertionError(f"accepted a bundle with bad {why}")
        cand = load_bundle(write(status="candidate"))
        assert "not activatable" in activation_error(cand)
        off = load_bundle(write(joint_offsets=[0.0, 1e-3, 0, 0, 0, 0, -2e-3]))
        assert off.calibrated and "embed the correction" in activation_error(off)
        q = np.array([0.1, -0.4, 0.2, -2.0, 0.1, 1.6, 0.5])
        assert np.allclose(off.to_raw_q(off.to_model_q(q)), q, atol=0)
        assert np.allclose(b.to_model_q(q), q)

    # One chain: bench_T_tcp and the derived inter-arm transform.
    fk = ee_fk(urdf, "fr3_hand_tcp")
    a = frames.world_T_arm({"arm": {"world_origin": [-0.0875, 0.1585, 0.012],
                                    "world_rpy": [0, 0, np.pi / 2]}})
    bb = frames.world_T_arm({"arm": {"world_origin": [-1.0375, 0.1585, 0.012],
                                     "world_rpy": [0, 0, np.pi / 2]}})
    q = np.array([0.1, -0.4, 0.2, -2.0, 0.1, 1.6, 0.5])
    assert np.allclose(chain(a, fk)(q), a @ fk(q))
    rel = base_to_base(a, bb)
    assert np.allclose(a @ rel, bb) and np.allclose(rel[:3, 3], [0.0, 0.95, 0.0], atol=1e-12)
    print("model_bundle: schema/hash/joint-order/frame/status validation, compatibility, "
          "single-application offsets round trip, chain and base-to-base: PASS")


if __name__ == "__main__":
    _self_check()
