"""Safety and persistence primitives for the local calibration console."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Mapping
from xml.etree import ElementTree as ET

import numpy as np
import yaml

from arm_control import frames
from arm_control.assets import asset_fingerprint


_ACTOR_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_CAPABILITIES = frozenset({"joint", "cartesian", "gripper", "wrench"})


@dataclass(frozen=True)
class ActorSpec:
    name: str
    joints: tuple[str, ...]
    capabilities: frozenset[str]
    urdf: str | None = None
    ee_frame: str | None = None

    @property
    def command_port(self) -> str:
        return f"{self.name}_command"

    @property
    def arm_port(self) -> str:
        return f"{self.name}_arm"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


class ActorCatalog:
    def __init__(self, actors: tuple[ActorSpec, ...]):
        self._actors = {actor.name: actor for actor in actors}

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "ActorCatalog":
        actors = []
        for name, value in raw.items():
            if not isinstance(name, str) or _ACTOR_NAME.fullmatch(name) is None:
                raise ValueError(f"invalid actor name: {name!r}")
            if not isinstance(value, Mapping):
                raise ValueError(f"actor {name!r} must be a mapping")
            joints_raw = value.get("joints")
            if (
                not isinstance(joints_raw, list)
                or not joints_raw
                or any(not isinstance(joint, str) or not joint for joint in joints_raw)
                or len(set(joints_raw)) != len(joints_raw)
            ):
                raise ValueError(f"actor {name!r} joints must be non-empty and unique")
            capabilities_raw = value.get("capabilities", ["joint"])
            if (
                not isinstance(capabilities_raw, list)
                or not capabilities_raw
                or any(capability not in _CAPABILITIES for capability in capabilities_raw)
            ):
                raise ValueError(f"actor {name!r} has unsupported capabilities")
            capabilities = frozenset(capabilities_raw)
            urdf = value.get("urdf")
            ee = value.get("ee_frame")
            if "cartesian" in capabilities and (
                not isinstance(urdf, str) or not isinstance(ee, str)
            ):
                raise ValueError(f"Cartesian actor {name!r} requires urdf and ee_frame")
            actors.append(
                ActorSpec(
                    name,
                    tuple(joints_raw),
                    capabilities,
                    urdf if isinstance(urdf, str) else None,
                    ee if isinstance(ee, str) else None,
                )
            )
        if not actors:
            raise ValueError("actor catalog must be non-empty")
        return cls(tuple(actors))

    def __getitem__(self, name: str) -> ActorSpec:
        return self._actors[name]

    def __iter__(self):
        return iter(self._actors.values())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._actors)


@dataclass
class ConsoleAuthority:
    actor_names: tuple[str, ...]
    deadman_timeout_s: float
    selected: str | None = None
    deadman_actor: str | None = None
    deadman_seen_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.actor_names or len(set(self.actor_names)) != len(self.actor_names):
            raise ValueError("actor_names must be non-empty and unique")
        if self.deadman_timeout_s <= 0.0:
            raise ValueError("deadman_timeout_s must be positive")

    @staticmethod
    def _stop(actor: str | None) -> list[tuple[str, str]]:
        return [] if actor is None else [("hold", actor), ("disarm", actor)]

    def select(self, actor: str, *, now: float) -> list[tuple[str, str]]:
        del now
        if actor not in self.actor_names:
            raise KeyError(f"unknown actor: {actor}")
        actions = self._stop(self.selected) if self.selected != actor else []
        self.selected = actor
        self.deadman_actor = None
        return actions

    def set_deadman(self, held: bool, *, now: float) -> list[tuple[str, str]]:
        if held:
            if self.selected is None:
                raise ValueError("select an actor before holding the deadman")
            self.deadman_actor = self.selected
            self.deadman_seen_s = float(now)
            return []
        actor = self.deadman_actor
        self.deadman_actor = None
        return self._stop(actor)

    def may_move(self, actor: str, *, now: float) -> bool:
        return (
            actor == self.selected == self.deadman_actor
            and float(now) - self.deadman_seen_s <= self.deadman_timeout_s
        )

    def expire(self, *, now: float) -> list[tuple[str, str]]:
        if self.deadman_actor is None or self.may_move(
            self.deadman_actor, now=now
        ):
            return []
        actor = self.deadman_actor
        self.deadman_actor = None
        return self._stop(actor)


class CalibrationStore:
    def __init__(self, allowed_roots: tuple[Path, ...]):
        if not allowed_roots:
            raise ValueError("allowed_roots must be non-empty")
        self.allowed_roots = tuple(Path(root).resolve() for root in allowed_roots)

    def _target(self, path: Path) -> Path:
        target = Path(path).resolve()
        if not any(target.is_relative_to(root) for root in self.allowed_roots):
            raise ValueError("calibration path is outside allowed roots")
        return target

    def revision(self, path: Path) -> str:
        target = self._target(path)
        return hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else "missing"

    def save_yaml(
        self,
        path: Path,
        value: Mapping[str, object],
        *,
        expected_revision: str,
    ) -> str:
        target = self._target(path)
        if self.revision(target) != expected_revision:
            raise ValueError("stale revision")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_dump(dict(value), sort_keys=False).encode()
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if target.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                backup = target.with_name(f"{target.name}.{stamp}.bak")
                shutil.copy2(target, backup)
                with backup.open("rb") as handle:
                    os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
            directory = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return self.revision(target)


@dataclass(frozen=True)
class GraspProfile:
    module_type: str
    calibration_status: str
    reference_link: str
    link_T_ee: np.ndarray
    approach_offset_m: np.ndarray
    retreat_offset_m: np.ndarray


def _vector(raw: object, size: int, label: str) -> np.ndarray:
    value = np.asarray(raw, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"{label} must contain {size} finite values")
    return value


def validate_grasp_profile(
    raw: Mapping[str, object],
    *,
    module_urdf: str | Path,
) -> GraspProfile:
    """Validate and normalize one module-local, asset-bound grasp profile."""
    urdf = Path(module_urdf).resolve(strict=True)
    if raw.get("version") != 1:
        raise ValueError("grasp profile version must be 1")
    module_type = raw.get("module_type")
    if not isinstance(module_type, str) or not module_type:
        raise ValueError("module_type is required")
    status = raw.get("calibration_status")
    if status not in {"draft", "approved"}:
        raise ValueError("calibration_status must be draft or approved")
    if raw.get("source_sha256") != asset_fingerprint(urdf):
        raise ValueError("source fingerprint does not match module asset")
    grasp = raw.get("grasp")
    if not isinstance(grasp, Mapping):
        raise ValueError("grasp must be a mapping")
    reference = grasp.get("reference_link")
    links = {link.get("name") for link in ET.parse(urdf).getroot().findall("link")}
    if not isinstance(reference, str) or reference not in links:
        raise ValueError("grasp.reference_link is not a module link")
    pose = grasp.get("link_T_ee")
    if not isinstance(pose, Mapping):
        raise ValueError("grasp.link_T_ee must be a mapping")
    position = _vector(pose.get("pos"), 3, "grasp.link_T_ee.pos")
    quaternion = _vector(pose.get("quat"), 4, "grasp.link_T_ee.quat")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("grasp.link_T_ee.quat must be non-zero")
    link_T_ee = frames.T_from_spec({"pos": position, "quat": quaternion / norm})
    return GraspProfile(
        module_type,
        str(status),
        reference,
        link_T_ee,
        _vector(grasp.get("approach_offset_m"), 3, "grasp.approach_offset_m"),
        _vector(grasp.get("retreat_offset_m"), 3, "grasp.retreat_offset_m"),
    )
