"""Generic, policy-free workcell scene loading and runtime state."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

import yaml

from arm_control import CONTROL_ROOT


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _names(raw: Any, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw or any(not isinstance(v, str) or not v for v in raw):
        raise ValueError(f"{label} must be a non-empty list of names")
    if len(set(raw)) != len(raw):
        raise ValueError(f"{label} contains duplicate names")
    return tuple(raw)


def _vec(raw: Any, label: str, *, positive: bool = False) -> tuple[float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError(f"{label} must have exactly three values")
    values = tuple(float(value) for value in raw)
    if not all(isfinite(value) for value in values):
        raise ValueError(f"{label} must be finite")
    if positive and not all(value > 0.0 for value in values):
        raise ValueError(f"{label} must be strictly positive")
    return values


def _path(raw: Any, source: Path, label: str) -> Path:
    if not raw:
        raise ValueError(f"{label}.path is required")
    path = Path(str(raw))
    if not path.is_absolute():
        local = source.parent / path
        path = local if local.exists() else CONTROL_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}.path not found: {path}")
    return path


@dataclass(frozen=True)
class ActorSpec:
    name: str
    path: Path
    joints: tuple[str, ...]
    pos: tuple[float, float, float]
    rpy: tuple[float, float, float]
    q: tuple[float, ...]


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    path: Path
    pos: tuple[float, float, float]
    rpy: tuple[float, float, float]


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    shape: str
    size: tuple[float, float, float] | None
    pos: tuple[float, float, float]
    rpy: tuple[float, float, float]
    path: Path | None = None


@dataclass(frozen=True)
class Attachment:
    object_name: str
    body: str
    parent_frame: str
    child_frame: str
    mate_pose: tuple[float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if not all((self.object_name, self.body, self.parent_frame, self.child_frame)):
            raise ValueError("attachment names must be non-empty")
        if len(self.mate_pose) != 7 or not all(isfinite(float(value)) for value in self.mate_pose):
            raise ValueError("attachment.mate_pose must be seven finite values")


@dataclass(frozen=True)
class SceneSpec:
    actors: tuple[ActorSpec, ...]
    objects: tuple[ObjectSpec, ...]
    obstacles: tuple[ObstacleSpec, ...]

    def state(self) -> "SceneState":
        return SceneState(actor_q={actor.name: list(actor.q) for actor in self.actors})

    @property
    def actor_names(self) -> tuple[str, ...]:
        return tuple(actor.name for actor in self.actors)

    @property
    def object_names(self) -> tuple[str, ...]:
        return tuple(obj.name for obj in self.objects)


@dataclass
class SceneState:
    actor_q: dict[str, list[float]]
    attachments: dict[str, Attachment] | None = None
    constraints: dict[str, bool] | None = None
    revision: int = 0

    def __post_init__(self) -> None:
        self.actor_q = {name: list(q) for name, q in self.actor_q.items()}
        self.attachments = dict(self.attachments or {})
        self.constraints = dict(self.constraints or {})

    def attach(self, spec: SceneSpec, attachment: Attachment) -> None:
        if attachment.object_name not in spec.object_names:
            raise KeyError(f"unknown scene object: {attachment.object_name}")
        if attachment.object_name in self.attachments:
            raise ValueError(f"object already attached: {attachment.object_name}")
        self.attachments[attachment.object_name] = attachment
        self.revision += 1

    def detach(self, spec: SceneSpec, object_name: str) -> Attachment:
        if object_name not in spec.object_names:
            raise KeyError(f"unknown scene object: {object_name}")
        try:
            attachment = self.attachments.pop(object_name)
        except KeyError as exc:
            raise ValueError(f"object is not attached: {object_name}") from exc
        self.revision += 1
        return attachment

    def set_actor_q(self, spec: SceneSpec, actor_name: str, q: list[float]) -> None:
        actor = next((actor for actor in spec.actors if actor.name == actor_name), None)
        if actor is None:
            raise KeyError(f"unknown scene actor: {actor_name}")
        values = [float(value) for value in q]
        if len(values) != len(actor.joints) or not all(isfinite(value) for value in values):
            raise ValueError(f"{actor_name}.q must contain {len(actor.joints)} finite values")
        self.actor_q[actor_name] = values
        self.revision += 1

    def set_constraint(self, name: str, active: bool) -> None:
        if not name:
            raise ValueError("constraint name must be non-empty")
        self.constraints[name] = bool(active)
        self.revision += 1


def load_scene(path: str | Path) -> SceneSpec:
    source = Path(path).resolve()
    raw = _mapping(yaml.safe_load(source.read_text()) or {}, str(source))
    if raw.get("version") != 1:
        raise ValueError("scene version must be 1")
    scene = _mapping(raw.get("scene"), "scene")
    actor_entries = _mapping(scene.get("actors"), "scene.actors")
    object_entries = _mapping(scene.get("objects", {}), "scene.objects")
    obstacle_entries = _mapping(scene.get("obstacles", {}), "scene.obstacles")
    names = [*actor_entries, *object_entries, *obstacle_entries]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ValueError("scene names must be non-empty and unique")

    actors: list[ActorSpec] = []
    for name, raw_actor in actor_entries.items():
        actor = _mapping(raw_actor, f"scene.actors.{name}")
        joints = _names(actor.get("joints"), f"scene.actors.{name}.joints")
        q = tuple(float(value) for value in actor.get("q", [0.0] * len(joints)))
        if len(q) != len(joints) or not all(isfinite(value) for value in q):
            raise ValueError(f"scene.actors.{name}.q must contain {len(joints)} finite values")
        actors.append(ActorSpec(name, _path(actor.get("path"), source, f"scene.actors.{name}"), joints,
                                _vec(actor.get("pos", (0, 0, 0)), f"scene.actors.{name}.pos"),
                                _vec(actor.get("rpy", (0, 0, 0)), f"scene.actors.{name}.rpy"), q))

    objects = tuple(
        ObjectSpec(name, _path(_mapping(value, f"scene.objects.{name}").get("path"), source,
                               f"scene.objects.{name}"),
                   _vec(_mapping(value, f"scene.objects.{name}").get("pos", (0, 0, 0)), f"scene.objects.{name}.pos"),
                   _vec(_mapping(value, f"scene.objects.{name}").get("rpy", (0, 0, 0)), f"scene.objects.{name}.rpy"))
        for name, value in object_entries.items()
    )
    obstacles: list[ObstacleSpec] = []
    for name, value in obstacle_entries.items():
        obstacle = _mapping(value, f"scene.obstacles.{name}")
        shape = obstacle.get("shape")
        pos = _vec(obstacle.get("pos", (0, 0, 0)), f"scene.obstacles.{name}.pos")
        rpy = _vec(obstacle.get("rpy", (0, 0, 0)), f"scene.obstacles.{name}.rpy")
        if shape == "box":
            size = _vec(obstacle.get("size"), f"scene.obstacles.{name}.size", positive=True)
            obstacles.append(ObstacleSpec(name, shape, size, pos, rpy))
        elif shape == "mesh":
            path = _path(obstacle.get("path"), source, f"scene.obstacles.{name}")
            obstacles.append(ObstacleSpec(name, shape, None, pos, rpy, path))
        else:
            raise ValueError(f"scene.obstacles.{name}.shape must be 'box' or 'mesh'")
    if not actors:
        raise ValueError("scene.actors must be non-empty")
    return SceneSpec(tuple(actors), objects, tuple(obstacles))


def _demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        model = root / "model.xml"
        model.write_text("<mujoco model='minimal'><worldbody/></mujoco>")
        scene_path = root / "scene.yaml"
        text = """version: 1
scene:
  actors:
    welder: {path: model.xml, joints: [a], pos: [0, 0, 0], rpy: [0, 0, 0]}
    turntable: {path: model.xml, joints: [b], pos: [1, 0, 0], rpy: [0, 0, 0], q: [0.2]}
  objects:
    first: {path: model.xml, pos: [0, 0, 0], rpy: [0, 0, 0]}
    second: {path: model.xml, pos: [0, 1, 0], rpy: [0, 0, 0]}
  obstacles:
    wall: {shape: box, size: [1, 1, 1], pos: [0, 0, 0], rpy: [0, 0, 0]}
assembly: {ignored: true}
"""
        scene_path.write_text(text)
        spec = load_scene(scene_path)
        assert spec.actor_names == ("welder", "turntable")
        assert spec.object_names == ("first", "second")
        state = spec.state()
        state.attach(spec, Attachment("first", "part", "welder_tip", "inward", (0, 0, 0, 1, 0, 0, 0)))
        assert state.detach(spec, "first").object_name == "first"
        assert state.revision == 2 and scene_path.read_text() == text
        scene_path.write_text(text.replace("[0, 0, 0]", "[nan, 0, 0]", 1))
        try:
            load_scene(scene_path)
        except ValueError:
            pass
        else:
            raise AssertionError("non-finite pose accepted")
    print("scene: OK")


if __name__ == "__main__":
    _demo()
