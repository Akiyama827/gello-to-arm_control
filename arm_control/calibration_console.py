"""Safety and persistence primitives for the local calibration console."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import yaml


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
