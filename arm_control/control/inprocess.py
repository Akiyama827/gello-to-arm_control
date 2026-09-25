"""In-process controller I/O: drive an ``ArmController`` without Dora.

Shared by every host that steps a controller in its own process (the
supervisor, the pair runtimes, bench checks).
"""
from __future__ import annotations

import pyarrow as pa


class QueueControllerNode:
    """An in-process controller node: collects outputs, feeds inputs.

    This is the local seam for driving an ``ArmController`` inside one
    process (bench checks, single-process demos). It implements exactly the
    two methods ``ArmController.run``/handlers use -- ``send_output`` and
    ``next`` -- plus ``drain_controller_events`` for a handle to poll.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, pa.Array]] = []
        self._inputs: list[dict] = []

    def send_output(self, topic: str, arrow) -> None:
        self.sent.append((topic, arrow))

    def inject(self, event: dict) -> None:
        """Queue one Dora-shaped event (``{"type": "INPUT", ...}``)."""
        self._inputs.append(event)

    def next(self, *, timeout: float = 0.0):
        if self._inputs:
            return self._inputs.pop(0)
        return None

    def drain_controller_events(self) -> list[pa.Array]:
        events = [arrow for topic, arrow in self.sent if topic == "controller_event"]
        self.sent = [(t, a) for t, a in self.sent if t != "controller_event"]
        return events
