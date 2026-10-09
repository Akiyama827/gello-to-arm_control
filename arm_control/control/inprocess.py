"""In-process controller I/O: drive an ``ArmController`` without Dora.

Shared by every host that steps a controller in its own process (the
supervisor, the pair runtimes, bench checks).
"""
from __future__ import annotations

import pyarrow as pa


class QueueControllerNode:
    """An in-process controller node: collects outputs, feeds inputs.

    This is the local seam for driving an ``ArmController`` inside one
    process (the supervisor, bench checks). It implements the one method the
    controller's handlers use, ``send_output``, plus ``drain_controller_events``
    for a host to poll.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, pa.Array]] = []

    def send_output(self, topic: str, arrow) -> None:
        self.sent.append((topic, arrow))

    def drain_controller_events(self) -> list[pa.Array]:
        events = [arrow for topic, arrow in self.sent if topic == "controller_event"]
        self.sent = [(t, a) for t, a in self.sent if t != "controller_event"]
        return events
