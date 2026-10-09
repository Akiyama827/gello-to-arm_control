"""The supervisor-hosted world (observe/1), republished on the Dora topics the in-process ``mujoco_interface`` plant node serves.

When a graph's plant moves into its arm's supervisor (scene hosting, 2026-10-08-scene-hosting-design.md), the nodes that read
the world -- simulated perception (``sim_qpos``), an in-hand slip monitor (``inhand_pose``) -- keep their inputs: this node reads
observe/1 through the SDK (read-only, no ownership) and publishes the same messages at the same rates, so a graph swaps the source
without touching a consumer. It never commands anything.

``sim_qpos``: ``{"qpos": [...]}`` at most every 0.2 s; ``inhand_pose``: ``{"pose_xyzquat": [...]}`` (the module in the EE frame)
at most every 0.1 s, only while the world reports one (a world without a module reports none: nothing is sent, as the plant
node does).
"""
from __future__ import annotations

import os
import time

#: the plant node's publication periods for the same topics (plants/mujoco/adapter.py)
PERIODS_S = {"sim_qpos": 0.2, "inhand_pose": 0.1}
#: how often observe/1 is read (the supervisor's ring holds about a second at its 100 Hz default)
POLL_S = 0.05


class SceneObserver:
    """Turn observe/1 batches into ``(topic, body)`` messages, rate-limited per topic on the caller's clock."""

    def __init__(self) -> None:
        self.cursor = 0
        self._sent = {topic: float("-inf") for topic in PERIODS_S}

    def messages(self, batch: dict, now: float) -> list[tuple[str, dict]]:
        self.cursor = int(batch["latest_seq"])
        if not batch["samples"]:
            return []
        latest = batch["samples"][-1]
        out = []
        for topic, field, key in (("sim_qpos", "qpos", "qpos"), ("inhand_pose", "inhand_pose", "pose_xyzquat")):
            value = latest.get(field)
            if value is not None and now - self._sent[topic] >= PERIODS_S[topic]:
                self._sent[topic] = now
                out.append((topic, {key: list(value)}))
        return out


def main() -> None:
    """Dora node (``ARM_CONTROL_CONFIG`` names the arm by ``rt.instance_id``; ``ARM_CONTROL_SOCKET`` the supervisor socket)."""
    from dora import Node

    from arm_control.config import load_robot_config
    from arm_control.messages import pack_json_message
    from arm_control.node_utils import ShutdownFlag, install_signal_handlers, next_event_gil_friendly
    from arm_control.sdk import ArmClient, SupervisorError

    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    instance_id = str((load_robot_config().get("rt") or {}).get("instance_id", ""))
    if not instance_id:
        raise SystemExit("config has no rt.instance_id: this node observes one supervised world")
    arm = ArmClient(instance_id, socket_path=os.environ.get("ARM_CONTROL_SOCKET") or None)
    node, observer, last = Node(), SceneObserver(), 0.0
    try:
        print(f"[scene_observer] {instance_id}: observe/1 at {arm.observe(0, 1)['rate_hz']:g} Hz", flush=True)
        while not shutdown.stop_requested:
            event = next_event_gil_friendly(node)
            if event is not None and event["type"] == "STOP":
                break
            now = time.monotonic()
            if now - last < POLL_S:
                continue
            last = now
            for topic, body in observer.messages(arm.observe(observer.cursor), now):
                node.send_output(topic, pack_json_message(topic, body))
    except (SupervisorError, OSError, TimeoutError) as exc:  # a refused observe/1 (Refused) or a vanished supervisor
        raise SystemExit(f"[scene_observer] {instance_id}: the world is no longer observable: {exc}") from None
    finally:
        arm.close()


def _self_check() -> None:
    observer = SceneObserver()
    sample = dict(seq=3, qpos=[.1, .2], inhand_pose=[0, 0, .1, 1, 0, 0, 0])
    first = observer.messages(dict(latest_seq=3, samples=[dict(sample, seq=2), sample]), now=10.0)
    assert first == [("sim_qpos", {"qpos": [.1, .2]}), ("inhand_pose", {"pose_xyzquat": [0, 0, .1, 1, 0, 0, 0]})], first
    assert observer.cursor == 3
    assert observer.messages(dict(latest_seq=4, samples=[dict(sample, seq=4)]), now=10.05) == []  # both inside their periods
    later = observer.messages(dict(latest_seq=5, samples=[dict(sample, seq=5)]), now=10.15)
    assert [t for t, _ in later] == ["inhand_pose"], later                                         # 0.1 s passed, not 0.2 s
    assert observer.messages(dict(latest_seq=6, samples=[dict(sample, seq=6, inhand_pose=None)]), now=10.5) == [
        ("sim_qpos", {"qpos": [.1, .2]})]                                                           # none reported: none sent
    assert observer.messages(dict(latest_seq=6, samples=[]), now=11.0) == [] and observer.cursor == 6
    print("scene_observer self-check: PASS (latest sample, per-topic periods, no inhand_pose reported -> none sent, cursor)")


if __name__ == "__main__":
    _self_check()
