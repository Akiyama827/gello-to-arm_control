"""The supervisor-hosted world (observe/1), republished on the Dora topics the in-process ``mujoco_interface`` plant node serves.

When a graph's plant moves into its arm's supervisor (scene hosting, 2026-10-08-scene-hosting-design.md), the nodes that read
the world -- simulated perception (``sim_qpos``), an in-hand slip monitor (``inhand_pose``) -- keep their inputs: this node reads
observe/1 through the SDK (read-only, no ownership) and publishes the same messages at the same rates, so a graph swaps the source
without touching a consumer. It never commands an arm.

``sim_qpos``: ``{"qpos": [...]}`` at most every 0.2 s; ``inhand_pose``: ``{"pose_xyzquat": [...]}`` (the module in the EE frame)
at most every 0.1 s, only while the world reports one (a world without a module reports none: nothing is sent, as the plant
node does); ``constraint_force``: ``{"reactions": ...}`` at most every 0.05 s; ``scene_state``: the world's scene echo every 2 s
and at once on a new revision (the current-value channel the plant node serves).

A ``scene_command`` input (a workcell world's scene change, from the graph's grafter) goes through scene_action/1: the node
takes the scene lease at its first command and answers each on ``scene_result`` (and the new ``scene_state``), as the plant
node did.
"""
from __future__ import annotations

import os
import time

#: the plant node's publication periods for the same topics (plants/mujoco/adapter.py)
PERIODS_S = {"sim_qpos": 0.2, "inhand_pose": 0.1, "constraint_force": 0.05, "scene_state": 2.0}
#: how often observe/1 is read (the supervisor's ring holds about a second at its 100 Hz default)
POLL_S = 0.05


class SceneObserver:
    """Turn observe/1 batches into ``(topic, body)`` messages, rate-limited per topic on the caller's clock."""

    def __init__(self) -> None:
        self.cursor = 0
        self.scene_revision = None  # the last scene revision published
        self._sent = {topic: float("-inf") for topic in PERIODS_S}

    def messages(self, batch: dict, now: float) -> list[tuple[str, dict]]:
        self.cursor = int(batch["latest_seq"])
        if not batch["samples"]:
            return []
        latest = batch["samples"][-1]
        out = []
        for topic, field, key in (("sim_qpos", "qpos", "qpos"), ("inhand_pose", "inhand_pose", "pose_xyzquat"),
                                  ("constraint_force", "constraint_force", "reactions")):
            value = latest.get(field)
            if value is not None and now - self._sent[topic] >= PERIODS_S[topic]:
                self._sent[topic] = now
                out.append((topic, {key: value if isinstance(value, dict) else list(value)}))
        scene = latest.get("scene_state")
        if scene is not None and (scene["revision"] != self.scene_revision
                                  or now - self._sent["scene_state"] >= PERIODS_S["scene_state"]):
            out.append(("scene_state", self.scene_sent(scene, now)))
        return out

    def scene_sent(self, scene: dict, now: float) -> dict:
        self.scene_revision, self._sent["scene_state"] = scene["revision"], now
        return scene


def main() -> None:
    """Dora node (``ARM_CONTROL_CONFIG`` names the arm by ``rt.instance_id``; ``ARM_CONTROL_SOCKET`` the supervisor socket)."""
    from dora import Node

    from arm_control.config import load_robot_config
    from arm_control.contracts.scene import pack_scene_result, pack_scene_state, unpack_scene_command
    from arm_control.messages import pack_json_message
    from arm_control.node_utils import ShutdownFlag, install_signal_handlers, next_event_gil_friendly
    from arm_control.sdk import ArmClient, Refused, SupervisorError

    def publish(topic, body):
        node.send_output(topic, pack_scene_state(**body) if topic == "scene_state" else pack_json_message(topic, body))

    def scene_command(value):
        request = unpack_scene_command(value)
        try:
            if not leased:
                leased.append(arm.scene_claim())
            reply = arm.scene_apply(request["request_id"], request["revision"], request["state"])
        except Refused as exc:
            node.send_output("scene_result", pack_scene_result(request_id=request["request_id"], revision=request["revision"],
                                                               ok=False, reason=str(exc)))
            return
        scene = reply["scene_state"]
        node.send_output("scene_result", pack_scene_result(request_id=request["request_id"], revision=scene["revision"], ok=True))
        publish("scene_state", observer.scene_sent(scene, time.monotonic()))

    shutdown = ShutdownFlag()
    install_signal_handlers(shutdown)
    instance_id = str((load_robot_config().get("rt") or {}).get("instance_id", ""))
    if not instance_id:
        raise SystemExit("config has no rt.instance_id: this node observes one supervised world")
    # a scene change recompiles the world while the call waits: seconds, not the SDK's default 5 s
    arm = ArmClient(instance_id, socket_path=os.environ.get("ARM_CONTROL_SOCKET") or None, timeout_s=60.0)
    node, observer, last, leased = Node(), SceneObserver(), 0.0, []
    try:
        print(f"[scene_observer] {instance_id}: observe/1 at {arm.observe(0, 1)['rate_hz']:g} Hz", flush=True)
        while not shutdown.stop_requested:
            event = next_event_gil_friendly(node)
            if event is not None and event["type"] == "STOP":
                break
            if event is not None and event["type"] == "INPUT" and event["id"] == "scene_command":
                scene_command(event["value"])
            now = time.monotonic()
            if now - last < POLL_S:
                continue
            last = now
            for topic, body in observer.messages(arm.observe(observer.cursor), now):
                publish(topic, body)
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
    scene = dict(revision=4, actor_q={}, attachments={}, constraints={})
    first = observer.messages(dict(latest_seq=7, samples=[dict(sample, seq=7, scene_state=scene, constraint_force={"w": 1.})]),
                              now=12.0)
    assert ("scene_state", scene) in first and ("constraint_force", {"reactions": {"w": 1.}}) in first, first
    assert [t for t, _ in observer.messages(dict(latest_seq=8, samples=[dict(sample, seq=8, scene_state=scene)]), now=12.5)
            if t == "scene_state"] == []                                                            # same revision, inside 2 s
    newer = dict(scene, revision=5)
    assert ("scene_state", newer) in observer.messages(dict(latest_seq=9, samples=[dict(sample, seq=9, scene_state=newer)]),
                                                       now=12.6)                                    # a new revision: at once
    print("scene_observer self-check: PASS (latest sample, per-topic periods, no inhand_pose reported -> none sent, cursor, "
          "scene_state on a new revision or every 2 s, constraint_force)")


if __name__ == "__main__":
    _self_check()
