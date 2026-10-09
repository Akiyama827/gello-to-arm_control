"""``python -m arm_control.contracts``: round-trip the planner/controller wire pair and the scene codecs (moved from the
retired ``arm_control.messages`` alias)."""
from __future__ import annotations

import numpy as np

from arm_control.contracts.motor import (
    pack_cartesian_block,
    pack_motor_command,
    unpack_motor_command,
)
from arm_control.contracts.motion import (
    pack_plan,
    unpack_plan,
    pack_control_update,
    unpack_control_update,
    pack_controller_event,
    unpack_controller_event,
)
from arm_control.contracts.scene import (
    scene_command_state,
    scene_state_payload,
    scene_state_from_payload,
)


def _self_check() -> None:
    """Round-trip the planner/controller wire pair."""
    n, samples = 7, 5
    times = np.linspace(0.0, 1.0, samples)
    q = np.tile(np.arange(n, dtype=float), (samples, 1))
    plan = unpack_plan(
        pack_plan(
            plan_id="p1", phase="final_approach", gated=True,
            times=times, positions=q, velocities=q * 0.0,
            kp=np.full(n, 1200.0), kd=np.full(n, 30.0),
            cartesian_poses=np.tile([0, 0, 0, 1, 0, 0, 0], (samples, 1)),
            cartesian={"frame": "base", "task_R": np.eye(3), "kc": np.ones(6), "dc": np.ones(6)},
            completion={"vel_tol": 0.002, "goal_q": np.arange(n, dtype=float),
                        "residual_max": 0.01},
        )
    )
    assert plan["plan_id"] == "p1" and plan["gated"] is True
    assert plan["positions"].shape == (samples, n), plan["positions"].shape
    assert plan["cartesian_poses"].shape == (samples, 7)
    assert plan["cartesian"]["task_R"].shape == (3, 3) and plan["cartesian"]["frame"] == "base"
    # The Cartesian spec is the robot BASE frame's, checked at pack time: a world-frame or out-of-envelope spec never ships.
    for bad in ({"task_R": np.eye(3), "kc": np.ones(6), "dc": np.ones(6)},
                {"frame": "world", "task_R": np.eye(3), "kc": np.ones(6), "dc": np.ones(6)},
                {"frame": "base", "task_R": np.eye(3), "kc": np.full(6, 2000.0), "dc": np.ones(6)}):
        try:
            pack_plan(plan_id="p", phase="x", gated=False, times=times, positions=q, velocities=q,
                      kp=np.ones(n), kd=np.ones(n), cartesian=bad)
        except ValueError:
            continue
        raise AssertionError(f"cartesian={bad} must be refused")
    # motor_command: the 21-value tail round-trips as the same spec, with no pose in it.
    command = unpack_motor_command(pack_motor_command(q[0], q[0], q[0], q[0], q[0],
                                                      pack_cartesian_block(plan["cartesian"])), n)
    assert command["cartesian"]["frame"] == "base" and "pose" not in command["cartesian"]
    assert np.array_equal(command["cartesian"]["task_R"], np.eye(3)) and np.array_equal(command["cartesian"]["kc"], np.ones(6))
    assert plan["completion"]["goal_q"].shape == (n,), plan["completion"]
    assert plan["completion"]["vel_tol"] == 0.002

    # A misspelled tolerance must not silently leave the leg on the default
    # rule: that is a leg judged done while it is still moving.
    for bad in ({"vel_tol": 0.1, "residual": 0.01}, {"vel_tol": 0.1, "goal_q": q[0]}):
        try:
            pack_plan(plan_id="p", phase="x", gated=False, times=times,
                      positions=q, velocities=q, kp=np.ones(n), kd=np.ones(n),
                      completion=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"completion={bad} must be refused")

    # A leg with no Cartesian block and no completion rule stays None, not
    # zeros -- the controller branches on exactly this.
    bare = unpack_plan(
        pack_plan(plan_id="p2", phase="lift", gated=False, times=times,
                  positions=q, velocities=q, kp=np.ones(n), kd=np.ones(n))
    )
    assert bare["cartesian"] is None and bare["completion"] is None
    assert bare["cartesian_poses"] is None

    # control: absent keys mean "unchanged", so an empty dict must survive and
    # an unknown key must be refused at PACK time, not silently ignored.
    assert unpack_control_update(pack_control_update()) == {}
    assert unpack_control_update(pack_control_update(arm=True, execute="p1")) == {
        "arm": True, "execute": "p1"
    }
    try:
        pack_control_update(payload={"mass_kg": 1.0}, bogus=1)
    except ValueError as exc:
        assert "bogus" in str(exc), exc
    else:
        raise AssertionError("pack_control_update accepted an unknown field")

    ev = unpack_controller_event(
        pack_controller_event(kind="ready", q=np.zeros(n))
    )
    assert ev["kind"] == "ready" and ev["q"].shape == (n,)
    ev = unpack_controller_event(
        pack_controller_event(kind="leg_result", plan_id="p1", ok=False, reason="x")
    )
    _check_scene_codec()
    assert ev["plan_id"] == "p1" and ev["ok"] is False and ev["q"] is None
    try:
        pack_controller_event(kind="nope")
    except ValueError:
        pass
    else:
        raise AssertionError("pack_controller_event accepted an unknown kind")
    print("contracts self-check ok")


def _check_scene_codec() -> None:
    """The two encoders differ ON PURPOSE; assert exactly how, in one place."""
    from arm_control.scene import Attachment, SceneState

    empty = SceneState(actor_q={}, attachments={}, constraints={}, revision=3)
    # A command with nothing to say about actors must not say "no actors".
    assert "actor_q" not in scene_command_state(empty), scene_command_state(empty)
    assert "revision" not in scene_command_state(empty)
    # The echo is authoritative: every key, every time.
    echo = scene_state_payload(empty)
    assert set(echo) == {"revision", "actor_q", "attachments", "constraints"}, echo
    assert echo["revision"] == 3 and echo["actor_q"] == {}

    full = SceneState(
        actor_q={"base": [0.1, 0.0]},
        attachments={"m": Attachment("part_a", "body", "p", "c", (0, 0, 0, 1, 0, 0, 0))},
        constraints={"hold": False},
        revision=7,
    )
    assert scene_command_state(full)["actor_q"] == {"base": [0.1, 0.0]}
    # Round trip both ways. A command's revision arrives as an ARGUMENT (it is
    # a sibling on the wire); demanding it inside the payload is the
    # KeyError('revision') this consolidation exists to prevent.
    back = scene_state_from_payload(empty, scene_command_state(full), revision=7)
    assert back.revision == 7 and back.actor_q == {"base": [0.1, 0.0]}
    assert back.attachments["m"].object_name == "part_a"
    assert back.constraints == {"hold": False}
    echoed = scene_state_from_payload(empty, scene_state_payload(full))
    assert echoed.revision == 7 and echoed.actor_q == {"base": [0.1, 0.0]}
    # Absent keys keep the CURRENT value rather than clearing it -- which is
    # what makes the omit-when-empty rule safe on the receiving side.
    kept = scene_state_from_payload(full, {"revision": 9, "attachments": {}})
    assert kept.actor_q == {"base": [0.1, 0.0]} and kept.constraints == {"hold": False}


if __name__ == "__main__":
    _self_check()
