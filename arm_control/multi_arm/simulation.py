"""Two stock controllers, one MuJoCo world, advanced in simulation lockstep.

This runtime opens no hardware transports. Its common physics tick is a
simulation property and is never presented as a hardware synchronization result.
"""

from __future__ import annotations

import numpy as np
import mujoco

from arm_control.config import gripper_joints, ee_frame
from arm_control import frames
from arm_control.control.arm_controller import ArmController
from arm_control.control.factory import build_executor, gripper_command_cfg
from arm_control.messages import pack_motor_state, unpack_motor_command
from arm_control.multi_arm.interface import LocalArmHandle, QueueControllerNode
from arm_control.multi_arm.config import scene_for_mode
from arm_control.plants.mujoco.backend import MuJoCoBackend


class SharedSimulation:
    backend_name = "shared_sim"
    timing_status = "Simulation lockstep; hardware timing unqualified"
    fresh = True
    hand_busy = False

    def cancel_hands(self):
        self.set_fingers(self.finger_positions())

    def stop(self):
        self.cancel_hands()
        for handle in self.handles.values():
            handle.set_armed(False)

    @property
    def scene_model(self):
        return self.backend.model

    @property
    def scene_data(self):
        return self.backend.data

    def joint_state(self):
        return self.backend.motor_state()

    def telemetry(self):
        return {}

    def __init__(self, config, *, period=0.01):
        self.config, self.period, self.t = config, float(period), 0.0
        self.mode, self.mode_revision = "independent", 0
        scene = config.scene
        state = scene.state()
        if config.shared_object:
            state.constraints.update(
                {name: False for name in config.shared_object.grasp_welds}
            )
        self.finger_names = tuple(
            f"{a}__{j}"
            for a in scene.actor_names
            for j in gripper_joints(config.robots[a])
        )
        self.backend = MuJoCoBackend.from_workcell_scene(
            scene,
            state,
            control_period=period,
            ground_z=config.ground_z,
            enable_self_collision=True,
            launch_viewer=False,
            gravcomp_prefixes=tuple(f"{a}__" for a in scene.actor_names),
            gripper_joints=self.finger_names,
        )
        self.backend.load()
        if config.shared_object:
            obj = config.shared_object
            body = self.backend.model.body(obj.body).id
            joint = self.backend.model.body_jntadr[body]
            if (
                joint < 0
                or self.backend.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE
                or not np.allclose(self.object_pose(), obj.world_T_object, atol=1e-8)
            ):
                raise ValueError(
                    "shared object body must be its free root at the configured pose"
                )
        self.controllers, self.handles, self.slices = {}, {}, {}
        offset = 0
        for actor in scene.actors:
            a = actor.name
            self.slices[a] = slice(offset, offset + len(actor.joints))
            offset += len(actor.joints)
            cfg = config.robots[a]
            c = ArmController(
                QueueControllerNode(),
                build_executor(
                    cfg,
                    arm_id=a,
                    gravity=frames.world_T_arm(cfg)[:3, :3].T
                    @ self.backend.model.opt.gravity,
                ),
                arm_id=a,
                gripper=gripper_command_cfg(cfg),
                state_period_s=period,
                plant_reports_health=False,
                clock=lambda: self.t,
                model_token=config.model_tokens[a],
            )
            self.controllers[a] = c
            self.handles[a] = LocalArmHandle(c, clock=lambda: self.t)
        if offset != self.backend.num_motors:
            raise ValueError(
                "shared simulation supports actor motors only; object motors need a consumer adapter"
            )
        self.command = {
            k: np.zeros(offset) for k in ("position", "velocity", "torque", "kp", "kd")
        }
        self.command["position"] = self.backend.motor_state()["position"].copy()
        self.finger_limits = {}
        for a in scene.actor_names:
            ranges = [
                self.backend.model.jnt_range[self.backend.model.joint(f"{a}__{j}").id]
                for j in gripper_joints(config.robots[a])
            ]
            self.finger_limits[a] = (
                (float(max(r[0] for r in ranges)), float(min(r[1] for r in ranges)))
                if ranges
                else (0.0, 0.0)
            )
        self.fingers = {a: hi for a, (_, hi) in self.finger_limits.items()}
        self.set_fingers(self.fingers)
        from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld

        world = MuJoCoCollisionWorld.from_scene(
            scene_for_mode(config, self.mode),
            scene.state(),
            scene.actor_names,
            ground_z=config.ground_z,
            workspace_xy=config.workspace_xy,
            held_positions={
                f"{a}__{j}": self.fingers[a]
                for a in scene.actor_names
                for j in gripper_joints(config.robots[a])
            },
        )
        if config.shared_object:
            world.set_free_body_pose(config.shared_object.body, self.object_pose())
        q = np.concatenate([a.q for a in scene.actors])
        if world.in_collision(q):
            raise ValueError(
                f"initial scene collides: {world.collision_contacts(q)[:3]}"
            )
        self._collision = world
        self.step()

    @property
    def paused(self):
        # A shared simulated load needs support while disarmed. Pause this
        # entire plant; never claim physical
        # holding from disarmed controllers or silently remove payload gravity.
        return self.mode == "shared" and not all(
            c.armed for c in self.controllers.values()
        )

    def set_shared_grasps(self, enabled):
        """Switch simulated support after caller's grasp/support confirmation.

        This mutates constraints and payload accounting only; no model reload,
        joint reset, finger command or object repositioning is performed.
        """
        obj = self.config.shared_object
        if obj is None:
            raise ValueError("this setup has no shared object")
        mode = "shared" if enabled else "independent"
        if mode == self.mode:
            return
        fingers = self.finger_positions()
        self._collision.set_held_positions(
            {
                f"{a}__{j}": fingers[a]
                for a in self.config.scene.actor_names
                for j in gripper_joints(self.config.robots[a])
            }
        )
        self._collision.set_free_body_pose(obj.body, self.object_pose())
        q = self.backend.motor_state()["position"]
        if self._collision.in_collision(q):
            raise ValueError(
                "current scene collides; resolve it before changing grasp constraints"
            )
        if not enabled:
            support = self.backend.model.equality(obj.support).id
            self.backend.model.eq_data[support, 3:10] = frames.T_to_pose_xyzquat(
                self.object_pose()
            )
        for name in obj.grasp_welds:
            self.backend.set_constraint(name, enabled)
        self.backend.set_constraint(obj.support, not enabled)
        self.backend.data.qacc_warmstart[:] = 0.0
        body = self.backend.model.body(obj.body).id
        mass = float(self.backend.model.body_subtreemass[body])
        world_com = self.backend.data.subtree_com[body]
        object_com = np.linalg.inv(self.object_pose()) @ np.r_[world_com, 1.0]
        for a, c in self.controllers.items():
            tcp_com = np.linalg.inv(obj.grasps.object_T_tcp[a]) @ object_com
            c.executor.set_payload(
                mass / 2 if enabled else 0.0,
                ee_frame(self.config.robots[a]),
                com_offset=tcp_com[:3],
            )
        self.mode = mode
        self.mode_revision += 1

    def object_pose(self):
        obj = self.config.shared_object
        if obj is None:
            return None
        body = self.backend.data.body(obj.body)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = body.xmat.reshape(3, 3), body.xpos
        return T

    def set_fingers(self, values):
        for a, width in values.items():
            if (
                a not in self.fingers
                or not np.isfinite(width)
                or not self.finger_limits[a][0] <= width <= self.finger_limits[a][1]
            ):
                raise ValueError("unknown arm or finger width outside model limits")
            self.fingers[a] = float(width)
        widths = [
            self.fingers[a]
            for a in self.config.scene.actor_names
            for _ in gripper_joints(self.config.robots[a])
        ]
        if widths:
            self.backend.apply_gripper_command(widths)

    def finger_positions(self):
        positions = self.backend.gripper_positions()
        result, offset = {}, 0
        for a in self.config.scene.actor_names:
            n = len(gripper_joints(self.config.robots[a]))
            result[a] = float(np.mean(positions[offset : offset + n])) if n else 0.0
            offset += n
        return result

    def step(self):
        if self.paused:
            self.backend.data.qvel[:] = 0.0
            mujoco.mj_forward(self.backend.model, self.backend.data)
        state = self.backend.motor_state()
        if not all(np.isfinite(state[k]).all() for k in ("position", "velocity")):
            raise ValueError("non-finite simulated motor state")
        for a, sl in self.slices.items():
            c = self.controllers[a]
            zeros = np.zeros(c.n_arm)
            c.on_motor_state(
                pack_motor_state(
                    state["position"][sl],
                    state["velocity"][sl],
                    state["position_cmd"][sl],
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                    zeros,
                )
            )
            remaining = []
            for topic, value in c.node.sent:
                if topic == "motor_command":
                    command = unpack_motor_command(value, c.n_arm)
                    for k in self.command:
                        self.command[k][sl] = command[k]
                elif topic == "controller_event":
                    remaining.append((topic, value))
            c.node.sent = remaining
            if not c.armed:
                for k in ("torque", "kp", "kd"):
                    self.command[k][sl] = 0.0
        if not self.paused:
            self.backend.step(self.command)
        self.t += self.period

    def close(self):
        self.backend.close()
