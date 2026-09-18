"""用同一个 MuJoCo 模型做"两臂最近距离"的碰撞守卫。

当 leader 与 follower 都在同一个 :class:`mujoco.MjModel` 里时，可以直接用
``mj_geomDistance`` 逐对计算几何最近距离（负值 = 已穿透），比球体近似保真得多。
这正好接上 ``SafetyMonitor`` 的 ``CollisionGuard`` 协议：

* :meth:`set_other_state` 记下当前小臂状态（每 tick 由 TeleopLoop 传入）；
* :meth:`min_distance` 用**候选大臂目标** + 当前小臂位形，在私有
  :class:`mujoco.MjData` 上 ``mj_forward`` 后算最小距离。

私有 ``MjData`` 与查看器线程用的那份互不干扰；``MjModel`` 只读，可多线程共享。
"""
from __future__ import annotations

from typing import Optional, Sequence

import mujoco
import numpy as np


class MjGeomDistanceGuard:
    """同场景几何最近距离碰撞守卫（实现 ``CollisionGuard`` 协议）。"""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        follower_arm_qpos_adr: Sequence[int],
        leader_arm_qpos_adr: Sequence[int],
        leader_grip_qpos_adr: Optional[int] = None,
        leader_geom_ids: Sequence[int] = (),
        follower_geom_ids: Sequence[int] = (),
        distmax_m: float = 0.30,
        grip_travel_m: float = 0.04,
    ) -> None:
        self._model = model
        self._data = mujoco.MjData(model)
        self._follower_adr = [int(a) for a in follower_arm_qpos_adr]
        self._leader_adr = [int(a) for a in leader_arm_qpos_adr]
        self._leader_grip_adr = None if leader_grip_qpos_adr is None else int(leader_grip_qpos_adr)
        self._leader_geoms = [int(g) for g in leader_geom_ids]
        self._follower_geoms = [int(g) for g in follower_geom_ids]
        self._distmax = float(distmax_m)
        self._grip_travel = float(grip_travel_m)
        self._other: Optional[np.ndarray] = None
        self._fromto = np.zeros(6, dtype=float)

    def set_other_state(self, other_state: Sequence[float]) -> None:
        self._other = np.asarray(other_state, dtype=float).ravel()

    def min_distance(self, arm_q: Sequence[float]) -> float:
        other = self._other
        if other is None:
            return float("inf")

        d = self._data
        for i, adr in enumerate(self._leader_adr):
            if i < other.size:
                d.qpos[adr] = float(other[i])
        if self._leader_grip_adr is not None and other.size > len(self._leader_adr):
            g = float(np.clip(other[len(self._leader_adr)], 0.0, 1.0))
            d.qpos[self._leader_grip_adr] = g * self._grip_travel

        arm = np.asarray(arm_q, dtype=float).ravel()
        for i, adr in enumerate(self._follower_adr):
            if i < arm.size:
                d.qpos[adr] = float(arm[i])

        mujoco.mj_forward(self._model, d)

        best = self._distmax
        for g1 in self._leader_geoms:
            for g2 in self._follower_geoms:
                dist = mujoco.mj_geomDistance(
                    self._model, d, g1, g2, self._distmax, self._fromto
                )
                if dist < best:
                    best = float(dist)
                    if best <= -self._distmax:
                        return best
        return float(best)

    def close(self) -> None:
        pass


def select_geoms_by_name(
    model: mujoco.MjModel,
    prefix: str,
    *,
    group: Optional[int] = None,
) -> list[int]:
    """按 geom 名字前缀（可选 group）挑几何 id，供守卫使用。"""
    out: list[int] = []
    for i in range(model.ngeom):
        name = model.geom(i).name or ""
        if not name.startswith(prefix):
            continue
        if group is not None and int(model.geom_group[i]) != int(group):
            continue
        out.append(i)
    return out
