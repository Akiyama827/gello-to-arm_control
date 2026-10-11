"""冒烟测试用假 plant：把收到的 jog 目标当实测位形回读（完美跟踪）。

不接任何硬件，只验证 leader_teleop 节点的 Dora 收发与安全层回读链路。
运行约 4.5s 后自行退出，让 `dora run` 结束。
"""
import time

import numpy as np
from dora import Node

from arm_control.messages import pack_json_message, pack_motor_state, unpack_jog

N = 7


def main() -> None:
    node = Node()
    q = np.zeros(N)
    start = time.monotonic()
    last_pub = 0.0
    node.send_output("motor_health", pack_json_message("motor_health", {"armed": True}))
    while time.monotonic() - start < 4.5:
        event = node.next(timeout=0.05)
        if event is not None and event["type"] == "STOP":
            break
        if event is not None and event["type"] == "INPUT" and event["id"] == "jog":
            q = np.asarray(unpack_jog(event["value"])["q"], dtype=float)
        now = time.monotonic()
        if now - last_pub >= 0.01:
            last_pub = now
            z = np.zeros(N)
            # position=q（完美跟踪），其余槽填零
            node.send_output("motor_state", pack_motor_state(q, z, q, z, z, z, z, z))
    print("[fake_plant] 退出", flush=True)


if __name__ == "__main__":
    main()
