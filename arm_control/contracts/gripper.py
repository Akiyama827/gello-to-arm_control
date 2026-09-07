"""Grasp request/result wire contracts shared by real and simulated bridges."""
from __future__ import annotations

import pyarrow as pa

from ._arrow import pack_json_message, unpack_json_message


# --------------------------------------------------------------------------- #
# Grasp request/result — the orchestrator <-> bridge wire contract. Lives here
# with the other codecs (NOT with docking policy): both real bridges, the sim
# bridge, and the orchestrator must agree on it, and this module is the one
# thing all of them already import.
# --------------------------------------------------------------------------- #


def pack_grasp_request(
    *,
    request_id: str,
    target_id: str,
    mode: str = "close",
    gripper_body: str = "gripper",
) -> pa.Array:
    return pack_json_message(
        "grasp_request",
        {
            "request_id": request_id,
            "target_id": target_id,
            "mode": mode,
            "gripper_body": gripper_body,
        },
    )


def unpack_grasp_request(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="grasp_request")


def pack_grasp_result(*, request_id: str, target_id: str, ok: bool, reason: str = "") -> pa.Array:
    return pack_json_message(
        "grasp_result",
        {
            "request_id": request_id,
            "target_id": target_id,
            "ok": bool(ok),
            "reason": reason,
        },
    )


def unpack_grasp_result(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema="grasp_result")
