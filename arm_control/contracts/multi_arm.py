"""Multi-arm coordination wire contracts.

The coordination layer above two (or more) single-arm controllers speaks these
schemas. They AGGREGATE existing per-arm primitives -- a ``plan`` payload, a
``controller_event`` payload -- and never replace them: there is no flat
14-DOF representation anywhere in this module. Arm identity is preserved by
keying every member value by arm id (``"left"``, ``"right"``, or whatever the
consuming project names its arms).
"""

from __future__ import annotations

import pyarrow as pa

from ._arrow import pack_json_message, unpack_json_message

BIMANUAL_PLAN_SCHEMA = "bimanual_plan"
BIMANUAL_COMMAND_SCHEMA = "bimanual_command"
BIMANUAL_STATE_SCHEMA = "bimanual_state"
PAIR_RESULT_SCHEMA = "pair_result"
BIMANUAL_ARM_COMMAND_SCHEMA = "bimanual_arm_command"


def plan_payload_from_json(raw: str) -> pa.Array:
    """Rebuild one ``plan`` wire payload from its JSON text.

    A bimanual plan nests per-member ``plan`` payloads inside one envelope.
    The member payload is opaque to coordination -- it is passed through to
    the member controller byte-identically -- so it travels as the same
    one-element Arrow string array a single-arm graph sends.
    """
    return pa.array([raw], type=pa.string())


def pack_pair_plan(*, pair_id: str, plans: dict[str, pa.Array]) -> pa.Array:
    """Planner -> coordinator. One pair: ``pair_id`` plus one plan per arm.

    ``plans`` maps arm id -> an existing ``pack_plan`` payload. The member
    plan ids are NOT rewritten here: the pair keeps whatever ids the producer
    staged under, and the coordinator correlates ACKs/results against them.

    Exactly two distinct arm ids are required. Names come from the consumer;
    extra members are rejected, never trimmed.
    """
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("bimanual_plan.pair_id must be a non-empty string")
    if not isinstance(plans, dict) or not plans:
        raise ValueError(
            "bimanual_plan.plans must be a non-empty mapping of arm id -> plan payload"
        )
    if len(plans) != 2:
        raise ValueError(
            "bimanual_plan.plans must name exactly "
            f"two distinct arms for the bimanual layer, got "
            f"{sorted(plans)}"
        )
    members = {}
    for arm_id, payload in plans.items():
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("bimanual_plan.plans keys must be non-empty arm ids")
        if len(payload) != 1:
            raise ValueError(
                f"bimanual_plan member {arm_id!r} must be a one-element plan payload"
            )
        raw = payload[0].as_py()
        if not isinstance(raw, str):
            raise ValueError(
                f"bimanual_plan member {arm_id!r} must be a plan JSON payload"
            )
        members[arm_id] = raw
    return pack_json_message(
        BIMANUAL_PLAN_SCHEMA, {"pair_id": pair_id, "plans": members}
    )


def unpack_pair_plan(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema=BIMANUAL_PLAN_SCHEMA)
    pair_id = body.get("pair_id")
    plans = body.get("plans")
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("bimanual_plan.pair_id must be a non-empty string")
    if (
        not isinstance(plans, dict)
        or len(plans) != 2
        or any(not isinstance(k, str) or not k for k in plans)
        or any(not isinstance(v, str) for v in plans.values())
    ):
        raise ValueError(
            "bimanual_plan.plans must name exactly "
            f"two distinct arms for the bimanual layer, got "
            f"{sorted(plans) if isinstance(plans, dict) else plans!r}"
        )
    return body


def pack_arm_command(
    *, arm_id: str, action: str, value=None, reason: str = ""
) -> pa.Array:
    """Planner/operator -> coordinator. One PER-ARM control-plane command.

    ``action`` is one of:

    ``arm``      ``value`` bool: arm (True) or disarm (False) that arm's
                 controller through its normal control path. Arming is a
                 separate lifecycle operation -- a plan never arms an arm.
    ``gripper``  ``value`` float: commanded finger width in metres.
    ``mode``     ``value`` mapping: ``{"law": "joint"}`` or
                 ``{"law": "soft", "spec": {...}}`` where the plant supports it.
    ``gains``    ``value`` mapping: optional ``kp`` / ``kd`` joint vectors.

    The coordinator forwards each command to THAT arm's handle only; it never
    becomes a second motor-command authority itself.
    """
    if not isinstance(arm_id, str) or not arm_id:
        raise ValueError("bimanual_arm_command.arm_id must be a non-empty string")
    if action not in ("arm", "gripper", "mode", "gains"):
        raise ValueError(
            f"bimanual_arm_command action must be arm/gripper/mode/gains, got {action!r}"
        )
    body: dict = {"arm_id": arm_id, "action": action, "reason": str(reason)}
    if value is not None:
        body["value"] = value
    return pack_json_message(BIMANUAL_ARM_COMMAND_SCHEMA, body)


def unpack_arm_command(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema=BIMANUAL_ARM_COMMAND_SCHEMA)
    if not isinstance(body.get("arm_id"), str) or not body["arm_id"]:
        raise ValueError("bimanual_arm_command.arm_id must be a non-empty string")
    if body.get("action") not in ("arm", "gripper", "mode", "gains"):
        raise ValueError(
            f"bimanual_arm_command action must be arm/gripper/mode/gains, got {body.get('action')!r}"
        )
    return body


def pack_pair_command(*, pair_id: str, action: str, reason: str = "") -> pa.Array:
    """Planner -> coordinator. ``action`` is ``execute`` or ``cancel``."""
    if not isinstance(pair_id, str) or not pair_id:
        raise ValueError("bimanual_command.pair_id must be a non-empty string")
    if action not in ("execute", "cancel"):
        raise ValueError(
            f"bimanual_command action must be execute or cancel, got {action!r}"
        )
    return pack_json_message(
        BIMANUAL_COMMAND_SCHEMA,
        {"pair_id": pair_id, "action": action, "reason": str(reason)},
    )


def unpack_pair_command(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema=BIMANUAL_COMMAND_SCHEMA)
    if not isinstance(body.get("pair_id"), str) or not body["pair_id"]:
        raise ValueError("bimanual_command.pair_id must be a non-empty string")
    if body.get("action") not in ("execute", "cancel"):
        raise ValueError(
            f"bimanual_command action must be execute or cancel, got {body.get('action')!r}"
        )
    return body


def pack_bimanual_state(payload: dict) -> pa.Array:
    """Coordinator -> consumers. The aggregated state echo.

    ``payload`` is the mapping produced by
    ``arm_control.multi_arm.types.BimanualState.to_payload``; this module only
    owns the envelope so producers and consumers agree on the schema name.
    """
    if not isinstance(payload, dict):
        raise ValueError("bimanual_state payload must be a mapping")
    return pack_json_message(BIMANUAL_STATE_SCHEMA, payload)


def unpack_bimanual_state(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema=BIMANUAL_STATE_SCHEMA)


def pack_pair_result(payload: dict) -> pa.Array:
    """Coordinator -> consumers. One terminal pair result.

    Same split as :func:`pack_bimanual_state`: the content is owned by
    ``multi_arm.types.PairState.to_payload``; the envelope lives here.
    """
    if not isinstance(payload, dict):
        raise ValueError("pair_result payload must be a mapping")
    return pack_json_message(PAIR_RESULT_SCHEMA, payload)


def unpack_pair_result(payload: pa.Array) -> dict:
    return unpack_json_message(payload, expected_schema=PAIR_RESULT_SCHEMA)
