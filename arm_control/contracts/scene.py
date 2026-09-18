"""Revision-checked scene commands, results, and state snapshots."""
from __future__ import annotations

import pyarrow as pa

from ._arrow import pack_json_message, unpack_json_message


def _scene_message(schema: str, body: dict) -> pa.Array:
    request_id = body.get("request_id")
    if schema != "scene_state" and (not isinstance(request_id, str) or not request_id):
        raise ValueError(f"{schema}.request_id must be a non-empty string")
    revision = body.get("revision")
    if not isinstance(revision, int) or revision < 0:
        raise ValueError(f"{schema}.revision must be a non-negative integer")
    return pack_json_message(schema, body)


def pack_scene_command(*, request_id: str, revision: int, state: dict) -> pa.Array:
    if not isinstance(state, dict):
        raise ValueError("scene_command.state must be a mapping")
    return _scene_message("scene_command", {"request_id": request_id, "revision": revision, "state": state})


def unpack_scene_command(payload: pa.Array) -> dict:
    body = unpack_json_message(payload, expected_schema="scene_command")
    _scene_message("scene_command", body)
    if not isinstance(body.get("state"), dict):
        raise ValueError("scene_command.state must be a mapping")
    return body


def pack_scene_result(*, request_id: str, revision: int, ok: bool, reason: str = "") -> pa.Array:
    return _scene_message("scene_result", {"request_id": request_id, "revision": revision, "ok": bool(ok), "reason": str(reason)})


def pack_scene_state(*, revision: int, actor_q: dict, attachments: dict, constraints: dict) -> pa.Array:
    return _scene_message("scene_state", {"revision": revision, "actor_q": actor_q, "attachments": attachments, "constraints": constraints})


# --------------------------------------------------------------------------- #
# SceneState <-> wire. These lived twice -- once in the caller's scene grafter,
# once in its MuJoCo interface -- written independently and genuinely divergent
# by the time anyone noticed: one took `revision` as an argument and one
# demanded it inside the payload, one omitted `actor_q` when empty and one
# always sent it. That divergence produced the KeyError('revision') that made
# every scene command fail. One copy, here, next to the messages they encode.
#
# There are TWO encoders because there are two messages, and they legitimately
# differ. Naming both is the point: a single encoder with a flag is how the
# rules drifted back into the callers last time.


def scene_command_state(state) -> dict:
    """The ``state`` field of a ``scene_command``: what the sender WANTS.

    No ``revision``: on a command that is a SIBLING of this dict, not a member
    (see ``pack_scene_command``), and a copy inside would be a second source of
    truth for the value the whole optimistic-concurrency check turns on.

    ``actor_q`` is OMITTED while empty, never sent as ``{}``. The plant reads a
    present-but-empty actor_q as "set every actor to nothing" and would drop the
    base tilt on the next recompile; absent means "no opinion, keep yours". A
    node learns actor_q only from a scene_state echo, and the plant publishes
    one only after a command -- so the FIRST command any node sends is always
    the empty case, and this is not a rare path.
    """
    payload: dict = {
        "attachments": {
            name: {
                "object_name": a.object_name,
                "body": a.body,
                "parent_frame": a.parent_frame,
                "child_frame": a.child_frame,
                "mate_pose": list(a.mate_pose),
            }
            for name, a in state.attachments.items()
        },
        "constraints": dict(state.constraints),
    }
    if state.actor_q:
        payload["actor_q"] = dict(state.actor_q)
    return payload


def scene_state_payload(state) -> dict:
    """A full ``scene_state`` echo: what the plant HAS. All four keys, always.

    This one is authoritative rather than a request, so nothing may be elided:
    a consumer bootstrapping from an echo needs every field, and ``revision``
    sits inside because there is no command to carry it alongside.
    """
    return {"revision": int(state.revision), "actor_q": dict(state.actor_q),
            **{k: v for k, v in scene_command_state(state).items()
               if k != "actor_q"}}


def scene_state_from_payload(current, payload: dict, revision: int | None = None):
    """Decode either message back onto ``current`` (absent keys keep its values).

    ``revision`` is passed in for a scene_COMMAND, whose state dict carries
    none by the rule above; a scene_STATE echo has it inside and may leave the
    argument out.
    """
    from arm_control.scene import Attachment, SceneState

    return SceneState(
        actor_q={
            name: list(values)
            for name, values in dict(
                payload.get("actor_q", current.actor_q)
            ).items()
        },
        attachments={
            name: Attachment(**value)
            for name, value in dict(payload.get("attachments", {})).items()
        },
        constraints={
            name: bool(value)
            for name, value in dict(
                payload.get("constraints", current.constraints)
            ).items()
        },
        revision=int(payload["revision"] if revision is None else revision),
    )
