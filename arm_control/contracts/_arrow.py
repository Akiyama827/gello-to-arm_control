"""Shared Arrow arrays and schema-tagged JSON envelopes."""
from __future__ import annotations

import json

import numpy as np
import pyarrow as pa


def _pack(arr: np.ndarray) -> pa.Array:
    return pa.array(np.asarray(arr, dtype=np.float64).ravel())


def _unpack(arrow: pa.Array) -> np.ndarray:
    return np.asarray(arrow, dtype=np.float64)


def pack_json_message(schema: str, payload: dict) -> pa.Array:
    """Pack a low-rate structured message as a one-element Arrow string array."""
    if not schema:
        raise ValueError("schema must be non-empty")
    body = {"schema": schema}
    body.update(payload)
    return pa.array([json.dumps(body, sort_keys=True, separators=(",", ":"))], type=pa.string())


def unpack_json_message(arrow: pa.Array, *, expected_schema: str | None = None) -> dict:
    if len(arrow) != 1:
        raise ValueError(f"unpack_json_message: expected 1 string value, got {len(arrow)}")
    raw = arrow[0].as_py()
    if not isinstance(raw, str):
        raise ValueError("unpack_json_message: expected Arrow string payload")
    body = json.loads(raw)
    if not isinstance(body, dict):
        raise ValueError("unpack_json_message: expected JSON object")
    schema = body.get("schema")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(f"unpack_json_message: expected schema {expected_schema}, got {schema}")
    return body


def _check_length(name: str, values: np.ndarray, expected: int) -> None:
    if values.size != expected:
        raise ValueError(f"{name}: expected {expected} float64 values, got {values.size}")
