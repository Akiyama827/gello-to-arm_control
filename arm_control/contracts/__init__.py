"""Wire contracts shared by producers and consumers."""
from arm_control.contracts._arrow import pack_json_message, unpack_json_message

__all__ = ["pack_json_message", "unpack_json_message"]
