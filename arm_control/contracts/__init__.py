"""Wire contracts shared by producers and consumers."""


def __getattr__(name):
    # The JSON codecs' public home. Lazy, so that importing a light contract (contracts.impedance, used by the numpy-only
    # SDK) never loads pyarrow.
    if name in ("pack_json_message", "unpack_json_message"):
        from arm_control.contracts import _arrow

        return getattr(_arrow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["pack_json_message", "unpack_json_message"]
