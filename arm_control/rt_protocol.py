"""Compatibility API/CLI for the byte-level remote-RT protocol (not Dora contracts)."""
# Explicit aliases preserve existing imports and object identity.
from arm_control.plants.remote_rt.protocol import (
    MAGIC_CMD as MAGIC_CMD, MAGIC_STATE as MAGIC_STATE, MAGIC_CTL as MAGIC_CTL,
    VERSION as VERSION, MAX_JOINTS as MAX_JOINTS,
    FLAG_ARMED as FLAG_ARMED, FLAG_FAULTED as FLAG_FAULTED,
    FLAG_HOLDING as FLAG_HOLDING, FLAG_WRENCH_VALID as FLAG_WRENCH_VALID,
    ONLINE_MASK_SHIFT as ONLINE_MASK_SHIFT, ONLINE_MASK_BITS as ONLINE_MASK_BITS,
    CTL_HELLO as CTL_HELLO, CTL_ARM as CTL_ARM, CTL_DISARM as CTL_DISARM,
    CTL_PING as CTL_PING, CTL_PONG as CTL_PONG, CTL_STATUS as CTL_STATUS,
    CTL_FAULT as CTL_FAULT, CTL_SET_ACTIVE as CTL_SET_ACTIVE,
    FAULT_CMD_LOST as FAULT_CMD_LOST, FAULT_CTL_LOST as FAULT_CTL_LOST,
    FAULT_PLANT as FAULT_PLANT,
    CMD_SIZE as CMD_SIZE, STATE_SIZE as STATE_SIZE, CTL_SIZE as CTL_SIZE,
    with_online_mask as with_online_mask, online_mask as online_mask,
    pack_command as pack_command, State as State, unpack_state as unpack_state,
    pack_control as pack_control, Control as Control, unpack_control as unpack_control,
    golden_lines as golden_lines, _demo as _demo,
)

if __name__ == '__main__':
    import sys

    if '--hex' in sys.argv:
        print('\n'.join(golden_lines()))
    else:
        _demo()
