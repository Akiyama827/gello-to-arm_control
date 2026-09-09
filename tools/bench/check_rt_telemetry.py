"""Desired-velocity telemetry and legacy handshake refusal, without sockets."""
import struct
from unittest.mock import patch

import numpy as np

from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError


def main():
    frame = bytearray.fromhex(p.golden_lines()[1].split()[1])
    # Adjacent fields use different sentinels to catch an incorrect wrench offset.
    struct.pack_into('<6d', frame, 32 + 6 * p.MAX_JOINTS * 8, *range(10, 16))
    state = p.unpack_state(frame)
    assert state.wrench == list(range(10, 16))
    backend = RtBackend(RtConfig(), [f'j{i}' for i in range(7)])
    backend._state = state
    assert np.array_equal(backend.motor_state()['velocity_cmd'], np.arange(7) * .01)
    for bad_n in (0, 17, 65535):
        bad = frame.copy()
        struct.pack_into('<H', bad, 6, bad_n)
        try:
            p.unpack_state(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid joint count accepted')
    for version in (1, 2):
        hello = bytearray(p.pack_control(ctl_type=p.CTL_HELLO, seq=0, arg=7,
                                        t_mono_ns=0, text='franka pose_hold=2'))
        struct.pack_into('<H', hello, 4, version)

        class LegacySocket:
            closed = False

            def settimeout(self, _):
                pass

            def recv(self, _):
                return hello

            def close(self):
                self.closed = True

            def sendall(self, _):
                raise AssertionError('client sent a command to a legacy peer')

        sock = LegacySocket()
        with patch('socket.create_connection', return_value=sock), \
                patch('socket.socket', side_effect=AssertionError('UDP opened before handshake')):
            try:
                backend.open()
            except RtLinkError as exc:
                assert 'protocol mismatch' in str(exc)
            else:
                raise AssertionError('legacy server admitted')
        assert sock.closed and backend._tcp is None
    print('RT telemetry: PASS (field offsets, backend mapping, malformed state, legacy refusal before commands)')


if __name__ == '__main__':
    main()
