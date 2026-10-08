"""Offline mode-selection golden/malformed parity. Pass mode_select_selfcheck."""
from __future__ import annotations

from pathlib import Path
import struct
import subprocess
import sys

from arm_control.plants.remote_rt import protocol as p

DIGEST = bytes(range(1, 33))


def golden() -> dict[str, bytes]:
    return {
        'MODE_SELECT': p.pack_mode_select(
            session_id=123, generation=6, expected_predecessor=5, arm_epoch=4,
            desired_mode=p.MODE_REFERENCE, model_digest=DIGEST, seq=42,
            t_mono_ns=123456789),
        'MODE_STATUS': p.pack_mode_status(
            session_id=123, generation=6, predecessor=5, arm_epoch=4,
            result=p.MODE_SEL_OK, selected_mode=p.MODE_REFERENCE,
            requested_mode=p.MODE_REFERENCE, seq=43, t_mono_ns=123456789),
    }


def main(binary: str):
    expected = golden()
    lines = [line.split() for line in subprocess.check_output([binary], text=True).splitlines()]
    native = {parts[0]: parts[1] for parts in lines
              if parts[0] in ('MODE_SELECT', 'MODE_STATUS')}
    assert native == {k: v.hex() for k, v in expected.items()}, native.keys()
    sizes = {parts[0]: parts[1:] for parts in lines if parts[0] == 'SIZES'}
    assert sizes == {'SIZES': ['104', '104']}, sizes

    # Round trips.
    sel = p.unpack_mode_select(expected['MODE_SELECT'])
    assert (sel['session_id'], sel['generation'], sel['expected_predecessor'],
            sel['arm_epoch'], sel['desired_mode'], sel['seq']) == (123, 6, 5, 4, 2, 42)
    assert sel['model_digest'] == DIGEST
    rep = p.unpack_mode_status(expected['MODE_STATUS'])
    assert (rep['result'], rep['selected_mode'], rep['requested_mode'], rep['seq']) == (0, 2, 2, 43)
    soft = p.pack_mode_select(session_id=1, generation=1, expected_predecessor=0,
                              arm_epoch=2, desired_mode=p.MODE_STREAMED)
    assert p.unpack_mode_select(soft)['model_digest'] == b'\0' * 32

    cases = []

    def reject(name, data):
        cases.append((name, data))

    # Structural refusals the C++ predicate also refuses.
    def mutate(pkt, offset, fmt, value):
        out = bytearray(pkt)
        struct.pack_into(fmt, out, offset, value)
        return bytes(out)

    reject('digest-on-streamed', mutate(soft, 56, '<32s', DIGEST))
    reject('reference-without-digest',
           mutate(expected['MODE_SELECT'], 56, '<32s', b'\0' * 32))
    reject('unselected-mode', mutate(expected['MODE_SELECT'], 52, '<H', p.MODE_UNSELECTED))
    reject('reserved0', mutate(expected['MODE_SELECT'], 54, '<H', 1))
    reject('request-flags', mutate(expected['MODE_SELECT'], 88, '<I', 1))
    reject('reserved-tail', mutate(expected['MODE_SELECT'], 92, '<B', 1))
    reject('zero-generation', mutate(expected['MODE_SELECT'], 32, '<Q', 0))
    reject('zero-session', mutate(expected['MODE_SELECT'], 24, '<Q', 0))
    reject('zero-epoch', mutate(expected['MODE_SELECT'], 48, '<I', 0))
    reject('generation-gap', mutate(expected['MODE_SELECT'], 40, '<Q', 4))
    reject('bad-version', mutate(expected['MODE_SELECT'], 4, '<H', p.REFERENCE_VERSION))
    reject('nonzero-arg', mutate(expected['MODE_SELECT'], 20, '<I', 1))
    reject('unknown-result', mutate(expected['MODE_STATUS'], 52, '<I', 9))
    reject('status-reserved', mutate(expected['MODE_STATUS'], 64, '<B', 1))

    ok = 0
    for name, data in cases:
        unpack = (p.unpack_mode_select
                  if data[4:6] == struct.pack('<H', p.CTL_MODE_SELECT)
                  else p.unpack_mode_status)
        try:
            unpack(data)
        except ValueError:
            ok += 1
        else:
            raise AssertionError(f'{name} accepted')
    # The unmutated frames still decode.
    p.unpack_mode_select(expected['MODE_SELECT'])
    p.unpack_mode_status(expected['MODE_STATUS'])

    # A bare server advertises neither the capability nor the mode flags.
    assert not p.CAP_MODE_SELECT & (p.CAP_POSE_HOLD | p.CAP_OBSERVER | p.CAP_REFERENCE_HORIZON)
    assert not (p.FLAG_MODE_STREAMED | p.FLAG_MODE_REFERENCE) & (
        p.FLAG_ARMED | p.FLAG_FAULTED | p.FLAG_HOLDING | p.FLAG_WRENCH_VALID | p.FLAG_CLAIMED)
    print(f'mode select: {len(expected)} byte-exact goldens; {ok} malformed cases PASS')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
