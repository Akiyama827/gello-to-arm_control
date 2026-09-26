"""Offline v7 golden/malformed parity. Pass reference_protocol_selfcheck binary."""
from __future__ import annotations

import math
from pathlib import Path
import struct
import subprocess
import sys

from arm_control.plants.remote_rt import protocol as p


def golden() -> dict[str, bytes]:
    out = {}

    def ctl(name, op, **kw):
        out[name] = p.pack_reference_control(
            ctl_type=op, seq=42, t_mono_ns=123456789, negotiated=True, **kw
        )

    digest = bytes(range(1, 33))
    ctl('MODE', p.CTL_REFERENCE_MODE, session_id=123, joint_order_hash=456,
        model_digest=digest, enable=1)
    ctl('BEGIN', p.CTL_REFERENCE_BEGIN, session_id=123, upload_id=2,
        reference_id=3, joint_order_hash=456, model_digest=digest, arm_epoch=4,
        expected_revision=5, revision=6, byte_count=33792, knots=128, n=16, flags=3)
    ctl('CHUNK', p.CTL_REFERENCE_CHUNK, upload_id=2, offset=88, length=3,
        data=bytes([1, 2, 3]))
    ctl('SEAL', p.CTL_REFERENCE_SEAL, upload_id=2, sha256=bytes(range(32, 0, -1)))
    ctl('COMMIT', p.CTL_REFERENCE_COMMIT, session_id=123, upload_id=2, reference_id=3,
        arm_epoch=4, expected_revision=5, revision=6, protected_until=1.25, latest_phase=1.)
    ctl('DEADLINE', p.CTL_REFERENCE_COMMIT, session_id=123, upload_id=2, reference_id=4,
        arm_epoch=4, expected_revision=0, revision=1, protected_until=0., latest_phase=0.,
        selection_budget_ns=123456789)
    ctl('ABORT', p.CTL_REFERENCE_ABORT, session_id=123, id=2, arm_epoch=4, revision=0)
    ctl('CANCEL', p.CTL_CANCEL_REFERENCE, session_id=123, id=3, arm_epoch=4, revision=6)
    ctl('REPLY', p.CTL_REFERENCE_STATUS, upload_id=2, reference_id=3, revision=6,
        result=p.REF_PREFIX, operation=p.CTL_REFERENCE_COMMIT)
    out['DYNAMIC'] = p.pack_reference_command(
        n=7, seq=42, arm_epoch=4, t_mono_ns=123456789, session_id=123,
        reference_id=3, revision=6, age_us=100, phase=1.25, valid_for_us=20000,
        phase_window_us=20000, tau_ff=[j*.5 for j in range(7)],
        kp=[100+j for j in range(7)], kd=[10+j for j in range(7)], negotiated=True)
    base = bytearray(p.STATE_SIZE)
    struct.pack_into('<IHH', base, 0, p.MAGIC_STATE, p.REFERENCE_VERSION, 7)
    struct.pack_into('<I', base, 12, 41)
    struct.pack_into('<Q', base, 32, 123)
    struct.pack_into('<I', base, 48, 4)
    out['STATE'] = bytes(base) + struct.pack(
        p._REFERENCE_STATE_TAIL_FMT, 3, 6, p.REFERENCE_RUNNING, 1.26, 1.25, 1000, 42,
        *(v for scale in (.25, .5, .75) for v in p._padded([j*scale for j in range(7)])))
    return out


def main(binary: str):
    expected = golden()
    native = dict(line.split() for line in subprocess.check_output([binary], text=True).splitlines())
    assert native == {k: v.hex() for k, v in expected.items()}
    cases = []

    def add(name, data, accepted=False, negotiated=True):
        kind = 'D' if name == 'DYNAMIC' else 'S' if name == 'STATE' else 'C'
        decode = {'D': p.unpack_reference_command, 'S': p.unpack_reference_state,
                  'C': p.unpack_reference_control}[kind]
        try:
            decode(data, negotiated=negotiated)
            actual = True
        except ValueError:
            actual = False
        assert actual == accepted, (name, len(data), 'python', accepted)
        cases.append((kind, negotiated, data, accepted))

    def change(name, offset, fmt, value):
        data = bytearray(expected[name])
        struct.pack_into(fmt, data, offset, value)
        add(name, data)

    for name, data in expected.items():
        add(name, data, True)
        add(name, data, name == 'MODE', negotiated=False)
        for length in (0, 7, 24, len(data)-1):
            add(name, data[:length])
        add(name, data+b'\0')
        change(name, 0, '<I', 0)
        for version in (1, 4, 5 if name != 'MODE' else 7, 6, 8):
            change(name, 4, '<H', version)
        if name not in ('STATE', 'DYNAMIC'):
            change(name, 12, '<I', 1)
            change(name, 6, '<H', 65535)
    # Every reserved byte, including CHUNK padding, independently rejected.
    for name, (start, end) in {
        'MODE': (76,128), 'BEGIN': (112,128), 'CHUNK': (43,128),
        'SEAL': (64,128), 'COMMIT': (88,128), 'ABORT': (48,128),
        'CANCEL': (48,128), 'REPLY': (56,128), 'STATE': (872,888),
    }.items():
        for offset in range(start, end):
            change(name, offset, '<B', 1)
    for name, offset in [('CHUNK',38), ('COMMIT',60), ('REPLY',52)]:
        change(name, offset, '<B', 1)
    for offset, fmt, values in [
        (104,'<H',[0,1,129,65535]), (106,'<H',[0,17]),
        (100,'<I',[0,33791,33793,0xFFFFFFFF]), (108,'<I',[4,8]),
        (96,'<I',[0,5,7]), (88,'<I',[0]),
    ]:
        for value in values:
            change('BEGIN',offset,fmt,value)
    for name in ('MODE','BEGIN','COMMIT','ABORT','CANCEL'):
        change(name,24,'<Q',0)
    change('MODE',72,'<I',0)
    change('CHUNK',32,'<I',0xFFFFFFFF)
    for value in (0,89,65535):
        change('CHUNK',36,'<H',value)
    for value in (-1.,math.nan,math.inf):
        for offset in (64,72):
            change('COMMIT',offset,'<d',value)
    change('COMMIT',72,'<d',1.5)
    change('CANCEL',44,'<I',0)
    change('ABORT',44,'<I',1)
    change('REPLY',44,'<I',10)
    for name in ('DYNAMIC','STATE'):
        for value in (0,17):
            change(name,6,'<H',value)
    for offset in (12,40,56,60):
        change('DYNAMIC',offset,'<I',0)
    for offset in (24,32):
        change('DYNAMIC',offset,'<Q',0)
    change('DYNAMIC',44,'<I',20001)
    for value in (-1.,math.nan,math.inf):
        change('DYNAMIC',48,'<d',value)
    for offset in (64,192,320):
        for value in (math.nan,math.inf):
            change('DYNAMIC',offset,'<d',value)
        change('DYNAMIC',offset+7*8,'<d',1.)
    for offset in (192,320):
        change('DYNAMIC',offset,'<d',-1.)
    change('STATE',900,'<I',5)
    for offset in (904,912):
        change('STATE',offset,'<d',-1.)
    for offset in (928,1056,1184):
        change('STATE',offset,'<d',math.nan)
        change('STATE',offset+7*8,'<d',1.)
    stream = b''.join(struct.pack('<BBI',ord(k),neg,len(d))+d for k,neg,d,_ in cases)
    result = subprocess.check_output([binary, '--decode'], input=stream)
    assert result == bytes(ord('1') if ok else ord('0') for *_,ok in cases)
    # Legacy parsers refuse v7; v5 state/control goldens retain their API.
    for name, decode in [('STATE',p.unpack_state),('BEGIN',p.unpack_control)]:
        try:
            decode(expected[name])
        except ValueError:
            pass
        else:
            raise AssertionError('legacy parser accepted v7')
    p._demo()
    print(f'reference protocol: {len(expected)} byte-exact goldens; '
          f'{len(cases)} cross-language acceptance/malformed cases PASS')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
