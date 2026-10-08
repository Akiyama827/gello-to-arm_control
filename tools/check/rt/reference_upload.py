"""B2 socketpair check: real control_session + PC client, no plant/servo/devices."""
from __future__ import annotations

import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import numpy as np
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError


class CountSocket:
    def __init__(self, sock):
        self.sock, self.frames = sock, []

    def sendall(self, data):
        self.frames.append(bytes(data))
        self.sock.sendall(data)

    def recv(self, n):
        return self.sock.recv(n)


def main(binary):
    left, right = socket.socketpair()
    left.settimeout(.1)
    left.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    proc = subprocess.Popen([binary, str(right.fileno())], pass_fds=(right.fileno(),),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    right.close()
    rt = RtBackend(RtConfig(instance_id='fake', ack_timeout_s=.5), [f'j{i}' for i in range(16)])
    rt._tcp = CountSocket(left)
    try:
        assert rt._await_control({p.CTL_HELLO}, 2)
        identity = rt._await_control({p.CTL_IDENTITY}, 2)
        rt.identity = p.unpack_identity(identity.raw)
        rt.session_id = rt.identity.session_id
        rt._send_control(p.CTL_CLAIM, text=p.pack_identity(rt.identity))
        assert rt._await_control({p.CTL_STATUS}, 2).arg & p.FLAG_CLAIMED
        ctl = threading.Thread(target=rt._ctl_loop, daemon=True)
        ctl.start()
        assert rt.negotiate_reference('approved-model:g123')
        assert rt.negotiate_reference('approved-model:g123')
        try:
            rt.negotiate_reference('different-model')
        except RtLinkError:
            pass
        else:
            raise AssertionError('mutable session model binding')
        ack = rt._control_roundtrip(p.CTL_ARM)
        assert ack.arg & p.FLAG_ARMED
        rt._arm_epoch = ack.seq
        trajectory = JointTrajectory(np.arange(128)*.01,
                                     np.arange(128)[:, None]*.001 + np.zeros((128,16)),
                                     np.full((128,16),.1))
        upload = rt.upload_reference(trajectory, reference_id=10)
        ops = [p.unpack_reference_control(f, negotiated=True)[0] for f in rt._tcp.frames
               if f[4:6] == b'\x07\x00']
        assert ops.count(p.CTL_REFERENCE_CHUNK) == 384
        assert ops[-1] == p.CTL_REFERENCE_SEAL
        # Sealed is still only a candidate; another BEGIN cannot overwrite it.
        try:
            rt.upload_reference(trajectory, reference_id=11)
        except RtLinkError as exc:
            assert str(p.REF_BUSY) in str(exc)
        else:
            raise AssertionError('second staging candidate accepted')
        rt.abort_reference_upload(upload)
        rt.abort_reference_upload(upload)  # idempotent exact ABORT
        # A stopped receiver cannot accumulate 384 upload frames ahead of DISARM.
        os.kill(proc.pid, signal.SIGSTOP)
        before = len(rt._tcp.frames)
        errors = []

        def send():
            try:
                rt.upload_reference(trajectory, reference_id=12)
            except RtLinkError as exc:
                errors.append(str(exc))

        sender = threading.Thread(target=send)
        sender.start()
        deadline = time.monotonic()+1
        while len(rt._tcp.frames)==before and time.monotonic()<deadline:
            time.sleep(.001)
        assert len(rt._tcp.frames)==before+1
        flood = [threading.Thread(target=send) for _ in range(32)]
        for t in flood:
            t.start()
        for t in flood:
            t.join(1)
            assert not t.is_alive()
        assert len(rt._tcp.frames)==before+1
        start = time.monotonic()
        rt._send_control(p.CTL_DISARM)
        elapsed = time.monotonic()-start
        sender.join(1)
        assert not sender.is_alive() and len(errors)==33
        assert len(rt._tcp.frames)==before+2 and elapsed < .5
        assert sum('already in progress' in e for e in errors)==32
        os.kill(proc.pid, signal.SIGCONT)
        deadline=time.monotonic()+1
        while rt._status_q.empty() and time.monotonic()<deadline:
            time.sleep(.001)
        assert not rt._status_q.get_nowait().arg & p.FLAG_ARMED
        # Old ARM epoch candidate was discarded even though BEGIN preceded stop.
        rt._arm_epoch = rt._control_roundtrip(p.CTL_ARM).seq
        assert rt.upload_reference(trajectory, reference_id=13)>upload
        print(f'PASS B2 real session: 384 acknowledged chunks, immutable binding, busy/abort, '
              f'epoch reset, 32-producer flood; stalled-peer DISARM send {elapsed*1000:.3f} ms')
    finally:
        os.kill(proc.pid, signal.SIGCONT)
        rt._stop.set()
        left.close()
        proc.wait(timeout=3)
        assert proc.returncode==0, proc.stderr.read().decode()


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
