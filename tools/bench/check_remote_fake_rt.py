"""The supervisor and SDK against an arm_rt_server on ANOTHER host running the FAKE backend (no robot, no FCI).

    python tools/bench/check_remote_fake_rt.py --host 172.16.1.2 --udp 47950 --tcp 47951 [--native]

The server is started by the operator on scratch ports (``--backend fake``, instance ``sdk_arm``, joints j1,j2,j3). This drives it
only through ``arm_control.sdk`` via a supervisor process on this PC: claim, ARM (of the fake backend), a move through SENT and APPLIED,
a cancel, state_history/trace reads, DISARM, and reports the state-stream ages and rates measured over the real RT link. It
arms nothing physical; it is the rehearsal of the deployed server build, not a qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_sdk_supervisor as base  # noqa: E402

from arm_control.execution.contract import Lifecycle, Phase, Start  # noqa: E402
from arm_control.sdk import ArmClient  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--udp', type=int, required=True)
    parser.add_argument('--tcp', type=int, required=True)
    parser.add_argument('--native', action='store_true', help='bind the analytical model and negotiate v7 (the server needs --enable-reference)')
    args = parser.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix='remote_fake_rt_'))
    cfg_path = base.write_config(tmp, args.udp, args.tcp)
    text = cfg_path.read_text().replace('host: 127.0.0.1', f'host: {args.host}')
    if not args.native:
        text = text.replace('  native_reference: true\n', '')
    else:
        gravity = dict(instance_id='sdk_arm', revision=1, vector=[0., 0., -9.81], units='m/s^2', frame='base',
                       source=dict(type='nominal_assumption', evidence='analytical fixture'),
                       reviewed_against=dict(bundle_id='fixture', revision=1, installation_epoch='fixture',
                                             model_sha256=hashlib.sha256((tmp / 'three.urdf').read_bytes()).hexdigest(),
                                             base_sha256=hashlib.sha256(b'identity base fixture').hexdigest()))
        digest = hashlib.sha256(json.dumps(gravity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        token = 'analytical-fixture:g' + digest
        base.MODEL_TOKEN = token
        text += 'model: ' + json.dumps(dict(token=token, bundle_id='fixture', revision=1, gravity=gravity)) + '\n'
    cfg_path.write_text(text)
    sock = tmp / 'sdk_arm.sock'
    log = open(tmp / 'supervisor.log', 'w')
    sup = subprocess.Popen([sys.executable, '-m', 'arm_control.supervisor.service', '--config', str(cfg_path), '--socket', str(sock)],
                           stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    arm = None
    try:
        end = time.monotonic() + 30
        while not sock.exists() or sup.poll() is None and not _accepts(sock):
            assert sup.poll() is None, (tmp / 'supervisor.log').read_text()[-2000:]
            assert time.monotonic() < end, 'the supervisor did not come up (is the remote server running?)'
            time.sleep(.1)
        arm = ArmClient('sdk_arm', socket_path=sock)
        print(f'extensions: {arm.extensions}; backend identity: {arm.identity}')
        arm.claim()
        arm.arm()
        assert arm.authority().lifecycle is Lifecycle.ARMED
        q0 = np.asarray(arm.state().q)
        status = arm.execute(base.move('r1', q0, q0 + [.2, -.1, .1], 1.5, start=Start.ON_ADMIT))
        done = arm.wait(status.execution_id, 15)
        # and back: the fake plant keeps its pose between sessions and its gravity feedforward is pose dependent, so the
        # sessions only compare when each one ends where it began
        back = arm.execute(base.move('r1b', np.asarray(arm.state().q), q0, 1.5, start=Start.ON_ADMIT))
        returned = arm.wait(back.execution_id, 15)
        assert returned.phase is Phase.COMPLETED, (returned.phase, returned.reason)
        print(f'out move {done.phase.value}; back at the start within {np.max(np.abs(np.asarray(arm.state().q) - q0)):.4f} rad')
        assert done.phase is Phase.COMPLETED, (done.phase, done.reason, [(r.phase.value, r.reason) for r in arm.receipts(status.execution_id)], (tmp / 'supervisor.log').read_text()[-1500:])
        base.completed_chain(arm, status.execution_id)
        # state stream over the real link: ages (supervisor clock) and rate
        ages, seqs = [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            history = arm.state_history()
            ages.append(history['samples'][-1]['age_ns'] / 1e6)
            seqs.append(history['samples'][-1]['state_seq'])
            time.sleep(.02)
        rate = (seqs[-1] - seqs[0]) / (time.monotonic() - t0)
        ages.sort()
        trace = arm.trace_samples(0)
        print(f'state_seq advances {rate:.0f}/s (the RT tick; published packets are every state-hz); newest-sample age ms: '
              f'p50={statistics.median(ages):.2f} p99={ages[int(len(ages) * .99) - 1]:.2f} max={ages[-1]:.2f}')
        print(f'trace: latest_seq={trace["latest_seq"]} dropped={trace["dropped"]} errors={trace["errors"]}')
        assert trace['errors'] == 0 and trace['samples']
        running = arm.execute(base.move('r2', np.asarray(arm.state().q), np.asarray(arm.state().q) + [.3] * 3, 4.0, start=Start.ON_ADMIT))
        time.sleep(.4)
        assert arm.cancel(running.execution_id).phase is Phase.CANCELLED
        arm.disarm()
        assert arm.authority().lifecycle is Lifecycle.CLAIMED
    finally:
        if arm is not None:
            arm.close()
        sup.terminate()
        try:
            sup.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sup.kill()
    print(f'PASS: supervisor + SDK against the remote fake-backend arm_rt_server at {args.host} ({"native v7" if args.native else "v5/v6"})')


def _accepts(sock):
    import socket
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(sock))
        return True
    except OSError:
        return False
    finally:
        probe.close()


if __name__ == '__main__':
    main()
