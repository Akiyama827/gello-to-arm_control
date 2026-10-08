"""The supervisor and SDK against an arm_rt_server on ANOTHER host running the FAKE backend (no robot, no FCI).

    python tools/hw/remote_fake_rt.py --host 172.16.1.2 --udp 47950 --tcp 47951 [--native]

The operator starts the server on scratch ports (``--backend fake``, instance ``sdk_arm``, joints j1,j2,j3; with ``--enable-reference``
for ``--native``). This drives it only through ``arm_control.sdk`` via a supervisor process on this PC: claim, ARM (of the fake
backend), a move through SENT and APPLIED and back, a cancel and back, state_history/trace reads, DISARM. It REFUSES to ARM anything
whose identity is not ``rt:fake``, DISARMs on every exit path (including SIGTERM/SIGHUP) before the supervisor is stopped, and
``--native`` fails unless the RT actually selected a native reference. It is the rehearsal of a deployed server build, not a
qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "check"))  # check_sdk_supervisor
import check_sdk_supervisor as base  # noqa: E402

from arm_control.execution.contract import Lifecycle, Phase, Start  # noqa: E402
from arm_control.sdk import ArmClient  # noqa: E402


def _accepts(sock) -> bool:
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(sock))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _move_and_return(arm, name, q0, delta, duration, native):
    """Out and back to ``q0``: the fake plant keeps its pose between sessions and its gravity feedforward is pose dependent, so
    sessions only compare when each one ends where it began."""
    out = arm.execute(base.move(name, q0, q0 + delta, duration, start=Start.ON_ADMIT))
    done = arm.wait(out.execution_id, 15)
    assert done.phase is Phase.COMPLETED, (name, done.phase, done.reason, [(r.phase.value, r.reason) for r in arm.receipts(out.execution_id)])
    base.completed_chain(arm, out.execution_id)
    if native:  # the RT must have SELECTED a native reference: the flag alone proves nothing
        selected = next(r for r in arm.receipts(out.execution_id) if r.phase is Phase.SELECTED)
        assert selected.evidence.get('reference_id') and selected.evidence['reference_id'] > 0, selected.evidence
    back = arm.execute(base.move(name + 'b', np.asarray(arm.state().q), q0, duration, start=Start.ON_ADMIT))
    returned = arm.wait(back.execution_id, 15)
    assert returned.phase is Phase.COMPLETED, (name + 'b', returned.phase, returned.reason)
    return float(np.max(np.abs(np.asarray(arm.state().q) - q0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--udp', type=int, required=True)
    parser.add_argument('--tcp', type=int, required=True)
    parser.add_argument('--native', action='store_true', help='bind the analytical model and require a native (v7) reference selection')
    args = parser.parse_args()
    # a killed tool must not leave a server ARMED: SIGTERM/SIGHUP unwind through the finally below
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda *_: sys.exit(128 + 15))
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
        base.MODEL_TOKEN = 'analytical-fixture:g' + digest
        text += 'model: ' + json.dumps(dict(token=base.MODEL_TOKEN, bundle_id='fixture', revision=1, gravity=gravity)) + '\n'
    cfg_path.write_text(text)
    sock = tmp / 'sdk_arm.sock'
    log = open(tmp / 'supervisor.log', 'w')
    sup = subprocess.Popen([sys.executable, '-m', 'arm_control.supervisor.service', '--config', str(cfg_path), '--socket', str(sock)],
                           stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    arm, ok = None, False
    try:
        end = time.monotonic() + 30
        while not (sock.exists() and _accepts(sock)):
            assert sup.poll() is None, (tmp / 'supervisor.log').read_text()[-2000:]
            assert time.monotonic() < end, 'the supervisor did not come up (is the remote server running?)'
            time.sleep(.1)
        arm = ArmClient('sdk_arm', socket_path=sock)
        print(f'extensions: {arm.extensions}; backend identity: {arm.identity}')
        assert arm.identity.backend == 'rt:fake', f'refusing to ARM a backend that is not the fake one: {arm.identity}'
        arm.claim()
        arm.arm()
        assert arm.authority().lifecycle is Lifecycle.ARMED
        q0 = np.asarray(arm.state().q)
        residual = _move_and_return(arm, 'r1', q0, np.array([.2, -.1, .1]), 1.5, args.native)
        print(f'out move completed ({"native v7 reference selected" if args.native else "streamed"}); back at the start within {residual:.4f} rad')
        # the age of the newest state packet since THIS host received it (not link latency: it is bounded by the publish period),
        # and how fast the RT tick counter advances over the window
        ages, seqs, stamps = [], [], []
        t0 = time.monotonic()
        while time.monotonic() - t0 < 5.0:
            history = arm.state_history()
            ages.append(history['samples'][-1]['age_ns'] / 1e6)
            seqs.append(history['samples'][-1]['state_seq'])
            stamps.append(time.monotonic())
            time.sleep(.02)
        rate = (seqs[-1] - seqs[0]) / (stamps[-1] - stamps[0])
        ages.sort()
        print(f'state_seq advances {rate:.0f}/s over the window (the RT tick counter); age of the newest packet since PC receipt, ms: '
              f'p50={statistics.median(ages):.2f} p98={ages[int(len(ages) * .98)]:.2f} max={ages[-1]:.2f}')
        # trace/1: two reads a moment apart see no gap (the first read of a stream has none by definition)
        first = arm.trace_samples(0)
        time.sleep(.05)
        second = arm.trace_samples(first['latest_seq'])
        assert first['errors'] == 0 and first['samples'] and second['dropped'] == 0 and second['samples'], (first['errors'], second['dropped'])
        print(f'trace: latest_seq={second["latest_seq"]} errors={second["errors"]} dropped_between_prompt_reads={second["dropped"]}')
        running = arm.execute(base.move('r2', np.asarray(arm.state().q), np.asarray(arm.state().q) + [.3] * 3, 4.0, start=Start.ON_ADMIT))
        time.sleep(.4)
        assert arm.cancel(running.execution_id).phase is Phase.CANCELLED
        # a cancelled move leaves the plant displaced: bring it back so the next session starts where this one did
        back = arm.execute(base.move('r2b', np.asarray(arm.state().q), q0, 2.0, start=Start.ON_ADMIT))
        assert arm.wait(back.execution_id, 15).phase is Phase.COMPLETED
        end_residual = float(np.max(np.abs(np.asarray(arm.state().q) - q0)))
        print(f'after the cancel and the move back, the plant is within {end_residual:.4f} rad of where the session began')
        assert end_residual < 0.05, end_residual
        arm.disarm()
        assert arm.authority().lifecycle is Lifecycle.CLAIMED
        ok = True
    finally:
        if arm is not None:
            try:  # whatever happened, leave the server DISARMED before the supervisor goes away (it abandons, never disarms)
                arm.disarm(timeout_s=3.0)
            except Exception as exc:  # noqa: BLE001
                print(f'WARNING: could not confirm DISARM on exit: {exc!r}', flush=True)
            arm.close()
        sup.terminate()
        try:
            sup.wait(timeout=10)
        except subprocess.TimeoutExpired:
            sup.kill()
        log.close()
        if ok:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f'kept for diagnosis: {tmp}', flush=True)
    print(f'PASS: supervisor + SDK against the remote fake-backend arm_rt_server at {args.host} ({"native v7" if args.native else "v5/v6"})')


if __name__ == '__main__':
    main()
