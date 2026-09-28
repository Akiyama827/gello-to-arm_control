"""CTRL-034 offline counterpart: PC-stall injection and the retire-on-expiry policy.

Decision (a): once a native reference's dynamics envelope expires, that
reference is FINISHED. The arm holds its measured position with the captured
gains and zero feedforward; the running reference is RETIRED
(REFERENCE_CANCELLED) with a distinct journal cause; envelopes for it are never
re-admitted; and motion resumes only through a NEW reference the PC plans from
the measured position. The PC recovery budget is `fault_ms - expiry`.

Retirement applies only while the reference is RUNNING (never at a stationary
END), and only to a reference the RT retired on expiry: an ordinary PC cancel
keeps its retained id/phase fresh so the continuing envelopes still hold the
deadman (native_reference.py). A COMMIT whose FIRST envelope arrives later than
the freshness ceiling is retired too, so the late envelope cannot be sampled at
the advanced phase and step the arm.

Fake backend only, one `arm_rt_server --enable-reference --hold-ms 100
--fault-ms 1000 --state-hz 1000`. No hardware, no deployment, no service action.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as p
from arm_control.plants.remote_rt.reference import ReferenceHorizon
from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError

FRESHNESS_MS = 50      # min(hold_ms, valid_for_us); valid_for/phase_window = 50 ms
HOLD_MS, FAULT_MS = 100, 1000
SPEED = 0.4            # planner speed cap, rad/s
POLL_S = 0.001
SLACK_MS = 25
# The arm holds the captured measured position; the retired reference would
# have advanced by SPEED * elapsed. A resume jump is therefore ~SPEED * elapsed,
# and the resume assertion below bounds the observed step at half of that.
STEP_LIMIT = 0.05
# A first envelope later than the ceiling must be REFUSED, not sampled at the
# advanced phase. At 1 kHz the normal per-frame setpoint step is SPEED/1000 =
# 0.4 mrad; before this bound the reviewer measured ~107 mrad in ONE frame.
GAP_STEP_LIMIT = 0.005


def unused_port(kind):
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        out = pred()
        if out:
            return out
        time.sleep(0.002)
    raise AssertionError('timed out')


def ramp(q0, duration, speed=SPEED):
    q0 = np.asarray(q0, float)
    q = np.stack([q0, q0 + speed * duration])
    v = np.full((2, 3), speed)
    return JointTrajectory(np.array([0.0, duration]), q, v)


class Pump:
    """Native v7 envelopes for whatever reference RT reports; `pause` is the stall."""

    def __init__(self, rt):
        self.rt, self.stop = rt, threading.Event()
        self.pause = False
        self.errors = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        command = dict(position=np.zeros(3), velocity=np.zeros(3),
                       torque=np.zeros(3), kp=np.ones(3) * 2, kd=np.ones(3) * .1)
        while not self.stop.is_set():
            if not self.pause:
                state, rx = self.rt.latest_state()
                try:
                    if state and state.reference and state.reference['reference_id']:
                        self.rt.apply_reference_envelope(
                            command, reference_id=state.reference['reference_id'],
                            revision=state.reference['revision'],
                            phase=state.reference['phase'], sample_received_at=rx,
                            valid_for_s=.05, phase_window_s=.05)
                except (ValueError, OSError, RtLinkError) as exc:
                    self.errors.append(str(exc))
            self.stop.wait(.002)

    def close(self):
        self.stop.set()
        self.thread.join(2)
        assert not self.errors, self.errors


class Recorder:
    """1 ms samples of what the servo selected, for edge analysis."""

    def __init__(self, rt):
        self.rt, self.stop = rt, threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        while not self.stop.is_set():
            state = self.rt.latest_state()[0]
            if state is not None:
                ref = state.reference or {}
                self.rows.append({
                    't': time.monotonic(), 'q_cmd': tuple(state.q_cmd),
                    'qd_cmd': tuple(state.qd_cmd), 'holding': bool(state.holding),
                    'status': ref.get('status'), 'age_us': ref.get('envelope_age_us'),
                    'faulted': bool(state.faulted), 'fault_code': state.fault_code,
                })
            self.stop.wait(POLL_S)

    def close(self):
        self.stop.set()
        self.thread.join(2)

    def mark(self):
        return len(self.rows)

    def since(self, mark):
        return self.rows[mark:]


@contextmanager
def server(binary):
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    argv = [binary, '--backend', 'fake', '--n', '3', '--instance-id', 'fake',
            '--joint-names', 'j0,j1,j2', '--udp-port', str(udp), '--tcp-port', str(tcp),
            '--bind', '127.0.0.1', '--hold-ms', str(HOLD_MS), '--fault-ms', str(FAULT_MS),
            '--state-hz', '1000', '--enable-reference']
    log = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.log')
    proc = subprocess.Popen(argv, stdout=log, stderr=log)
    rt = RtBackend(RtConfig(host='127.0.0.1', udp_port=udp, tcp_port=tcp,
                            instance_id='fake'), ['j0', 'j1', 'j2'])
    try:
        time.sleep(.2)
        rt.open()
        assert rt.negotiate_reference('pc-stall-fixture-model')
        rt.enable_all()
        yield rt, log.name
    except BaseException:
        log.flush()
        print(Path(log.name).read_text()[-4000:])
        raise
    finally:
        rt.close()
        proc.terminate()
        proc.wait(3)
        log.close()


def commit(rt, reference_id, q0, *, duration=3.0):
    # Terminal so a reference can reach END, which is the state a new reference
    # may replace without a cancel (used by the COMMIT->gap negative case).
    load = rt.upload_reference(ramp(q0, duration), reference_id=reference_id,
                               terminal=True)
    rt.commit_reference(load, reference_id=reference_id)
    return wait_for(lambda: (rt.latest_state()[0] or None)
                    if (rt.latest_state()[0] and rt.latest_state()[0].reference
                        and rt.latest_state()[0].reference['reference_id'] == reference_id
                        and rt.latest_state()[0].reference['status'] == p.REFERENCE_RUNNING)
                    else None)


def hold_edge(rows, t0):
    """(first holding sample, ms after t0) or (None, None)."""
    for row in rows:
        if row['holding']:
            return row, (row['t'] - t0) * 1000.0
    return None, None


def inject(rt, pump, recorder, stall_ms, *, resume):
    """Pause the PC stream, optionally resume, and return (t0, rows)."""
    assert not rt.latest_state()[0].holding, 'already holding before the stall'
    mark, t0 = recorder.mark(), time.monotonic()
    pump.pause = True
    if resume:
        time.sleep(stall_ms / 1000.0)
        pump.pause = False
        time.sleep(0.25)
    return t0, recorder.since(mark)


def horizon_expiry_report(binary):
    """Item 2: the supervisor-facing horizon reports an RT expiry retire as a
    failed segment (REFERENCE_EXPIRED) and never plans a continuation; a LOCAL
    cancel is not misreported as an expiry."""
    command = dict(position=np.zeros(3), velocity=np.zeros(3),
                   torque=np.zeros(3), kp=np.ones(3) * 2, kd=np.ones(3) * .1)

    def run_to_selection(rt, h):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not (h.selected and not h.error):
            state, received = rt.latest_state()
            h.poll(state, received)
            if h.ready and not h.selected:
                h.approve()
            if h.selected:
                rt.apply_reference_envelope(
                    command, reference_id=h.id, revision=h.feedback['revision'],
                    phase=h.point.time, sample_received_at=received,
                    valid_for_s=.05, phase_window_s=.05)
            time.sleep(0.005)
        assert h.selected and not h.error, h.error

    with server(binary) as (rt, _log):
        h = ReferenceHorizon(rt)
        try:
            h.stage("expiry", ramp(np.zeros(3), 2.0), stream=True)
            run_to_selection(rt, h)
            # Stop feeding: RT retires the reference when the envelope expires.
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not h.error:
                h.poll(*rt.latest_state()[:2])
                time.sleep(0.005)
            assert h.error == 'REFERENCE_EXPIRED', h.error
            assert not h.selected
            print('  report  unsolicited RT retire -> horizon.error=REFERENCE_EXPIRED')
        finally:
            h.close()
    with server(binary) as (rt, _log):
        h = ReferenceHorizon(rt)
        try:
            h.stage("local-cancel", ramp(np.zeros(3), 2.0), stream=True)
            run_to_selection(rt, h)
            h.cancel()                     # OUR cancel, not an expiry
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and h.selected:
                h.poll(*rt.latest_state()[:2])
                time.sleep(0.005)
            assert not h.selected and h.error != 'REFERENCE_EXPIRED', h.error
            print('  report  local cancel -> not reported as an expiry')
        finally:
            h.close()


def main(binary):
    with server(binary) as (rt, logpath):
        recorder = Recorder(rt)
        pump = Pump(rt)
        try:
            # ---- S1: 30 ms never expires, never retires --------------------
            wait_for(lambda: commit(rt, 1, np.zeros(3)))
            wait_for(lambda: not rt.latest_state()[0].holding)
            t0, rows = inject(rt, pump, recorder, 30, resume=True)
            edge, at_ms = hold_edge(rows, t0)
            assert edge is None, f'30 ms stall caused a hold at {at_ms:.0f} ms'
            state = rt.latest_state()[0]
            assert state.reference['status'] == p.REFERENCE_RUNNING, state.reference
            assert not state.faulted
            print('  S1   30 ms -> no hold, reference still RUNNING')

            # ---- S2/S3: expire at ~50 ms, RETIRE, no resume, no jump -------
            for stall_ms, ref_id, tag in ((60, 2, 'S2'), (150, 3, 'S3')):
                t0, rows = inject(rt, pump, recorder, stall_ms, resume=True)
                edge, at_ms = hold_edge(rows, t0)
                assert edge is not None, f'{stall_ms} ms stall produced no hold'
                assert abs(at_ms - FRESHNESS_MS) <= SLACK_MS, \
                    f'hold began at {at_ms:.0f} ms, expected ~{FRESHNESS_MS}'
                # The wire status is the existing REFERENCE_CANCELLED; the
                # journal (checked below) is what names the expiry cause.
                wait_for(lambda: rt.latest_state()[0].reference['status']
                         == p.REFERENCE_CANCELLED, timeout=2)
                q_held = np.array(edge['q_cmd'])
                after = rt.latest_state()[0]
                assert after.holding and not after.faulted, after.flags
                assert np.allclose(after.q_cmd, q_held, atol=STEP_LIMIT), (
                    'q_cmd moved after the retire: the old reference resumed')
                # A NEW reference from the MEASURED position resumes without a
                # setpoint step.
                mark = recorder.mark()
                commit(rt, ref_id, np.array(rt.latest_state()[0].q), duration=2.0)
                wait_for(lambda: not rt.latest_state()[0].holding, timeout=3)
                running = [r for r in recorder.since(mark)
                           if r['status'] == p.REFERENCE_RUNNING]
                steps = [abs(b['q_cmd'][0] - a['q_cmd'][0])
                         for a, b in zip(running, running[1:])]
                # If the RETIRED reference had resumed, q_cmd would jump by
                # SPEED x elapsed (~0.12-0.16 rad here). What remains is the
                # measured-position drift of the soft hold between the capture
                # and the new commit, so bound the step at half of that jump.
                jump_if_resumed = SPEED * (stall_ms + 250) / 1000.0
                assert steps and max(steps) <= 0.5 * jump_if_resumed, \
                    f'{tag} {stall_ms} ms: resume step {max(steps, default=0):.4f} rad '\
                    f'(old-reference resume would be ~{jump_if_resumed:.3f})'
                assert not rt.latest_state()[0].faulted
                print(f'  {tag}  {stall_ms:>3} ms -> hold at {at_ms:.0f} ms, '
                      f'CANCELLED, resumes only from a new measured-state commit')

            # ---- F3: a first envelope later than the ceiling RETIRES --------
            # A new reference may replace one that has reached its terminal END
            # (a normal first envelope is admitted well inside the ceiling and
            # stays RUNNING). If that FIRST envelope takes longer than the
            # freshness ceiling, the RT retires the selection so the late
            # envelope cannot be sampled at the advanced reference phase. The
            # RT's COMMIT acknowledgement can itself take ~100 ms, so the retire
            # latency is measured from the recorder's first RUNNING sample.
            wait_for(lambda: rt.latest_state()[0].reference['status']
                     == p.REFERENCE_END, timeout=5)
            gap_id = rt.latest_state()[0].reference['reference_id'] + 1
            pump.pause = True                       # no envelope for the new reference
            mark = recorder.mark()
            load = rt.upload_reference(
                ramp(np.array(rt.latest_state()[0].q), 2.0),
                reference_id=gap_id, terminal=True)
            rt.commit_reference(load, reference_id=gap_id)
            wait_for(lambda: rt.latest_state()[0].reference['status']
                     == p.REFERENCE_CANCELLED, timeout=2)
            before_gap = recorder.since(mark)
            running_row = next((r for r in before_gap
                                if r['status'] == p.REFERENCE_RUNNING), None)
            cancelled_row = next((r for r in before_gap
                                  if r['status'] == p.REFERENCE_CANCELLED), None)
            assert running_row is not None and cancelled_row is not None
            retire_ms = (cancelled_row['t'] - running_row['t']) * 1000.0
            assert retire_ms <= FRESHNESS_MS + 2 * SLACK_MS, \
                f'first-envelope gap retired after {retire_ms:.0f} ms, '\
                f'expected ~{FRESHNESS_MS}'
            # From here the arm must HOLD: the late first envelope is refused.
            # The window starts after the retire so the legitimate new-plan
            # re-selection at COMMIT (a step from the old terminal setpoint to
            # the measured start) is not mistaken for the stale-reference jump.
            step_mark = recorder.mark()
            time.sleep(0.35)                        # first envelope finally arrives
            pump.pause = False
            time.sleep(0.25)
            gap_state = rt.latest_state()[0]
            assert gap_state.reference['status'] == p.REFERENCE_CANCELLED, gap_state.reference
            assert not gap_state.faulted
            rows = recorder.since(step_mark)
            steps = [abs(b['q_cmd'][0] - a['q_cmd'][0]) for a, b in zip(rows, rows[1:])]
            assert steps and max(steps) <= GAP_STEP_LIMIT, \
                f'gap: single-frame step {max(steps, default=0):.4f} rad on the late '\
                f'first envelope (normal per-frame is {SPEED / 1000:.4f})'
            print(f'  gap  first envelope late -> retired {retire_ms:.0f} ms after RUNNING, '
                  f'max step {max(steps):.4f} rad (was ~0.12), still armed')

            # ---- S4: no new commit -> CMD_LOST at the recovery budget ------
            # A fresh reference from the current measured hold; its first
            # envelope is admitted normally and is NOT retired.
            s4_id = gap_state.reference['reference_id'] + 1
            commit(rt, s4_id, np.array(rt.latest_state()[0].q), duration=3.0)
            wait_for(lambda: not rt.latest_state()[0].holding)
            assert rt.latest_state()[0].reference['status'] == p.REFERENCE_RUNNING
            mark, t0 = recorder.mark(), time.monotonic()
            pump.pause = True                       # never resumed
            wait_for(lambda: rt.latest_state()[0].faulted, timeout=4)
            # Let the 1 ms recorder capture the fault row before slicing.
            wait_for(lambda: any(r['faulted'] for r in recorder.since(mark)), timeout=1)
            rows = recorder.since(mark)
            hold_row, _ = hold_edge(rows, t0)
            faulted_row = next((r for r in rows if r['faulted']), None)
            assert hold_row is not None and faulted_row is not None
            assert faulted_row['fault_code'] == p.FAULT_CMD_LOST, faulted_row
            budget_ms = (faulted_row['t'] - hold_row['t']) * 1000.0
            assert abs(budget_ms - (FAULT_MS - FRESHNESS_MS)) <= 2 * SLACK_MS, \
                f'CMD_LOST {budget_ms:.0f} ms after expiry, expected ~{FAULT_MS - FRESHNESS_MS}'
            print(f'  S4  no commit -> CMD_LOST {budget_ms:.0f} ms after the '
                  f'{FRESHNESS_MS} ms expiry (budget {FAULT_MS - FRESHNESS_MS} ms)')

            # The retire is distinguishable in the evidence by its cause.
            journal = Path(logpath).read_text()
            assert 'reference envelope expired' in journal, journal[-2000:]
            print('  journal names "reference envelope expired" as the cause')
        finally:
            recorder.close()
            pump.close()
    horizon_expiry_report(binary)
    print(f'pc stall: S1-S4 PASS (ceiling {FRESHNESS_MS} ms, hold {HOLD_MS}, '
          f'fault {FAULT_MS}, speed {SPEED} rad/s)')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
