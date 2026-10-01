"""CTRL-052: the four thin clients against the real supervisor over the fake RT (no hardware).

Planner, action-chunk policy with delayed responses, deadman teleop and an RL step-window all go through ArmClient only.
Checks: a late or cancelled-generation chunk never reaches the robot or restarts a session; the waiting chunk is bounded
and replaced; a released/stale deadman stops the arm without re-arming; an RL step reports actual elapsed time and
infrastructure truncation, and reset() refuses (NeedsOperator) instead of re-arming; the backend never imports clients.
"""
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import numpy as np

from arm_control.clients.chunks import ActionChunkClient, Chunk
from arm_control.clients.common import NeedsOperator, StreamContext
from arm_control.clients.planner import run_reviewed
from arm_control.clients.rl import StepWindow
from arm_control.clients.teleop import DeadmanTeleop
from arm_control.execution.contract import Lifecycle, Phase, Start, TrajectoryRequest
from arm_control.motion import JointTrajectory
from controller_extension import configure
from reference_supervisor import until
from stream_races import main


def cases(arm, runtime, scene, service):
    h = runtime._native
    scene.facts = replace(scene.facts, policy=replace(scene.facts.policy, max_candidate_age_s=1., max_future_s=10.))
    ctx = StreamContext('fake', tuple(runtime.joints), scene.facts.model_token, scene.facts.scene_token,
                        scene.facts.policy.stream_policy_id)

    def idle():
        until(lambda: runtime._stream_job is None and not h.busy, 8)

    def stream_phase(session):
        return arm.status(session.execution_id).phase

    # -- planner: admitted, completed; a wrong model identity is refused by admission, not by the client ----------
    q = arm.state().q
    request = TrajectoryRequest('plan', 'fake', tuple(runtime.joints),
                                JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3))),
                                model_token=scene.facts.model_token, start=Start.ON_ADMIT)
    report, status = run_reviewed(arm, request, timeout_s=8.)
    assert report.admitted == ['plan'] and status.phase is Phase.COMPLETED, (report, status)
    idle()
    report, status = run_reviewed(arm, replace(request, request_id='bad', model_token='other'), timeout_s=2.)
    assert report.refused and not report.admitted and status.phase is Phase.REJECTED, (report, status)

    # -- action chunks ------------------------------------------------------------------------------------------
    idle()
    client = ActionChunkClient(arm, ctx, max_age_s=2., hold_s=6.)
    generation = client.start()
    now = arm.now()

    def chunk(name, *, age=0., **kw):
        base = np.asarray(arm.state().q)
        fields = dict(chunk_id=name, joints=ctx.joints, values=base[None, :] + .002, dt_s=.5,
                      observed_at=replace(now, ns=now.ns - int(age * 1e9)), generation=generation)
        fields.update(kw)
        return Chunk(**fields)

    until(lambda: arm.stream_state(client.session.execution_id).selected_revision == 1, 8)
    assert client.submit(chunk('c1')) == 'sent', client.report
    assert client.submit(chunk('c2')) == 'waiting' and client.submit(chunk('c3')) == 'waiting'
    assert client.report.replaced == ['c2'], client.report  # bounded: one waiting, the newer replaces it
    until(lambda: client.pump() and client.inflight is None and client.waiting is None, 8)
    assert client.report.admitted[:2] == ['c1', 'c3'] and client.report.selected_revision >= 2, client.report
    # a response requested before a restart is late even though a session is open (generation rule alone)
    old_generation = chunk('prev')
    generation = client.start()
    assert client.submit(old_generation) == 'late' and client.session is not None
    until(lambda: arm.stream_state(client.session.execution_id).selected_revision == 1, 8)
    # nothing sent in this session yet, so only the age rule can drop this one
    assert client.submit(chunk('old', age=5.)) == 'late'
    assert client.submit(chunk('fresh')) == 'sent'
    # older observation than what was sent, wrong joints, wrong units, delta without base: nothing reaches the RT
    n_admitted = len(client.report.admitted)
    assert client.submit(chunk('older', observed_at=replace(now, ns=now.ns - 1))) == 'late'
    assert client.submit(chunk('names', joints=tuple(reversed(ctx.joints)))) == 'refused'
    assert client.submit(chunk('units', units='deg')) == 'refused'
    assert client.submit(chunk('delta', mode='delta')) == 'refused'
    assert len(client.report.admitted) == n_admitted
    assert [c for c, _ in client.report.late] == ['prev', 'old', 'older'], client.report.late
    # a response for the previous generation, after cancel: dropped, and the session is not restarted
    stale = chunk('after-cancel')
    until(lambda: client.pump() and client.inflight is None, 8)
    client.cancel('operator')
    assert stream_phase(client.session) if client.session else True
    assert client.submit(stale) == 'late' and client.session is None
    idle()
    assert arm.authority().lifecycle is Lifecycle.ARMED  # cancelling a stream never disarms

    # -- deadman teleop (a fake clock: no sleeping) ----------------------------------------------------------------
    clock = [100.]
    tele = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                         clock=lambda: clock[0])
    tele.start()
    until(lambda: arm.stream_state(tele.session.execution_id).selected_revision == 1, 8)
    target = np.asarray(arm.state().q) + .003
    assert tele.command(target, held=True) == 'moved' and tele.report.admitted == ['teleop-1']
    clock[0] += .1
    assert tele.tick() is None
    clock[0] += .5  # the GUI stalled: no heartbeat past the timeout
    assert tele.tick() == 'expired'
    until(lambda: stream_phase(tele.session) is Phase.CANCELLED, 8)
    assert tele.command(target, held=True) == 'expired'  # a stopped session never resumes by itself
    idle()
    tele.start()
    until(lambda: arm.stream_state(tele.session.execution_id).selected_revision == 1, 8)
    assert tele.command(target, held=False) == 'released'
    until(lambda: stream_phase(tele.session) is Phase.CANCELLED, 8)
    assert arm.authority().lifecycle is Lifecycle.ARMED
    idle()

    # -- RL step-window ---------------------------------------------------------------------------------------------
    env = StepWindow(arm, ctx, hold_s=6.)
    env.reset()
    until(lambda: arm.stream_state(env.session.execution_id).selected_revision == 1, 8)
    result = env.step(np.asarray(arm.state().q) + .002, .3)
    assert result.outcome == 'completed' and result.elapsed_s >= .3 and not result.terminated and not result.truncated, result
    assert result.selected_revision >= 2, result
    stopped = env.step(np.asarray(arm.state().q) + .002, 5., intervention=lambda: True)
    assert stopped.outcome == 'intervention' and stopped.elapsed_s < 1., stopped
    try:
        env.step([float('nan')] * 3, .3)
    except ValueError:
        pass
    else:
        raise AssertionError('a non-finite action was sent')
    arm.cancel(env.session.execution_id)  # infrastructure ends the episode stream
    ended = env.step(np.asarray(arm.state().q) + .002, .3)
    assert ended.truncated and ended.outcome in ('expired', 'faulted', 'refused') and not ended.terminated, ended
    try:
        env.step(np.asarray(arm.state().q), .3)
    except NeedsOperator:
        pass
    else:
        raise AssertionError('a closed episode was silently restarted')
    idle()
    arm.disarm()
    try:
        env.reset()
    except NeedsOperator:
        pass
    else:
        raise AssertionError('reset() accepted a disarmed robot')
    assert arm.authority().lifecycle is not Lifecycle.ARMED, 'reset() re-armed the robot'
    arm.arm()

    # -- the backend never imports the clients --------------------------------------------------------------------
    code = ('import sys, arm_control, arm_control.sdk, arm_control.supervisor.runtime, arm_control.supervisor.service;'
            'bad=[m for m in sys.modules if m.startswith("arm_control.clients")]; assert not bad, bad')
    subprocess.run([sys.executable, '-c', code], check=True)
    print('PASS: planner, action-chunk, deadman and RL-step clients run over ArmClient only; late/replaced chunks and '
          'stale deadman stop safely; reset never re-arms; the backend does not import clients')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
