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
from arm_control.clients.common import NeedsOperator, Report, StreamContext, classify, note_outcome, resolved
from arm_control.clients.planner import run_reviewed
from arm_control.clients.rl import StepWindow
from arm_control.clients.teleop import DeadmanTeleop
from arm_control.execution.contract import Lifecycle, Phase, Start, TrajectoryRequest
from arm_control.execution.stream import RevisionStatus, SelectionState
from arm_control.motion import JointTrajectory
from arm_control.sdk import Refused, SupervisorError
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
    assert client.report.admitted[:2] == ['c1', 'c3'] and client.report.selected[:2] == ['c1', 'c3'], client.report
    assert client.report.requested[:2] == ['c1', 'c3'] and client.report.selected_revision >= 2, client.report
    # a response requested before a restart is late even though a session is open (generation rule alone)
    old_generation = chunk('prev')
    generation = client.start()
    assert client.submit(old_generation) == 'late' and client.session is not None
    until(lambda: arm.stream_state(client.session.execution_id).selected_revision == 1, 8)
    # nothing sent in this session yet, so only the age rule can drop this one
    assert client.submit(chunk('old', age=5.)) == 'late'
    assert client.submit(chunk('fresh')) == 'sent'
    n_before = len(client.report.requested)
    # older observation than what was sent, wrong joints, wrong units, delta without base: nothing reaches the RT
    n_requested = n_before
    assert client.submit(chunk('older', observed_at=replace(now, ns=now.ns - 1))) == 'late'
    assert client.submit(chunk('names', joints=tuple(reversed(ctx.joints)))) == 'refused'
    assert client.submit(chunk('units', units='deg')) == 'refused'
    assert client.submit(chunk('delta', mode='delta')) == 'refused'
    assert len(client.report.requested) == n_requested
    assert [c for c, _ in client.report.late] == ['prev', 'old', 'older'], client.report.late
    # a delta chunk composes on base_q (arithmetic), is sent and the RT selects it
    base_q = np.asarray(arm.state().q)
    delta = chunk('delta-ok', mode='delta', base_q=tuple(base_q), values=np.full((2, 3), .001),
                  observed_at=arm.now())
    assert np.allclose(client._knots(delta), base_q + .001)
    until(lambda: client.pump() and client.inflight is None, 8)
    assert client.submit(delta) == 'sent'
    until(lambda: client.pump() and 'delta-ok' in client.report.selected, 8)
    # an async admission refusal is reported as refused, never as admitted or selected
    until(lambda: client.pump() and client.inflight is None, 8)
    far = chunk('far', values=np.asarray(arm.state().q)[None, :] + 50., observed_at=arm.now())
    assert client.submit(far) == 'sent'
    until(lambda: client.pump() and client.inflight is None, 8)
    assert 'far' in client.report.requested and 'far' not in client.report.admitted + client.report.selected
    assert any(c == 'far' for c, _ in client.report.refused), client.report
    # Hermite construction: anchor velocity at the first knot, central differences inside, 127-knot cap
    seen = {}
    real_revise, real_anchor = arm.revise_stream, arm.stream_anchor
    arm.stream_anchor = lambda eid: seen.setdefault('anchor', replace(real_anchor(eid), qd=(.01, .02, .03)))
    arm.revise_stream = lambda rev: (seen.__setitem__('rev', rev), real_revise(rev))[1]
    try:
        until(lambda: client.pump() and client.inflight is None, 8)
        knots = np.asarray(arm.state().q) + np.array([[.001, 0, 0], [.002, 0, 0], [.004, 0, 0]])
        client.session.revise('hermite', knots, .5)
    finally:
        arm.revise_stream, arm.stream_anchor = real_revise, real_anchor
    sfx = seen['rev'].suffix
    assert np.allclose(sfx.velocities[0], seen['anchor'].qd) and sfx.times[0] == seen['anchor'].cut_phase
    assert np.allclose(sfx.velocities[1], (sfx.positions[2] - sfx.positions[0]) / 1.0)
    assert np.allclose(sfx.velocities[2], (sfx.positions[3] - sfx.positions[1]) / 1.0)
    assert np.allclose(sfx.velocities[3], 0.)
    try:
        client.session.revise('too-long', np.zeros((128, 3)), .1)
    except ValueError:
        pass
    else:
        raise AssertionError('128 knots + anchor exceeds stream capacity but was accepted')
    # revision status unreadable (history pruned): the in-flight slot is released and the fate is reported unknown
    until(lambda: client.pump() and client.inflight is None, 8)
    assert client.submit(chunk('lost', observed_at=arm.now())) == 'sent'
    real_rev = arm.stream_revision

    def pruned(*a, **k):
        raise Refused('revision history unavailable')
    arm.stream_revision = pruned
    try:
        client.pump()
    finally:
        arm.stream_revision = real_rev
    assert client.inflight is None and any('fate unknown' in n for n in client.report.notes), client.report
    assert 'lost' in client.report.unknown and 'lost' not in client.report.refused
    # a revision that expires or is cancelled after admission is resolved (not stuck in flight) and not selected
    at = arm.now()

    def revision_status(phase, selection):
        return RevisionStatus('e', 'r', 1, None, phase, selection, at, 'why')
    for phase in (Phase.EXPIRED, Phase.CANCELLED, Phase.SUPERSEDED):
        for selection in (SelectionState.NOT_REQUESTED, SelectionState.PENDING):
            assert classify(revision_status(phase, selection)) == 'ended' and resolved(revision_status(phase, selection))
    assert classify(revision_status(Phase.ADMITTED, SelectionState.PENDING)) == 'admitted'
    assert classify(revision_status(Phase.REQUESTED, SelectionState.NOT_REQUESTED)) == 'pending'
    folded = Report()
    note_outcome(folded, 'x', revision_status(Phase.EXPIRED, SelectionState.NOT_REQUESTED))
    assert folded.refused and not folded.admitted and not folded.selected
    # a REAL expiry after admission (the RT never reports ready): resolved, reported, the waiting chunk moves on,
    # teleop is not 'busy' forever
    idle()
    real_stage = h.stage_suffix
    h.stage_suffix = lambda *a, **k: None
    try:
        assert client.submit(chunk('stuck', observed_at=arm.now())) == 'sent'
        stuck_rid = client.inflight[1]
        until(lambda: arm.stream_revision(client.session.execution_id, stuck_rid).phase is Phase.EXPIRED, 12)
        assert client.submit(chunk('next', observed_at=arm.now())) == 'waiting'
    finally:
        h.stage_suffix = real_stage
    until(lambda: client.pump() and client.inflight is None or client.inflight and client.inflight[0] != 'stuck', 8)
    assert any(c == 'stuck' and 'ended without selection' in why for c, why in client.report.refused), client.report
    # the reply to a revision is lost AFTER the supervisor took it: still 'sent' (in flight, fate unknown), an older
    # observation is late, and pump() later finds it selected
    idle()
    until(lambda: client.pump() and client.inflight is None and client.waiting is None, 12)  # drain: 'next' may still be in flight
    lost_obs = arm.now()
    real_revise3 = arm.revise_stream

    def lost_reply_chunk(rev):
        real_revise3(rev)
        raise SupervisorError('connection closed before the reply')
    arm.revise_stream = lost_reply_chunk
    try:
        assert client.submit(chunk('lostreply', observed_at=lost_obs)) == 'sent'
    finally:
        arm.revise_stream = real_revise3
    assert 'lostreply' in client.report.unknown and client.inflight is not None and 'lostreply' not in client.report.refused
    assert client.submit(chunk('beforelost', observed_at=replace(lost_obs, ns=lost_obs.ns - 1))) == 'late'
    until(lambda: client.pump() and 'lostreply' in client.report.selected, 8)
    # an unreadable revision (fate unknown) is reported separately and nothing older may follow it
    until(lambda: client.pump() and client.inflight is None, 8)
    idle()  # the previous (unreadable) revision may still hold the supervisor's candidate slot
    unknown_obs = arm.now()
    verdict = client.submit(chunk('maybe', observed_at=unknown_obs))
    assert verdict == 'sent', (verdict, client.report.late[-2:], client.report.refused[-2:], client.inflight)
    real_rev2 = arm.stream_revision
    arm.stream_revision = lambda *a, **k: (_ for _ in ()).throw(Refused('revision history unavailable'))
    try:
        client.pump()
    finally:
        arm.stream_revision = real_rev2
    assert 'maybe' in client.report.unknown and 'maybe' not in client.report.refused
    assert client.submit(chunk('retro', observed_at=replace(unknown_obs, ns=unknown_obs.ns - 1))) == 'late'
    # an unconfirmed cancel is reported and keeps the session; start() then refuses to open a second stream
    real_cancel2 = arm.cancel
    arm.cancel = lambda *a, **k: (_ for _ in ()).throw(TimeoutError('link stall'))
    try:
        assert client.cancel('operator') is False
        assert any('CANCEL NOT CONFIRMED' in n for n in client.report.notes) and client.session is not None
        assert client.submit(chunk('during-cancel', generation=client.generation, observed_at=arm.now())) == 'late'
        assert client.report.late[-1][0] == 'during-cancel' and 'outstanding' in client.report.late[-1][1]
        try:
            client.start()
        except NeedsOperator:
            pass
        else:
            raise AssertionError('start() opened a second stream over an unconfirmed one')
    finally:
        arm.cancel = real_cancel2
    assert client.cancel('operator') is True and client.session is None
    generation = client.start()
    until(lambda: arm.stream_state(client.session.execution_id).selected_revision == 1, 8)
    assert client.cancel_pending is False and client.submit(chunk('after-retry', observed_at=arm.now())) == 'sent'
    until(lambda: client.pump() and client.inflight is None, 8)
    # a response for the previous generation, after cancel: dropped, and the session is not restarted
    stale = chunk('after-cancel')
    until(lambda: client.pump() and client.inflight is None, 8)
    client.cancel('operator')
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
    assert tele.command(target, held=True) == 'requested' and len(tele.report.requested) == 1
    # a second command while the first is unresolved is 'busy' or 'requested', never an exception; refusal is reported
    assert tele.command(target, held=True) in ('busy', 'requested')
    until(lambda: not tele._settle(), 8)
    assert tele.report.admitted and tele.report.selected, tele.report
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
    # a supervisor refusal of the revision itself (slot unavailable) is returned as 'refused', never raised
    tele_r = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_r.start()
    until(lambda: arm.stream_state(tele_r.session.execution_id).selected_revision == 1, 8)
    real_revise = arm.revise_stream

    def busy(rev):
        raise Refused('stream candidate slot unavailable')
    arm.revise_stream = busy
    try:
        assert tele_r.command(target, held=True) == 'refused' and tele_r.report.refused
    finally:
        arm.revise_stream = real_revise
    tele_r.session.close('done')
    idle()
    # the supervisor ended the stream under us: 'expired', not 'busy'/'refused' forever
    tele_e = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_e.start()
    until(lambda: arm.stream_state(tele_e.session.execution_id).selected_revision == 1, 8)
    assert tele_e.command(target, held=True) == 'requested'
    arm.cancel(tele_e.session.execution_id)
    assert tele_e.command(target, held=True) == 'expired'
    idle()
    # a pending revision whose status is unreadable (history pruned) is a reported unknown, not 'busy' forever
    tele_u = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_u.start()
    until(lambda: arm.stream_state(tele_u.session.execution_id).selected_revision == 1, 8)
    assert tele_u.command(target, held=True) == 'requested'
    real_rev3 = arm.stream_revision
    arm.stream_revision = lambda *a, **k: (_ for _ in ()).throw(Refused('revision history unavailable'))
    try:
        assert tele_u._settle() is False and tele_u.report.unknown
    finally:
        arm.stream_revision = real_rev3
    tele_u.session.close('done')
    idle()
    # the reply to a teleop revision is lost after the supervisor took it: 'unknown', then settled by the next command
    tele_l = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_l.start()
    until(lambda: arm.stream_state(tele_l.session.execution_id).selected_revision == 1, 8)
    real_revise4 = arm.revise_stream

    def lost_reply_tele(rev):
        real_revise4(rev)
        raise SupervisorError('connection closed before the reply')
    arm.revise_stream = lost_reply_tele
    try:
        assert tele_l.command(target, held=True) == 'unknown'
    finally:
        arm.revise_stream = real_revise4
    assert tele_l.report.unknown and not tele_l.report.refused, tele_l.report
    until(lambda: not tele_l._settle(), 8)
    assert tele_l.report.selected == tele_l.report.unknown, tele_l.report
    tele_l.session.close('done')
    idle()
    assert len(tele_l.report.unknown) == len(set(tele_l.report.unknown)), tele_l.report  # an id is listed once
    dup = Report()
    dup.mark_unknown('x')
    dup.mark_unknown('x')
    assert dup.unknown == ['x'], dup.unknown
    # a failure BEFORE a revision exists (the state read) sent nothing: 'refused', not 'unknown'
    tele_p = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_p.start()
    until(lambda: arm.stream_state(tele_p.session.execution_id).selected_revision == 1, 8)
    real_state_p = arm.state

    def no_state():
        raise SupervisorError('no reply to the state read')
    arm.state = no_state
    try:
        assert tele_p.command(target, held=True) == 'refused'
    finally:
        arm.state = real_state_p
    assert not tele_p.report.requested and not tele_p.report.unknown and tele_p.report.refused and tele_p.pending is None, \
        tele_p.report
    tele_p.session.close('done')
    idle()
    # a REAL expiry for teleop: not 'busy' forever
    idle()
    tele_x = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_x.start()
    until(lambda: arm.stream_state(tele_x.session.execution_id).selected_revision == 1, 8)
    h.stage_suffix = lambda *a, **k: None
    try:
        assert tele_x.command(np.asarray(arm.state().q) + .003, held=True) == 'requested'
        until(lambda: arm.stream_revision(tele_x.session.execution_id, tele_x.pending[1]).phase is Phase.EXPIRED, 12)
    finally:
        h.stage_suffix = real_stage
    outs = []

    def again():
        outs.append(tele_x.command(np.asarray(arm.state().q) + .003, held=True))
        clock[0] += .01
        return outs[-1] == 'requested'
    until(again, 10)  # a transient 'slot unavailable' refusal may come first; never 'busy' forever
    assert 'busy' not in outs, outs
    tele_x.session.close('done')
    idle()
    # start() never opens a second stream over one whose stop was not confirmed
    tele_s = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                           clock=lambda: clock[0])
    tele_s.start()
    until(lambda: arm.stream_state(tele_s.session.execution_id).selected_revision == 1, 8)
    real_cancel5 = arm.cancel
    arm.cancel = lambda *a, **k: (_ for _ in ()).throw(TimeoutError('link stall'))
    try:
        clock[0] += 1.
        assert tele_s.tick() == 'stop_failed'
        try:
            tele_s.start()
        except NeedsOperator:
            pass
        else:
            raise AssertionError('teleop.start() opened a second stream over an unconfirmed stop')
    finally:
        arm.cancel = real_cancel5
    clock[0] += 1.
    assert tele_s.tick() == 'expired'
    idle()
    # a heartbeat older than the timeout stops the arm even when tick() was never called (command's own check);
    # a second instance on the same ArmClient reuses no request id
    tele2 = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                          clock=lambda: clock[0])
    tele2.start()
    until(lambda: arm.stream_state(tele2.session.execution_id).selected_revision == 1, 8)
    assert tele2.command(target, held=True) == 'requested' and tele2.report.requested[0] != tele.report.requested[0]
    until(lambda: not tele2._settle(), 8)
    clock[0] += 1.
    assert tele2.command(target, held=True) == 'expired'
    until(lambda: stream_phase(tele2.session) is Phase.CANCELLED, 8)
    idle()
    # a far setpoint commits at most one deadman timeout of travel; a stop that is not confirmed is retried and said so
    tele3 = DeadmanTeleop(arm, ctx, deadman_timeout_s=.3, max_speed_rad_s=.5, min_step_s=.2, hold_s=6.,
                          clock=lambda: clock[0])
    tele3.start()
    until(lambda: arm.stream_state(tele3.session.execution_id).selected_revision == 1, 8)
    recorded = {}
    real_revise = arm.revise_stream
    arm.revise_stream = lambda rev: (recorded.__setitem__('rev', rev), real_revise(rev))[1]
    try:
        assert tele3.command(np.asarray(arm.state().q) + 2., held=True) == 'requested'
    finally:
        arm.revise_stream = real_revise
    horizon = recorded['rev'].suffix.times[-1] - recorded['rev'].suffix.times[0]
    assert horizon <= .3 + 1e-9 and any('limited' in n for n in tele3.report.notes), (horizon, tele3.report.notes)
    real_cancel = arm.cancel
    calls = []

    def flaky_cancel(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError('link stall')
        return real_cancel(*a, **k)
    arm.cancel = flaky_cancel
    try:
        clock[0] += 1.
        assert tele3.tick() == 'stop_failed' and not tele3.session.closed
        assert any('STOP NOT CONFIRMED' in n for n in tele3.report.notes)
        assert tele3.tick() == 'expired' and tele3.session.closed  # retried
    finally:
        arm.cancel = real_cancel
    idle()

    # -- RL step-window ---------------------------------------------------------------------------------------------
    env = StepWindow(arm, ctx, hold_s=6.)
    env.reset()
    until(lambda: arm.stream_state(env.session.execution_id).selected_revision == 1, 8)
    result = env.step(np.asarray(arm.state().q) + .002, .3)
    assert result.outcome == 'completed' and result.elapsed_s >= .3 and not result.terminated and not result.truncated, result
    assert result.selected_revision >= 2, result
    # a step the supervisor refuses AFTER admission is 'refused', never 'completed'; a second episode object on the
    # same ArmClient reuses no request id
    far = env.step(np.asarray(arm.state().q) + 50., .3)
    assert far.outcome == 'refused' and far.truncated and not far.terminated, far
    assert not [a for a, _ in env.report.refused if a not in env.report.requested], env.report
    until(lambda: not h.busy and runtime._stream_job is None, 8)
    real_rev4 = arm.stream_revision
    arm.stream_revision = lambda eid, rid: replace(real_rev4(eid, rid), selection_state=SelectionState.PENDING,
                                                   phase=Phase.ADMITTED)
    try:
        unselected = env.step(np.asarray(arm.state().q) + .002, .3)
    finally:
        arm.stream_revision = real_rev4
    assert unselected.outcome == 'not_selected' and unselected.truncated and not unselected.terminated, unselected
    until(lambda: not h.busy and runtime._stream_job is None, 8)
    arm.stream_revision = lambda eid, rid: replace(real_rev4(eid, rid), selection_state=SelectionState.REFUSED)
    try:
        early = env.step(np.asarray(arm.state().q) + .002, 3.)
    finally:
        arm.stream_revision = real_rev4
    assert early.outcome == 'refused' and early.elapsed_s < 1.5, early  # not held to the end of the window
    until(lambda: not h.busy and runtime._stream_job is None, 8)
    arm.stream_revision = lambda *a, **k: (_ for _ in ()).throw(Refused('revision history unavailable'))
    try:
        broken = env.step(np.asarray(arm.state().q) + .002, .3)
    finally:
        arm.stream_revision = real_rev4
    assert broken.truncated and broken.outcome == 'unknown' and env.report.unknown, broken  # never raised; sent => unknown
    until(lambda: not h.busy and runtime._stream_job is None, 8)
    # the reply to the revision is lost AFTER it was sent: 'unknown' (motion may have happened), never 'refused'
    real_revise2 = arm.revise_stream

    def lost_reply(rev):
        real_revise2(rev)
        raise SupervisorError('reply lost')
    arm.revise_stream = lost_reply
    try:
        lost = env.step(np.asarray(arm.state().q) + .002, .3)
    finally:
        arm.revise_stream = real_revise2
    assert lost.outcome == 'unknown' and lost.truncated and env.report.unknown, lost
    until(lambda: not h.busy and runtime._stream_job is None, 8)
    # the next step fails BEFORE anything is sent: 'refused', not a stale 'unknown' (the flag is per step)
    real_state = arm.stream_state
    arm.stream_state = lambda *a, **k: (_ for _ in ()).throw(SupervisorError('link down'))
    try:
        presend = env.step(np.asarray(arm.state().q) + .002, .3)
    finally:
        arm.stream_state = real_state
    assert presend.outcome == 'refused' and presend.truncated, presend
    until(lambda: not h.busy and runtime._stream_job is None, 8)
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
    env2 = StepWindow(arm, ctx, hold_s=6.)  # a second episode object on the same ArmClient: no request id is reused
    env2.reset()
    until(lambda: arm.stream_state(env2.session.execution_id).selected_revision == 1, 8)
    assert env2.step(np.asarray(arm.state().q) + .002, .3).outcome == 'completed'
    env2.session.close('done')
    idle()
    real_cancel6 = arm.cancel
    arm.cancel = lambda *a, **k: (_ for _ in ()).throw(TimeoutError('link stall'))
    try:
        env.session.closed = False
        try:
            env.reset()
        except NeedsOperator:
            pass
        else:
            raise AssertionError('reset() opened a new stream over one it could not cancel')
    finally:
        arm.cancel = real_cancel6
    env.session.closed = True
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
