"""Adversarial stream/1 admission checks over SDK/socket and fake-only RT.

The gates block trusted scene work or an upload reply, never the control loop.
All model/scene changes are analytical fault injection, not deployment facts.
Transient payload changes use the canonical SDK owner.
"""
from dataclasses import replace
from pathlib import Path
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

from arm_control.config import load_robot_config
from arm_control.execution.contract import Phase, Start
from arm_control.execution.controller import PayloadValue
from arm_control.execution.stream import StreamOpen, StreamRevision, SelectionState
from arm_control.motion import JointTrajectory
from arm_control.sdk import ArmClient
from arm_control.supervisor import runtime as runtime_module
from arm_control.supervisor.runtime import RobotRuntime
from arm_control.supervisor.service import Service
from native_reference import unused_port
from reference_supervisor import until
from stream_supervisor import Scene, refusal, write_config


def cases(arm, runtime, scene, service):
    base = replace(scene.facts, policy=replace(scene.facts.policy, max_candidate_age_s=1., max_future_s=10.))
    scene.facts = base
    sequence = 100

    def request(name, *, start=Start.ON_RELEASE):
        nonlocal sequence
        sequence += 3
        q = arm.state().q
        return StreamOpen(name, sequence, 'fake', tuple(runtime.joints), base.model_token,
                          base.scene_token, base.policy.stream_policy_id,
                          JointTrajectory([0., 3., 6., 9.], np.array([q]*4), np.zeros((4, 3))), start)

    def idle():
        until(lambda: runtime._stream_job is None and not runtime._native.busy)

    mutations = {
        'model': replace(base, model_token='other-model'),
        'scene': replace(base, scene_token='scene-2'),
        'policy': replace(base, policy=replace(base.policy, max_future_s=9.5)),
    }
    # Facts change while the scene worker is blocked. Its late success must
    # neither admit the old snapshot nor restore the execution after cancellation.
    for name, changed in mutations.items():
        idle()
        scene.entered.clear()
        scene.resume.clear()
        req = request(f'worker-{name}')
        view = arm.open_stream(req)
        eid = view.execution.execution_id
        assert view.execution.phase is Phase.REQUESTED
        assert scene.entered.wait(1.)
        scene.facts = changed
        result = arm.wait(eid, 1.)
        assert result.phase is Phase.REJECTED, (name, result)
        scene.resume.set()
        idle()
        assert arm.status(eid) == result
        assert arm.open_stream(req).execution == result
        refusal(lambda: arm.open_stream(replace(req, request_id=req.request_id+'-new')))
        scene.facts = base
    print('PASS: model/scene/policy worker races and late-result rejection')

    # Same facts change after native SEAL but before COMMIT. The task snapshot
    # was valid at admission; now the admitted execution must retire unselected.
    upload = runtime.link.upload_reference
    for name, changed in mutations.items():
        idle()
        entered, resume = threading.Event(), threading.Event()
        def delayed_reply(*args, **kwargs):
            result = upload(*args, **kwargs)
            entered.set()
            assert resume.wait(2.), 'fixture upload gate not released'
            return result
        runtime.link.upload_reference = delayed_reply
        req = request(f'selection-{name}', start=Start.ON_ADMIT)
        view = arm.open_stream(req)
        eid = view.execution.execution_id
        assert entered.wait(1.)
        candidate = runtime._native.id
        assert arm.stream_revision(eid, req.request_id).selection_state is SelectionState.NOT_REQUESTED
        scene.facts = changed
        result = arm.wait(eid, 1.)
        assert result.phase is Phase.CANCELLED, (name, result)
        resume.set()
        runtime.link.upload_reference = upload
        idle()
        assert arm.status(eid) == result
        observed = runtime._native.observed
        assert observed is None or observed['reference_id'] != candidate
        scene.facts = base
    print('PASS: model/scene/policy changes after SEAL cannot select')

    # A valid start can become invalid after trusted admission. Inject only
    # the observed velocity in the readback, leaving fake plant/command laws alone.
    idle()
    entered, resume = threading.Event(), threading.Event()
    latest = runtime.link.latest_state
    def delayed_start(*args, **kwargs):
        result = upload(*args, **kwargs)
        entered.set()
        assert resume.wait(2.)
        return result
    runtime.link.upload_reference = delayed_start
    req = request('start-velocity-race', start=Start.ON_ADMIT)
    eid = arm.open_stream(req).execution.execution_id
    assert entered.wait(1.)
    def wrong_velocity():
        state, received = latest()
        return replace(state, dq=[.5]*3), received
    runtime.link.latest_state = wrong_velocity
    until(lambda: np.allclose(arm.state().dq, .5))
    resume.set()
    result = arm.wait(eid, 1.)
    assert result.phase is Phase.REJECTED and 'velocity' in result.reason, result
    runtime.link.latest_state = latest
    runtime.link.upload_reference = upload
    idle()
    print('PASS: changed start velocity refuses after SEAL before COMMIT')

    # Payload truth is now owned by the runtime, never the provider.
    arm.set_payload(PayloadValue(.5))
    req = request('pc-payload')
    view = arm.open_stream(req)
    assert arm.wait(view.execution.execution_id, 1.).phase is Phase.REJECTED
    arm.set_payload(PayloadValue(0.))
    idle()
    print('PASS: canonical PC payload refuses loaded streams')

    # One candidate and one admission job despite many syntactically valid intents.
    # The SDK serializes requests; a stalled scene worker must not serialize cancel.
    scene.entered.clear()
    scene.resume.clear()
    req = request('flood-held')
    view = arm.open_stream(req)
    eid = view.execution.execution_id
    assert scene.entered.wait(1.)
    before = len(runtime._executions), len(runtime._stream_attempts)
    for i in range(32):
        refusal(lambda i=i: arm.open_stream(replace(req, open_seq=req.open_seq+10+i,
                                                    request_id=f'flood-{i}')))
        assert (len(runtime._executions), len(runtime._stream_attempts)) == before
    assert runtime._stream_marks[arm.session][0] == req.open_seq
    begin = time.monotonic()
    arm.cancel(eid)
    assert time.monotonic()-begin < .5
    scene.resume.set()
    idle()
    print('PASS: 32 rejected producer intents retain one worker/candidate; cancel responsive')

    # The same bound with a selected reference continuing under an unselected
    # revision: candidate abort must not cancel or rewrite the selected curve.
    idle()
    req = request('selected-flood', start=Start.ON_ADMIT)
    eid = arm.open_stream(req).execution.execution_id
    until(lambda: arm.stream_state(eid).selected_revision == 1)
    idle()
    anchor = arm.stream_anchor(eid)
    suffix = JointTrajectory([anchor.cut_phase, anchor.cut_phase+.2],
                             [anchor.q, np.array(anchor.q)+.001], [anchor.qd, np.zeros(3)])
    proposal = StreamRevision('blocked-revision', eid, 10, 1, anchor.anchor_id,
                              base.scene_token, suffix)
    scene.entered.clear()
    scene.resume.clear()
    selected_bytes = runtime._native.trajectory.positions.tobytes()
    arm.revise_stream(proposal)
    assert scene.entered.wait(1.)
    before = len(runtime._executions), len(runtime._stream_attempts)
    cycle = runtime.controller._cycle_id
    for i in range(32):
        refusal(lambda i=i: arm.revise_stream(replace(proposal, proposal_seq=20+i,
                                                      request_id=f'revision-flood-{i}')))
        assert (len(runtime._executions), len(runtime._stream_attempts)) == before
        assert arm.stream_state(eid).selected_revision == 1
    assert runtime.controller._cycle_id > cycle
    assert runtime._stream_marks[arm.session][1] == 10
    assert arm.abort_stream_revision(eid, proposal.request_id).phase is Phase.REJECTED
    scene.resume.set()
    idle()
    assert runtime._native.trajectory.positions.tobytes() == selected_bytes
    assert arm.stream_state(eid).selected_revision == 1
    arm.cancel(eid)
    print('PASS: selected reference survives 32 rejected revision intents and candidate abort; ticks continue')

    # Small budget to force actual eviction without 512 slow socket cycles.
    # The production eviction implementation and session marks are unchanged.
    retained = runtime_module.RETAINED
    runtime_module.RETAINED = 8
    arm.set_payload(PayloadValue(.5))
    oldest = None
    try:
        for i in range(12):
            idle()
            req = request(f'evict-open-{i}')
            view = arm.open_stream(req)
            assert arm.wait(view.execution.execution_id, 1.).phase is Phase.REJECTED
            oldest = oldest or req
        idle()
        until(lambda: len(runtime._executions)+len(runtime._stream_attempts) <= 8)
        assert (arm.session, oldest.request_id) not in runtime._stream_attempts
        refusal(lambda: arm.open_stream(oldest))
        refusal(lambda: arm.open_stream(replace(oldest, request_id='evicted-replay')))
        high = runtime._stream_marks[arm.session][0]
        arm.unclaim()
        arm.claim()
        assert runtime._stream_marks[arm.session][0] == high
    finally:
        runtime_module.RETAINED = retained
        arm.set_payload(PayloadValue(0.))
        scene.facts = base
    print('PASS: metadata eviction and unclaim/reclaim cannot replay an open sequence')

    # Revision eviction must protect live selected and pending metadata, but not
    # failed attempts. Reject a wrong cut q after ledger insertion (consumes seq).
    idle()
    req = request('revision-eviction', start=Start.ON_ADMIT)
    eid = arm.open_stream(req).execution.execution_id
    until(lambda: arm.stream_state(eid).selected_revision == 1)
    idle()
    runtime_module.RETAINED = 8
    first = None
    try:
        for i in range(12):
            anchor = arm.stream_anchor(eid)
            wrong = JointTrajectory([anchor.cut_phase, anchor.cut_phase+.2],
                                    [np.array(anchor.q)+.01, anchor.q], [anchor.qd, anchor.qd])
            revision = StreamRevision(f'evict-rev-{i}', eid, 100+i*3, 1,
                                      anchor.anchor_id, base.scene_token, wrong)
            assert arm.revise_stream(revision).phase is Phase.REJECTED
            assert arm.stream_state(eid).splice is None
            first = first or revision
        until(lambda: len(runtime._executions)+len(runtime._stream_attempts) <= 8)
        assert (arm.session, first.request_id) not in runtime._stream_attempts
        refusal(lambda: arm.revise_stream(first))
        refusal(lambda: arm.revise_stream(replace(first, request_id='evicted-revision-replay')))
        assert arm.stream_state(eid).selected_revision == 1
        assert arm.stream_revision(eid, req.request_id).selection_state is SelectionState.SELECTED
    finally:
        runtime_module.RETAINED = retained
        arm.cancel(eid)
    print('PASS: revision eviction retains live identity and rejects consumed proposal sequence')

    # A disconnected owner cannot resubmit or receive a late admission into a
    # new service session. Worker capacity remains bounded until old work ends.
    idle()
    scene.entered.clear()
    scene.resume.clear()
    req = request('disconnected-worker')
    eid = arm.open_stream(req).execution.execution_id
    assert scene.entered.wait(1.)
    old_session = arm.session
    arm.close()
    until(lambda: runtime._owner is None)
    with ArmClient('fake', socket_path=service.path) as other:
        assert other.session != old_session
        other.claim()
        new_req = replace(req, request_id='new-session', open_seq=1)
        refusal(lambda: other.open_stream(new_req))  # old worker still occupies its one slot
        assert other.status(eid).phase is Phase.REJECTED
        scene.resume.set()
        idle()
        result = other.open_stream(new_req)
        assert result.execution.phase is Phase.REQUESTED
        until(lambda: other.status(result.execution.execution_id).phase is Phase.ADMITTED)
        assert runtime._stream_marks[other.session] == [1, 0]
        assert old_session not in runtime._stream_marks
        assert other.status(eid).phase is Phase.REJECTED
        refusal(lambda: other.stream_revision(eid, req.request_id))
        other.cancel(result.execution.execution_id)
    print('PASS: reconnect resets namespace only; old owned work cannot resurrect')


def main(binary, scenario=cases, configure=None):
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+') as log:
        root = Path(tmp)
        path = write_config(root, udp, tcp)
        for file in (path, root/'three.urdf'):
            file.write_text(re.sub(r'\bj([123])\b', lambda m: f'j{int(m[1])-1}',
                                   file.read_text().replace('sdk_arm', 'fake')))
        scene = Scene()
        cfg = load_robot_config(path)
        if configure is not None:
            configure(cfg)
        runtime = RobotRuntime(cfg, stream_admission=scene)
        runtime.controller.model_token = scene.facts.model_token
        stop, service = threading.Event(), None
        proc = subprocess.Popen([binary, str(udp), str(tcp)], stdout=log, stderr=log)
        try:
            time.sleep(.2)
            runtime.start()
            service = Service(runtime, root/'races.sock')
            serving = threading.Thread(target=service.serve, args=(stop,), daemon=True)
            serving.start()
            with ArmClient('fake', socket_path=service.path) as arm:
                arm.claim()
                arm.arm()
                scenario(arm, runtime, scene, service)
        except Exception:
            log.seek(0)
            print(log.read())
            raise
        finally:
            scene.resume.set()
            stop.set()
            if service is not None:
                serving.join(1.)
                service.close()
            runtime.close()
            proc.terminate()
            proc.wait(3.)
    print(json.dumps({'result': 'PASS', 'scope': 'actual SDK/runtime/native fake-RT admission races'}))


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
