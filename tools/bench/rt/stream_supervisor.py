"""stream/1 through the actual SDK/socket/runtime/horizon and fake-only RT.

No physical device/config: generated analytical model and loopback ports only.
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
from arm_control.execution.controller import PayloadValue
from arm_control.execution import codec
from arm_control.execution.contract import Phase, Start
from arm_control.execution.stream import StreamOpen, StreamRevision, SelectionState
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt.client import RtLinkError
from arm_control.sdk import ArmClient, Refused
from arm_control.supervisor.admission import StreamFacts, StreamPolicy
from arm_control.supervisor.runtime import RobotRuntime
from arm_control.supervisor.service import Service
from native_reference import unused_port
from reference_supervisor import until

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_sdk_supervisor import write_config


class Scene:
    def __init__(self):
        self.facts = StreamFacts('analytical-three-joint-fixture', 'scene-1',
                                 StreamPolicy('analytical-envelope-1', .05, .4, 3., (.1,)*3))
        self.entered = threading.Event()
        self.resume = threading.Event()
        self.resume.set()

    def snapshot(self):
        return self.facts

    def check(self, curve, facts):
        self.entered.set()
        assert self.resume.wait(2.), 'fixture worker not released'
        return ''


def refusal(fn):
    try:
        fn()
    except Refused:
        return
    raise AssertionError('expected refusal')


def main(binary):
    udp, tcp = unused_port(socket.SOCK_DGRAM), unused_port(socket.SOCK_STREAM)
    with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryFile(mode='w+') as log:
        root = Path(tmp)
        proc = subprocess.Popen([binary, str(udp), str(tcp)], stdout=log, stderr=log)
        path = write_config(root, udp, tcp)
        for file in (path, root/'three.urdf'):
            text = file.read_text().replace('sdk_arm', 'fake')
            file.write_text(re.sub(r'\bj([123])\b', lambda m: f'j{int(m[1])-1}', text))
        cfg = load_robot_config(path)
        scene = Scene()
        runtime = RobotRuntime(cfg, stream_admission=scene)
        runtime.controller.model_token = scene.facts.model_token
        stop = threading.Event()
        service = None
        try:
            time.sleep(.2)
            runtime.start()
            service = Service(runtime, root/'stream.sock')
            serving = threading.Thread(target=service.serve, args=(stop,), daemon=True)
            serving.start()
            with ArmClient('fake', socket_path=service.path) as arm:
                assert arm.extensions == {'stream': 1, 'controller': 1}
                arm.claim()
                arm.arm()
                def opened(seq, rid, start=Start.ON_ADMIT):
                    q = arm.state().q
                    curve = JointTrajectory([0., .7, 1.4, 2.1],
                                            np.array([q, q+.002, q+.004, q+.006]), np.zeros((4, 3)))
                    req = StreamOpen(rid, seq, 'fake', tuple(runtime.joints), scene.facts.model_token,
                                     scene.facts.scene_token, scene.facts.policy.stream_policy_id,
                                     curve, start)
                    assert codec.encode(codec.stream_open_from(codec.encode(req))) == codec.encode(req)
                    return req, arm.open_stream(req)

                # Deferred work cannot monopolize the request channel or loop.
                scene.resume.clear()
                req, view = opened(2, 'blocked', Start.ON_RELEASE)
                assert view.execution.phase is Phase.REQUESTED
                assert scene.entered.wait(1.)
                refusal(lambda: arm.release(view.execution.execution_id))
                begin = time.monotonic()
                arm.cancel(view.execution.execution_id)
                arm.unclaim()
                assert time.monotonic()-begin < .5
                scene.resume.set()
                until(lambda: runtime._stream_job is None)
                assert arm.status(view.execution.execution_id).terminal
                arm.claim()
                refusal(lambda: arm.open_stream(replace(req, request_id='replay')))

                req, view = opened(7, 'live')
                eid = view.execution.execution_id
                until(lambda: arm.stream_state(eid).selected_revision == 1)
                original = arm.stream_state(eid)
                assert original.splice is None and original.selected_terminal is False
                for _ in range(3):
                    assert arm.stream_state(eid).splice is None
                until(lambda: not runtime._native.busy)
                anchor = arm.stream_anchor(eid)
                assert arm.stream_anchor(eid) == anchor
                suffix = JointTrajectory([anchor.cut_phase, anchor.cut_phase+.6],
                                         [anchor.q, np.array(anchor.q)+.002], [anchor.qd, np.zeros(3)])
                commit = runtime.link.commit_reference
                def wrong_cut(*args, **kwargs):
                    return commit(*args, **{**kwargs, 'protected_until': kwargs['protected_until']+.0001})
                runtime.link.commit_reference = wrong_cut
                rejected = StreamRevision('wrong-cut', eid, 4, 1, anchor.anchor_id,
                                          scene.facts.scene_token, suffix)
                arm.revise_stream(rejected)
                until(lambda: arm.stream_revision(eid, 'wrong-cut').selection_state is SelectionState.REFUSED)
                assert arm.stream_state(eid).selected_revision == 1
                runtime.link.commit_reference = commit
                until(lambda: not runtime._native.busy)
                anchor = arm.stream_anchor(eid)
                suffix = JointTrajectory([anchor.cut_phase, anchor.cut_phase+.6],
                                         [anchor.q, np.array(anchor.q)+.002], [anchor.qd, np.zeros(3)])
                dispatch_entered, dispatch_resume = threading.Event(), threading.Event()
                def lost_ack(*args, **kwargs):
                    dispatch_entered.set()
                    assert dispatch_resume.wait(1.)
                    commit(*args, **kwargs)
                    raise RtLinkError('fixture lost COMMIT acknowledgement')
                runtime.link.commit_reference = lost_ack
                revision = StreamRevision('r1', eid, 9, 1, anchor.anchor_id, scene.facts.scene_token,
                                          suffix, terminal=True)
                attempt = arm.revise_stream(revision)
                assert attempt.phase is Phase.REQUESTED
                assert arm.stream_state(eid).splice is None
                assert arm.revise_stream(revision).request_id == attempt.request_id
                refusal(lambda: arm.revise_stream(replace(revision, terminal=False)))
                assert dispatch_entered.wait(1.)
                pending = arm.stream_revision(eid, 'r1')
                assert pending.phase is Phase.ADMITTED and pending.selection_state is SelectionState.PENDING
                assert arm.abort_stream_revision(eid, 'r1') == pending
                refusal(lambda: arm.stream_anchor(eid))
                refusal(lambda: arm.revise_stream(replace(revision, request_id='flood', proposal_seq=10)))
                dispatch_resume.set()
                until(lambda: arm.stream_revision(eid, 'r1').selection_state is SelectionState.SELECTED
                      or arm.stream_revision(eid, 'r1').phase.terminal)
                assert arm.stream_revision(eid, 'r1').selection_state is SelectionState.SELECTED, arm.stream_revision(eid, 'r1')
                runtime.link.commit_reference = commit
                assert arm.stream_state(eid).selected_terminal is True
                done = arm.wait(eid, 5.)
                assert done.phase is Phase.COMPLETED, done
                assert arm.stream_revision(eid, 'r1').phase is Phase.APPLIED
                print('PASS: actual SDK async open/cancel/unclaim, sequence persistence, pure read, anchor, retry, native splice, terminal APPLIED/settle')
                # Silence on a new nonterminal stream must not complete at its endpoint.
                _, view = opened(12, 'silent')
                eid = view.execution.execution_id
                done = arm.wait(eid, 5.)
                assert done.phase is Phase.FAULTED and 'UNDERFLOW' in done.reason, done
                assert arm.authority().lifecycle.value == 'armed'
                for i, mass in enumerate((.5, 1.)):
                    until(lambda: runtime._stream_job is None)
                    arm.set_payload(PayloadValue(mass))
                    _, view = opened(20+i, f'load{i}')
                    assert arm.wait(view.execution.execution_id, 1.).phase is Phase.REJECTED
                print('PASS: nonterminal silence underflow, known-zero admission, canonical nonzero load refusal')
        except Exception:
            print('attempts', [(k, v.status, v.upload_started) for k, v in runtime._stream_attempts.items()])
            print('native', runtime._native_pending, runtime._native.key, runtime._native.candidate_selection, runtime._native.candidate_error, runtime._native.busy)
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
    print(json.dumps({'result': 'PASS', 'hardware': 'none; fake-only RT'}))


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()))
