"""An unobserved pending selection combined with a cancel that is refused or slow.

CTRL-033 4b (repeated review). The 250 ms rule must free a stuck slot in every pending case, not only
the ones with no error: (R) a CANCEL that is refused while the selection is unobserved used to leave the
candidate 'pending' forever (busy, no commands, CMD_LOST, no recovery); (S) an operator cancel whose ack
is slow must keep its reason and stay non-terminal until the RT confirms the drop, not be rewritten to
'selection unresolved' by the 250 ms rule.
"""
from pathlib import Path
import sys
import time

import numpy as np

from arm_control.execution.contract import Phase
from arm_control.execution.stream import StreamOpen
from arm_control.motion import JointTrajectory
from arm_control.plants.remote_rt import protocol as rtp
from arm_control.plants.remote_rt.client import ReferenceRefused
from controller_extension import configure
from reference_supervisor import until
from stream_races import main


def cases(arm, runtime, scene, service):
    h, link = runtime._native, runtime.link
    real_cancel, poll, confirm = link.cancel_reference, h.poll, runtime._confirm_drops
    blind, mode, seq = [], {}, 0

    def cancel_reference(*args, **kwargs):
        if mode.get('refuse'):
            mode.pop('refuse')
            raise ReferenceRefused(rtp.CTL_CANCEL_REFERENCE, rtp.REF_BUSY)
        if mode.get('slow'):
            time.sleep(mode['slow'])
        return real_cancel(*args, **kwargs)
    link.cancel_reference = cancel_reference
    h.poll = lambda state, at: poll(None if blind else state, at)
    runtime._confirm_drops = lambda state: confirm(None if blind else state)

    def open_pending():
        nonlocal seq
        seq += 1
        q = arm.state().q
        curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
        blind.append(True)
        eid = arm.open_stream(StreamOpen(f'c{seq}', seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token,
            scene.facts.policy.stream_policy_id, curve)).execution.execution_id
        until(lambda: h.candidate_selection == 'pending', 5)
        return eid

    def reset():
        # A fresh ARM epoch between cases: a stream opened right after a cancel can be refused by BEGIN
        # while the RT still holds the previous reference (the known CTRL-034 item 4a race).
        arm.disarm()
        time.sleep(1.2)
        arm.arm()
        arm.select_control_preset('track')

    try:
        # S: a slow CANCEL ack: the operator's cancel keeps its reason and is not confirmed early.
        eid = open_pending()
        mode['slow'] = .6
        arm.select_control_preset('float')
        time.sleep(.45)  # past 250 ms, CANCEL still unacknowledged, PC blind
        status = arm.status(eid)
        if status.terminal:
            # Legitimate only with proof that COMMIT never reached the RT (the cancel can win the race
            # against COMMIT dispatch), and then with the operator's reason, never 'selection unresolved'.
            assert (status.phase is Phase.CANCELLED and status.reason.startswith('controller preset changed')
                    and arm.receipts(eid)[-1].evidence.get('drop_confirmed') in ('undispatched', 'refused')), (
                'early terminal receipt without proof', status.phase, status.reason, arm.receipts(eid)[-1])
        mode.pop('slow')
        blind.clear()
        until(lambda: arm.status(eid).phase is Phase.CANCELLED, 8)
        status = arm.status(eid)
        assert status.reason.startswith('controller preset changed'), status.reason  # the operator's reason
        assert arm.receipts(eid)[-1].evidence.get('drop_confirmed'), arm.receipts(eid)[-1]
        reset()

        # R: CANCEL refused while the selection is unobserved.
        eid = open_pending()
        mode['refuse'] = True
        arm.select_control_preset('float')
        time.sleep(.6)  # past the 250 ms rule, PC blind
        blind.clear()
        until(lambda: not h.busy and arm.status(eid).terminal, 5)  # was: 'pending' and busy forever
        reset()
        # the slot is usable again: a new stream is admitted and selected
        q = arm.state().q
        seq += 1
        curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
        second = arm.open_stream(StreamOpen(f'c{seq}', seq, 'fake', tuple(runtime.joints),
            scene.facts.model_token, scene.facts.scene_token,
            scene.facts.policy.stream_policy_id, curve)).execution.execution_id
        until(lambda: arm.stream_state(second).selected_revision == 1, 8)
        arm.select_control_preset('float')
        until(lambda: arm.status(second).terminal, 8)
        arm.select_control_preset('track')
    finally:
        link.cancel_reference, h.poll, runtime._confirm_drops = real_cancel, poll, confirm
    print('PASS: an unobserved pending selection is freed by the 250 ms rule even with a refused CANCEL '
          '(slot usable again), and a slow-acked operator cancel keeps its reason until the drop is confirmed')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
