"""CANCELLED is written only for a confirmed drop, never for a requested one.

A stream whose reference the RT selected is cancelled while the CANCEL is refused
(a stand-in for a lost link) and the PC is not shown the RT's state. The execution
must stay non-terminal, must not be reported CANCELLED however long it waits, and
becomes CANCELLED (`drop_confirmed='rt_observed'`) only once the RT's own state shows
the reference retired (its deadman, or the mode-change retire) or the ARM epoch/session
has changed ('rt_epoch': every RT reference dies with it). A CANCEL that is acknowledged
confirms it at once ('ack'). A proof is never downgraded to 'unconfirmed'.
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
    seq, refuse, blind = 0, [], []
    real_cancel, confirm = link.cancel_reference, runtime._confirm_drops

    def cancel_reference(*args, **kwargs):
        if refuse:
            raise ReferenceRefused(rtp.CTL_CANCEL_REFERENCE, refuse.pop())
        return real_cancel(*args, **kwargs)
    link.cancel_reference = cancel_reference
    runtime._confirm_drops = lambda state: confirm(None if blind else state)

    def stream():
        """Open a stream and wait until revision 1 is selected. A BEGIN refused while the
        RT still holds the previous reference is the open CTRL-034 item 4a; the stream
        then ends REJECTED and is reopened (a refusal is safe, retry)."""
        nonlocal seq
        for _ in range(4):
            seq += 1
            q = arm.state().q
            curve = JointTrajectory([0., 1., 2.], [q, q+.001, q+.002], np.zeros((3, 3)))
            eid = arm.open_stream(StreamOpen(f'a{seq}', seq, 'fake', tuple(runtime.joints),
                scene.facts.model_token, scene.facts.scene_token,
                scene.facts.policy.stream_policy_id, curve)).execution.execution_id
            until(lambda: arm.stream_state(eid).selected_revision == 1 or arm.status(eid).terminal, 10)
            if not arm.status(eid).terminal:
                until(lambda: not h.busy, 5)
                return eid
            until(lambda: not h.busy, 5)
            time.sleep(.3)
        raise AssertionError(('stream never selected', arm.status(eid), h.error))

    def evidence(eid):
        return arm.receipts(eid)[-1].evidence.get('drop_confirmed')

    try:
        # Acknowledged CANCEL: confirmed.
        eid = stream()
        arm.select_control_preset('float')
        until(lambda: arm.status(eid).phase is Phase.CANCELLED, 8)
        assert evidence(eid) in ('ack', 'rt_observed'), evidence(eid)
        arm.select_control_preset('track')

        # Refused CANCEL and no RT evidence: nothing is claimed, however long we wait.
        eid = stream()
        reference = h.id
        refuse.append(rtp.REF_BUSY)
        blind.append(True)
        arm.select_control_preset('float')
        time.sleep(.5)  # well past the RT's 100 ms envelope ceiling, inside --fault-ms
        status = arm.status(eid)
        assert not status.terminal and status.phase is not Phase.CANCELLED, (status.reason, arm.receipts(eid)[-2:])
        # The RT shows the reference retired (envelopes stopped, deadman): now confirmed.
        blind.clear()
        until(lambda: arm.status(eid).phase is Phase.CANCELLED, 8)
        assert evidence(eid) == 'rt_observed', evidence(eid)
        try:
            real_cancel(reference, 1)  # the scripted refusal never reached the RT
        except ReferenceRefused:
            pass  # already retired by the RT's deadman
        arm.select_control_preset('track')

        # The refused CANCEL above left the PC silent past --fault-ms (CMD_LOST is
        # latched): recover with DISARM/ARM, as an operator would.
        arm.disarm()
        arm.arm()
        arm.select_control_preset('track')

        # Refused CANCEL, no RT evidence, then a new ARM epoch: the old reference is gone
        # with its epoch, so the drop is confirmed (it must not stay non-terminal forever).
        eid = stream()
        refuse.append(rtp.REF_BUSY)
        blind.append(True)
        arm.select_control_preset('float')
        assert not arm.status(eid).terminal
        arm.disarm()
        arm.arm()
        blind.clear()
        until(lambda: arm.status(eid).phase is Phase.CANCELLED, 8)
        assert evidence(eid) in ('rt_epoch', 'rt_disarmed'), evidence(eid)
        arm.select_control_preset('track')

        # A proof is not overwritten by a later failed cancel job.
        h._record_drop('k', 'undispatched')
        h._record_drop('k', 'unconfirmed')
        assert h.dropped['k'] == 'undispatched'
        h._record_drop(None, 'ack')
        assert None not in h.dropped
        h.dropped.pop('k')
    finally:
        link.cancel_reference, runtime._confirm_drops = real_cancel, confirm
    print('PASS: CANCELLED only on a confirmed drop: ACK, or the RT showing the reference retired; '
          'a refused CANCEL with no RT evidence stays non-terminal')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
