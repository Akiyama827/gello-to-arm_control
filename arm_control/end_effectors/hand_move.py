"""Measured completion for a position-only Hand operation, without grasp ownership."""
import math

from arm_control.end_effectors.franka_hand import resolve_grasp_parameters


class HandMove:
    """A matching operation DONE plus a fresh measured endpoint is success."""

    def __init__(self):
        self.pending = None

    def start(self, request, state, now):
        if self.pending is not None:
            raise ValueError('hand move already pending')
        rid = request.get('request_id')
        if not isinstance(rid, str) or not rid:
            raise ValueError('hand move requires a request_id')
        if not (state.get('available') and state.get('measured')) or state.get('busy'):
            raise ValueError('hand unavailable, stale, or busy')
        if state.get('is_grasped'):
            raise ValueError('position move refused while holding an object')
        if any(key not in request for key in ('width_m', 'speed_mps')):
            raise ValueError('position move requires explicit width and speed')
        parameters = resolve_grasp_parameters({}, request)
        self.pending = dict(request_id=rid, width_m=parameters['width_m'],
                            deadline=now + 8., done_seq=None)
        return f"MOVE {parameters['width_m']:.8f} {parameters['speed_mps']:.8f}"

    def fail(self, reason):
        if self.pending is None:
            return None
        result = dict(request_id=self.pending['request_id'], ok=False, reason=reason)
        self.pending = None
        return result

    def done(self, result, state):
        if self.pending is None or result['request_id'] != self.pending['request_id']:
            return None
        if not result['ok']:
            return self.fail(result.get('reason', 'hand move failed'))
        self.pending['done_seq'] = state.get('sample_seq', 0)
        return None

    def poll(self, state, now):
        p = self.pending
        if p is None:
            return None
        if not state.get('available'):
            return self.fail('hand unavailable')
        if now >= p['deadline']:
            return self.fail('hand move timeout')
        if not state.get('measured'):
            return None
        if state.get('is_grasped'):
            return self.fail('unexpected object in position-only move')
        if p['done_seq'] is None or state.get('sample_seq', 0) <= p['done_seq']:
            return None
        if state.get('busy'):
            return None
        width = state.get('width')
        if (isinstance(width, bool) or not isinstance(width, (float, int))
                or not math.isfinite(width) or abs(width - p['width_m']) > .001):
            return self.fail('hand stopped outside position tolerance')
        self.pending = None
        return dict(request_id=p['request_id'], ok=True, reason='position reached')


def _self_check():
    assert 'HandMove' in globals(), 'position-only Hand completion is missing'
    move = HandMove()
    state = dict(width=.075, is_grasped=False, measured=True, available=True,
                 sample_seq=1, busy=False)
    request = dict(request_id='a', width_m=.045, speed_mps=.05)
    move.start(request, state, 0.)
    assert move.poll(dict(state, width=.045, sample_seq=2), .1) is None
    move.done(dict(request_id='other', ok=True), state)
    assert move.poll(dict(state, width=.045, sample_seq=3), .2) is None
    move.done(dict(request_id='a', ok=True), dict(state, sample_seq=3))
    assert move.poll(dict(state, width=.045, sample_seq=3), .3) is None
    result = move.poll(dict(state, width=.045, sample_seq=4), .4)
    assert result['ok'] and result['request_id'] == 'a'
    assert not hasattr(move, '_held'), 'position completion must not own a grasp'
    move.start(dict(request, request_id='b'), state, 1.)
    result = move.poll(dict(state, available=False), 1.1)
    assert not result['ok']
    move.start(dict(request, request_id='c'), state, 2.)
    assert not move.poll(state, 20.)['ok']
    for bad in ({'width_m': float('nan')}, {'speed_mps': 0}, {'request_id': ''}):
        try:
            move.start(dict(request, **bad), state, 0.)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid Hand move accepted')
    for width in (.075, float('nan')):
        move.start(request, state, 0.)
        move.done(dict(request_id='a', ok=True), state)
        assert not move.poll(dict(state, width=width, sample_seq=2), .1)['ok']
    move.start(request, state, 0.)
    try:
        move.start(dict(request, request_id='new'), state, .1)
    except ValueError:
        pass
    else:
        raise AssertionError('concurrent request replaced pending move')
    assert move.done(dict(request_id='a', ok=False), state)['ok'] is False
    print('hand_move self-check OK')


if __name__ == '__main__':
    _self_check()
