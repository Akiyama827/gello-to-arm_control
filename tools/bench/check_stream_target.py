"""Analytical target strategy check; explicit fixture values, no devices."""
from dataclasses import replace
import numpy as np

from arm_control.control.execution_policy import ExecutionPolicy
from arm_control.planning.stream_target import TargetStopGrid
from arm_control.supervisor.admission import StreamPolicy


def main():
    policy = StreamPolicy('analytical-envelope-1', .05, .1, .9, (.01,))
    strategy = TargetStopGrid((.1, .2, .4), (1., .5, .25))
    limits = ExecutionPolicy(command_rate_hz=100, state_timeout_sec=.25,
                             health_timeout_sec=1., start_pos_tol_rad=.1,
                             abort_pos_err_rad=.35, hold_relatch_rad=.3,
                             torque_limits=np.array([87.]), position_lower=np.array([-3.]),
                             position_upper=np.array([3.]), velocity_limits=np.array([.4]),
                             acceleration_limits=np.array([.8]))
    args = dict(cut_phase=.2, q=[.002], qd=[.01], observed_phase=0., stream_policy=policy,
                execution_policy=limits, kp=np.array([100.]), kd=np.array([10.]), check=lambda _: '')
    curve, alpha, duration = strategy.toward([.022], **args)
    assert (alpha, duration) == (1., .4)
    for phase in (.01, .03):
        another, _, _ = strategy.toward([.022], **{**args, 'observed_phase': phase})
        assert another.positions.tobytes() == curve.positions.tobytes()
        assert another.times.tobytes() == curve.times.tobytes()
    partial = strategy.toward([.022], **{**args, 'check': lambda t: 'collision' if any(q[0] > .013 for q in t.densify(.001)) else ''})
    assert partial[1] == .5
    for invalid in (lambda: replace(strategy, duration_grid_s=(.1, .1)),
                    lambda: replace(strategy, progress_fractions=(.5, .25)),
                    lambda: replace(policy, max_candidate_age_s=float('nan')),
                    lambda: strategy.toward([10.], **args),
                    lambda: strategy.toward([.022], **{**args, 'check': lambda _: 'collision'})):
        try:
            invalid()
        except ValueError:
            pass
        else:
            raise AssertionError('invalid/infeasible declaration passed')
    print('PASS: production target strategy uses explicit grid/envelope, continuous bounds, partial progress, scene refusal, no arrival-derived timing')


if __name__ == '__main__':
    main()
