"""Bounded native TrajOpt refinement and canonical MuJoCo validation.

Install ``arm_control[trajopt]`` to use this opt-in path. Native dependencies
are lazy. Unsupported projection, optimization, timing or validation raises;
there is no unchecked seed fallback.

Both native cubic geometry and the executor's linearly interpolated q are
collision-sampled at <=0.001 rad maximum joint motion. Exact cubic q/qd/qdd
extrema and linear commanded q/qd segment slopes satisfy supplied limits after
uniform time dilation. The executor still interpolates q and qd independently:
qd is not the derivative of commanded q; continuous commanded qdd and jerk
are NOT certified. Sampled collision checks are not continuous certification.
"""

import numpy as np


def _program(world, positions, *, state=False):
    from tesseract_robotics import tesseract_command_language as command
    from tesseract_robotics.tesseract_common import ManipulatorInfo

    program = command.CompositeInstruction('DEFAULT')
    info = ManipulatorInfo()
    info.manipulator, info.working_frame = 'manipulator', 'body_0'
    info.tcp_frame = f'body_{int(world.model.joint(world.joint_names[-1]).bodyid[0])}'
    program.setManipulatorInfo(info)
    for index, q in enumerate(positions):
        if state:
            waypoint = command.StateWaypoint(world.joint_names, q)
            waypoint.setVelocity(np.zeros(len(world.joint_names)))
            waypoint.setAcceleration(np.zeros(len(world.joint_names)))
            wrapped = command.StateWaypointPoly_wrap_StateWaypoint(waypoint)
        else:
            waypoint = command.JointWaypoint(world.joint_names, q, index in (0, len(positions) - 1))
            wrapped = command.JointWaypointPoly_wrap_JointWaypoint(waypoint)
        move = command.MoveInstruction(wrapped, command.MoveInstructionType_FREESPACE, 'DEFAULT')
        program.appendMoveInstruction(command.MoveInstructionPoly_wrap_MoveInstruction(move))
    return program


def _states(program):
    from tesseract_robotics import tesseract_command_language as command

    return [command.WaypointPoly_as_StateWaypointPoly(
        command.InstructionPoly_as_MoveInstructionPoly(instruction).getWaypoint())
        for instruction in program]


def _optimize(env, world, seed):
    from tesseract_robotics import tesseract_command_language as command
    from tesseract_robotics import tesseract_motion_planners_trajopt as trajopt
    from tesseract_robotics.tesseract_common import CollisionMarginPairOverrideType
    from tesseract_robotics.tesseract_collision import CollisionEvaluatorType
    from tesseract_robotics.tesseract_motion_planners import PlannerRequest

    profiles = command.ProfileDictionary()
    composite = trajopt.TrajOptDefaultCompositeProfile()
    composite.smooth_velocities = composite.smooth_accelerations = True
    composite.smooth_jerks = False
    composite.velocity_coeff = np.ones(seed.shape[1])
    composite.acceleration_coeff = np.full(seed.shape[1], 10.)
    model = world.model
    live = [i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    for config, extra in ((composite.collision_cost_config, .005),
                          (composite.collision_constraint_config, 0.)):
        config.enabled = True
        config.contact_manager_config.default_margin = world._pad + extra
        pairs = config.contact_manager_config.pair_margin_data
        for index, a in enumerate(live):
            for b in live[index + 1:]:
                if world._env_geom[a] or world._env_geom[b]:
                    pairs.setCollisionMargin(f'geom_{a}', f'geom_{b}', extra)
        config.contact_manager_config.pair_margin_data = pairs
        config.contact_manager_config.pair_margin_override_type = CollisionMarginPairOverrideType.REPLACE
        config.collision_margin_buffer = .01
        config.collision_coeff_data.setDefaultCollisionCoeff(20.)
        config.collision_check_config.type = CollisionEvaluatorType.LVS_DISCRETE
        config.collision_check_config.longest_valid_segment_length = .01
    solver = trajopt.TrajOptOSQPSolverProfile()
    solver.opt_params.max_iter = 100
    solver.opt_params.max_time = 120.
    solver.opt_params.num_threads = 1
    trajopt.ProfileDictionary_addTrajOptPlanProfile(profiles, 'TRAJOPT', 'DEFAULT', trajopt.TrajOptDefaultPlanProfile())
    trajopt.ProfileDictionary_addTrajOptCompositeProfile(profiles, 'TRAJOPT', 'DEFAULT', composite)
    trajopt.ProfileDictionary_addTrajOptSolverProfile(profiles, 'TRAJOPT', 'DEFAULT', solver)
    request = PlannerRequest()
    request.env, request.instructions, request.profiles = env, _program(world, seed), profiles
    request.format_result_as_input = False
    response = trajopt.TrajOptMotionPlanner('TRAJOPT').solve(request)
    if not response.successful:
        raise ValueError(f'TrajOpt optimization failed: {response.message}')
    result = np.asarray([state.getPosition() for state in _states(response.results)])
    if result.shape != seed.shape or not np.isfinite(result).all():
        raise ValueError('TrajOpt returned invalid knot positions')
    if not np.allclose(result[[0, -1]], seed[[0, -1]], atol=1e-6, rtol=0):
        raise ValueError('TrajOpt changed fixed endpoints')
    # Restore exact requested endpoints after solver tolerance; final checks
    # include these restored segments, so no unchecked geometry is introduced.
    result[[0, -1]] = seed[[0, -1]]
    return result


def _native_timing(env, world, positions, vmax, amax):
    from tesseract_robotics import tesseract_command_language as command
    from tesseract_robotics.tesseract_time_parameterization import IterativeSplineParameterization, ISPCompositeProfile

    program = _program(world, positions, state=True)
    profile = ISPCompositeProfile(1., 1.)
    profile.override_limits, profile.add_points = True, False
    profile.velocity_limits = np.column_stack((-vmax, vmax))
    profile.acceleration_limits = np.column_stack((-amax, amax))
    profiles = command.ProfileDictionary()
    profiles.addProfile('ISP', 'DEFAULT', profile)
    if not IterativeSplineParameterization('ISP').compute(program, env, profiles):
        raise ValueError('native iterative spline timing failed')
    states = _states(program)
    return tuple(np.asarray([getattr(state, getter)() for state in states])
                 for getter in ('getTime', 'getPosition', 'getVelocity', 'getAcceleration'))


def _spline_extrema(spline):
    """Exact per-joint cubic extrema, including either side of every knot."""
    extrema = []
    knots = np.concatenate((spline.x, np.nextafter(spline.x[1:], spline.x[:-1])))
    for order in range(3):
        roots = spline.derivative(order + 1).roots(extrapolate=False)
        times = np.concatenate([knots, *roots])
        values = spline(times[np.isfinite(times)], order)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite native spline extrema')
        extrema.append((np.min(values, axis=0), np.max(values, axis=0)))
    return extrema


def _check_positions(world, positions):
    if (not np.isfinite(positions).all() or np.any(positions < world.lower - 1e-9)
            or np.any(positions > world.upper + 1e-9)):
        raise ValueError('trajectory exceeds finite joint position limits')


def _check_collision(world, positions):
    for q in positions:
        if world.in_collision(q):
            raise ValueError('trajectory fails canonical MuJoCo collision validation')


def _validate_timing(world, positions, t, q, v, a, vmax, amax):
    from scipy.interpolate import CubicHermiteSpline
    from arm_control.motion import JointTrajectory

    if (t.shape != (len(positions),) or any(values.shape != positions.shape for values in (q, v, a))
            or not all(np.isfinite(values).all() for values in (t, q, v, a))
            or t[0] != 0 or np.any(np.diff(t) <= 0)):
        raise ValueError('native timing returned invalid times or states')
    if not np.array_equal(q, positions):
        raise ValueError('native add_points=False changed trajectory geometry')
    if np.max(np.abs(v[[0, -1]])) > 1e-9:
        raise ValueError('native endpoint velocity is nonzero')
    spline = CubicHermiteSpline(t, q, v)
    extrema = _spline_extrema(spline)
    if not np.allclose(spline(t, 2), a, atol=1e-8, rtol=0):
        raise ValueError('native states do not describe their full cubic spline')
    _check_positions(world, np.asarray(extrema[0]))
    dt = np.diff(t)[:, None]
    velocity_ratio = np.max(np.maximum(np.abs(extrema[1][0]), np.abs(extrema[1][1])) / vmax)
    acceleration_ratio = np.max(np.maximum(np.abs(extrema[2][0]), np.abs(extrema[2][1])) / amax)
    secant_ratio = np.max(np.abs(np.diff(q, axis=0) / dt) / vmax)
    command_acceleration_ratio = np.max(np.abs(np.diff(v, axis=0) / dt) / amax)
    factor = max(1., velocity_ratio, np.sqrt(acceleration_ratio), secant_ratio,
                 np.sqrt(command_acceleration_ratio))
    if not np.isfinite(factor):
        raise ValueError('nonfinite required time dilation')
    t, v, a = t * factor, v / factor, a / factor**2
    if not all(np.isfinite(values).all() for values in (t, v, a)):
        raise ValueError('nonfinite dilated trajectory')
    spline = CubicHermiteSpline(t, q, v)
    extrema = _spline_extrema(spline)
    dt = np.diff(t)[:, None]
    for values, limits in ((np.asarray(extrema[1]), vmax), (np.asarray(extrema[2]), amax),
                           (v, vmax), (a, amax), (np.diff(q, axis=0) / dt, vmax),
                           (np.diff(v, axis=0) / dt, amax)):
        if np.any(np.abs(values) > limits * (1 + 1e-9)):
            raise ValueError('dilated trajectory exceeds velocity or acceleration limits')
    # ponytail: serial dense oracle checks; batch only if validation dominates.
    max_speed = float(np.max(np.abs(extrema[1])))
    for index, (start, end) in enumerate(zip(q[:-1], q[1:])):
        intervals = max(1, int(np.ceil(np.max(np.abs(end - start)) / .001)))
        _check_collision(world, start + np.linspace(0, 1, intervals + 1)[:, None] * (end - start))
        intervals = max(1, int(np.ceil((t[index + 1] - t[index]) * max_speed / .001)))
        samples = spline(np.linspace(t[index], t[index + 1], intervals + 1))
        _check_positions(world, samples)
        _check_collision(world, samples)
    return JointTrajectory(t, q, v)


def refine_trajectory(world, seed_positions, vmax, amax):
    """Refine >=2 finite seed knots against a fresh canonical collision world.

    ``world`` is the raw MuJoCoCollisionWorld snapshot, not a callback/cache.
    Limits have one finite positive value per planned joint. Raises on failure.
    Two/three-knot seeds gain segment midpoints for native ISP's four-knot
    minimum; every original knot and the piecewise linear geometry are retained.
    """
    seed, vmax, amax = (np.asarray(value, dtype=float) for value in (seed_positions, vmax, amax))
    joints = len(world.joint_names)
    if joints == 0 or seed.ndim != 2 or seed.shape[1] != joints or len(seed) < 2 or not np.isfinite(seed).all():
        raise ValueError('TrajOpt requires finite (N>=2, planned joints) seed positions')
    if any(values.shape != (joints,) or not np.isfinite(values).all() or np.any(values <= 0) for values in (vmax, amax)):
        raise ValueError('TrajOpt limits must be finite positive per-joint vectors')
    if any(np.shape(values) != (joints,) or not np.isfinite(values).all() for values in (world.lower, world.upper)) or np.any(world.lower >= world.upper):
        raise ValueError('TrajOpt requires finite ordered joint bounds')
    _check_positions(world, seed)
    while len(seed) < 4:
        index = int(np.argmax(np.linalg.norm(np.diff(seed, axis=0), axis=1)))
        seed = np.insert(seed, index + 1, (seed[index] + seed[index + 1]) / 2, axis=0)
    from arm_control.planning.projection import project_world

    env = project_world(world)
    positions = _optimize(env, world, seed)
    _check_positions(world, positions)
    trajectory = _validate_timing(world, positions, *_native_timing(env, world, positions, vmax, amax), vmax, amax)
    lengths = [float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) for path in (seed, positions)]
    print(f'[trajopt] PASS knots={len(positions)} path_rad={lengths[0]:.6f}->{lengths[1]:.6f} '
          f'duration_s={trajectory.duration_sec:.6f} fixed_endpoints=PASS '
          'cubic_limits=PASS command_segment_limits=PASS canonical_collision_0.001rad=PASS', flush=True)
    return trajectory


def _self_check(native=False):
    from types import SimpleNamespace
    from scipy.interpolate import CubicHermiteSpline

    world = SimpleNamespace(joint_names=['a', 'b'], lower=-np.ones(2),
                            upper=np.ones(2), in_collision=lambda q: False)
    q = np.column_stack((np.linspace(0, .3, 4), np.zeros(4)))
    for seed, velocity in ((q, [np.nan, 1]), (q[:1], [1, 1]),
                           (q * np.nan, [1, 1])):
        try:
            refine_trajectory(world, seed, velocity, [1, 1])
        except ValueError:
            pass
        else:
            raise AssertionError('invalid input accepted')
    t = np.arange(4, dtype=float)
    v = np.zeros_like(q)
    spline = CubicHermiteSpline(t, q, v)
    # Piecewise rest-to-rest cubic has interior speed peaks, not at knots.
    extrema = _spline_extrema(spline)
    assert np.isclose(extrema[1][1][0], .15)
    trajectory = _validate_timing(world, q, t, q.copy(), v,
                                  spline(t, 2), np.ones(2)*.1,
                                  np.ones(2)*.2)
    assert trajectory.duration_sec > 3
    assert np.array_equal(trajectory.positions, q)
    assert np.all(trajectory.velocities[[0, -1]] == 0)
    world.in_collision = lambda q: .149 < q[0] < .151
    try:
        _validate_timing(world, q, t, q.copy(), v, spline(t, 2),
                         np.ones(2), np.ones(2))
    except ValueError:
        pass
    else:
        raise AssertionError('interior collision accepted')
    if native:
        import mujoco
        from arm_control.planning.mujoco_collision import MuJoCoCollisionWorld
        from arm_control.planning.projection import project_world

        world = MuJoCoCollisionWorld.__new__(MuJoCoCollisionWorld)
        world.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body pos="0 0 1">'
            '<joint name="a" range="-90 90"/><geom type="capsule" size=".05 .2"/>'
            '<body pos=".4 0 0"><joint name="b" range="-90 90"/>'
            '<geom type="sphere" size=".05"/></body></body></worldbody></mujoco>')
        world.data = mujoco.MjData(world.model)
        world.joint_names, world._qadr = ['a', 'b'], np.array([0, 1])
        world._held_qpos = world.model.qpos0.copy()
        world.lower, world.upper = world.model.jnt_range.T.copy()
        world._env_geom, world._pad = np.zeros(world.model.ngeom, dtype=bool), -.002
        seed = np.linspace([0, 0], [.3, -.2], 8)
        trajectory = refine_trajectory(world, seed, np.ones(2), np.ones(2))
        assert trajectory.num_joints == 2
        assert np.array_equal(trajectory.positions[[0, -1]], seed[[0, -1]])
        for count in (2, 3):
            short_seed = np.linspace([0, 0], [.02, -.01], count)
            trajectory = refine_trajectory(world, short_seed, np.ones(2), np.ones(2))
            assert len(trajectory.times) == 4
            assert np.array_equal(trajectory.positions[[0, -1]], short_seed[[0, -1]])
        world.model.geom_margin[0] = .01
        try:
            project_world(world)
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported projection margin accepted')
    print('TrajOpt self-check: PASS')


if __name__ == '__main__':
    import sys

    _self_check(native='--native' in sys.argv)
