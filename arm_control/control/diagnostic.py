"""A PC controller's per-tick diagnostic record (CTRL-010/051), shared by every runtime that ticks one."""


def controller_diagnostic(link, c, packet, received, health, reference_time=None):
    """One arm's controller diagnostic, the exact packets a PC controller consumed on a tick (CTRL-010/051).

    The CTRL-010 fields are labeled by origin: `*_pc` are the PC's own accepted command and its algebraic PD+FF against the
    measurement that command's cycle consumed; `*_rt` come from the same RT state packet (the servo's post-clamp post-slew
    output and measured torque). Vendor-acknowledged torque (franka tau_J_d) is not on the wire and stays explicitly absent,
    never zero-filled; the authoritative limiter activity is the RT-side loop-gap diagnostic, never a PC-side cross-tick
    comparison. Shared by the direct hardware runtime and the supervisor's trace ring."""
    motor = link.motor_state(state=packet) if packet is not None else None
    command = c.last_command
    return dict(
        rt_host=link.config.host,
        received_at_pc=received,
        state_t_mono_ns=None if packet is None else packet.t_mono_ns,
        state_seq=None if packet is None else packet.state_seq,
        last_cmd_seq=None if packet is None else packet.last_cmd_seq,
        rt_session=None if packet is None else f"{packet.session_id:016x}",
        health=dict(health),
        controller_measurement_valid=c._last_state_at is not None,
        controller_observed_at_pc=c._last_state_at,
        q_controller=c.last_state.position.tolist(),
        dq_controller=c.last_state.velocity.tolist(),
        q_rt=None if motor is None else motor["position"].tolist(),
        dq_rt=None if motor is None else motor["velocity"].tolist(),
        q_cmd_rt=None if motor is None else motor["position_cmd"].tolist(),
        dq_cmd_rt=None if motor is None else motor["velocity_cmd"].tolist(),
        q_cmd_pc=None if command is None else command.q_des.tolist(),
        dq_cmd_pc=None if command is None else command.qd_des.tolist(),
        tau_ff_pc=None if command is None else command.tau_ff.tolist(),
        integral_bias_pc=c.integral.bias.tolist(),
        raw_pc=(None if command is None else
                (command.kp * (command.q_des - c.last_state.position)
                 + command.kd * (command.qd_des - c.last_state.velocity)
                 + command.tau_ff).tolist()),
        tau_cmd_rt=None if motor is None else motor["torque_cmd"].tolist(),
        tau_meas_rt=None if motor is None else motor["torque"].tolist(),
        plan_id=c.current_plan_id,
        trajectory_elapsed_s=(c.reference.elapsed(reference_time)
                              if c.reference is not None and reference_time is not None else None),
    )
