"""Thin application clients over the SDK (CTRL-052). Stdlib + numpy only; the backend never imports this package.

Four contracts, each one module: ``planner`` (a reviewed trajectory), ``chunks`` (an action-chunk policy with variable
delay), ``teleop`` (a deadman setpoint) and ``rl`` (a step-window). They only call :class:`arm_control.sdk.ArmClient`;
none opens a transport, imports UI internals or recovers the robot. Admitted high-level actions and what the RT actually
selected are reported as separate fields (:class:`~arm_control.clients.common.Report`).
"""
