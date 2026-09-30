# CTRL-033: console control adapter (option B)

`arm_control/supervisor/console_adapter.py` translates the console's Dora `control`
message fields (gains, pose_hold/Soft, payload) into SDK calls (`select_control_preset`,
`set_payload`), so the SDK is the single owner of mode and load. Everything else
(arm, execute, cancel, stop, reason, hold, cancel_plan_id) is returned in `forward`,
unchanged.

- Gains resolve only against the named non-Soft presets (exact kp/kd); a non-preset
  gain is refused, not approximated. Soft matches the canonicalized spec minus `id`.
- SDK refusals (Soft not capable/calibrated, payload during an execution, invalid
  mass) come back as refusal strings; nothing is swallowed.
- Not wired into any deployed graph: production graphs run a Dora `arm_controller`
  node, not the supervisor. Wiring is CTRL-053; raw-jog retirement and a console
  rewrite are CTRL-052. Nothing operator-visible changes, and this does not close
  CTRL-033's console item.
- Bench `tools/bench/rt/console_adapter.py`: the fake plant is not Soft-capable, so
  the Soft case checks that the SDK's own refusal text is returned; a successful Soft
  selection through the adapter is not exercised offline.
