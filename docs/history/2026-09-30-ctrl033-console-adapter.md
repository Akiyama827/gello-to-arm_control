# CTRL-033: console control adapter (option B)

`arm_control/supervisor/console_adapter.py` translates the console's Dora `control`
message fields (gains, pose_hold/Soft, payload) into SDK calls (`select_control_preset`,
`set_payload`), so the SDK is the single owner of mode and load. Everything else
(arm, execute, cancel, hold, stop, reason, cancel_plan_id) is returned in `forward` as an
ordered list of small updates. Fields are walked in `ArmController.on_control` order
(arm, payload, execute, gains, pose_hold, cancel, hold, stop) and each is applied
independently, as on that path (a refused payload does not undo a preset change).

- Gains resolve only against the named non-Soft presets (exact kp/kd); a non-preset
  gain is refused, not approximated. Soft matches the canonicalized spec minus `id`.
- SDK refusals (Soft not capable/calibrated, payload during an execution, invalid
  mass) come back as refusal strings; nothing is swallowed.
- **Differences from the Dora path (round-1 review; intentional, not "no change"):**
  (1) gains/Soft must be an exact named preset: partial or non-preset gains are refused;
  `gains={}`/`None` are ignored as on Dora. (2) Payload during a live execution is
  refused (SDK rule); Dora applied it. The console sends payload on every attachment
  change, so once wired this is operator-visible. (3) A same-preset gains request
  cancels the executions given by the caller's `live()` (Dora cancels a running leg on
  any gains message; the SDK alone returns early). (4) A Soft request while already in
  that Soft preset is refused: the SDK cannot re-capture the measured pose, Dora does.
  Malformed fields are refusals, never exceptions. Preset equality is exact; whether a
  real console/supervisor config pair resolves identically is untested (bench uses the
  fixture config). The bench compares against the SDK and the documented Dora rules, not
  against `ArmController.on_control` run on the same message.
- Not wired into any deployed graph: production graphs run a Dora `arm_controller`
  node, not the supervisor. Wiring is CTRL-053; raw-jog retirement and a console
  rewrite are CTRL-052. Because of the differences above, wiring it is not a no-op for
  the operator; this does not close CTRL-033's console item.
- Bench `tools/bench/rt/console_adapter.py`: the fake plant is not Soft-capable, so
  the Soft case checks that the SDK's own refusal text is returned; a successful Soft
  selection through the adapter is not exercised offline.
