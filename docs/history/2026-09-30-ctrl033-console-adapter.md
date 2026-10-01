# CTRL-033: console control adapter (option B)

`arm_control/supervisor/console_adapter.py` translates the console's Dora `control`
message fields (gains, pose_hold/Soft, payload) into SDK calls (`select_control_preset`,
`set_payload`), so the SDK is the single owner of mode and load. Everything else
(arm, execute, cancel, hold, stop, reason, cancel_plan_id) is returned in `forward` as an
ordered list of small updates. Payload, gains and Soft are applied independently in `on_control` relative order (a
refused payload does not undo a preset change). SDK calls take effect during `apply()`;
forwarded groups only when the caller sends them, so arm/execute in the same message land
after them (Dora does them first). The console never sends mixed messages.

- Gains resolve only against the named non-Soft presets (exact kp/kd); a non-preset
  gain is refused, not approximated. Soft matches the canonicalized spec minus `id`.
- SDK refusals (Soft not capable/calibrated, payload during an execution, invalid
  mass) come back as refusal strings; nothing is swallowed.
- **Differences from the Dora path (round-1 review; intentional, not "no change"):**
  (1) gains/Soft must be an exact named preset: partial or non-preset gains are refused;
  `gains={}`/`None`/`{kp: None, kd: None}` are ignored as on Dora. (2) Payload during a live execution is
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

**CTRL-053 gates (from review):** wiring needs (a) a differential bench that feeds the same
message through `ArmController.on_control` and the adapter, (b) a real `live()` supplier
(the default cancels nothing), (c) a `send` interleave if mixed messages ever appear, and
(d) a note that same-preset gains block up to 2 s per live execution in a Dora handler.

## CTRL-053 gates (b)-(e), 2026-10-01

- (b) `LiveExecutions(arm)` is the `live()` supplier: it tracks the executions this client starts (`track(id)`) and returns those still not terminal; an execution whose status cannot be read counts as running, but one the supervisor reports as unknown (it retains a bounded number) is dropped. The supervisor has no list-executions op, so an execution another client started is not seen (there is one claimed session).
- (c) `apply(update, send=...)` sends each forwarded group at its `on_control` position (arm, payload, execute, gains, Soft, cancel, hold, stop); a raising `send` is reported as a refusal and the rest still runs. Without `send`, behaviour is unchanged.
- (d) `cancel_budget_s` (default 2 s) bounds the total cancel wait per `apply`. The budget counts from the first cancel. When it runs out the request is refused (remaining legs keep running, earlier ones stay cancelled) and the refusal says how many were cancelled; a cancel that times out (the leg not yet terminal) is reported the same way, noting its own cancel is pending. The budget is approximate: one channel call can run past it, and `LiveExecutions` holds its lock across its status sweep. It bounds only the same-preset path; the changed-preset path is not bounded (the supervisor cancels without waiting and switches at once).
- (e) payload while an execution is current (a staged ON_RELEASE one, not a running one) is refused with the SDK's reason (asserted). A same-Soft re-request is refused by the adapter, but this is NOT exercised: the fake plant is not Soft-capable. It stays unproven until a Soft-capable plant (hardware or sim) runs it.
