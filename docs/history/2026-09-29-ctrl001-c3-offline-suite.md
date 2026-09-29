# CTRL-001 C3 offline suite check — 2026-09-29

Source: `arm_control` `95ea1d906b99f4d6344985dab0d869ff7bae4f06`, clean tree.
Command: `python tools/bench/rt_offline_suite.py --jobs 4` with local loopback
sockets available. The suite uses `WITH_FRANKA=OFF`; it contacted no RT host or
robot. Its generated `rt/build-offline/offline-summary.json` and per-check logs
are local build artifacts, not hardware evidence.

| Check group | Result |
|---|---|
| Configure/build | 2/2 PASS |
| CTest | 19/19 self-checks PASS |
| SDK streamed/native | 2/2 PASS |
| Fast Python RT benches | 13/14 PASS |
| Whole entry point | **18 PASS, 1 FAIL** |

`controller_extension.py` is the failure. After a preset change cancels a stream
with COMMIT pending, its final `until(lambda: not runtime._native.busy)` times
out after 20 seconds; the RT fake then reports CMD_LOST. The same case failed
when run directly against the integrated `3c82090` source and the local
`reference_runtime_selfcheck`, independently of the new CTest registration.
The stream horizon's `busy` predicate includes `candidate_selection == 'pending'`
after cancellation. This is a reproducible CTRL-033 mode/stream retirement
issue to investigate in that behavior slice, not a reason to mark C3 green or
to alter the controller in a tooling cleanup commit.

A first sandboxed run denied loopback sockets and is not a product result. An
earlier C3 runner attempt also passed the wrong binary to four benches; that
entry-point mapping was corrected before the pinned run above. All other
pinned checks passed. No deployed service, FCI, ARM, or hardware action occurred.
