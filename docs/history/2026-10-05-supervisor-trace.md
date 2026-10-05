# trace/1: the supervisor's controller ticks, for lineage and fault traces (CTRL-051, dual-arm slice 5)

Under the supervisor the PC controller that produces the command terms (`tau_ff_pc`, `raw_pc`, `q_cmd_pc`, ...) lives in the
supervisor process, so the console cannot record them itself. trace/1 closes that gap without a second recording format.

**Record.** `control/diagnostic.controller_diagnostic` is the per-arm record `RemoteHardware.diagnostic_sample` always built
(moved verbatim, now shared). The supervisor builds one per loop tick into a bounded ring (`TRACE_TICKS = 64`, about 0.64 s at
100 Hz) and stamps it with a sequence (`seq`), the PC time, the authority lifecycle, the native reference id/revision and
`execution_id`. `plan_id` is the REQUEST id (the identity a client or pair requested and a lineage joins on); the controller
runs an execution id, kept as `execution_id`. Building a record can never stop the loop (an exception is counted in `errors`).

**Read.** Negotiated extension `trace` (hello `extensions.trace == 1`), `ArmClient.trace_samples(after)`: the ticks after
sequence `after`, oldest first, plus `latest_seq`, `capacity`, `errors` and `dropped`: the ticks that fell off the ring
before this reader came back (0 on a first read, which has no gap to measure). The loop never waits for a reader: the ring
overwrites and the gap is reported. Read-only, no ownership needed. Replies are shape-checked in the SDK like the others.

**Console.** `SupervisedHardware.step()` reads each arm's new ticks; `diagnostic_sample()` returns the newest per arm with
`trace_seq`, `trace_dropped` (cumulative ticks missed) and `trace_errors`. The console records one sample per console tick as
before, so ticks between console samples are not in its fault trace; the count says how many the ring lost, not how many the
console skipped. `reconstruct_lineage` over those samples joins complete for a pair executed through two supervisors
(bench). One semantic fix found on the way: an arm whose controller holds NO plan (`''` or `None`: finished, or not yet
loaded) is idle, not running a foreign plan; only a different non-empty plan is foreign.

Reading it back: a read that fails marks the held newest sample `trace_stale` (the supervisor stopped answering; the
console never presents it as current) and a working read clears the mark. The client never reconnects, so a restarted
supervisor is a lost channel, not a sequence that starts again. `trajectory_elapsed_s` is always None in these samples (the
supervisor has no shared reference-time batch). A `complete` lineage means the identities JOINED (every declared member plan
was seen, and each command link came from a joined arm), not that both arms finished; an arm whose plan is cleared mid-span
(cancel, fault) does not make it incomplete, the pair lifecycle says how it ended. The trace read is one more call on the
channel that carries control, so a socket timeout in it closes that channel like any other call. Recording failures: a record
that cannot be built is counted in `errors` and does not advance `seq`, so it shows as `errors`, never as a gap.

Not done: the RT host's own trace export (SITL/RT) and a supervisor-written trace file on fault (the console still writes
its own `PairFaultTrace`); the ring is read by polling, so a console tick slower than 0.64 s loses ticks (counted).
Bench: `check_supervised_hardware.py` (lineage, drop detection, prompt reader), `rt/sdk_channel.py` (reply shape),
`check_trace_retention.py` (idle-vs-foreign lineage).
