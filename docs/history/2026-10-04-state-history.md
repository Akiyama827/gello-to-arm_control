# state_history/1: RT-stamped recent states through the supervisor (dual-arm slice 3)

Negotiated opt-in (hello `extensions.state_history == 1`, advertised when the link keeps recent packets). Read-only: no
claim needed, like `state`. `ArmClient.state_history()` returns the supervisor's last 32 distinct RT packets (oldest first):
RT stamp `t_mono_ns` (RT host CLOCK_MONOTONIC, named by `clock_domain`, `None` if unattested), `state_seq`, session/ARM/stop
epochs, armed/faulted, `q`, `dq`, and `age_ns` on the supervisor's clock at `now_ns`; plus the current `authority`
(session id, ARM epoch). No RT or wire change: the packets are the ones the supervisor's link already receives.

`multi_arm/supervised_feedback.align_arms(arms)` feeds these to the unchanged `align_feedback` (common RT stamp, bracketing,
no extrapolation, the same limits). Freshness uses the client's clock with each age taken from a read before the call, so
the error is toward refusal. An arm with an unattested clock domain is refused (a supervisor hides the host name the direct
path falls back on); different domains, stale or non-increasing histories and a changed authority are refused.

Limits: 32 packets, about 128 ms at the default 250 Hz RT publish rate (measured: 32 samples spanning 126 ms in review).
The binding limit is `align_feedback`'s 50 ms freshness (`max_age_s`), about 12 packets: a client that reads late or slowly
gets a refusal, never an extrapolation. Reply size about 7 KB at 3 joints. The closure record (`multi_arm/closure.py`) still reads the direct link; moving it is a later slice.
Not a qualification: fake RT only. Bench: `tools/bench/check_state_history.py` (in the offline suite).
