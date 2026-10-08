# observe/1 over SimLink (scene hosting, step 1)

Operator 2026-10-08: build the design in 2026-10-08-scene-hosting-design.md (D1 (a): the supervisor admits the base, gripper and
scene actions; D2 after D1; D3: `sim_bench` first). Step 1 proves the observation contract on the existing single-model plant,
before any scene work.

* `MuJoCoBackend.observe()`: this tick's world by name: plant time, model revision, and the position and velocity of every motor and
  finger joint.
* `SimLink(observe=...)`, from the config's `sim_observe: {rate_hz, ring_s}`. It is off unless stated, and refused if malformed
  (0 < rate_hz <= sim_servo.state_hz, ring_s > 0).
  * It records one observation every `ceil(state_hz / rate_hz)` ticks, right after that tick's plant step, so every field is the
    world that step left.
  * The ring holds `ceil(ring_s * rate)` observations and never makes the tick wait. A failing observation counts in `errors`,
    and the tick goes on.
* `observations(after, limit)` returns samples oldest first. `latest_seq` is the next cursor: the last sample returned, or, with
  none, `after` clamped to the newest. `dropped` counts the samples that fell off the ring before this read.
* `RobotRuntime` advertises `observe: 1` only for an observing link. `service` adds the `observe` operation (limit capped at 1024),
  and `ArmClient.observe(after, limit)` decodes and validates the batch.
* Fields a single-model plant cannot provide (scene revision, gripper, attachments, objects, constraint forces, wrench, topology)
  are None, never stale. The scene host fills them in step 2.

Checked: `check_sim_supervisor.py`.
* `observe_unit`: refusals; decimation; plant-time equality with the stepping tick; the cursor, `dropped`, `limit` and the stale-cursor
  clamp; a failing observation.
* The process part: `observe/1` advertised; samples read without a claim; the cursor advances.

Mutants killed: an observation taken before the step, no decimation, no `dropped`, no `limit`.

Review round 1 (BLOCK): the design's "a slow reader cannot slow the plant" was violated. Building and serializing up to 1024
samples per read, in the supervisor's process, competed with the tick for the interpreter: four tight readers slowed the plant to
0.44x real time. Fixed:
* a read returns at most `OBSERVE_LIMIT_MAX` = 64 samples;
* the start index is arithmetic (sequences are contiguous). A read costs O(start + limit), and a caught-up reader returns at once;
* `ring_s` is bounded at 10 s.

The bench now measures the plant's own clock (observation sim time against wall time) while four reader PROCESSES poll at the
cap: 1.000x. With the cap removed (mutant) the same bench measures 0.713x and fails. Reader threads in one process do not show
the effect; they throttle each other.

Also from the review:
* the unit drives the arm, so a stale position cannot pass for the tick's (that mutant survived before);
* `observe/1` is not advertised for a link that does not observe;
* the cap is tested;
* the SDK decoder checks `rate_hz`, `model_revision` and `actor_q`;
* sequences are per process, and the docstring says so;
* the design note's wording matches the code (`limit`, `capacity`, rules 7-8).

Round 2 (ACCEPT), folded in: the load test asserts its readers are alive; caught-up reads return at once; the design states the
minimum poll rate and describes the consistency rule as built; one malformed-reply test for the SDK decoder.
