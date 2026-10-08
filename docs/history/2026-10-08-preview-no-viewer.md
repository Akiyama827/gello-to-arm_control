# The plan preview never takes its host process down (no Rerun viewer)

Found by the project suite (`check_pipeline_sim`, 2026-10-08) once the operator's Rerun viewer was closed: the arm planner
worker died at start, `RuntimeError: gRPC has been unable to connect for 6s, uri: rerun+http://127.0.0.1:9876/proxy`, raised
from `rr.init` inside `planning/preview_rerun.init_preview_stream` (Rerun flushes an earlier recording of the process, whose
gRPC sink has no viewer). The whole add transaction failed. Earlier passes had a viewer running. The same code path runs in
the hardware graphs: a closed viewer would have stopped the planner there too.

* `init_preview_stream`: init, spawn/connect and the blueprint are one guarded step; any Rerun failure prints `preview
  disabled` once and installs a disabled default recording (`RecordingStream(..., default_enabled=False)`), so every later
  preview log is a no-op. The old spawn-only fallback (`memory_recording()`) could not work here: it needs the recording
  `rr.init` failed to create (measured: it raised "No recording was provided or set as current").
* `simulation/rerun_scene` mirror thread: the same guard; it says so once and stops instead of dying with a traceback.

Checked: with no viewer listening, `check_pipeline_sim sim-add` (project) passes 2/2 (it failed 3/3 before); the planner logs
`preview disabled` once. Not changed: the CAVEAT about the gRPC batcher blocking a logging thread when a viewer is attached
but slow.
