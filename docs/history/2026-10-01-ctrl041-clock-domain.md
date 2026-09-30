# CTRL-041 slice 1: HELLO clock domain and one-domain alignment (2026-10-01)

Approved design: `Control/docs/ctrl041-design.md` (parent). This slice is the provenance part;
the closure-error decomposition and diagnostics schema version are later slices.

- **Server** (`rt/src/clock_domain.hpp`, `control_session.cpp`): HELLO gains one `key=value` token,
  `clk=mono:<host12>:<boot12>`, placed before `rt=` (older clients ignore unknown tokens; the
  wire version is unchanged). Both fields are HMAC-SHA256 keyed by the machine-id and truncated
  to 6 bytes (as machine-id(5) requires): host = HMAC(machine-id, label), boot = HMAC(machine-id,
  label + boot_id). The raw machine-id and boot-id are never sent or logged. Unreadable ids: no
  token (unattested). New flags `--machine-id-file`, `--boot-id-file` (defaults `/etc/machine-id`,
  `/proc/sys/kernel/random/boot_id`) let device-free checks present other hosts or a reboot.
- **Client** (`client.py`): `RtBackend.clock_domain` = `('mono', host, boot)` or `None`.
- **Alignment** (`multi_arm/feedback.py`): both arms attested and equal => one clock (host strings
  may differ); attested but different, or attested on one arm only => refused; both unattested =>
  the old host-string rule, and the report says `clock_domain: unattested`. No cross-host
  mapping (v1). The report carries the domain.
- **Checks:** `clock_domain_selfcheck` (ctest: format, determinism, reboot/host separation, no raw
  ids, unreadable), `tools/bench/rt/clock_domain.py` (suite: the client's value equals an
  independent Python HMAC; boot/host/unreadable cases; ephemeral ports, readiness poll),
  `check_bimanual_sync.py` (alignment cases).
- **Not done / not claimed:** no RT-host verification; a boot_id changing mid-run is caught only
  through the session/epoch change on reconnect (a reconnect re-reads HELLO); publication time is
  still not capture time; the five-quantity closure record and diagnostics schema version are
  not started; `check_two_instances` still has its fixed ports and sleep (CTRL-013 item d).
