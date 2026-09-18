# Historical engineering records

These files preserve the library documentation as it stood before the
architecture/ownership documentation cleanup (source commit
`835686dc7af2e346ed5663b959a4122020d0057a`), plus the delivered plans and
verification records of work completed since. Their bytes are unchanged.

They contain obsolete paths, superseded process contracts, machine-specific
observations, and commands that are not current operating instructions. Do not
use historical safety claims or launch commands to authorize hardware motion.
For current ownership and commands, read [Architecture](../ARCHITECTURE.md) and
[Runbook](../RUNBOOK.md).

| Record | Historical subject |
|---|---|
| [Previous README](README-pre-ownership-refactor.md) | Earlier package layout and launch surfaces |
| [Console manual](console-manual.md) | Operator UI behavior and troubleshooting at that revision |
| [Operator-console design](operator-console.md) | Command ownership, jog envelope, and deadman rationale |
| [Cartesian teleop](cartesian-teleop.md) | Earlier page/IK and calibration workflow |
| [Motion bench checklist](motion-bench-checklist.md) | Old physical bring-up plan; not a current safety procedure |
| [Cartesian soft plan](2026-09-07-cartesian-soft-plan.md) | Delivered plan for the soft/Cartesian mode |
| [Cartesian soft verification](2026-09-07-cartesian-soft-verification.md) | That plan's acceptance evidence |
| [Native DM MIT plan](2026-09-09-dm-native-mit-plan.md) | Delivered plan for native MIT forwarding (`pack_native_mit`); moved out of `docs/superpowers/` on 2026-09-11 |

Old relative links and command examples remain historical evidence rather than
being rewritten to imply that the original documents described today's tree.
