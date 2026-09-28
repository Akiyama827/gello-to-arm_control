"""CTRL-034 B1 acceptance: run the native benches under a constrained CPU.

Pins every bench (and the fake server it spawns, which inherits the affinity)
to two CPUs and runs a background CPU hog on the same set. Reports pass/fail per
iteration plus the decision / transport / RT-total envelope-age p99.9 and max
each bench printed, so a run on a different machine can be compared directly.

No hardware, no deployment: fake backend only.

    python tools/bench/rt/constrained_cpu.py --build /tmp/arm_rt_build --count 20
    python tools/bench/rt/constrained_cpu.py --build /tmp/arm_rt_build --count 10 --hogs 0
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time

# (label, script, reference_runtime_selfcheck?, script arguments)
BENCHES = (
    ('supervisor', 'reference_supervisor.py', True, ()),
    ('horizon-0.05', 'reference_horizon.py', True, ('0.05',)),
    ('horizon-0.4', 'reference_horizon.py', True, ('0.4',)),
    ('pc-stall', 'pc_stall.py', False, ()),
)
AGE = re.compile(
    r'age\s+(.+?):\s+decision\s+n=\d+\s+p99\.9=(\d+)us\s+max=(\d+)us\s+\|\s+'
    r'transport\s+p99\.9=(\d+)us\s+max=(\d+)us\s+\|\s+'
    r'rt_total\s+p99\.9=(\d+)us\s+max=(\d+)us\s+over_\d+ms=(\d+)')
SELECTION = re.compile(r'selected_to_tracking n=(\d+) p99\.9=(\d+)us max=(\d+)us')


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', default='/tmp/arm_rt_build',
                        help='directory containing the built RT binaries')
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--cpus', default='0,1')
    parser.add_argument('--server-cpus', default=None,
                        help='pin the fake RT separately from the PC bench')
    parser.add_argument('--hogs', type=int, default=1,
                        help='background CPU hogs on the same CPU set (0 = unpinned noise only)')
    parser.add_argument('--no-pin', action='store_true',
                        help='run the benches unpinned (no taskset, no hog)')
    parser.add_argument('--bench', action='append', choices=[b[0] for b in BENCHES])
    parser.add_argument('--timeout', type=float, default=300.0)
    args = parser.parse_args()

    build = Path(args.build).resolve()
    binaries = {
        'runtime': build / 'reference_runtime_selfcheck',
        'server': build / 'arm_rt_server',
    }
    for path in binaries.values():
        assert path.exists(), f'missing {path}'

    prefix = [] if args.no_pin else ['taskset', '-c', args.cpus]
    wrapper_dir = tempfile.TemporaryDirectory()
    if args.server_cpus is not None:
        for label, binary in tuple(binaries.items()):
            wrapper = Path(wrapper_dir.name) / label
            wrapper.write_text(
                '#!/bin/sh\nexec taskset -c ' + args.server_cpus + ' ' + str(binary)
                + ' "$@"\n')
            wrapper.chmod(0o755)
            binaries[label] = wrapper
    # Each hog owns its session so an interrupted run cannot leave it behind.
    hogs = []
    if not args.no_pin:
        for _ in range(args.hogs):
            hogs.append(subprocess.Popen(
                ['taskset', '-c', args.cpus, 'sh', '-c', 'while :; do :; done'],
                start_new_session=True))
    time.sleep(0.3)
    selected = [b for b in BENCHES if not args.bench or b[0] in args.bench]
    failures = 0
    try:
        for label, script, native, extra in selected:
            binary = binaries['runtime' if native else 'server']
            passed, ages, selections = 0, [], []
            for i in range(args.count):
                proc = subprocess.run(
                    prefix + [sys.executable, str(here / script), str(binary), *extra],
                    capture_output=True, text=True, timeout=args.timeout)
                ok = proc.returncode == 0
                passed += ok
                for line in proc.stdout.splitlines():
                    found = AGE.search(line)
                    if found:
                        ages.append((found.group(1),
                                     int(found.group(2)), int(found.group(3)),
                                     int(found.group(4)), int(found.group(5)),
                                     int(found.group(6)), int(found.group(7)),
                                     int(found.group(8))))
                    found = SELECTION.search(line)
                    if found:
                        selections.append(tuple(map(int, found.groups())))
                if not ok:
                    failures += 1
                    text = (proc.stdout + '\n' + proc.stderr).splitlines()
                    tail = [ln for ln in text
                            if 'Error' in ln or 'expired' in ln][-2:]
                    print(f'  {label} run {i + 1}/{args.count} FAIL {tail}')
                else:
                    print(f'  {label} run {i + 1}/{args.count} PASS')
            print(f'{label}: {passed}/{args.count} PASS')
            for name in sorted({a[0] for a in ages}):
                rows = [a for a in ages if a[0] == name]
                print(f'  {name}: decision p99.9_max={max(r[1] for r in rows)}us '
                      f'max={max(r[2] for r in rows)}us | '
                      f'transport p99.9_max={max(r[3] for r in rows)}us '
                      f'max={max(r[4] for r in rows)}us | '
                      f'rt_total p99.9_max={max(r[5] for r in rows)}us '
                      f'max={max(r[6] for r in rows)}us over={sum(r[7] for r in rows)}')
            if selections:
                print(f'  selected_to_tracking: runs={len(selections)} '
                      f'p99.9_max={max(r[1] for r in selections)}us '
                      f'max={max(r[2] for r in selections)}us')
    finally:
        for hog in hogs:
            try:
                os.killpg(os.getpgid(hog.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            hog.wait(3)
        wrapper_dir.cleanup()
    print('CONSTRAINED-CPU ACCEPTANCE:', 'PASS' if failures == 0 else f'{failures} FAIL')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
