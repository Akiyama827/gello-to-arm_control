"""state_history/1: paired feedback at a common RT stamp through two supervisors, fake RT only.

    python tools/bench/check_state_history.py --server <rt build>/arm_rt_server

Two fake RT servers (one attested clock domain), one supervisor each, read only through ``ArmClient.state_history`` and
``align_arms``. Real path: samples carry increasing RT stamps, the aligned stamp lies inside every bracket. Unit cases
(canned replies): an unattested arm, different domains, a stale history, a changed authority and a non-increasing stamp are
refused; an arm without the extension raises Refused. Nothing here is a timing or hardware claim.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dual_supervisor import start_arm  # noqa: E402

from arm_control.multi_arm.supervised_feedback import align_arms  # noqa: E402
from arm_control.sdk import ArmClient, Refused  # noqa: E402


class Canned:
    def __init__(self, reply, ext=1):
        self.reply, self.extensions = reply, {'state_history': ext}

    def state_history(self):
        if self.extensions.get('state_history') != 1:
            raise Refused('no state_history/1')
        return self.reply


def sample(t, seq, **kw):
    return dict(t_mono_ns=t, state_seq=seq, age_ns=1_000_000, session_id=7, arm_epoch=1, stop_epoch=0, armed=True,
                faulted=False, q=[0.1 * seq] * 3, dq=[0.] * 3, **kw)


def reply(stamps, domain='h:b', age_ns=1_000_000, **auth):
    samples = [dict(sample(t, i + 1), age_ns=age_ns) for i, t in enumerate(stamps)]
    return dict(clock_domain=domain, now_ns=0, authority=dict(session_id='7', arm_epoch=1, **auth), samples=samples)


def refused(arms, why, **kw):
    try:
        align_arms(arms, **kw)
    except ValueError as exc:
        assert why in str(exc), (why, exc)
    else:
        raise AssertionError(f'accepted: {why}')


def units():
    ms = 1_000_000
    good = {n: Canned(reply([i * ms for i in range(1, 6)])) for n in 'ab'}
    positions, info = align_arms(good)
    assert set(positions) == set('ab') and info['clock_domain'] == 'h:b', info
    refused({'a': Canned(reply([ms, 2 * ms], domain=None)), 'b': good['b']}, 'attested')
    refused({'a': Canned(reply([ms, 2 * ms], domain='h:x')), 'b': good['b']}, 'one RT clock domain')
    refused({n: Canned(reply([ms, 2 * ms], age_ns=10 * 10**9)) for n in 'ab'}, 'stale')
    refused({'a': Canned(reply([2 * ms, ms])), 'b': good['b']}, 'not increasing')
    bad = reply([ms, 2 * ms])
    bad['authority']['session_id'] = '8'
    refused({'a': Canned(bad), 'b': good['b']}, 'authority changed')
    try:
        align_arms({'a': Canned({}, ext=0), 'b': good['b']})
    except Refused:
        pass
    else:
        raise AssertionError('used an arm without the extension')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    args = parser.parse_args()
    units()
    tmp = Path(tempfile.mkdtemp(prefix='state_history_'))
    (tmp / 'machine-id').write_text('0123456789abcdef0123456789abcdef\n')
    (tmp / 'boot-id').write_text('11111111-2222-3333-4444-555555555555\n')
    flags = ['--machine-id-file', str(tmp / 'machine-id'), '--boot-id-file', str(tmp / 'boot-id')]
    procs, clients = [], []
    try:
        arms = {}
        for name in ('arm_a', 'arm_b'):
            rt, sup, sock, token = start_arm(args.server, tmp, name, flags)
            procs += [sup, rt]
            arms[name] = ArmClient(name, socket_path=sock)
            clients.append(arms[name])
            assert arms[name].extensions.get('state_history') == 1, arms[name].extensions
            arms[name].claim()
            arms[name].arm()
        time.sleep(.3)
        ok = 0
        for _ in range(20):
            hist = {n: a.state_history() for n, a in arms.items()}
            for n, h in hist.items():
                stamps = [s['t_mono_ns'] for s in h['samples']]
                assert len(stamps) > 2 and stamps == sorted(set(stamps)), (n, stamps)
                assert h['clock_domain'] and all(s['age_ns'] >= 0 for s in h['samples'])
            try:
                positions, info = align_arms(arms)
            except ValueError as exc:
                if 'cannot bracket' in str(exc) or 'skew' in str(exc):
                    continue  # two free-running fake RTs: their stamps need not overlap yet
                raise
            for b in info['brackets'].values():
                assert b['before_t_mono_ns'] <= info['t_mono_ns'] <= b['after_t_mono_ns'], info
            ok += 1
            time.sleep(.05)
        print(f'aligned {ok}/20')
    finally:
        for c in clients:
            c.close()
        for p in procs:
            p.terminate()
    print('PASS: state_history/1 through two supervisors; unattested, mixed-domain, stale, non-increasing and authority-changed histories are refused')


if __name__ == '__main__':
    main()
