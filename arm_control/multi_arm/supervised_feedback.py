"""Paired feedback at a common RT stamp through two supervisors (state_history/1).

The same ``align_feedback`` as the direct-link path, fed from each supervisor's recent RT packets. RT stamps and the
bracketing are the RT's; only the freshness check uses the client's clock, with each packet's age taken from a read made
BEFORE the call (so the error is toward older, i.e. toward a refusal). An arm whose clock domain is unattested is refused:
there is no host name to fall back on through a supervisor.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from .feedback import align_feedback


def align_arms(arms: dict, *, clock=time.monotonic, **limits):
    """``arms``: name -> ArmClient advertising state_history/1. Returns what ``align_feedback`` returns."""
    links, histories = {}, {}
    for name, arm in arms.items():
        before = clock()
        reply = arm.state_history()
        domain = reply.get('clock_domain')
        if domain is None:
            raise ValueError(f'{name}: paired feedback needs an attested RT clock domain through a supervisor')
        auth = reply['authority']
        samples = [SimpleNamespace(**{**s, 'n': len(s['q'])}) for s in reply['samples']]
        by_id = {id(st): s for st, s in zip(samples, reply['samples'])}
        links[name] = SimpleNamespace(
            clock_domain=tuple(domain.split(':')), config=SimpleNamespace(host=domain),
            motor_health=lambda auth=auth: dict(session_id=f"{auth['session_id']}", arm_epoch=auth['arm_epoch']),
            motor_state=lambda state, by_id=by_id: dict(position=np.asarray(by_id[id(state)]['q'], float),
                                                       velocity=np.asarray(by_id[id(state)]['dq'], float)))
        histories[name] = [(st, before - s['age_ns'] / 1e9) for st, s in zip(samples, reply['samples'])]
    return align_feedback(links, histories, now=clock(), **limits)
