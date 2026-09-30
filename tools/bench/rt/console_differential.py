"""The same console `control` message through the legacy `ArmController.on_control` and through the SDK adapter.

CTRL-053 gate (a) for wiring `ConsoleControlAdapter` into a graph. Each message starts from the same baseline ('track',
payload 0, nothing running) and the resulting mode (law, kp, kd) and payload (mass, CoM) are recorded for both paths. The
messages the console really sends (named gain presets, valid payloads) must give IDENTICAL state. Every other message is a
documented, intended difference: the table below is the complete list, and the bench fails if a message moves between
"same" and "different", or if a legacy path that raised now applies (or the reverse). It proves equivalence of the
resulting state only; it does not touch a graph, a Dora node or hardware.
"""
from pathlib import Path
import sys

import numpy as np

from arm_control.contracts.motion import pack_control_update, unpack_control_update
from arm_control.supervisor.console_adapter import ConsoleControlAdapter
from controller_extension import configure
from stream_races import main


def cases(arm, runtime, scene, service):
    presets = runtime._presets
    adapter = ConsoleControlAdapter(arm, presets)
    c = runtime.controller
    p = presets['track']
    arm.select_control_preset('track')

    def observe():
        mode = runtime._call(lambda: dict(runtime._mode_feedback))
        mass, com = runtime._call(lambda: c.executor._payload)
        return (mode['law'], tuple(np.round(mode['kp'], 9)), tuple(np.round(mode['kd'], 9)),
                round(float(mass), 9), None if com is None else tuple(np.round(com, 9)))

    def reset(start='track'):
        arm.select_control_preset(start)
        arm.set_payload(adapter_payload(0.))

    def adapter_payload(mass, com=None):
        from arm_control.execution.controller import PayloadValue
        return PayloadValue(mass, com) if com is not None else PayloadValue(mass)

    def via_adapter(fields, start='track'):
        reset(start)
        result = adapter.apply(unpack_control_update(pack_control_update(**fields)))
        return observe(), ('refused' if result.refusals else 'applied')

    def via_legacy(fields, start='track'):
        reset(start)

        def run():
            c.on_control(pack_control_update(**fields))
            runtime._pump_controller()
        try:
            runtime._call(run)
        except Exception as exc:  # noqa: BLE001 - recorded: the legacy node would raise in its handler
            return observe(), 'raised ' + type(exc).__name__
        return observe(), 'applied'

    kp, kd = p.kp.tolist(), p.kd.tolist()
    same = {  # what the console really sends
        'gains track': dict(gains=dict(kp=kp, kd=kd)),
        'gains float': dict(gains=dict(kp=presets['float'].kp.tolist(), kd=presets['float'].kd.tolist())),
        'payload 0': dict(payload={'mass_kg': 0.}),
        'payload 0.5 with com': dict(payload={'mass_kg': .5, 'com_ee': [.01, .02, .03]}),
        'payload 0.5 no com': dict(payload={'mass_kg': .5}),
    }
    different = {  # documented intended differences (console_adapter module docstring)
        'gains not a preset': (dict(gains=dict(kp=(p.kp * 1.5).tolist(), kd=kd)), 'legacy applies, adapter refuses'),
        'payload negative mass': (dict(payload={'mass_kg': -1.}), 'legacy clamps to 0, adapter refuses'),
        'payload None mass': (dict(payload={'mass_kg': None}), 'legacy raises, adapter refuses'),
    }
    rows = []
    for name, fields in same.items():
        start = 'float' if name == 'gains track' else 'track'  # a real transition, not the same-preset early return
        a, b = via_adapter(fields, start), via_legacy(fields, start)
        assert a == b, (name, 'adapter', a, 'legacy', b)
        rows.append((name, 'same'))
    for name, (fields, note) in different.items():
        a, b = via_adapter(fields), via_legacy(fields)
        assert a != b, (name, 'documented as different but identical', a, b)
        rows.append((name, note))
    # Direct facts behind the table: the adapter refuses without touching state, the legacy path applies or raises.
    a, _ = via_adapter(different['gains not a preset'][0])
    assert a[1] == tuple(np.round(p.kp, 9)), 'a refused gains request changed the state'
    legacy_gains = via_legacy(different['gains not a preset'][0])
    assert legacy_gains[1] == 'applied' and legacy_gains[0][1] == tuple(np.round(p.kp * 1.5, 9)), 'legacy did not apply'
    assert via_legacy(different['payload negative mass'][0])[0][3] == 0.
    assert via_legacy(different['payload None mass'][0])[1].startswith('raised')
    reset()
    for name, verdict in rows:
        print(f'  {name}: {verdict}')
    print('PASS: console gain-preset and valid payload messages give identical state through on_control and the adapter; '
          f'{len(different)} documented differences')


if __name__ == '__main__':
    main(str(Path(sys.argv[1]).resolve()), cases, configure)
