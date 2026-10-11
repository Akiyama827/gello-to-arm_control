"""SocketChannel turns every unreadable supervisor reply into SupervisorError (outcome unknown), never ValueError.

A caller that took a request to the supervisor and then saw a ValueError could read it as "nothing was sent / refused" (the
clients treat ValueError as a refusal). Against a scripted one-shot server: undecodable bytes, JSON that is not an object,
an object without "ok", a mismatched id, an ok reply without a result, and an error reply without text all raise
SupervisorError (never ValueError/KeyError), the channel is closed afterwards, and well-formed ok/refused/timeout replies
still map as before. No supervisor, RT server or robot: the binary argument of the suite is ignored.
"""
from pathlib import Path
import json
import socket
import tempfile
import threading

from arm_control.sdk import ArmClient, Refused, SocketChannel, SupervisorError, _dec
from arm_control.execution.stream import RevisionStatus, SelectionState


def serve(path, reply):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)

    def run():
        conn, _ = server.accept()
        with conn:
            conn.makefile('rb').readline()
            conn.sendall(reply(1) + b'\n')
        server.close()
    threading.Thread(target=run, daemon=True).start()


def outcome(reply):
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 's.sock')
        serve(path, reply)
        channel = SocketChannel(path, timeout_s=2.)
        try:
            return channel.call('status'), channel
        except Exception as exc:  # noqa: BLE001 - the class is the point
            return exc, channel


def concurrent_close():
    """A malformed reply on one thread must not break another thread's call that has already sent its request:
    the other thread gets its own reply or a SupervisorError, never ValueError('readline of closed file')."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'c.sock')
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(path)
        server.listen(1)

        def run():
            conn, _ = server.accept()
            with conn:
                rx = conn.makefile('rb')
                rx.readline()
                conn.sendall(json.dumps({'id': 1, 'ok': True}).encode() + b'\n')  # no result: untrusted
                rx.readline()  # B's request arrives; it is never answered while the connection stays up
                threading.Event().wait(1.5)
        threading.Thread(target=run, daemon=True).start()
        channel = SocketChannel(path, timeout_s=2.)
        results = {}
        real_close, real_sock, sends = channel._close, channel._sock, []

        def slow_close():  # A is slow to drop the channel: outside the lock, B can send meanwhile
            threading.Event().wait(.3)
            real_close()
        channel._close = slow_close

        class SlowSend:  # B stalls between sendall and readline, so a close by A lands inside its call
            def __getattr__(self, name):
                return getattr(real_sock, name)

            def sendall(self, data):
                real_sock.sendall(data)
                sends.append(1)
                if len(sends) == 2:
                    threading.Event().wait(.6)
        channel._sock = SlowSend()

        def call(name, delay):
            try:
                threading.Event().wait(delay)
                results[name] = channel.call('status')
            except Exception as exc:  # noqa: BLE001
                results[name] = exc
        a, b = (threading.Thread(target=call, args=('A', 0.)), threading.Thread(target=call, args=('B', .05)))
        a.start()
        b.start()
        a.join(5)
        b.join(5)
        for name in 'AB':
            assert type(results[name]) is SupervisorError, (name, repr(results[name]))


def codec_layer():
    """A reply that is JSON but does not fit the contract is SupervisorError at the ArmClient layer too."""
    for name, decode, value in (('unknown enum', lambda v: SelectionState(v['s']), {'s': 'bogus'}),
                                ('missing field', lambda v: SelectionState(v['s']), {}),
                                ('extra field', lambda v: RevisionStatus(**v), {'unexpected': 1})):
        try:
            _dec(decode, value)
        except SupervisorError as exc:
            assert 'outcome unknown' in str(exc), (name, exc)
        else:
            raise AssertionError(f'{name}: decode error not wrapped')

    class Scripted:
        def call(self, op, wait_s=0.0, **args):
            return {'contract_version': 99} if op == 'hello' else {}

        def close(self):
            self.closed = True
    channel = Scripted()
    try:
        ArmClient('x', channel=channel)
    except SupervisorError:
        assert channel.closed  # a failed handshake never leaks the channel
    else:
        raise AssertionError('mismatched contract accepted')

    class NoVersion(Scripted):
        def call(self, op, wait_s=0.0, **args):
            return {}
    channel = NoVersion()
    try:
        ArmClient('x', channel=channel)
    except SupervisorError as exc:
        assert 'unreadable hello' in str(exc) and channel.closed, exc
    else:
        raise AssertionError('hello without a contract version accepted')


def wrapped_methods():
    """The real ArmClient methods (not just _dec): a well-formed envelope whose payload does not fit the contract is
    SupervisorError from every method that decodes one, including the ones that index or iterate the raw result."""
    from arm_control.execution import codec
    from arm_control.execution.contract import CONTRACT_VERSION, Capabilities, RobotIdentity

    class Garbage:
        def __init__(self, result):
            self.result = result

        def call(self, op, wait_s=0.0, **args):
            if op in ('arm', 'disarm') and self.result == 'AWAIT':
                return {'gen': 1}
            if op == 'await_authority':
                return self.after
            if op == 'hello':
                return dict(contract_version=CONTRACT_VERSION, identity=codec.encode(RobotIdentity('x', ('j',), 'fake')),
                            capabilities=codec.encode(Capabilities(CONTRACT_VERSION, ('joint',))), session='s',
                            extensions={'controller': 1, 'group': 1, 'state_history': 1})
            return self.result

        def close(self):
            pass
    from arm_control.execution.contract import Authority, Lifecycle
    armed = codec.encode(Authority(Lifecycle.CLAIMED))
    for after in ({'settled': 1, 'authority': armed}, 5, {}, {'settled': 'yes'}, {'settled': 0}, {'settled': 1, 'authority': None}, {'settled': True},
                  {'settled': True, 'authority': 3}, None):
        for method in ('arm', 'disarm'):
            client = ArmClient('x', channel=Garbage('AWAIT'))
            client._channel.after = after
            try:
                getattr(client, method)(timeout_s=.5)
            except SupervisorError:
                pass
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f'{method} await with {after!r}: {type(exc).__name__}: {exc}') from exc
            else:
                raise AssertionError(f'{method} accepted await reply {after!r}')
    calls = {'arm': lambda c: c.arm(timeout_s=.5), 'disarm': lambda c: c.disarm(timeout_s=.5),
             'receipts': lambda c: c.receipts('e'), 'presets': lambda c: c.controller_presets(),
             'state': lambda c: c.state(), 'status': lambda c: c.status('e')}
    for result in (5, 'x', [1, 2], {}, {'gen': 'one', 'settled': 'yes', 'authority': 3}, [{'bogus': 1}], None):
        for name, call in calls.items():
            client = ArmClient('x', channel=Garbage(result))
            try:
                call(client)
            except SupervisorError:
                pass
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f'{name} with {result!r}: {type(exc).__name__}: {exc}') from exc
            else:
                raise AssertionError(f'{name} accepted {result!r}')


def extension_replies():
    """group/1 and state_history/1 replies are shape-checked too: a wrong-shaped payload is SupervisorError, a good one passes."""
    from arm_control.execution.contract import CONTRACT_VERSION  # noqa: F401 (same fake as wrapped_methods)
    good_sample = dict(t_mono_ns=1, state_seq=1, age_ns=0, session_id=1, arm_epoch=1, stop_epoch=0, armed=True, faulted=False,
                       q=[0.], dq=[0.])
    good_history = dict(clock_domain='h:b', now_ns=1, authority=dict(session_id='1', arm_epoch=1), samples=[good_sample])
    cases = {
        'group_join': (lambda c: c.group_join(1, 0, 2), (5, 'x', [1], None), {'clock_domain': None}),
        'group_report': (lambda c: c.group_report('e'), (5, 'x', [1], None), {}),
        'group_cancel': (lambda c: c.group_cancel(1), (5, 'x', [1], None), {'cancelled': True}),
        'group_events': (lambda c: c.group_events(), (5, 'x', {}, [1], None), [{'kind': 'peer'}]),
        'trace_samples': (lambda c: c.trace_samples(0),
                          (5, 'x', [], {}, dict(samples=[], latest_seq=0, dropped=0, errors=0),
                           dict(samples=[], latest_seq='0', dropped=0, errors=0, capacity=64),
                           dict(samples=[1], latest_seq=0, dropped=0, errors=0, capacity=64),
                           dict(samples=[dict(seq='1', pc_monotonic=0., plan_id='', q_rt=None)], latest_seq=1, dropped=0,
                                errors=0, capacity=64),
                           dict(samples=[dict(seq=1, pc_monotonic=0.)], latest_seq=1, dropped=0, errors=0, capacity=64)),
                          dict(samples=[dict(seq=1, pc_monotonic=0., plan_id='', q_rt=None)], latest_seq=1, dropped=0,
                               errors=0, capacity=64)),
        'state_history': (lambda c: c.state_history(),
                          (5, 'x', [], {}, dict(good_history, clock_domain=3), dict(good_history, samples={}),
                           dict(good_history, samples=[dict(good_sample, q=5)]),
                           dict(good_history, samples=[{k: v for k, v in good_sample.items() if k != 'age_ns'}]),
                           dict(good_history, authority=[])), good_history),
    }

    class Fixed:
        def __init__(self, result):
            self.result = result

        def call(self, op, wait_s=0.0, **args):
            if op == 'hello':
                from arm_control.execution import codec
                from arm_control.execution.contract import Capabilities, RobotIdentity
                return dict(contract_version=CONTRACT_VERSION, identity=codec.encode(RobotIdentity('x', ('j',), 'fake')),
                            capabilities=codec.encode(Capabilities(CONTRACT_VERSION, ('joint',))), session='s',
                            extensions={'group': 1, 'state_history': 1, 'trace': 1})
            return self.result

        def close(self):
            pass
    for name, (call, bad, good) in cases.items():
        for result in bad:
            try:
                call(ArmClient('x', channel=Fixed(result)))
            except SupervisorError:
                pass
            except Exception as exc:  # noqa: BLE001
                raise AssertionError(f'{name} with {result!r}: {type(exc).__name__}: {exc}') from exc
            else:
                raise AssertionError(f'{name} accepted {result!r}')
        assert call(ArmClient('x', channel=Fixed(good))) == good, name


def main():
    bad = {
        'undecodable bytes': lambda i: b'\xff\xfe not json',
        'not an object': lambda i: b'[1, 2]',
        'null': lambda i: b'null',
        'no ok': lambda i: json.dumps({'id': i}).encode(),
        'wrong id': lambda i: json.dumps({'id': i + 7, 'ok': True, 'result': 1}).encode(),
        'ok without result': lambda i: json.dumps({'id': i, 'ok': True}).encode(),
        'boolean id': lambda i: json.dumps({'id': True, 'ok': True, 'result': 1}).encode(),
        'ok as string': lambda i: json.dumps({'id': i, 'ok': 'false'}).encode(),
        'too deeply nested': lambda i: b'[' * 200000,
    }
    for name, reply in bad.items():
        got, channel = outcome(reply)
        assert type(got) is SupervisorError and 'outcome unknown' in str(got), (name, repr(got))
        try:
            channel.call('status')
        except SupervisorError as exc:
            assert 'closed' in str(exc), (name, exc)  # the channel was dropped
        else:
            raise AssertionError(f'{name}: channel stayed usable after an untrusted reply')
    got, _ = outcome(lambda i: json.dumps({'id': i, 'ok': False}).encode())
    assert type(got) is SupervisorError and 'unspecified' in str(got), repr(got)
    got, _ = outcome(lambda i: json.dumps({'id': i, 'ok': False, 'kind': 'refused', 'error': 'no'}).encode())
    assert type(got) is Refused and str(got) == 'no', repr(got)
    got, _ = outcome(lambda i: json.dumps({'id': i, 'ok': False, 'kind': 'timeout', 'error': 't'}).encode())
    assert type(got) is TimeoutError, repr(got)
    for kind in (['refused'], {'k': 1}, 5, None):  # `kind` is any JSON value: never hashed, never a TypeError
        got, _ = outcome(lambda i, kind=kind: json.dumps({'id': i, 'ok': False, 'kind': kind, 'error': ['not text']}).encode())
        assert type(got) is SupervisorError and str(got) == repr(['not text']), (kind, repr(got))
    got, _ = outcome(lambda i: json.dumps({'id': i, 'ok': True, 'result': {'a': 1}}).encode())
    assert got == {'a': 1}, got
    concurrent_close()
    codec_layer()
    wrapped_methods()
    extension_replies()
    print('PASS: unreadable supervisor replies are SupervisorError (outcome unknown), the channel is dropped; '
          'well-formed ok/refused/timeout replies map as before')


if __name__ == '__main__':
    main()
