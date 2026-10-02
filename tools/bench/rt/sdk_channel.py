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
    got, _ = outcome(lambda i: json.dumps({'id': i, 'ok': True, 'result': {'a': 1}}).encode())
    assert got == {'a': 1}, got
    concurrent_close()
    codec_layer()
    print('PASS: unreadable supervisor replies are SupervisorError (outcome unknown), the channel is dropped; '
          'well-formed ok/refused/timeout replies map as before')


if __name__ == '__main__':
    main()
