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

from arm_control.sdk import Refused, SocketChannel, SupervisorError


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


def main():
    bad = {
        'undecodable bytes': lambda i: b'\xff\xfe not json',
        'not an object': lambda i: b'[1, 2]',
        'null': lambda i: b'null',
        'no ok': lambda i: json.dumps({'id': i}).encode(),
        'wrong id': lambda i: json.dumps({'id': i + 7, 'ok': True, 'result': 1}).encode(),
        'ok without result': lambda i: json.dumps({'id': i, 'ok': True}).encode(),
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
    print('PASS: unreadable supervisor replies are SupervisorError (outcome unknown), the channel is dropped; '
          'well-formed ok/refused/timeout replies map as before')


if __name__ == '__main__':
    main()
