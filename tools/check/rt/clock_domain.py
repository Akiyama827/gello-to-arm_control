"""HELLO clock domain (CTRL-041): what the client learns of the RT host's clock identity.

Spawns fake ``arm_rt_server`` processes with different machine-id / boot-id files and checks the
client-side ``clock_domain`` against an independent Python HMAC computation: the same host under
two boot ids shares the host field and differs in the boot field, another host differs in the
host field, an unreadable identity gives None (unattested), and the raw machine-id never appears
in the value. Ephemeral ports and a readiness poll (no fixed sleep). Fake backend only.

    PYTHONPATH=. python tools/check/rt/clock_domain.py path/to/arm_rt_server
"""
from __future__ import annotations

# ruff: noqa: E402  (path bootstrap must precede the arm_control import)

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import hashlib
import hmac
import socket
import subprocess
import tempfile
import time

from arm_control.plants.remote_rt.client import RtBackend, RtConfig, RtLinkError, parse_clock_domain

JOINTS = [f"j{i}" for i in range(1, 4)]
MACHINE_1, MACHINE_2 = "0123456789abcdef0123456789abcdef", "fedcba9876543210fedcba9876543210"
PLACEHOLDER = "uninitialized"
BOOT_1, BOOT_2 = "11111111-2222-3333-4444-555555555555", "66666666-7777-8888-9999-000000000000"


def expected(machine: str, boot: str) -> tuple[str, str, str]:
    def h(msg: str) -> str:
        return hmac.new(machine.encode(), msg.encode(), hashlib.sha256).hexdigest()[:12]
    return ("mono", h("arm_rt_server:clock-domain:host:v1"),
            h("arm_rt_server:clock-domain:boot:v1:" + boot))


def parse_checks():
    good = "aabbccddeeff:112233445566"
    assert parse_clock_domain(good) == ("mono", "aabbccddeeff", "112233445566")
    for bad in ("", "aabbccddeeff", "aabbccddeeff:1122", good[:-1], good + "0", "AABBCCDDEEFF:112233445566",
                "aabbccddeeff:11223344556g", "mono:aabbccddeeff:112233445566"):
        assert parse_clock_domain(bad) is None, bad


def free_port(kind: int) -> int:
    with socket.socket(socket.AF_INET, kind) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def domain_of(binary: str, tmp: Path, machine: str | None, boot: str | None):
    machine_file, boot_file = tmp / "machine-id", tmp / "boot-id"
    for path, value in ((machine_file, machine), (boot_file, boot)):
        path.unlink(missing_ok=True)
        if value is not None:
            path.write_text(value + "\n")
    udp, tcp = free_port(socket.SOCK_DGRAM), free_port(socket.SOCK_STREAM)
    server = subprocess.Popen(
        [binary, "--backend", "fake", "--n", "3", "--instance-id", "cd", "--joint-names",
         ",".join(JOINTS), "--udp-port", str(udp), "--tcp-port", str(tcp), "--bind", "127.0.0.1",
         "--machine-id-file", str(machine_file), "--boot-id-file", str(boot_file)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        cfg = RtConfig(host="127.0.0.1", udp_port=udp, tcp_port=tcp, instance_id="cd")
        deadline = time.monotonic() + 8
        while True:
            link = RtBackend(cfg, JOINTS)
            try:
                link.open()
                break
            except (RtLinkError, OSError):
                assert server.poll() is None and time.monotonic() < deadline, "server not ready"
                time.sleep(0.05)
        try:
            return link.clock_domain
        finally:
            link.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else str(_ROOT / "rt/build/arm_rt_server")
    parse_checks()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        a = domain_of(binary, tmp, MACHINE_1, BOOT_1)
        assert a == expected(MACHINE_1, BOOT_1), (a, expected(MACHINE_1, BOOT_1))
        again = domain_of(binary, tmp, MACHINE_1, BOOT_1)
        assert again == a, "same host and boot must give the same domain"
        rebooted = domain_of(binary, tmp, MACHINE_1, BOOT_2)
        assert rebooted == expected(MACHINE_1, BOOT_2) and rebooted[1] == a[1] and rebooted[2] != a[2]
        other = domain_of(binary, tmp, MACHINE_2, BOOT_1)
        assert other == expected(MACHINE_2, BOOT_1) and other[1] != a[1]
        assert MACHINE_1[:12] not in "".join(a)
        assert domain_of(binary, tmp, None, BOOT_1) is None, "unreadable machine-id: unattested"
        assert domain_of(binary, tmp, MACHINE_1, None) is None, "unreadable boot-id: unattested"
        assert domain_of(binary, tmp, PLACEHOLDER, BOOT_1) is None, "not a machine-id: unattested"
    print("PASS: HELLO clock domain is a keyed host+boot identity (matches an independent HMAC), "
          "changes with the boot id or host, and is absent when unreadable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
