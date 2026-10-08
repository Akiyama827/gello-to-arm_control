"""CLI target selection shared by fake-RT timing benches."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteTarget:
    host: str
    udp_port: int
    tcp_port: int
    instance_id: str


def parse_target(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('binary', nargs='?', help='local fake-server executable')
    parser.add_argument('--remote', metavar='HOST', help='connect to an existing fake RT server')
    parser.add_argument('--udp-port', type=int)
    parser.add_argument('--tcp-port', type=int)
    parser.add_argument('--instance-id')
    args = parser.parse_args()
    if args.remote:
        if not args.udp_port or not args.tcp_port or not args.instance_id:
            parser.error('--remote requires --udp-port, --tcp-port and --instance-id')
        if args.binary:
            parser.error('a local binary cannot be combined with --remote')
        return None, RemoteTarget(args.remote, args.udp_port, args.tcp_port,
                                  args.instance_id)
    if not args.binary:
        parser.error('provide a local binary or --remote HOST with ports and instance ID')
    if any(v is not None for v in (args.udp_port, args.tcp_port, args.instance_id)):
        parser.error('ports and instance ID require --remote')
    return args.binary, None
