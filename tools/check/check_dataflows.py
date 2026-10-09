#!/usr/bin/env python3
"""Every graph is wired: each node's path exists, and each input names an output its source node declares (or a Dora timer).
Static, no Dora: a graph rewrite (CTRL-053's per-graph migrations) is checked before anything is launched. ``problems`` is also
what a project's own graph check runs.

    python tools/check/check_dataflows.py
"""
from __future__ import annotations

from pathlib import Path
import re

import yaml

ROOT = Path(__file__).resolve().parents[2]
TIMER = re.compile(r"dora/timer/(millis|secs)/\d+")


def problems(graph: Path) -> list[str]:
    nodes = yaml.safe_load(graph.read_text())["nodes"]
    outputs = {node["id"]: set(node.get("outputs") or ()) for node in nodes}
    found = []
    for node in nodes:
        path = (graph.parent / node["path"]).resolve()
        if not path.is_file():
            found.append(f"{node['id']}: no such path {node['path']}")
        for name, source in (node.get("inputs") or {}).items():
            source = source["source"] if isinstance(source, dict) else source
            if TIMER.fullmatch(source):
                continue
            producer, _, topic = source.partition("/")
            if producer not in outputs:
                found.append(f"{node['id']}.{name}: no node {producer!r}")
            elif topic not in outputs[producer]:
                found.append(f"{node['id']}.{name}: {producer} declares no output {topic!r}")
    return found


def main() -> int:
    graphs = sorted((ROOT / "dataflows").glob("*.yml"))
    assert graphs, "no graphs"
    bad = {g.name: p for g in graphs if (p := problems(g))}
    assert not bad, bad
    # The check itself: a dangling input and a missing path are reported.
    sample = yaml.safe_load((ROOT / "dataflows/sim_franka_motion.yml").read_text())
    sample["nodes"][0]["inputs"]["ghost"] = "nobody/nothing"
    sample["nodes"][1]["path"] = "../nodes/missing_node.py"
    probe = ROOT / ".cache" / "check_dataflows_probe.yml"
    probe.parent.mkdir(exist_ok=True)
    probe.write_text(yaml.safe_dump(sample))
    try:
        found = problems(probe)
    finally:
        probe.unlink()
    assert any("no node 'nobody'" in f for f in found) and any("no such path" in f for f in found), found
    print(f"dataflows: {len(graphs)} library graphs wired (paths exist, every input names a declared output): PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
