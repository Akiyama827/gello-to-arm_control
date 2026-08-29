"""Assembly topology — what is docked to what, decoupled from where it is.

The BrickSim lesson (arXiv:2603.16853, Brick Topology Graph): keep connectivity
in a graph whose edges hold the *discrete* mate parameters, and treat the
physics engine's body poses as a DERIVED view that gets re-synced from the
graph. Absolute poses drift; a committed edge does not. Everything that needs
to know the current robot — the plant's kinematic chain, the planner's
obstacle set, the dock target for the next module — reads this, not the twin.

Two deliberate departures from BrickSim:

- **A chain, not a graph.** Every module in this system is one passive port in,
  one active port out, so an assembly is an ordered list root->tip. A general
  graph would be an interface with one implementation. If a branching module
  ever exists, this grows a parent field and stops being a list.
- **No breakage detector.** BrickSim needs a QP to decide when a snap-fit
  fails because their joints are passive friction. Ours is a SERVO latch
  (`Dock_Control/`, ACTIVE cmd 0xEn): it releases when commanded and not
  otherwise, so detachment is an FSM verb, not an emergent force event.

Clocking is our quantized yaw: the keyed dock admits exactly four 90-degree
mates. ``clocking`` counts quarter turns about the mate axis, CW seen from
outside the port — the same convention and sign as the bench-verified passive
one-hot sensor map (2026-08-03 E1b: 0deg 0x08, 90deg 0x04, 180deg 0x02,
270deg 0x01, expected bit = (3 - k) mod 4), so a sim clocking and a sensor
readout are directly comparable integers.

Engine-neutral by construction: no mujoco, no numpy, no Dora. The mate SE(3)
is not stored because it is fully determined by the port geometry (a static
model fact) and ``clocking`` (here) — storing it too would be a second source
of truth that can disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CLOCKINGS", "DockedModule", "AssemblyChain", "clocking_from_onehot"]

# The keyed dock's four mates. Not a config knob: it is the number of keys on
# the printed part.
CLOCKINGS = 4


def clocking_from_onehot(bits: int) -> int:
    """Quarter turns from the dock MCU's 4-bit passive sensor word.

    Bench map (E1b): {0deg: 0x08, 90deg: 0x04, 180deg: 0x02, 270deg: 0x01},
    i.e. bit (3 - k). Raises on empty (0x00) or mid-seat (multi-bit) words —
    those are seat states, not clockings, and the caller must not read a
    partial insertion as a mate.
    """
    if bits == 0:
        raise ValueError("one-hot word 0x00: port empty, no clocking to read")
    if bits & (bits - 1):
        raise ValueError(
            f"one-hot word {bits:#04x} has multiple bits set: mid-seat, "
            "not a settled clocking"
        )
    return 3 - bits.bit_length() + 1


@dataclass(frozen=True)
class DockedModule:
    """One committed edge, named from the child's side.

    ``slot`` is the identity: inventory slots are unique and a docked one can
    never be re-docked, so it needs no separate instance counter.
    ``active_port`` is the port this module leaves OPEN — recorded rather than
    reconstructed, because only the caller knows how the model names it (the
    composed prefix is config, not something this module should guess).
    """

    slot: str              # inventory slot this module was picked from
    module_id: str         # module TYPE ("row_module") — indexes module_grasps
    clocking: int          # quarter turns about the mate axis, 0..3
    active_port: str       # site name of the port left open at the tip

    def __post_init__(self) -> None:
        if not 0 <= self.clocking < CLOCKINGS:
            raise ValueError(
                f"{self.slot}: clocking {self.clocking} outside 0..{CLOCKINGS - 1}"
            )


@dataclass
class AssemblyChain:
    """Ordered root->tip list of docked modules, plus a revision counter.

    ``root_port`` is the site name of the base's own dock port; every later
    module mates onto the previous module's active port. ``revision`` is what
    downstream caches key on (the compiled model, the planner's obstacle set,
    the generated URDF) — it changes on every commit and never rolls back.
    """

    root_port: str
    modules: list[DockedModule] = field(default_factory=list)
    revision: int = 0

    @property
    def tip_port(self) -> str:
        """Site name of the one open port the next module mates onto."""
        if not self.modules:
            return self.root_port
        return self.modules[-1].active_port

    @property
    def slots(self) -> list[str]:
        """Inventory slots already spent, in dock order."""
        return [m.slot for m in self.modules]

    def attach(
        self, *, slot: str, module_id: str, clocking: int, active_port: str
    ) -> DockedModule:
        module = DockedModule(
            slot=slot,
            module_id=module_id,
            clocking=int(clocking),
            active_port=active_port,
        )
        if slot in {m.slot for m in self.modules}:
            raise ValueError(f"slot {slot!r} is already docked at {self.tip_port}")
        self.modules.append(module)
        self.revision += 1
        return module

    def detach(self) -> DockedModule:
        """Release the TIP module (the only one whose latch is reachable)."""
        if not self.modules:
            raise ValueError("nothing docked: no latch to release")
        module = self.modules.pop()
        self.revision += 1
        return module

    def state(self) -> dict[str, Any]:
        """Serializable snapshot — the `topology_state` Dora message body."""
        return {
            "revision": self.revision,
            "root_port": self.root_port,
            "tip_port": self.tip_port,
            "modules": [
                {
                    "slot": m.slot,
                    "module_id": m.module_id,
                    "clocking": m.clocking,
                    "active_port": m.active_port,
                }
                for m in self.modules
            ],
        }


def _demo() -> None:
    chain = AssemblyChain(root_port="base_dock_port")
    assert chain.tip_port == "base_dock_port"
    assert chain.revision == 0

    chain.attach(slot="s0", module_id="row_module", clocking=0,
                 active_port="s0_active_connector")
    assert chain.tip_port == "s0_active_connector"
    assert chain.revision == 1

    b = chain.attach(slot="s1", module_id="row_module", clocking=1,
                     active_port="s1_active_connector")
    assert chain.tip_port == "s1_active_connector"

    # A slot holds one physical module: docking it twice is a config error.
    try:
        chain.attach(slot="s0", module_id="row_module", clocking=0,
                     active_port="s0_active_connector")
    except ValueError as exc:
        assert "already docked" in str(exc)
    else:
        raise AssertionError("re-docking a spent slot must raise")

    # Detaching walks the open port back to the module before it.
    assert chain.detach() == b
    assert chain.tip_port == "s0_active_connector"
    assert chain.detach().slot == "s0"
    assert chain.tip_port == "base_dock_port"
    # revision counts EVENTS, so it never rolls back to an earlier value.
    assert chain.revision == 4

    chain.attach(slot="s0", module_id="row_module", clocking=0,
                 active_port="s0_active_connector")
    assert chain.slots == ["s0"]
    assert chain.state()["modules"] == [
        {"slot": "s0", "module_id": "row_module", "clocking": 0,
         "active_port": "s0_active_connector"}
    ]

    for k, bits in enumerate((0x08, 0x04, 0x02, 0x01)):
        assert clocking_from_onehot(bits) == k, (k, bits)
    for bad in (0x00, 0x0C):
        try:
            clocking_from_onehot(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad:#04x} must not read as a clocking")

    try:
        DockedModule(slot="s", module_id="m", clocking=4, active_port="p")
    except ValueError:
        pass
    else:
        raise AssertionError("clocking 4 must raise")

    print("topology: OK")


if __name__ == "__main__":
    _demo()
