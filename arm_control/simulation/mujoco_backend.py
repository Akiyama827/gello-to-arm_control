"""Compatibility imports; implementation is in plants.mujoco.backend."""
from arm_control.plants.mujoco.backend import (
    MuJoCoUnavailableError as MuJoCoUnavailableError,
    contacts_possible as contacts_possible,
    SceneModelSpec as SceneModelSpec,
    ObjectSlot as ObjectSlot,
    MuJoCoSceneSpec as MuJoCoSceneSpec,
    compose_workcell_scene as compose_workcell_scene,
    compose_scene as compose_scene,
    MuJoCoBackend as MuJoCoBackend,
    _load_model_spec as _load_model_spec,
    _prefixed as _prefixed,
    _transfer_state as _transfer_state,
)


if __name__ == "__main__":
    import sys
    from arm_control.plants.mujoco import backend

    if "--demo" in sys.argv:
        backend._demo()
    elif "--scene-demo" in sys.argv:
        backend._scene_demo()
    elif "--self-check" in sys.argv:
        backend._check_state_transfer()
        backend._check_contact_policy()
        backend._check_scene_requirements()
        backend._check_mate_is_the_callers()
    else:
        print(backend.__doc__)
