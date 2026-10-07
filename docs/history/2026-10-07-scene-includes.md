# Scene files can include a shared scene (config single source, step 4a, library)

Operator 2026-10-07: the deployment's two FR3 bench scenes repeated 42 keys of the same bench, and they had drifted. One of them lacked
the other arm's riser. They are to share one bench file. `arm_control.scene.load_scene` read scene files with plain YAML, so a scene
could not include anything.

`scene.scene_tree(source)` resolves a scene through the config-tree loader (`include:`, deep merge, own keys win), and `load_scene` uses
it. Asset paths in a scene are relative to the scene file, so an included scene file must be in the includer's own directory. An
include from elsewhere is refused at load, rather than resolving its paths against the wrong directory. Only the top level is
checked; a shared scene file that itself includes another is not supported.

Checked: `python -m arm_control.scene` (a scene including a shared one resolves whole, with its own override; an include from another
directory is refused), offline suite.
