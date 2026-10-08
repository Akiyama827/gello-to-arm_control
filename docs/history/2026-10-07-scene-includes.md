# Scene files can include a shared scene (config single source, step 4a, library)

Operator 2026-10-07: the deployment's two FR3 bench scenes repeated 42 keys of the same bench, and they had drifted. One of them lacked
the other arm's riser. They are to share one bench file. `arm_control.scene.load_scene` read scene files with plain YAML, so a scene
could not include anything.

`scene.scene_tree(source)` resolves a scene through the config-tree loader (`include:`, deep merge, own keys win), and `load_scene` uses
it. Asset paths in a scene are relative to the scene file, so an included scene file must be in the includer's own directory. An
include from elsewhere is refused at load, rather than resolving its paths against the wrong directory. `scene_files` walks the whole
include chain and checks this at every level. A name that is missing from the scene's directory is refused too: the config loader would
otherwise fall back to the deployment root.

**`scene_sha256`** (review round 1): a model token and a provenance record hashed only the scene file's own bytes, so an edit to an
included shared bench left them unchanged. `scene_sha256` hashes every file of the scene. A scene without includes hashes to
sha256 of its bytes, exactly as before, so its existing tokens are unchanged.

Checked: `python -m arm_control.scene` (a scene including a shared one resolves whole, with its own override; an include from another
directory is refused), offline suite.
