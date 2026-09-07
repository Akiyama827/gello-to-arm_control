// The grasp-profile editor: pick a module/storage/context, drag the tool to a
// grasp pose, watch the IK and contact checks, save the draft.
//
// A separate PAGE from the arm console since 2026-09-07. Both used to be one
// file switched on whether `state.module` existed, so driving a robot rendered
// this editor's Calibration rack -- selects, Tare F/T, Capture pose, Edit
// grasp, Save draft -- with every one of its routes answering 404.
//
// VISUALIZATION ONLY. The node behind this page has trajectory/arm/gripper
// outputs (calibration_workcell.yml wires them), but the PAGE has never
// commanded an actor: GraspEditorPanel serves no /click and no /deadman. The
// deadman button is disabled and says so.
import {
  THREE, $, post, getJSON, scene, camera, orbit, resize, startRendering,
  loadGeometry, buildRobot, setPoses, gizmo, gizmoTarget, isDraggingGizmo,
  initGizmoModeButtons, addSlider, syncSliders, isSliderDragging,
  setConnected, wxyzPoseMatrix,
} from "./core.js";

initGizmoModeButtons();

let built = false;
let building = false;
let editorReferenceMatrix = null;
let editorRevision = "";
let editorFingerSlider = null;
let editorCurrentTool = null;
let editorPregraspTool = null;
let editorRetreatTool = null;
let editorFixedGroup = null;
let editorArm = null;
let editorArmBaseColors = [];
let editorCheckTimer = null;
let editorEditRevision = 0;
let editorChecks = null;
let editorContexts = [];
let lastGizmoSend = 0;

gizmo.addEventListener("objectChange", () => {
  const now = performance.now();
  if (now - lastGizmoSend < 80) return;
  lastGizmoSend = now;
  syncPoseFields();
  sendEditorGrasp().catch((error) => {
    $("grasp-status").textContent = error.message;
  });
});

const csvVector = (id) => {
  const values = $(id).value.split(",").map((value) => Number(value.trim()));
  if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) {
    throw new Error(`${id} needs three finite comma-separated values`);
  }
  return values;
};

const syncPoseFields = () => {
  if (!editorReferenceMatrix) return;
  gizmoTarget.updateMatrix();
  const local = editorReferenceMatrix.clone().invert().multiply(gizmoTarget.matrix);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  local.decompose(position, quaternion, new THREE.Vector3());
  const euler = new THREE.Euler().setFromQuaternion(quaternion, "ZYX");
  $("pose-x").value = (position.x * 1000).toFixed(3);
  $("pose-y").value = (position.y * 1000).toFixed(3);
  $("pose-z").value = (position.z * 1000).toFixed(3);
  $("pose-roll").value = THREE.MathUtils.radToDeg(euler.x).toFixed(3);
  $("pose-pitch").value = THREE.MathUtils.radToDeg(euler.y).toFixed(3);
  $("pose-yaw").value = THREE.MathUtils.radToDeg(euler.z).toFixed(3);
};

const applyPoseFields = async () => {
  const mm = ["pose-x", "pose-y", "pose-z"].map((id) => Number($(id).value));
  const degrees = ["pose-roll", "pose-pitch", "pose-yaw"].map(
    (id) => Number($(id).value),
  );
  if ([...mm, ...degrees].some((value) => !Number.isFinite(value))) {
    throw new Error("XYZ and RPY must be finite numbers");
  }
  const [roll, pitch, yaw] = degrees.map(THREE.MathUtils.degToRad);
  const local = new THREE.Matrix4().compose(
    new THREE.Vector3(...mm.map((value) => value / 1000)),
    new THREE.Quaternion().setFromEuler(
      new THREE.Euler(roll, pitch, yaw, "ZYX"),
    ),
    new THREE.Vector3(1, 1, 1),
  );
  editorReferenceMatrix.clone().multiply(local).decompose(
    gizmoTarget.position,
    gizmoTarget.quaternion,
    new THREE.Vector3(),
  );
  await sendEditorGrasp();
};

async function sendEditorGrasp() {
  if (!editorReferenceMatrix) return;
  gizmoTarget.updateMatrix();
  const local = editorReferenceMatrix.clone().invert().multiply(gizmoTarget.matrix);
  const position = new THREE.Vector3();
  const quaternion = new THREE.Quaternion();
  local.decompose(position, quaternion, new THREE.Vector3());
  const response = await post("/grasp", {
    pos: position.toArray(),
    quat: [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
    finger_width_m: editorFingerSlider ? Number(editorFingerSlider[0].value) : 0,
    approach_offset_m: csvVector("grasp-approach"),
    retreat_offset_m: csvVector("grasp-retreat"),
  });
  updateEditorState(response.editor);
  $("grasp-status").textContent = "Draft changed in memory — Save draft to persist";
  scheduleEditorChecks();
  return response.editor;
}

const buildEditorTool = async (items, opacity, tint = null) => {
  const group = new THREE.Group();
  const parts = [];
  for (const item of items) {
    const baseColor = new THREE.Color(...item.color);
    const mesh = new THREE.Mesh(
      await loadGeometry(item.mesh),
      new THREE.MeshPhongMaterial({
        color: tint || baseColor,
        transparent: opacity < 1,
        opacity,
        depthWrite: opacity >= 1,
      }),
    );
    mesh.scale.set(...item.scale);
    mesh.position.set(...item.p);
    mesh.quaternion.set(...item.q);
    group.add(mesh);
    parts.push({ mesh, link: item.link, baseColor });
  }
  return { group, parts };
};

const updateToolParts = (tool, items) => items.forEach((item, index) => {
  const part = tool.parts[index];
  if (!part) return;
  part.mesh.position.set(...item.p);
  part.mesh.quaternion.set(...item.q);
});

const setToolPose = (tool, pose) => {
  const matrix = wxyzPoseMatrix(pose);
  matrix.decompose(tool.group.position, tool.group.quaternion, new THREE.Vector3());
};

const samePose = (left, right) => left.every(
  (value, index) => Math.abs(value - right[index]) < 1e-9,
);

const setCheckBadge = (name, status, detail) => {
  const badge = $(`${name}-ik`);
  if (!badge) return;
  badge.className = "status";
  badge.classList.add(
    status === "valid" ? "live"
      : status === "checking" ? "neutral"
        : status === "collision" ? "warn" : "fault",
  );
  badge.textContent = `${name[0].toUpperCase()}${name.slice(1)}: ${status}`;
  badge.title = detail || "";
};

const showSelectedArm = () => {
  const check = editorChecks ? editorChecks[$("context-select").value] : null;
  const visible = Boolean(
    editorArm
      && $("show-arm").checked
      && check
      && check.arm_geoms
      && !["unreachable", "error"].includes(check.status),
  );
  setPoses(editorArm, visible ? check.arm_geoms : null);
  if (editorArm) {
    editorArm.parts.forEach((part, index) => {
      part.material.color.copy(
        check && check.status === "collision"
          ? new THREE.Color(0xaf3932)
          : editorArmBaseColors[index],
      );
    });
  }
  if (editorCurrentTool) editorCurrentTool.group.visible = !visible;
};

const scheduleEditorChecks = () => {
  clearTimeout(editorCheckTimer);
  editorChecks = null;
  for (const name of editorContexts) setCheckBadge(name, "checking", "");
  showSelectedArm();
  editorCheckTimer = setTimeout(runEditorChecks, 150);
};

const runEditorChecks = async () => {
  try {
    const result = await post("/grasp/check", {
      edit_revision: editorEditRevision,
    });
    if (result.stale || result.edit_revision !== editorEditRevision) return;
    editorChecks = result.checks;
    for (const name of editorContexts) {
      const check = editorChecks[name];
      setCheckBadge(name, check.status, check.detail);
    }
    showSelectedArm();
  } catch (error) {
    for (const name of editorContexts) setCheckBadge(name, "error", error.message);
  }
};

const updateEditorState = (editor) => {
  if (!editor) return;
  editorContexts = editor.choices.contexts;
  editorEditRevision = editor.edit_revision;
  setToolPose(editorPregraspTool, editor.pregrasp_pose);
  setToolPose(editorRetreatTool, editor.retreat_pose);
  editorPregraspTool.group.visible = !samePose(editor.pregrasp_pose, editor.ee_pose);
  editorRetreatTool.group.visible = !samePose(editor.retreat_pose, editor.ee_pose);
  const report = editor.contacts.grasp;
  const forbidden = new Set(report.forbidden_tool_links);
  const intended = new Set(report.intended_tool_links);
  for (const part of editorCurrentTool.parts) {
    const color = forbidden.has(part.link)
      ? new THREE.Color(0xaf3932)
      : intended.has(part.link) ? new THREE.Color(0xd99a24) : part.baseColor;
    part.mesh.material.color.copy(color);
  }
  const label = (name) => {
    const ok = editor.contacts[name].ok;
    return `<span class="${ok ? "ok" : "blocked"}">${name}: ${ok ? "clear" : "blocked"}</span>`;
  };
  $("contact-status").innerHTML = ["pregrasp", "grasp", "retreat"].map(label).join(" · ");
  $("plan-summary").textContent = report.ok
    ? "Editor collision preview clear"
    : `Forbidden contact: ${report.forbidden_tool_links.join(", ")}`;
  syncPoseFields();
};

const refreshEditorTool = async () => {
  const geometry = await (await fetch("/scene", { cache: "no-store" })).json();
  updateToolParts(editorCurrentTool, geometry.tool || []);
  updateToolParts(editorPregraspTool, geometry.tool || []);
  updateToolParts(editorRetreatTool, geometry.tool || []);
};

const setOptions = (select, values, selected) => {
  select.replaceChildren();
  select.dataset.current = selected;
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selected;
    select.appendChild(option);
  }
};

const syncSelectionControls = (editor) => {
  setOptions($("module-select"), editor.choices.modules, editor.selection.module);
  setOptions($("storage-select"), editor.choices.storages, editor.selection.storage);
  setOptions($("context-select"), editor.choices.contexts, editor.selection.context);
};

const rebuildEditorScene = async (state, frameCamera) => {
  const geometry = await (await fetch("/scene")).json();
  if (editorFixedGroup) scene.remove(editorFixedGroup);
  if (editorPregraspTool) scene.remove(editorPregraspTool.group);
  if (editorRetreatTool) scene.remove(editorRetreatTool.group);
  if (editorArm) scene.remove(editorArm.group);
  gizmoTarget.clear();
  editorFixedGroup = new THREE.Group();
  scene.add(editorFixedGroup);
  for (const item of geometry.fixed || [...(geometry.fixture || []), ...(geometry.module || [])]) {
    const mesh = new THREE.Mesh(
      await loadGeometry(item.mesh),
      new THREE.MeshPhongMaterial({
        color: new THREE.Color(...item.color),
        transparent: item.emphasized === false,
        opacity: item.emphasized === false ? 0.34 : 1,
      }),
    );
    mesh.scale.set(...item.scale);
    mesh.position.set(...item.p);
    mesh.quaternion.set(...item.q);
    editorFixedGroup.add(mesh);
  }
  for (const [name, pose] of Object.entries(geometry.frames || {})) {
    const axes = new THREE.AxesHelper(0.055);
    axes.position.set(...pose.p);
    axes.quaternion.set(...pose.q);
    axes.name = name;
    if (name === state.module.ee_frame) gizmoTarget.add(axes);
    else editorFixedGroup.add(axes);
  }
  editorCurrentTool = await buildEditorTool(geometry.tool || [], 1);
  editorPregraspTool = await buildEditorTool(geometry.tool || [], 0.28, new THREE.Color(0x55aa7a));
  editorRetreatTool = await buildEditorTool(geometry.tool || [], 0.24, new THREE.Color(0x6d9fc0));
  editorArm = await buildRobot(geometry.arm || [], null, 1);
  editorArmBaseColors = (geometry.arm || []).map((item) => new THREE.Color(...item.color));
  gizmoTarget.add(editorCurrentTool.group);
  scene.add(editorPregraspTool.group, editorRetreatTool.group);
  editorReferenceMatrix = wxyzPoseMatrix(state.module.reference_pose);
  const ee = wxyzPoseMatrix(state.module.ee_pose);
  ee.decompose(gizmoTarget.position, gizmoTarget.quaternion, new THREE.Vector3());
  updateEditorState(state.module);
  showSelectedArm();
  if (frameCamera) {
    const bounds = new THREE.Box3().setFromObject(editorFixedGroup).expandByObject(gizmoTarget);
    const center = bounds.getCenter(new THREE.Vector3());
    const size = Math.max(bounds.getSize(new THREE.Vector3()).length(), 0.08);
    orbit.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(1, -1, 0.75).normalize().multiplyScalar(size * 2.2));
    camera.near = Math.max(size / 100, 0.001);
    camera.far = Math.max(size * 30, 2);
    camera.updateProjectionMatrix();
    orbit.update();
  }
};

const changeSelection = async () => {
  try {
    const result = await post("/selection", {
      module: $("module-select").value,
      storage: $("storage-select").value,
      context: $("context-select").value,
    });
    const state = { module: result.editor };
    editorRevision = result.revision;
    syncSelectionControls(result.editor);
    $("grasp-approach").value = result.editor.approach_offset_m.join(", ");
    $("grasp-retreat").value = result.editor.retreat_offset_m.join(", ");
    await rebuildEditorScene(state, false);
    scheduleEditorChecks();
  } catch (error) {
    $("grasp-status").textContent = error.message;
    syncSelectionControls({
      choices: {
        modules: [...$("module-select").options].map((option) => option.value),
        storages: [...$("storage-select").options].map((option) => option.value),
        contexts: [...$("context-select").options].map((option) => option.value),
      },
      selection: {
        module: $("module-select").dataset.current,
        storage: $("storage-select").dataset.current,
        context: $("context-select").dataset.current,
      },
    });
  }
};

for (const id of ["module-select", "storage-select", "context-select"]) {
  $(id).addEventListener("change", changeSelection);
}
$("show-arm").addEventListener("change", showSelectedArm);
for (const id of ["pose-x", "pose-y", "pose-z", "pose-roll", "pose-pitch", "pose-yaw"]) {
  $(id).addEventListener("blur", () => {
    applyPoseFields().catch((error) => {
      $("grasp-status").textContent = error.message;
    });
  });
  $(id).addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    applyPoseFields().catch((error) => {
      $("grasp-status").textContent = error.message;
    });
  });
}


// -- first paint -------------------------------------------------------------
const build = async (state) => {
  building = true;
  await rebuildEditorScene(state, true);
  syncSelectionControls(state.module);
  $("actor-select").innerHTML = `<option>${state.module.type} grasp</option>`;
  const deadman = $("deadman");
  if (deadman) {
    deadman.disabled = true;
    deadman.querySelector("span").textContent = "Visualization only";
    deadman.querySelector("small").textContent = "No motion output is open";
  }
  $("buttons").textContent = "Profile editing cannot command an actor.";
  $("finger-opening").replaceChildren();
  editorFingerSlider = addSlider(
    $("finger-opening"),
    state.module.finger_opening,
    3,
    async (value) => {
      $("grip-readout").value = `${value.toFixed(3)} m`;
      try {
        await sendEditorGrasp();
        await refreshEditorTool();
      } catch (error) {
        $("grasp-status").textContent = error.message;
      }
    },
  );
  $("grip-readout").value = `${state.module.finger_opening.value.toFixed(3)} m`;
  $("save-grasp").disabled = false;
  $("grasp-fields").hidden = false;
  $("grasp-approach").value = state.module.approach_offset_m.join(", ");
  $("grasp-retreat").value = state.module.retreat_offset_m.join(", ");
  $("joint-summary").textContent = "Module fixed at q = 0";
  $("wrench-summary").textContent = "Not connected in visualization mode";
  editorRevision = state.module.revision;
  $("grasp-status").textContent = `${state.module.status} / ${state.module.reference_link}`;
  scheduleEditorChecks();
  $("save-grasp").onclick = async () => {
    if (!confirm("Save this draft grasp and create a dated backup?")) return;
    await sendEditorGrasp();
    const saved = await post("/grasp/save", {
      confirm: true,
      expected_revision: editorRevision,
    });
    editorRevision = saved.revision;
    $("grasp-status").textContent = "Draft saved";
  };
  built = true;
  building = false;
  resize();
};

const poll = async () => {
  try {
    const state = await getJSON("/state");
    setConnected(true, state.ident);
    if (state.ident) document.title = `${state.ident} — grasp editor`;
    if (!state.module) {
      // Only meaningful against a node that serves an editor state. Say so,
      // rather than rendering an empty rack.
      $("grasp-status").textContent =
        "This node serves no grasp profile — open the arm console instead.";
      return;
    }
    if (!built && !building) await build(state);
    $("log").textContent = state.log.join("\n");
    if (!isDraggingGizmo()) {
      editorReferenceMatrix = wxyzPoseMatrix(state.module.reference_pose);
      editorRevision = state.module.revision;
    }
    if (built && !isSliderDragging() && editorFingerSlider) {
      syncSliders([editorFingerSlider], [state.module.finger_opening], 3);
    }
  } catch (error) {
    setConnected(false);
    console.error(error);
  }
};
setInterval(poll, 150);
poll();

startRendering();
