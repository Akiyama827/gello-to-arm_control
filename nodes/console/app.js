import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { TransformControls } from "three/addons/TransformControls.js";
import { STLLoader } from "three/addons/STLLoader.js";

const $ = (id) => document.getElementById(id);
const post = async (route, payload) => {
  const response = await fetch(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `${route}: ${response.status}`);
  return body;
};

const view = $("view");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202a2e);
const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.9, -0.9, 0.62);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.appendChild(renderer.domElement);
const resize = () => {
  const width = view.clientWidth;
  const height = view.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
};
addEventListener("resize", resize);

scene.add(new THREE.HemisphereLight(0xf5fbfa, 0x33464d, 1.25));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.55);
keyLight.position.set(1, -1, 2);
scene.add(keyLight);
const grid = new THREE.GridHelper(2, 20, 0x718083, 0x344145);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.14));
const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0, 0.16);

const loader = new STLLoader();
const geometryCache = new Map();
const loadGeometry = (url) => {
  if (!geometryCache.has(url)) {
    geometryCache.set(url, new Promise((resolve) => loader.load(
      url,
      resolve,
      undefined,
      (error) => {
        console.error("Mesh load failed", url, error);
        resolve(new THREE.BufferGeometry());
      },
    )));
  }
  return geometryCache.get(url);
};

const buildRobot = async (geometries, tint, opacity) => {
  const group = new THREE.Group();
  const parts = [];
  for (const item of geometries) {
    const geometry = await loadGeometry(item.mesh);
    const color = tint || new THREE.Color(...item.color);
    const material = new THREE.MeshPhongMaterial({
      color,
      transparent: opacity < 1,
      opacity,
      depthWrite: opacity >= 1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.set(...item.scale);
    group.add(mesh);
    parts.push(mesh);
  }
  group.visible = false;
  scene.add(group);
  return { group, parts };
};

const buildStatic = async (items) => {
  for (const item of items) {
    const mesh = new THREE.Mesh(
      await loadGeometry(item.mesh),
      new THREE.MeshPhongMaterial({ color: new THREE.Color(...item.color) }),
    );
    mesh.scale.set(...item.scale);
    mesh.position.set(...item.p);
    mesh.quaternion.set(...item.q);
    scene.add(mesh);
  }
};

const setPoses = (robot, poses) => {
  if (!robot) return;
  if (!poses) {
    robot.group.visible = false;
    return;
  }
  robot.group.visible = true;
  poses.forEach((pose, index) => {
    if (!robot.parts[index]) return;
    robot.parts[index].position.set(...pose.p);
    robot.parts[index].quaternion.set(...pose.q);
  });
};

const gizmoTarget = new THREE.Object3D();
scene.add(gizmoTarget);
const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.attach(gizmoTarget);
gizmo.setSpace("world");
gizmo.setSize(0.62);
scene.add(gizmo);
let draggingGizmo = false;
let lastGizmoSend = 0;
let rotationDelta = 0;
let editorMode = false;
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
const previousQuaternion = new THREE.Quaternion();
gizmo.addEventListener("dragging-changed", (event) => {
  draggingGizmo = event.value;
  orbit.enabled = !event.value;
  if (event.value) {
    previousQuaternion.copy(gizmoTarget.quaternion);
    rotationDelta = 0;
  }
});
gizmo.addEventListener("objectChange", () => {
  if (editorMode) {
    const now = performance.now();
    if (now - lastGizmoSend < 80) return;
    lastGizmoSend = now;
    syncPoseFields();
    sendEditorGrasp().catch((error) => {
      $("grasp-status").textContent = error.message;
    });
    return;
  }
  const axis = gizmo.axis;
  if (!["X", "Y", "Z"].includes(axis)) return;
  const index = { X: 0, Y: 1, Z: 2 }[axis];
  const now = performance.now();
  if (gizmo.mode === "translate") {
    if (now - lastGizmoSend < 50) return;
    lastGizmoSend = now;
    post("/cart", { axis: index, value: gizmoTarget.position.getComponent(index) })
      .catch(releaseDeadman);
    return;
  }
  const delta = gizmoTarget.quaternion.clone().multiply(previousQuaternion.clone().invert());
  previousQuaternion.copy(gizmoTarget.quaternion);
  const vector = new THREE.Vector3(delta.x, delta.y, delta.z);
  if (vector.length() < 1e-9) return;
  let angle = 2 * Math.atan2(vector.length(), delta.w);
  if (angle > Math.PI) angle -= 2 * Math.PI;
  rotationDelta += angle * Math.sign(vector.getComponent(index));
  if (now - lastGizmoSend < 50 || Math.abs(rotationDelta) < 1e-4) return;
  lastGizmoSend = now;
  post("/cart", { axis: index + 3, delta: rotationDelta }).catch(releaseDeadman);
  rotationDelta = 0;
});

const selectMode = (mode) => {
  gizmo.setMode(mode);
  $("mode-t").classList.toggle("selected", mode === "translate");
  $("mode-r").classList.toggle("selected", mode === "rotate");
};
$("mode-t").onclick = () => selectMode("translate");
$("mode-r").onclick = () => selectMode("rotate");

let sliderDrag = false;
const addSlider = (parent, descriptor, decimals, oninput) => {
  const row = document.createElement("div");
  row.className = "slider-row";
  const label = document.createElement("label");
  label.textContent = descriptor.name;
  const input = document.createElement("input");
  input.type = "range";
  input.min = descriptor.min;
  input.max = descriptor.max;
  input.step = descriptor.step || 0.01;
  input.value = descriptor.value;
  const output = document.createElement("output");
  output.value = Number(descriptor.value).toFixed(decimals);
  input.oninput = () => {
    output.value = Number(input.value).toFixed(decimals);
    oninput(Number(input.value));
  };
  input.onpointerdown = () => { sliderDrag = true; };
  input.onpointerup = () => { sliderDrag = false; };
  row.append(label, input, output);
  parent.appendChild(row);
  return [input, output];
};

const syncSliders = (pairs, descriptors, decimals) => pairs.forEach((pair, index) => {
  const descriptor = descriptors[index];
  if (!descriptor || Math.abs(Number(pair[0].value) - descriptor.value) < 1e-4) return;
  pair[0].value = descriptor.value;
  pair[1].value = descriptor.value.toFixed(decimals);
});

let deadmanHeld = false;
let deadmanTimer = null;
const deadmanButton = $("deadman");
const authorityText = $("authority-text");
const sendDeadman = () => post("/deadman", { held: deadmanHeld }).catch(() => {
  deadmanHeld = false;
  deadmanButton.classList.remove("held");
});
function holdDeadman(event) {
  if (event) event.preventDefault();
  if (deadmanHeld) return;
  deadmanHeld = true;
  deadmanButton.classList.add("held");
  authorityText.textContent = "Deadman held — selected actor may move";
  sendDeadman();
  deadmanTimer = setInterval(sendDeadman, 100);
}
function releaseDeadman() {
  if (!deadmanHeld) return;
  deadmanHeld = false;
  clearInterval(deadmanTimer);
  deadmanTimer = null;
  deadmanButton.classList.remove("held");
  authorityText.textContent = "Deadman released — hold and disarm requested";
  sendDeadman();
}
deadmanButton.addEventListener("pointerdown", holdDeadman);
addEventListener("pointerup", releaseDeadman);
addEventListener("pointercancel", releaseDeadman);
addEventListener("blur", releaseDeadman);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseDeadman();
});
addEventListener("keydown", (event) => {
  if (event.code === "Space" && !event.repeat && event.target === document.body) holdDeadman(event);
});
addEventListener("keyup", (event) => {
  if (event.code === "Space") releaseDeadman();
});

let built = false;
let building = false;
let jointSliders = [];
let gripperSlider = null;
let measuredRobot = null;
let targetRobot = null;
let planRobot = null;
let planFrames = [];
let planIndex = 0;
let planVersion = -1;

const wxyzPoseMatrix = (pose) => new THREE.Matrix4().compose(
  new THREE.Vector3(...pose.slice(0, 3)),
  new THREE.Quaternion(pose[4], pose[5], pose[6], pose[3]),
  new THREE.Vector3(1, 1, 1),
);

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
  setCheckBadge("storage", "checking", "");
  setCheckBadge("dock", "checking", "");
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
    for (const name of ["storage", "dock"]) {
      const check = editorChecks[name];
      setCheckBadge(name, check.status, check.detail);
    }
    showSelectedArm();
  } catch (error) {
    setCheckBadge("storage", "error", error.message);
    setCheckBadge("dock", "error", error.message);
  }
};

const updateEditorState = (editor) => {
  if (!editor) return;
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

const buildModuleEditor = async (state) => {
  editorMode = true;
  await rebuildEditorScene(state, true);
  syncSelectionControls(state.module);
  $("actor-select").innerHTML = `<option>${state.module.type} grasp</option>`;
  deadmanButton.disabled = true;
  deadmanButton.querySelector("span").textContent = "Visualization only";
  deadmanButton.querySelector("small").textContent = "No motion output is open";
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
  $("edit-grasp").disabled = false;
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

const build = async (state) => {
  building = true;
  if (state.mode === "module_editor") {
    await buildModuleEditor(state);
    return;
  }
  const geometry = await (await fetch("/scene")).json();
  measuredRobot = await buildRobot(geometry.geoms, null, 1);
  targetRobot = await buildRobot(geometry.geoms, new THREE.Color(0xd99a24), 0.45);
  planRobot = await buildRobot(geometry.geoms, new THREE.Color(0x55aa7a), 0.55);
  await buildStatic(geometry.static || []);

  jointSliders = state.sliders.map((descriptor) => addSlider(
    $("debug"),
    descriptor,
    3,
    () => post("/sliders", { values: jointSliders.map(([input]) => Number(input.value)) }),
  ));
  if (state.gripper.max > state.gripper.min) {
    gripperSlider = addSlider($("grip"), state.gripper, 3, (value) => {
      $("grip-readout").value = `${value.toFixed(3)} m`;
      post("/gripper", { value }).catch(releaseDeadman);
    });
  } else {
    $("grip").textContent = "No Hand configured";
  }
  for (const name of state.buttons) {
    const button = document.createElement("button");
    button.textContent = name;
    button.onclick = () => post("/click", { button: name }).catch(releaseDeadman);
    $("buttons").appendChild(button);
  }
  built = true;
  building = false;
  resize();
};

const updateGate = (state) => {
  const gate = $("gate");
  if (state.armed === null || state.armed === undefined) {
    gate.hidden = true;
    return;
  }
  gate.hidden = false;
  const badge = $("gatebadge");
  badge.className = "status";
  if (state.fault) {
    badge.textContent = "Faulted";
    badge.classList.add("fault");
    badge.title = state.fault;
  } else if (state.armed) {
    badge.textContent = "Armed";
    badge.classList.add("live");
  } else {
    badge.textContent = "Disarmed";
    badge.classList.add("fault");
  }
};

const poll = async () => {
  try {
    const state = await (await fetch("/state", { cache: "no-store" })).json();
    $("connection").textContent = "Live";
    $("connection").className = "status live";
    if (state.ident) {
      $("ident").textContent = state.ident;
      document.title = `${state.ident.includes("real") ? "REAL" : "SIM"} calibration console`;
    }
    if (!built && !building) await build(state);
    if (state.mode === "module_editor") {
      $("log").textContent = state.log.join("\n");
      if (!draggingGizmo && editorMode) {
        editorReferenceMatrix = wxyzPoseMatrix(state.module.reference_pose);
        editorRevision = state.module.revision;
      }
      return;
    }
    if (built && !sliderDrag) {
      syncSliders(jointSliders, state.sliders, 3);
      if (gripperSlider) syncSliders([gripperSlider], [state.gripper], 3);
    }
    setPoses(measuredRobot, state.measured_geoms);
    setPoses(targetRobot, state.target_geoms);
    if (!draggingGizmo && state.ee) {
      gizmoTarget.position.set(...state.ee.p);
      gizmoTarget.quaternion.set(...state.ee.q);
    }
    if (state.plan_version !== planVersion) {
      planVersion = state.plan_version;
      const plan = await (await fetch("/plan", { cache: "no-store" })).json();
      planFrames = plan.frames || [];
      planIndex = 0;
      $("plan-summary").textContent = planFrames.length ? `${planFrames.length} preview frames` : "None";
      if (!planFrames.length && planRobot) planRobot.group.visible = false;
    }
    $("joint-summary").textContent = state.sliders.length
      ? `${state.sliders.length} joints / ${state.measured_geoms ? "valid" : "waiting"}`
      : "No joints";
    $("log").textContent = state.log.join("\n");
    updateGate(state);
  } catch (error) {
    $("connection").textContent = "Disconnected";
    $("connection").className = "status fault";
    releaseDeadman();
    console.error(error);
  }
};
setInterval(poll, 150);
poll();

$("arm-btn").onclick = () => post("/click", { button: "ARM" });
$("disarm-btn").onclick = () => post("/click", { button: "DISARM" });
$("hold-disarm").onclick = async () => {
  releaseDeadman();
  await post("/click", { button: "Stop (hold)" }).catch(() => {});
  await post("/click", { button: "DISARM" }).catch(() => {});
};

setInterval(() => {
  if (!planRobot || !planFrames.length) return;
  setPoses(planRobot, planFrames[planIndex]);
  planIndex = (planIndex + 1) % planFrames.length;
}, 100);

const animate = () => {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
};
resize();
animate();
