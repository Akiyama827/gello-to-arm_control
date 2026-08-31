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
  if (!response.ok) throw new Error(`${route}: ${response.status}`);
  return response.json();
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

const build = async (state) => {
  building = true;
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
