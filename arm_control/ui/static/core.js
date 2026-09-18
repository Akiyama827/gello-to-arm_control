// Shared machinery for both console pages: the 3D scene, mesh loading, robot
// building, the gizmo, sliders, the deadman, and the fetch helpers.
//
// Two pages import this: console.js (drive an arm) and editor.js (edit grasp
// profiles). They used to be ONE file serving one page, switched at runtime on
// whether `state.module` existed -- which is why the operator console showed a
// Calibration rack whose every control 404'd, and why the tab was titled
// "Workcell calibration console" while you were driving a robot.
//
// Nothing here knows which page it is on. Anything that does belongs in the
// page module.
import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { TransformControls } from "three/addons/TransformControls.js";
import { STLLoader } from "three/addons/STLLoader.js";

export { THREE };

export const $ = (id) => document.getElementById(id);

export const post = async (route, payload) => {
  const response = await fetch(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `${route}: ${response.status}`);
  return body;
};

export const getJSON = async (route) =>
  (await fetch(route, { cache: "no-store" })).json();

// -- scene -------------------------------------------------------------------
const view = $("view");
export const scene = new THREE.Scene();
scene.background = new THREE.Color(0x202a2e);
export const camera = new THREE.PerspectiveCamera(48, 1, 0.01, 20);
camera.up.set(0, 0, 1);
camera.position.set(0.9, -0.9, 0.62);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
view.appendChild(renderer.domElement);

export const resize = () => {
  const width = view.clientWidth;
  const height = view.clientHeight;
  if (!width || !height) return;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
};
addEventListener("resize", resize);
// Panel drags change the viewport width without a window resize event.
new ResizeObserver(resize).observe(view);

scene.add(new THREE.HemisphereLight(0xf5fbfa, 0x33464d, 1.25));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.55);
keyLight.position.set(1, -1, 2);
scene.add(keyLight);
const grid = new THREE.GridHelper(2, 20, 0x718083, 0x344145);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.14));
export const orbit = new OrbitControls(camera, renderer.domElement);
orbit.target.set(0, 0, 0.16);

export const startRendering = () => {
  const animate = () => {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  };
  resize();
  animate();
};

// -- meshes ------------------------------------------------------------------
const loader = new STLLoader();
const geometryCache = new Map();
export const loadGeometry = (url) => {
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

export const buildRobot = async (geometries, tint, opacity) => {
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

export const buildStatic = async (items) => {
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

export const setPoses = (robot, poses) => {
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

export const wxyzPoseMatrix = (pose) => new THREE.Matrix4().compose(
  new THREE.Vector3(...pose.slice(0, 3)),
  new THREE.Quaternion(pose[4], pose[5], pose[6], pose[3]),
  new THREE.Vector3(1, 1, 1),
);

// -- gizmo -------------------------------------------------------------------
export const gizmoTarget = new THREE.Object3D();
scene.add(gizmoTarget);
export const gizmo = new TransformControls(camera, renderer.domElement);
gizmo.attach(gizmoTarget);
gizmo.setSpace("world");
gizmo.setSize(0.62);
scene.add(gizmo);

let draggingGizmo = false;
let rotationDelta = 0;
const previousQuaternion = new THREE.Quaternion();
export const isDraggingGizmo = () => draggingGizmo;
gizmo.addEventListener("dragging-changed", (event) => {
  draggingGizmo = event.value;
  orbit.enabled = !event.value;
  if (event.value) {
    previousQuaternion.copy(gizmoTarget.quaternion);
    rotationDelta = 0;
  }
});

/** Accumulated rotation about world axis `index` since the last read, or null
 *  while it is still below the send threshold. */
export const takeRotationDelta = (index) => {
  const delta = gizmoTarget.quaternion.clone()
    .multiply(previousQuaternion.clone().invert());
  previousQuaternion.copy(gizmoTarget.quaternion);
  const vector = new THREE.Vector3(delta.x, delta.y, delta.z);
  if (vector.length() < 1e-9) return null;
  let angle = 2 * Math.atan2(vector.length(), delta.w);
  if (angle > Math.PI) angle -= 2 * Math.PI;
  rotationDelta += angle * Math.sign(vector.getComponent(index));
  if (Math.abs(rotationDelta) < 1e-4) return null;
  const value = rotationDelta;
  rotationDelta = 0;
  return value;
};

export const initGizmoModeButtons = () => {
  const select = (mode) => {
    gizmo.setMode(mode);
    $("mode-t").classList.toggle("selected", mode === "translate");
    $("mode-r").classList.toggle("selected", mode === "rotate");
  };
  $("mode-t").onclick = () => select("translate");
  $("mode-r").onclick = () => select("rotate");
  select("translate");
};

// -- sliders -----------------------------------------------------------------
let sliderDrag = false;
export const isSliderDragging = () => sliderDrag;

export const addSlider = (parent, descriptor, decimals, oninput) => {
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

export const syncSliders = (pairs, descriptors, decimals) =>
  pairs.forEach((pair, index) => {
    const descriptor = descriptors[index];
    if (!descriptor || Math.abs(Number(pair[0].value) - descriptor.value) < 1e-4) return;
    pair[0].value = descriptor.value;
    pair[1].value = descriptor.value.toFixed(decimals);
  });

// -- connection badge --------------------------------------------------------
export const setConnected = (live, ident) => {
  const badge = $("connection");
  if (badge) {
    badge.textContent = live ? "Live" : "Disconnected";
    badge.className = live ? "status live" : "status fault";
  }
  if (live && ident && $("ident")) $("ident").textContent = ident;
};
