// The operator console: drive ONE arm. Plan a move and review it, jog by hand,
// switch the control law, arm and disarm.
//
// Everything here commands an arm. Nothing here edits a stored profile -- that
// is editor.js, a separate page. They share core.js and nothing else.
import {
  THREE, $, post, getJSON, resize, startRendering,
  buildRobot, buildStatic, setPoses, gizmo, gizmoTarget, isDraggingGizmo,
  takeRotationDelta, initGizmoModeButtons, addSlider, syncSliders,
  isSliderDragging, setConnected,
} from "./core.js";

initGizmoModeButtons();

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
let lastGizmoSend = 0;

// -- deadman -----------------------------------------------------------------
// Held authority for REVIEWED motion. Releasing it drops authority: Stop AND
// DISARM -- which is why jog does NOT use it (see below).
//
// Lives here, not in core.js: the grasp editor has a deadman BUTTON but it
// is disabled and posts nothing, so a shared implementation put /deadman
// calls in a page whose panel answers 404.
let deadmanHeld = false;
let deadmanTimer = null;
let deadmanButton = null;
let authorityText = null;

const sendDeadman = () => post("/deadman", { held: deadmanHeld }).catch(() => {
  deadmanHeld = false;
  if (deadmanButton) deadmanButton.classList.remove("held");
});

function holdDeadman(event) {
  if (event) event.preventDefault();
  if (deadmanHeld || !deadmanButton || deadmanButton.disabled) return;
  deadmanHeld = true;
  deadmanButton.classList.add("held");
  if (authorityText) authorityText.textContent = "Deadman held — selected actor may move";
  sendDeadman();
  deadmanTimer = setInterval(sendDeadman, 100);
}

function releaseDeadman() {
  if (!deadmanHeld) return;
  deadmanHeld = false;
  clearInterval(deadmanTimer);
  deadmanTimer = null;
  if (deadmanButton) deadmanButton.classList.remove("held");
  if (authorityText) authorityText.textContent = "Deadman released — hold and disarm requested";
  sendDeadman();
}

const initDeadman = () => {
  deadmanButton = $("deadman");
  authorityText = $("authority-text");
  if (!deadmanButton) return null;
  deadmanButton.addEventListener("pointerdown", holdDeadman);
  addEventListener("pointerup", releaseDeadman);
  addEventListener("pointercancel", releaseDeadman);
  addEventListener("blur", releaseDeadman);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) releaseDeadman();
  });
  addEventListener("keydown", (event) => {
    if (event.code === "Space" && !event.repeat && event.target === document.body) {
      holdDeadman(event);
    }
  });
  addEventListener("keyup", (event) => {
    if (event.code === "Space") releaseDeadman();
  });
  return deadmanButton;
};


const deadmanButton = initDeadman();

// -- gizmo: drag the target, one axis at a time ------------------------------
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
  const delta = takeRotationDelta(index);
  if (delta === null || now - lastGizmoSend < 50) return;
  lastGizmoSend = now;
  post("/cart", { axis: index + 3, delta }).catch(releaseDeadman);
});

// -- jog ---------------------------------------------------------------------
// Press-and-hold. The button IS the deadman: a jog needs one finger and stops
// the instant it lifts. Because the page only asserts "still held" while it is
// down, a closed tab or a dead network stops the arm with nothing having to
// send a stop.
let jogAxis = null;
let jogTimer = null;

const sendJog = () => {
  if (!jogAxis) return;
  post("/jog", { axis: jogAxis.axis, dir: jogAxis.dir, held: true }).catch(releaseJog);
};

function releaseJog() {
  if (!jogAxis) return;
  const last = jogAxis;
  jogAxis = null;
  clearInterval(jogTimer);
  jogTimer = null;
  post("/jog", { axis: last.axis, dir: last.dir, held: false }).catch(() => {});
}
addEventListener("pointerup", releaseJog);
addEventListener("pointercancel", releaseJog);
addEventListener("blur", releaseJog);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) releaseJog();
});

const buildJogPad = (jog) => {
  const pad = $("jog-pad");
  pad.replaceChildren();
  const holdJog = (axis, dir) => (event) => {
    if (event) event.preventDefault();
    // NOT holdDeadman(): that deadman drops authority on release (Stop AND
    // DISARM), which would disarm the arm between every nudge.
    jogAxis = { axis, dir };
    sendJog();
    if (jogTimer) clearInterval(jogTimer);
    jogTimer = setInterval(sendJog, 100);
  };
  const addRow = (parent, label, entries) => {
    const row = document.createElement("div");
    row.className = "jog-row";
    const tag = document.createElement("span");
    tag.textContent = label;
    row.appendChild(tag);
    for (const [text, axis, dir] of entries) {
      const button = document.createElement("button");
      button.textContent = text;
      button.addEventListener("pointerdown", holdJog(axis, dir));
      row.appendChild(button);
    }
    parent.appendChild(row);
  };
  for (const axis of jog.axes) {
    addRow(pad, axis.toUpperCase(), [["−", axis, -1], ["+", axis, 1]]);
  }
  // Joint jog lives behind a disclosure: it is the escape hatch (exempt from
  // the singularity gate), not the everyday control.
  const joints = $("jog-joints");
  joints.replaceChildren();
  for (let j = 0; j < jog.joints; j += 1) {
    addRow(joints, `J${j + 1}`, [["−", `j${j}`, -1], ["+", `j${j}`, 1]]);
  }
  $("jog-joints-wrap").hidden = jog.joints === 0;
};

// -- first paint -------------------------------------------------------------
const build = async (state) => {
  building = true;
  const geometry = await getJSON("/scene");
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
    $("grip-wrap").hidden = true;
  }
  if (state.jog) buildJogPad(state.jog);

  const strip = $("buttons");
  const gains = $("gains");
  strip.replaceChildren();
  gains.replaceChildren();
  for (const name of state.buttons) {
    const button = document.createElement("button");
    button.textContent = name.startsWith("Gains: ") ? name.slice(7) : name;
    button.onclick = () => post("/click", { button: name }).catch(releaseDeadman);
    // Gain presets are a different KIND of action from plan/execute: they
    // change the control law under whatever is running. Own row, own heading.
    (name.startsWith("Gains: ") ? gains : strip).appendChild(button);
  }
  $("gains-wrap").hidden = gains.childElementCount === 0;
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
    const state = await getJSON("/state");
    setConnected(true, state.ident);
    if (state.ident) document.title = `${state.ident} — arm console`;
    if (!built && !building) await build(state);
    if (state.jog) {
      const note = state.jog.note || (state.jog.held ? "jogging" : "");
      $("jog-note").textContent = note;
      $("jog-note").className = note.includes("refused") ? "jog-note refused" : "jog-note";
    }
    if (built && !isSliderDragging()) {
      syncSliders(jointSliders, state.sliders, 3);
      if (gripperSlider) syncSliders([gripperSlider], [state.gripper], 3);
    }
    setPoses(measuredRobot, state.measured_geoms);
    setPoses(targetRobot, state.target_geoms);
    if (!isDraggingGizmo() && state.ee) {
      gizmoTarget.position.set(...state.ee.p);
      gizmoTarget.quaternion.set(...state.ee.q);
    }
    if (state.plan_version !== planVersion) {
      planVersion = state.plan_version;
      const plan = await getJSON("/plan");
      planFrames = plan.frames || [];
      planIndex = 0;
      $("plan-summary").textContent = planFrames.length
        ? `${planFrames.length} preview frames`
        : "None";
      if (!planFrames.length && planRobot) planRobot.group.visible = false;
    }
    $("joint-summary").textContent = state.sliders.length
      ? `${state.sliders.length} joints / ${state.measured_geoms ? "valid" : "waiting"}`
      : "No joints";
    $("log").textContent = state.log.join("\n");
    updateGate(state);
  } catch (error) {
    setConnected(false);
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

// The rail is resizable; the scene takes the rest.
const RAIL_MIN = 200;
const RAIL_MAX = 520;
const shell = document.querySelector(".console-shell");
const gutter = shell.querySelector(".gutter");
gutter.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  const startX = event.clientX;
  const startWidth = $("rail").getBoundingClientRect().width;
  gutter.setPointerCapture(event.pointerId);
  gutter.classList.add("dragging");
  const onMove = (move) => {
    const width = Math.min(RAIL_MAX, Math.max(RAIL_MIN, startWidth + (startX - move.clientX)));
    shell.style.setProperty("--rail-w", `${width}px`);
  };
  const onUp = () => {
    gutter.classList.remove("dragging");
    gutter.releasePointerCapture(event.pointerId);
    gutter.removeEventListener("pointermove", onMove);
    gutter.removeEventListener("pointerup", onUp);
    gutter.removeEventListener("pointercancel", onUp);
  };
  gutter.addEventListener("pointermove", onMove);
  gutter.addEventListener("pointerup", onUp);
  gutter.addEventListener("pointercancel", onUp);
});

startRendering();
