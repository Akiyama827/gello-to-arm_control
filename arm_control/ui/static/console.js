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
let handPosting = false;
let handError = "";
let currentPose = null;
let poseDirty = false;
let poseEdits = 0;
let posePosting = false;
let posePending = null;
let poseError = "";
let speedDirty = false;
let speedPosting = false;
let modeError = "";
const poseAxes = ["x", "y", "z", "roll", "pitch", "yaw"];
const poseInputs = poseAxes.map((axis) => $(`pose-${axis}`));
const speedInputs = [$("jog-cart-speed"), $("jog-joint-input")];
const poseValues = (pose) => [...pose.xyz_mm, ...pose.rpy_deg];
const fillPose = (pose) => poseValues(pose).forEach((value, i) => {
  poseInputs[i].value = Number(value.toFixed(6));
});
const updateMoveButtons = () => {
  for (const button of $("buttons").children) {
    if (/^(Plan|Execute)/i.test(button.textContent)) {
      button.disabled = poseDirty || posePosting || Boolean(posePending);
      button.title = button.disabled ? "Apply the desired pose successfully before planning or executing" : "";
    }
  }
};
for (const input of poseInputs) input.addEventListener("input", () => {
  poseDirty = true;
  poseEdits += 1;
  poseError = "";
  $("pose-status").textContent = "Unsaved desired pose · Apply target before Plan";
  updateMoveButtons();
});
$("copy-current").onclick = () => {
  if (!currentPose) return;
  fillPose(currentPose);
  poseDirty = true;
  poseEdits += 1;
  poseError = "";
  $("pose-status").textContent = "Current pose copied to Desired · not applied";
  updateMoveButtons();
};
// Presets FILL Desired and nothing else -- same contract as Copy current, so
// Apply/Plan/Execute still gate the motion. The list is supplied by the
// deployment; an empty one hides the row entirely.
const buildPresets = (presets) => {
  const row = $("preset-row");
  const select = $("preset-select");
  select.replaceChildren();
  if (!presets || !presets.length) {
    row.hidden = true;
    return;
  }
  for (const [index, preset] of presets.entries()) {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = preset.name;
    select.append(option);
  }
  row.hidden = false;
  $("fill-preset").onclick = () => {
    const preset = presets[Number(select.value)];
    if (!preset) return;
    fillPose(preset);
    poseDirty = true;
    poseEdits += 1;
    poseError = "";
    $("pose-status").textContent = `"${preset.name}" filled into Desired \u00b7 not applied`;
    updateMoveButtons();
  };
};

$("apply-pose").onclick = async () => {
  if (posePosting || posePending) return;
  if (!poseInputs.every((input) => input.reportValidity())) {
    poseError = "Enter all six finite pose values before applying";
    $("pose-status").textContent = poseError;
    return;
  }
  const values = poseInputs.map((input) => Number(input.value));
  if (!values.every(Number.isFinite)) return;
  const edits = poseEdits;
  posePosting = true;
  poseError = "";
  $("apply-pose").disabled = true;
  $("pose-status").textContent = "Applying target; robot does not move";
  updateMoveButtons();
  try {
    const result = await post("/pose", { xyz_mm: values.slice(0, 3), rpy_deg: values.slice(3) });
    posePending = { revision: result.revision, edits };
  } catch (error) {
    poseError = error.message;
    $("pose-status").textContent = poseError;
  } finally {
    posePosting = false;
    $("apply-pose").disabled = Boolean(posePending);
    updateMoveButtons();
  }
};
const updatePose = (pose) => {
  currentPose = pose?.current || null;
  const values = currentPose ? poseValues(currentPose) : [];
  poseAxes.forEach((axis, i) => {
    $(`current-${axis}`).textContent = currentPose ? values[i].toFixed(2) : "—";
  });
  $("current-status").textContent = currentPose ? "Current · FK estimate from measured joints" : "Current pose unavailable / stale";
  $("copy-current").disabled = !currentPose;
  if (posePending && pose && (pose.revision > posePending.revision ||
      (pose.revision === posePending.revision && !pose.status.startsWith("Applying")))) {
    if (pose.revision === posePending.revision && pose.status.startsWith("Target applied") && poseEdits === posePending.edits) {
      poseDirty = false;
    }
    posePending = null;
  }
  if (pose?.target && !poseDirty && !posePosting && !posePending && !poseInputs.includes(document.activeElement)) {
    fillPose(pose.target);
  }
  if (!posePosting) $("pose-status").textContent = poseError || [
    poseDirty && !posePending ? "Desired is an unapplied draft" : "",
    pose ? (pose.status || "Edit Desired, then Apply target") : "Target status unavailable",
  ].filter(Boolean).join(" · ");
  $("apply-pose").disabled = posePosting || Boolean(posePending);
  updateMoveButtons();
};

for (const input of speedInputs) input.addEventListener("input", () => {
  speedDirty = true;
  input.setCustomValidity("");
  $("speed-status").textContent = "Unsaved speeds · Save to apply on the next press";
});
$("save-speeds").onclick = async () => {
  if (speedPosting) return;
  for (const input of speedInputs) {
    input.setCustomValidity(Number(input.value) > 0 ? "" : "Speed must be greater than zero");
    if (!input.reportValidity()) {
      $("speed-status").textContent = "Enter positive speeds at or below the displayed maximum";
      return;
    }
  }
  const draft = speedInputs.map((input) => input.value);
  speedPosting = true;
  $("save-speeds").disabled = true;
  try {
    await post("/jog_speed", {
      speed_m_s: Number(draft[0]) / 1000,
      joint_speed_rad_s: Number(draft[1]) * Math.PI / 180,
    });
    speedDirty = speedInputs.some((input, i) => input.value !== draft[i]);
    $("speed-status").textContent = speedDirty ? "Saved; newer edits remain unsaved" : "Speeds saved · apply on the next press";
  } catch (error) {
    $("speed-status").textContent = error.message;
  } finally {
    speedPosting = false;
    $("save-speeds").disabled = false;
  }
};
const updateSpeeds = (jog) => {
  // Truncate displayed limits: rounding up would advertise a forbidden speed.
  const floorLimit = (value) => Math.floor(value * 1e9) / 1e9;
  const maxima = [jog.max_speed_m_s * 1000, jog.max_joint_speed_rad_s * 180 / Math.PI].map(floorLimit);
  const values = [jog.speed_m_s * 1000, jog.joint_speed_rad_s * 180 / Math.PI];
  speedInputs.forEach((input, i) => {
    input.max = maxima[i];
    if (!speedDirty && !speedPosting && !speedInputs.includes(document.activeElement)) input.value = floorLimit(values[i]);
  });
  $("jog-speed").textContent = `Accepted ${values[0].toFixed(3)} mm/s · max ${maxima[0]} mm/s · hold − / + · robot base XYZ`;
  $("jog-joint-speed").textContent = `Accepted ${values[1].toFixed(6)}°/s · max ${maxima[1]}°/s`;
};
const updateMode = (mode) => {
  const selected = mode?.selected;
  $("mode-selected").textContent = selected ? selected[0].toUpperCase() + selected.slice(1) : "Unknown";
  for (const button of $("gains").children) {
    const pressed = Boolean(selected && button.dataset.mode === selected);
    button.classList.toggle("selected", pressed);
    button.setAttribute("aria-pressed", String(pressed));
  }
  $("mode-status").textContent = modeError || [
    mode?.pending ? `Pending: ${mode.pending}` : "",
    mode?.reason || "",
  ].filter(Boolean).join(" · ");
};

const updateHand = (hand) => {
  for (const id of ["hand-grasp", "hand-open"]) $(id).disabled = handPosting || !hand?.enabled;
  if (gripperSlider) gripperSlider[0].disabled = handPosting || Boolean(hand?.busy);
  $("hand-status").textContent = handError || [hand?.status, hand?.reason].filter(Boolean).join(" · ");
  $("hand-feedback").textContent = hand?.measured_width_mm == null
    ? "Measured opening unavailable (not the slider target)"
    : `Measured opening: ${hand.measured_width_mm.toFixed(1)} mm total`;
};

const sendHand = async (mode) => {
  if (handPosting) return;
  const payload = { mode };
  if (mode === "close") {
    if (!$("hand-force").reportValidity() || !$("hand-width").reportValidity()) return;
    payload.force_n = Number($("hand-force").value);
    payload.width_m = Number($("hand-width").value) / 1000;
    // What the hand is about to hold. Nothing else in a manual graph knows,
    // so an unset selector means "declare nothing" -- never a stale carry-over.
    payload.payload_id = $("hand-payload").value;
  }
  handPosting = true;
  handError = "";
  $("hand-grasp").disabled = true;
  $("hand-open").disabled = true;
  try {
    await post("/hand", payload);
  } catch (error) {
    handError = error.message;
    $("hand-status").textContent = handError;
  } finally {
    handPosting = false;
  }
};
$("hand-grasp").onclick = () => sendHand("close");
$("hand-open").onclick = () => sendHand("release");

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
      .catch(console.error);
    return;
  }
  const delta = takeRotationDelta(index);
  if (delta === null || now - lastGizmoSend < 50) return;
  lastGizmoSend = now;
  post("/cart", { axis: index + 3, delta }).catch(console.error);
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
  updateSpeeds(jog);
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
      post("/gripper", { value }).catch(console.error);
    });
  } else {
    $("grip").textContent = "No Hand configured";
    $("grip-wrap").hidden = true;
  }
  const hand = state.hand;
  if (hand?.defaults) {
    $("hand-force").min = hand.force_min_n;
    $("hand-force").max = hand.force_max_n;
    $("hand-force").value = hand.defaults.force_n;
    $("hand-width").max = hand.width_max_mm;
    $("hand-width").value = hand.defaults.width_m * 1000;
    const sel = $("hand-payload");
    const want = (hand.payloads || []).map((p) => `${p.id} (${p.mass_kg.toFixed(3)} kg)`).join("|");
    if (sel.dataset.built !== want) {
      sel.dataset.built = want;
      sel.innerHTML = '<option value="">none</option>';
      for (const p of hand.payloads || []) {
        const o = document.createElement("option");
        o.value = p.id;
        o.textContent = `${p.id} (${p.mass_kg.toFixed(3)} kg)`;
        sel.appendChild(o);
      }
    }
    sel.hidden = !(hand.payloads || []).length;
    $("hand-settings").textContent =
      `Commanded force · total jaw width · ${hand.defaults.speed_mps * 1000} mm/s. ` +
      `Acceptance: −${hand.defaults.epsilon_inner_m * 1000} / +${hand.defaults.epsilon_outer_m * 1000} mm. ` +
      `Open: ${hand.defaults.open_width_m * 1000} mm. DISARM blocks new actions; an executing Hand move/grasp may continue. It does not open a held object.`;
  }
  if (state.jog) buildJogPad(state.jog);
  buildPresets(state.pose_presets);

  const strip = $("buttons");
  const gains = $("gains");
  strip.replaceChildren();
  gains.replaceChildren();
  for (const name of state.buttons) {
    const button = document.createElement("button");
    button.textContent = name.startsWith("Gains: ") ? name.slice(7) : name;
    if (name.startsWith("Gains: ")) {
      button.dataset.mode = name.slice(7).toLowerCase();
      button.setAttribute("aria-pressed", "false");
    }
    button.onclick = () => {
      if (name === "Stop (hold)") releaseJog();
      if (button.dataset.mode) modeError = "";
      post("/click", { button: name }).catch((error) => {
        if (button.dataset.mode) {
          modeError = error.message;
          $("mode-status").textContent = modeError;
        } else console.error(error);
      });
    };
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
  gate.hidden = false;
  const badge = $("gatebadge");
  badge.className = "status";
  if (state.fault) {
    badge.textContent = "Faulted";
    badge.classList.add("fault");
    badge.title = state.fault;
  } else if (state.armed === null || state.armed === undefined) {
    badge.textContent = "Unknown";
    badge.classList.add("neutral");
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
      updateSpeeds(state.jog);
      const note = state.jog.note || (state.jog.held ? "jogging" : "");
      $("jog-note").textContent = note;
      $("jog-note").className = note.includes("refused") ? "jog-note refused" : "jog-note";
    }
    if (built && !isSliderDragging()) {
      syncSliders(jointSliders, state.sliders, 3);
      if (gripperSlider) syncSliders([gripperSlider], [state.gripper], 3);
    }
    updateHand(state.hand);
    updatePose(state.pose);
    updateMode(state.mode);
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
    $("control-mode").textContent = state.control_mode === "soft"
      ? "Soft: EE position + orientation hold; compliant nullspace. Select Track before jogging or planning."
      : "Joint control · Soft requires a Cartesian-capable plant";
    updateGate(state);
  } catch (error) {
    setConnected(false);
    updateHand(null);
    updatePose(null);
    updateMode(null);
    releaseJog();
    console.error(error);
  }
};
setInterval(poll, 150);
poll();

$("arm-btn").onclick = () => post("/click", { button: "ARM" });
$("disarm-btn").onclick = () => {
  releaseJog();
  post("/click", { button: "DISARM" }).catch(console.error);
};
$("hold-disarm").onclick = async () => {
  releaseJog();
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
