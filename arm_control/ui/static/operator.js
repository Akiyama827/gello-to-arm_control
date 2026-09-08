/* Operator gate page.
 *
 * Polls /state and posts an allowed action to /action. It deliberately
 * knows nothing about arms, phases or docks beyond the strings the server
 * sends: the meaning lives in the adapter that builds the workspace, and
 * duplicating any of it here would give the bench two places to disagree.
 */
"use strict";

const el = (id) => document.getElementById(id);
const POLL_MS = 400;

let lastLog = 0;

function setStatus(node, text, kind) {
  node.textContent = text;
  node.className = "status " + kind;
}

function renderPhases(state) {
  const list = el("phases");
  const accepted = new Set(state.accepted || []);
  list.replaceChildren(...(state.phases || []).map((phase) => {
    const li = document.createElement("li");
    const mark = document.createElement("span");
    mark.className = "mark";
    // Accepted is NOT "before the current phase": a phase can run without its
    // gate accepting, and that disagreement is the thing worth seeing.
    const done = accepted.has(phase);
    const held = state.holding === phase;
    mark.textContent = done ? "✓" : held ? "▸" : phase === state.current ? "•" : "·";
    li.append(mark, document.createTextNode(phase));
    if (done) li.classList.add("done");
    if (phase === state.current) li.classList.add("current");
    if (held) li.classList.add("held");
    list.append(li);
    return li;
  }));
}

function renderStep(state) {
  const held = el("held");
  const none = el("held-none");
  if (!state.step) {
    held.hidden = true;
    none.hidden = false;
    return;
  }
  none.hidden = true;
  held.hidden = false;
  el("held-phase").textContent = state.step.phase;
  const bits = [];
  if (state.step.summary) bits.push(state.step.summary);
  if (state.step.duration_s != null) bits.push(state.step.duration_s.toFixed(2) + " s");
  el("held-summary").textContent = bits.join(" — ");
  const table = el("held-detail");
  table.replaceChildren(...Object.entries(state.step.detail || {}).map(([k, v]) => {
    const tr = document.createElement("tr");
    const name = document.createElement("td");
    const value = document.createElement("td");
    name.textContent = k;
    value.textContent = v;
    tr.append(name, value);
    return tr;
  }));
}

function renderGate(state) {
  const badge = el("gatebadge");
  if (state.stopped) setStatus(badge, "STOPPED", "bad");
  else if (state.holding) setStatus(badge, "HELD — awaiting review", "hold");
  else if (state.started) setStatus(badge, "RUNNING", "ok");
  else setStatus(badge, "DISARMED", "neutral");

  // RE-PLAN only means something for a phase held at a gate; GO doubles as the
  // initial arm, so it stays live before the sequence starts.
  el("plan").disabled = !state.holding || state.stopped;
  el("play").disabled = !state.holding || state.stopped
    || !Number.isFinite(state.step?.duration_s);
  el("go").disabled = state.stopped;
  el("stop").disabled = state.stopped;

  const failure = el("failure");
  failure.hidden = !state.failure;
  failure.textContent = state.failure || "";
}

function renderLog(state) {
  const entries = state.log || [];
  if (entries.length === lastLog) return;
  lastLog = entries.length;
  const list = el("log");
  list.replaceChildren(...entries.slice().reverse().map((line) => {
    const li = document.createElement("li");
    li.textContent = line;
    return li;
  }));
}

async function poll() {
  try {
    const response = await fetch("/state", { cache: "no-store" });
    if (!response.ok) throw new Error("HTTP " + response.status);
    const state = await response.json();
    setStatus(el("connection"), "Connected", "ok");
    el("phase-now").textContent = state.current || "idle";
    renderPhases(state);
    renderStep(state);
    renderGate(state);
    renderLog(state);
  } catch (err) {
    setStatus(el("connection"), "Disconnected", "bad");
  }
}

async function send(action) {
  try {
    const response = await fetch("/action", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      setStatus(el("connection"), body.error || "Refused", "bad");
      return;
    }
  } catch (err) {
    setStatus(el("connection"), "Disconnected", "bad");
    return;
  }
  poll();
}

for (const action of ["go", "plan", "play", "stop"]) {
  el(action).addEventListener("click", () => send(action));
}
poll();
setInterval(poll, POLL_MS);
