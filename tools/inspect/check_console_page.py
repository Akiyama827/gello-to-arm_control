"""Run directly: python tools/inspect/check_console_page.py (no browser needed)."""
from html.parser import HTMLParser
from pathlib import Path
import json
import subprocess


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.elements, attrs["id"]
            self.elements[attrs["id"]] = (tag, attrs)


static = Path(__file__).resolve().parents[2] / "arm_control/ui/static"
page = Page()
page.feed((static / "console.html").read_text())
for axis in ("x", "y", "z", "roll", "pitch", "yaw"):
    tag, attrs = page.elements[f"pose-{axis}"]
    assert tag == "input" and attrs["type"] == "number" and "required" in attrs
    assert page.elements[f"current-{axis}"][0] == "output"
for name in ("jog-cart-speed", "jog-joint-input"):
    tag, attrs = page.elements[name]
    assert tag == "input" and attrs["type"] == "number" and "required" in attrs
for name in ("pose-status", "speed-status", "mode-status"):
    assert page.elements[name][1]["aria-live"] == "polite"
js = (static / "console.js").read_text()
for contract in ('post("/pose",', 'post("/jog_speed",', 'setAttribute("aria-pressed"',
                 'pose.revision', 'max_joint_speed_rad_s', 'reportValidity()'):
    assert contract in js, contract
subprocess.run(["node", "--input-type=module", "--check"], input=js, text=True, check=True)
controls = js[js.index("let currentPose ="):js.index("const updateHand =")]
setup = r"""
const assert = (value, message) => { if (!value) throw Error(message); };
const elements = Object.fromEntries(ids.map(id => [id, {
  value: '', textContent: '', children: [], dataset: {}, disabled: false,
  addEventListener(name, callback) { this[name] = callback; },
  setCustomValidity(message) { this.validationMessage = message; },
  reportValidity() { return !this.validationMessage && this.value !== '' && Number.isFinite(Number(this.value)) && (!this.max || Number(this.value) <= Number(this.max)); },
  setAttribute(name, value) { this[name] = value; },
  classList: { toggle() {} },
}]));
const $ = id => elements[id];
const document = { activeElement: null };
const calls = [];
let rejectPost = false;
const post = async (route, payload) => {
  calls.push({ route, payload });
  if (rejectPost) throw Error('Server refused');
  return { ok: true, revision: 2 };
};
"""
checks = r"""
const xyz = [100, 200, 300], rpy = [10, 20, 30];
const state = { current: { xyz_mm: xyz, rpy_deg: rpy }, target: { xyz_mm: xyz, rpy_deg: rpy }, revision: 1, status: '' };
$('buttons').children = [{ textContent: 'Plan' }, { textContent: 'Execute' }, { textContent: 'Stop (hold)' }];
updatePose(state);
assert($('pose-x').value === 100, 'accepted target fills Desired');
document.activeElement = $('pose-x');
$('pose-x').value = '123';
updatePose(state);
assert($('pose-x').value === '123', 'poll preserves focused input');
$('pose-x').input();
document.activeElement = null;
updatePose(state);
assert($('pose-x').value === '123' && $('buttons').children[0].disabled, 'dirty draft survives blur and blocks Plan');
assert(!$('buttons').children[2].disabled, 'Stop remains available');
$('copy-current').onclick();
assert(calls.length === 0 && $('pose-x').value === 100, 'Copy current is local only');
await $('apply-pose').onclick();
assert(calls[0].route === '/pose' && calls[0].payload.xyz_mm.length === 3 && calls[0].payload.rpy_deg.length === 3, 'atomic six-field pose');
updatePose({ ...state, revision: 2, status: 'Applying target; robot does not move' });
assert($('apply-pose').disabled && $('buttons').children[0].disabled, 'IK pending keeps draft and blocks motion');
updatePose({ ...state, revision: 2, status: 'Target applied; Plan + preview before Execute' });
assert(!$('apply-pose').disabled && !$('buttons').children[0].disabled, 'accepted target unblocks planning');
$('pose-x').value = '111'; $('pose-x').input();
await $('apply-pose').onclick();
$('pose-x').value = '222'; $('pose-x').input();
updatePose({ ...state, revision: 2, status: 'Target applied; Plan + preview before Execute' });
assert($('pose-x').value === '222' && $('buttons').children[0].disabled, 'edits during IK survive acceptance');
await $('apply-pose').onclick();
$('pose-x').value = '333'; $('pose-x').input();
updatePose({ ...state, revision: 3, status: 'Applying target; robot does not move' });
assert(!$('apply-pose').disabled, 'newer revision releases superseded Apply even with Applying status');
assert($('pose-x').value === '333' && $('buttons').children[0].disabled, 'superseding revision preserves later draft');
updatePose(null);
assert($('copy-current').disabled && $('current-x').textContent === '—', 'missing current is unavailable');
rejectPost = true;
await $('apply-pose').onclick();
assert($('pose-status').textContent === 'Server refused', 'pose server errors visible');
rejectPost = false;
const jog = { speed_m_s: .01, joint_speed_rad_s: .15, max_speed_m_s: .01, max_joint_speed_rad_s: .15 };
updateSpeeds(jog);
assert(Number($('jog-joint-input').max) * Math.PI / 180 <= .15, 'joint ceiling never rounds upward');
$('jog-cart-speed').value = '5'; $('jog-cart-speed').input();
$('jog-joint-input').value = '4'; $('jog-joint-input').input();
updateSpeeds(jog);
assert($('jog-cart-speed').value === '5', 'speed draft survives polls');
await $('save-speeds').onclick();
assert(calls.at(-1).route === '/jog_speed' && calls.at(-1).payload.speed_m_s === .005 && calls.at(-1).payload.joint_speed_rad_s === 4 * Math.PI / 180, 'both speeds sent in SI');
const count = calls.length;
$('jog-cart-speed').value = '0'; $('jog-cart-speed').input();
await $('save-speeds').onclick();
assert(calls.length === count, 'zero speed rejected');
const gain = { dataset: { mode: 'track' }, setAttribute(name, value) { this[name] = value; }, classList: { toggle() {} } };
$('gains').children = [gain];
updateMode({ selected: 'track', pending: 'float', reason: '' });
assert(gain['aria-pressed'] === 'true' && $('mode-status').textContent === 'Pending: float', 'pending does not replace confirmed selection');
updateMode({ selected: 'track', pending: null, reason: 'refused' });
assert(gain['aria-pressed'] === 'true' && $('mode-status').textContent === 'refused', 'refusal preserves confirmed selection');
updateMode(null);
assert(gain['aria-pressed'] === 'false' && $('mode-selected').textContent === 'Unknown', 'unknown mode clears highlights');
"""
subprocess.run(
    ["node", "--input-type=module"],
    input=f"const ids = {json.dumps(list(page.elements))};\n" + setup + controls + checks,
    text=True, check=True,
)
print("PASS: console HTML, JavaScript syntax, pose drafts/revisions, speed units/limits, confirmed mode")
