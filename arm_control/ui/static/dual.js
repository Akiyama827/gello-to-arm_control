import { $, THREE, gizmo, gizmoTarget, isDraggingGizmo, initGizmoModeButtons, camera, orbit, post, getJSON, buildRobot, buildStatic, setPoses, startRendering } from './core.js';

let state = null, built = false, requesting = false, currentPreview = null;
let preview = [], playing = false, playbackStart = 0, dirty = false, uiError = '';
const controls = new Map(), robots = new Map();
let objectInputs = [], objectMeshes = null, modeRevision = null;
const sharedMode = s => s?.mode === 'shared';
const radians = Math.PI / 180;
camera.position.set(1.65, -1.65, 1.2);
orbit.target.set(.3, 0, .35);
gizmo.visible = false;
gizmo.enabled = false;
startRendering();
initGizmoModeButtons();

// Numeric RPY is fixed-axis xyz (Rz * Ry * Rx), matching the Python API.
function poseMatrix(values) {
  const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(...values.slice(3).map(v=>v*radians), 'ZYX'));
  return new THREE.Matrix4().compose(new THREE.Vector3(...values.slice(0,3).map(v=>v/1000)), q, new THREE.Vector3(1,1,1));
}
function matrixValues(matrix) {
  const p = new THREE.Vector3(), q = new THREE.Quaternion();
  matrix.decompose(p, q, new THREE.Vector3());
  const e = new THREE.Euler().setFromQuaternion(q, 'ZYX');
  return [...p.toArray().map(v=>v*1000), e.x/radians, e.y/radians, e.z/radians];
}
function baseMatrix(id) {
  const b = state.arms[id].world_T_base;
  return new THREE.Matrix4().compose(new THREE.Vector3(...b.p), new THREE.Quaternion(...b.q), new THREE.Vector3(1,1,1));
}
function selectedDraft() {
  if (sharedMode(state)) return {inputs:objectInputs, frame:'world'};
  const c = controls.get($('drag-target').value);
  return c?.kind.value==='pose' && c.selected.checked ? {inputs:c.inputs, frame:c.frame.value} : null;
}
function syncGizmo() {
  if (isDraggingGizmo()) return;
  const draft = selectedDraft();
  if (!draft || draft.inputs.some(i=>!i.validity.valid)) { gizmo.visible=gizmo.enabled=false; return; }
  let T = poseMatrix(draft.inputs.map(i=>Number(i.value)));
  if (draft.frame==='base') T.premultiply(baseMatrix($('drag-target').value));
  T.decompose(gizmoTarget.position, gizmoTarget.quaternion, gizmoTarget.scale);
  if (objectMeshes && sharedMode(state)) {
    objectMeshes.target.position.copy(gizmoTarget.position);
    objectMeshes.target.quaternion.copy(gizmoTarget.quaternion);
    objectMeshes.target.visible=true;
  }
  for (const [id,c] of controls) c.panel.classList.toggle('drag-selected', !sharedMode(state) && id===$('drag-target').value);
  if (built) render(state);
}
function draftChanged() { changed(); syncGizmo(); }
$('drag-target').onchange = syncGizmo;
gizmo.addEventListener('objectChange', () => {
  if (!isDraggingGizmo() || !gizmo.enabled) return;
  const draft = selectedDraft();
  if (!draft) return;
  let T = new THREE.Matrix4().compose(gizmoTarget.position, gizmoTarget.quaternion, new THREE.Vector3(1,1,1));
  if (draft.frame==='base') T.premultiply(baseMatrix($('drag-target').value).invert());
  matrixValues(T).forEach((v,i)=>{draft.inputs[i].value=v.toFixed(3);});
  changed();
  if (objectMeshes && sharedMode(state)) {
    objectMeshes.target.position.copy(gizmoTarget.position);
    objectMeshes.target.quaternion.copy(gizmoTarget.quaternion);
    objectMeshes.target.visible=true;
  }
});

function element(tag, parent, text, props = {}) {
  const el = document.createElement(tag);
  if (text !== null) el.textContent = text;
  Object.assign(el, props);
  parent.appendChild(el);
  return el;
}
async function action(name, extra = {}) {
  if (!state || (requesting && !['cancel','disarm'].includes(name))) return;
  requesting = true;
  if (built) render(state);
  try {
    const reply = await post('/action', {session: state.session, revision: state.revision, action: name, ...extra});
    state.revision = Math.max(state.revision, reply.revision);
    uiError = '';
    return reply;
  } catch (error) {
    uiError = error.message;
    $('error').textContent = uiError;
  } finally { requesting = false; }
}
function changed() {
  dirty = true;
  $('reviewed').checked = false;
  $('stage').disabled = $('execute').disabled = true;
  $('play').disabled = $('scrub').disabled = true;
  playing = false;
  for (const group of robots.values()) {
    group.target.group.visible = false;
    group.preview.group.visible = false;
  }
  if (objectMeshes) objectMeshes.preview.visible=false;
}
function valuesFor(a, ctl) {
  if (ctl.kind.value === 'joint') return a.q_deg;
  return ctl.frame.value === 'world' ? a.world_pose : a.base_pose;
}
function fill(a, ctl) {
  const values = valuesFor(a, ctl);
  ctl.fields.replaceChildren();
  ctl.inputs = values.map((value, i) => {
    const name = ctl.kind.value === 'joint' ? `J${i+1} (°)` : ['X (mm)','Y (mm)','Z (mm)','Roll (°)','Pitch (°)','Yaw (°)'][i];
    const label = element('label', ctl.fields, name);
    const input = element('input', label, null, {type:'number', required:true, step:'any', value:Number(value).toFixed(2)});
    input.addEventListener('input', draftChanged);
    return input;
  });
  changed();
  if (built) syncGizmo();
}
function targets() {
  for (const c of controls.values()) for (const input of c.inputs) {
    if (!input.reportValidity()) return null;
  }
  return Object.fromEntries([...controls].map(([id,c]) => [id, {kind:c.kind.value,frame:c.frame.value,values:c.inputs.map(i=>Number(i.value))}]));
}
async function build(initial) {
  if (!Object.values(initial.arms).every(a=>a.world_T_base)) {
    throw new Error('Restart the dual console to load the new target controls.');
  }
  const geometry = await getJSON('/geometry');
  for (const [id,a] of Object.entries(initial.arms)) {
    const panel = element('section', $('arms'), null, {className:'arm-panel'});
    const head = element('div', panel, null, {className:'arm-heading'});
    const selected = element('input', head, null, {type:'checkbox',checked:true});
    selected.setAttribute('aria-label',`Move ${id}`);
    selected.onchange = draftChanged;
    element('h3', head, id);
    const badge = element('output', head, 'Loading…');
    const row = element('div', panel, null, {className:'arm-actions'});
    const arm = element('button', row, 'Arm', {disabled:true});
    const disarm = element('button', row, 'Disarm', {disabled:true});
    arm.onclick = ()=>action('arm',{arm_id:id});
    disarm.onclick = ()=>action('disarm',{arm_id:id});
    const mode = element('div',panel,null,{className:'target-mode'});
    const kindLabel = element('label',mode,'Target');
    const kind = element('select',kindLabel,null);
    for (const [value,text] of [['pose','TCP pose'],['joint','Joint angles']]) element('option',kind,text,{value});
    const frameLabel = element('label',mode,'Coordinates');
    const frame = element('select',frameLabel,null);
    for (const [value,text] of [['world','Shared world'],['base','Arm base']]) element('option',frame,text,{value});
    const fields = element('div',panel,null,{className:'target-grid'});
    const targetTools = element('div',panel,null,{className:'target-tools'});
    const sync = element('button',targetTools,'Use measured pose');
    const hand = element('div',panel,null,{className:'hand-row'});
    const label = element('label',hand,'Hand opening (mm)');
    const width = element('input',label,null,{type:'number',min:'0',max:'80',step:'any',value:String(a.target_finger_m*2000)});
    width.min=String(a.finger_limits_mm[0]);width.max=String(a.finger_limits_mm[1]);
    const setHand = element('button',hand,'Set opening');
    const measured = element('p',panel,'',{className:'detail'});
    const c = {panel,selected,badge,arm,disarm,kind,frame,previousFrame:'world',fields,inputs:[],sync,width,setHand,measured,groups:[mode,fields,targetTools,hand]};
    controls.set(id,c);

    kind.onchange=()=>{frame.disabled=kind.value==='joint';fill(state.arms[id],c);};
    frame.onchange=()=>{
      let T=poseMatrix(c.inputs.map(i=>Number(i.value)));
      if(c.previousFrame==='base')T.premultiply(baseMatrix(id));
      if(frame.value==='base')T.premultiply(baseMatrix(id).invert());
      matrixValues(T).forEach((v,i)=>{c.inputs[i].value=v.toFixed(3);});
      c.previousFrame=frame.value;draftChanged();
    };
    sync.onclick=()=>fill(state.arms[id],c);
    const drag=element('button',targetTools,'Drag this TCP');
    drag.onclick=()=>{if(kind.value!=='pose'){kind.value='pose';fill(state.arms[id],c);}selected.checked=true;$('drag-target').value=id;syncGizmo();};
    c.drag=drag;
    width.required=true;
    setHand.onclick=()=>{if(width.reportValidity())action('gripper',{arm_id:id,width_mm:Number(width.value)});};
    fill(a,c);

    robots.set(id,{measured:await buildRobot(geometry.arms[id],null,1),
      target:await buildRobot(geometry.arms[id],new THREE.Color(0xe9ac42),.28),
      preview:await buildRobot(geometry.arms[id],new THREE.Color(0x72c48f),.55)});
    robots.get(id).target.group.visible=false;
    robots.get(id).preview.group.visible=false;
  }
  const scenery = await buildStatic(geometry.static || []);
  const bounds = new THREE.Box3().setFromObject(scenery);
  for (const [id,robot] of robots) {
    setPoses(robot.measured, initial.arms[id].visual.geoms);
    bounds.expandByObject(robot.measured.group);
  }
  if (!bounds.isEmpty()) {
    const center = bounds.getCenter(new THREE.Vector3());
    const extent = bounds.getSize(new THREE.Vector3()).length();
    const fov = Math.min(camera.fov, camera.fov * camera.aspect) * Math.PI / 180;
    orbit.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(1,-1,.8).normalize().multiplyScalar(extent / (2*Math.tan(fov/2))));
    orbit.update();
  }
  if (initial.shared_object) {

    objectMeshes={};
    for (const [name,color,opacity] of [['measured',0x8ca3bf,initial.shared_object.virtual ? .45 : 1],['target',0xe9ac42,.3],['preview',0x72c48f,.55]]) {
      const group=await buildStatic(geometry.object);
      group.traverse(o=>{if(o.isMesh){o.material.color.setHex(color);o.material.transparent=opacity<1;o.material.opacity=opacity;o.material.depthWrite=opacity===1;}});
      group.visible=name==='measured';
      objectMeshes[name]=group;
    }
    objectInputs=initial.shared_object.values.map((v,i)=>{
      const label=element('label',$('object-fields'),['X (mm)','Y (mm)','Z (mm)','Roll (°)','Pitch (°)','Yaw (°)'][i]);
      const input=element('input',label,null,{type:'number',step:'any',required:true,value:v.toFixed(3)});
      input.oninput=draftChanged;
      return input;
    });
    $('object-sync').onclick=()=>{state.shared_object.values.forEach((v,i)=>{objectInputs[i].value=v.toFixed(3);});draftChanged();};
  }
  $('models').replaceChildren();
  for (const [id,a] of Object.entries(initial.arms)) element('p',$('models'),`${id}: ${a.model_token}`);
  built=true;
  resetMode(initial);
}
function resetMode(s) {
  modeRevision=s.mode_revision;
  $('control-mode').value=s.mode;
  $('mode-confirmed').checked=false;
  $('drag-target').replaceChildren();
  if (sharedMode(s)) {
    element('option',$('drag-target'),s.shared_object.name,{value:'object'});
    s.shared_object.values.forEach((v,i)=>{objectInputs[i].value=v.toFixed(3);});
  } else {
    for (const [id,c] of controls) {
      element('option',$('drag-target'),`${id} TCP`,{value:id});
      fill(s.arms[id],c);
    }
  }
  changed();
  syncGizmo();
}
function renderMode(s, fresh, blocked) {
  const virtual=!!s.shared_object?.virtual;
  const wanted=$('control-mode').value;
  const transition=wanted!==s.mode && !!s.shared_object;
  $('active-mode').textContent=sharedMode(s)?(virtual?'Virtual coupling active':'Shared object active'):'Independent arms active';
  $('control-mode').querySelector('option[value="shared"]').textContent=virtual?'Virtual object':'Shared object';
  $('control-mode').disabled=!fresh||blocked||!s.shared_object;
  $('object-panel').hidden=!sharedMode(s);
  $('mode-transition').hidden=!transition;
  $('plan-grasps').hidden=wanted!=='shared';
  $('plan-grasps').disabled=!fresh||blocked;
  $('plan-grasps').textContent=virtual?'Plan to virtual grasps':'Plan to grasp poses';
  $('mode-confirmed').disabled=!fresh||blocked;
  $('apply-mode').textContent=wanted==='shared'?(virtual?'Enable virtual coupling':'Enable shared-object mode'):'Return to independent arms';
  $('apply-mode').disabled=!fresh||blocked||!!s.transition_reason||!$('mode-confirmed').checked;
  $('mode-confirmation').textContent=virtual?(wanted==='shared'?'Both Hands are empty; capture these TCP offsets.':'Return to independent arm targets.'):
    wanted==='shared'?'I verified both fixed grasps.':'The object is externally supported; release both grasp constraints.';
  const entry=s.shared_object?.entry_errors;
  const residual=virtual&&entry?Object.entries(entry).map(([a,e])=>`${a}: ${e.position_mm.toFixed(2)} mm / ${e.angle_deg.toFixed(2)}°`).join('; '):'';
  $('transition-reason').textContent=s.transition_reason || (virtual&&wanted==='shared'?
    `Ready to capture stopped TCP offsets (within 10 mm / 2°). ${residual}`:'Stopped and ready for confirmation.');
  const widths=s.shared_object?Object.entries(s.shared_object.grasp_widths_mm).map(([a,w])=>`${a}: ${w.toFixed(1)} mm`).join(', '):'';
  $('mode-detail').textContent=!s.shared_object?'No shared object is described in this setup.':
    virtual?(sharedMode(s)?'Both real arms follow the virtual plate using the captured TCP offsets. No physical object or added payload.':
      'Keep both Hands empty and open. Plan to virtual grasps, review / stage / execute, then capture the stopped TCP offsets. These offsets stay fixed during coupling; no Hand closure is required.'):
    sharedMode(s)?'Both TCPs follow the object. To return to independent control, stop and support the object first.':
    wanted==='shared'?`Object supported in simulation. Open both Hands, Plan to grasp poses, review / stage / execute, then close the Hands (${widths}) and confirm. Selecting this mode does not move the arms.`:
    'Individual TCP targets. The object remains supported in simulation; select Shared object to prepare both grasps.';
}
$('control-mode').onchange=()=>{ $('mode-confirmed').checked=false; if(state)render(state); };
$('refine').onchange=async()=>{
  const reply=await action('set_refinement',{refinement:$('refine').checked?'trajopt':'none'});
  if(reply)changed();
};
$('mode-confirmed').onchange=()=>state&&render(state);
$('apply-mode').onclick=async()=>{
  const wanted=$('control-mode').value;
  const reply=await action('set_mode',{mode:wanted,
    empty_hands_confirmed:!!state.shared_object?.virtual&&wanted==='shared'&&$('mode-confirmed').checked,
    virtual_release_confirmed:!!state.shared_object?.virtual&&wanted==='independent'&&$('mode-confirmed').checked,
    grasps_confirmed:wanted==='shared'&&$('mode-confirmed').checked,
    support_confirmed:wanted==='independent'&&$('mode-confirmed').checked});
  if(reply)changed();
};
$('plan-grasps').onclick=async()=>{
  for(const [id,c] of controls) {
    c.kind.value='pose';c.frame.value=c.previousFrame='world';c.selected.checked=true;
    fill(state.arms[id],c);
    state.shared_object.grasp_targets[id].forEach((v,i)=>{c.inputs[i].value=v.toFixed(3);});
  }
  const reply=await action('plan_grasps');
  if(reply){dirty=false;currentPreview=null;$('reviewed').checked=false;}
};
function setObject(group, pose) {
  group.position.set(...pose.p);group.quaternion.set(...pose.q);group.visible=true;
}
function showFrame(fraction) {
  if (!preview.length || dirty) return;
  const index=Math.min(preview.length-1,Math.round(fraction*(preview.length-1)));
  const f=preview[index];
  for (const [a,pose] of Object.entries(f.arms)) setPoses(robots.get(a).preview,pose.geoms);
  if (objectMeshes && f.object) setObject(objectMeshes.preview,f.object);
  $('preview-time').textContent=`${f.t.toFixed(2)} / ${state.duration.toFixed(2)} s`;
}
function render(s) {
  const active=['staging','ready','executing'].includes(s.pair.lifecycle);
  const blocked=active || s.planning || !!s.pair.cancellation_pending.length;
  const fresh=s.connected && !requesting;
  $('refine').disabled=!fresh||blocked;
  if(!requesting)$('refine').checked=s.refinement==='trajopt';
  gizmo.enabled=fresh && !blocked && !!selectedDraft();
  gizmo.visible=gizmo.enabled;
  for (const el of [$('drag-target'),$('mode-t'),$('mode-r'),$('object-sync'),...objectInputs]) el.disabled=!fresh||blocked;
  $('drag-hint').textContent=sharedMode(s)?'Drag the object; both TCPs follow fixed grasps. Plan to preview.':selectedDraft()?'Drag edits the selected TCP target. Plan to preview.':'Select a moving arm with a TCP pose target to drag.';
  if (s.shared_object) {
    const o=s.shared_object;
    setObject(objectMeshes.measured,o.pose);
    $('object-heading').textContent=o.virtual?'Virtual plate · empty Hands':'Shared object · fixed grasps';
    $('object-sync').textContent=o.virtual?'Use current virtual pose':'Use measured object pose';
    $('object-detail').textContent=`${o.virtual?'Virtual pose inferred from TCPs; real arms move.':o.paused?'Physics paused until both arms are armed.':'Simulated rigid grasps active.'} TCP agreement error: ${o.closure_mm.toFixed(3)} mm / ${o.closure_deg.toFixed(3)}°. Object target is in shared world coordinates.`;
    if (!sharedMode(s)) { objectMeshes.target.visible=false; objectMeshes.preview.visible=false; }
  }
  renderMode(s, fresh, blocked);
  const hardware=s.backend==='remote_rt';
  $('backend-label').textContent=hardware?'REAL HARDWARE · experimental dual-arm control':'Shared simulation';
  $('connection').textContent=s.connected?(hardware?'Hardware connected':'Simulation connected'):'Feedback stale';
  $('connection').className=`status ${s.connected?'live':'fault'}`;
  $('phase').textContent=active?s.pair.lifecycle:s.status;
  $('plan').disabled=!fresh || blocked || (sharedMode(s) && !s.mode_settled);
  $('reviewed').disabled=!s.preview_id || dirty || active;
  $('stage').disabled=!fresh || s.status!=='preview' || dirty || !$('reviewed').checked || !Object.values(s.arms).every(a=>a.armed&&a.ready);
  $('execute').disabled=!fresh || s.pair.lifecycle!=='ready' || dirty;
  $('error').textContent=uiError || s.error;
  $('timing').textContent=s.timing;
  const skew=s.pair.pair_id===s.preview_id?s.pair.release_skew_sec:null;
  $('pair-detail').textContent=s.pair.cancellation_pending.length?`Waiting for cancellation: ${s.pair.cancellation_pending.join(', ')}`:
    skew!==null?`Release send gap: ${(skew*1000).toFixed(3)} ms. Both member results are required.`:
    sharedMode(s)?'Arm both controllers before staging. Both grasps remain fixed throughout the move.':
    'Arm both controllers before staging. Unselected arms hold their measured position.';
  $('instruction').textContent = s.planning ? 'Planning both arms in the shared scene…'
    : s.pair.lifecycle === 'executing' ? 'Executing both paths. Stop both / hold cancels the pair.'
    : dirty && s.preview_id ? 'Targets changed or page reloaded. Stop any staged pair and re-plan.'
    : s.pair.lifecycle === 'ready' ? 'Both controllers accepted the plan. Ready to execute.'
    : s.status === 'preview' ? 'Review the green motion and amber targets, then stage the pair.'
    : s.status === 'complete' ? 'Both controllers completed. Set targets for a new plan.'
    : sharedMode(s)?'Set the object target, then plan both arms.':'Set targets, then plan both paths.';
  for (const [id,a] of Object.entries(s.arms)) {
    const c=controls.get(id);
    const health=s.telemetry?.[id], pending=!!health?.authority_pending, hand=health?.hand;
    c.badge.textContent=pending?'Awaiting authority ACK':a.fault_reason?'Faulted':a.armed?(a.executing?'Executing':'Armed'):'Disarmed';
    c.arm.disabled=!fresh || a.armed || blocked || pending;
    c.disarm.disabled=!a.armed && !pending && !a.fault_reason;
    for (const input of [c.selected,c.kind,c.frame,c.sync,c.drag,c.width,c.setHand,...c.inputs]) input.disabled=!fresh||blocked;
    if (c.kind.value==='joint') c.frame.disabled=true;
    if (sharedMode(s)) for (const input of [c.selected,c.kind,c.frame,c.sync,c.width,c.setHand,...c.inputs]) input.disabled=true;
    if (!a.armed || !a.ready) c.width.disabled=c.setHand.disabled=true;
    if(hand && (hand.busy || !hand.active_stop))c.setHand.disabled=true;
    for (const group of c.groups) group.hidden=sharedMode(s);
    c.measured.textContent=`Measured opening: ${(a.finger_m*2000).toFixed(1)} mm. TCP: ${a.world_pose.slice(0,3).map(v=>v.toFixed(1)).join(', ')} mm in world.`;
    if(a.fault_reason)c.measured.textContent+=` Fault: ${a.fault_reason}. Disarm before retrying.`;
    if(hand?.last_move?.ok===false)c.measured.textContent+=` Hand: ${hand.last_move.reason || 'move failed'}.`;
    setPoses(robots.get(id).measured,a.visual.geoms);
  }
}
$('plan').onclick = async () => {
  if (sharedMode(state)) {
    if (objectInputs.some(i=>!i.reportValidity())) return;
    const reply=await action('plan',{object_target:objectInputs.map(i=>Number(i.value))});
    if (reply) { dirty=false;currentPreview=null;$('reviewed').checked=false; }
    return;
  }
  const t = targets();
  if (!t) return;
  const reply = await action('plan', {
    targets: t,
    active_arms: [...controls].filter(([, c]) => c.selected.checked).map(([id]) => id),
  });
  if (reply) {
    dirty = false;
    currentPreview = null;
    $('reviewed').checked = false;
  }
};
$('stage').onclick = () => action('stage', {
  pair_id: state.preview_id, reviewed: $('reviewed').checked,
});
$('execute').onclick = () => action('execute', {pair_id: state.preview_id});
$('stop').onclick = () => { changed(); action('cancel'); };
$('reviewed').onchange = () => state && render(state);
$('play').onclick = () => {
  playing = !playing;
  playbackStart = performance.now() - Number($('scrub').value) * state.duration * 1000;
  $('play').textContent = playing ? 'Pause preview' : 'Play preview';
};
$('scrub').oninput = () => {
  playing = false;
  $('play').textContent = 'Play preview';
  showFrame(Number($('scrub').value));
};
function animate(now) {
  if (playing && state?.duration) {
    const t = (now - playbackStart) / (state.duration * 1000);
    $('scrub').value = Math.min(t, 1);
    showFrame(Math.min(t, 1));
    if (t >= 1) {
      playing = false;
      $('play').textContent = 'Play preview';
    }
  }
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);

async function poll() {
  try {
    const incoming = await getJSON('/state');
    if (state && incoming.session !== state.session) {
      location.reload();
      return;
    }
    // An earlier GET cannot roll back the revision acknowledged by a POST.
    if (requesting || (state && incoming.revision < state.revision)) return;
    state = incoming;
    if (!built) await build(state);
    if (modeRevision !== state.mode_revision) resetMode(state);
    if (state.preview_id !== currentPreview) {
      currentPreview = state.preview_id;
      preview = [];
      playing = false;
      $('reviewed').checked = false;
      if (currentPreview) {
        const p = await getJSON('/preview');
        if (p.pair_id === currentPreview) {
          preview = p.frames;
          // Reloaded/edited fields are not the request that produced this plan.
          if (!dirty) {
            for (const [a, pose] of Object.entries(preview.at(-1).arms)) {
              setPoses(robots.get(a).target, pose.geoms);
            }
            showFrame(0);
          }
        }
      } else {
        if(objectMeshes)objectMeshes.preview.visible=false;
        for (const r of robots.values()) {
          r.target.group.visible = false;
          r.preview.group.visible = false;
        }
      }
      $('play').disabled = $('scrub').disabled = !preview.length || dirty;
      $('play').textContent = 'Play preview';
      $('scrub').value = 0;
      if (!preview.length || dirty) $('preview-time').textContent = 'No preview';
    }
    render(state);
  } catch (error) {
    uiError = error.message;
    if (built && state) render({...state, connected:false});
    $('error').textContent = uiError;
    $('connection').textContent = 'Disconnected';
    $('connection').className = 'status fault';
    for (const id of ['plan', 'stage', 'execute']) $(id).disabled = true;
  } finally {
    setTimeout(poll, 150);
  }
}
poll();
