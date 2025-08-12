// ui/renderer.js
document.addEventListener('DOMContentLoaded', () => {
  const SERVER = "http://127.0.0.1:5001";

  function q(id){ return document.getElementById(id); }
  const els = {
    inputPath: q("inputPath"),
    annotateBtn: q("annotateBtn"),
    resetBtn: q("resetBtn"),
    logsBtn: q("logsBtn"),
    logsPane: q("logsPane"),
    logArea: q("logArea"),
    segmentsBanner: q("segmentsBanner"),
    segCount: q("segCount"),
    searchSeg: q("searchSeg"),
    strength: q("strength"),
    strengthVal: q("strengthVal"),
    groupBody: q("group-body"),
    groupArms: q("group-arms"),
    groupLegs: q("group-legs"),
    groupExt: q("group-extremities"),
    groupOther: q("group-other"),
    fitBtn: q("fitBtn"),
    fillBtn: q("fillBtn"),
    oneBtn: q("oneBtn"),
    viewport: q("viewport"),
    previewImg: q("previewImg"),
    statusBar: q("statusBar"),
  };

  const state = {
    configPath: "configs/default.yaml",
    inputPath: "",
    segments: [],
    controls: { geometry: {}, photo: {} },
    strength: 1.0,
    zoomMode: "fit",
  };

  function log(s){
    const t=new Date().toLocaleTimeString();
    if(els.logArea){ els.logArea.value+=`[${t}] ${s}\n`; els.logArea.scrollTop=els.logArea.scrollHeight; }
    console.log(s);
  }
  function setStatus(s){ if(els.statusBar) els.statusBar.textContent=s; }
  function debounce(f,m){ let h=null; return (...a)=>{ if(h) clearTimeout(h); h=setTimeout(()=>f(...a),m); }; }
  function toFileURL(p){ if(!p) return ""; return `file:///${String(p).replace(/\\/g,"/")}`; }

  async function postProcess(body){
    const payload = JSON.stringify(body);
    log(`POST ${SERVER}/process :: ${payload}`);
    const r = await fetch(`${SERVER}/process`, { method:"POST", headers:{ "Content-Type":"application/json" }, body: payload });
    const t = await r.text();
    log(`JSON bytes=${t.length}`);
    try { return JSON.parse(t); } catch { log(`[WARN] Non-JSON: ${t.slice(0,200)}...`); return {}; }
  }

  // --- Segment helpers ------------------------------------------------------
  function toSegmentsArray(raw){
    const arr = Array.isArray(raw) ? raw : [];
    return arr.map((s, i) => {
      if (typeof s === "string") return { id:s, label:niceLabel(s) };
      if (s && typeof s === "object"){
        const id = String(s.name || s.id || s.label || `seg${i}`);
        const label = String(s.label || s.name || s.id || `seg${i}`);
        return { id, label:niceLabel(label) };
      }
      return { id:`seg${i}`, label:`seg${i}` };
    });
  }
  function niceLabel(s){
    return s.replace(/_/g, " ").replace(/\b(l|r)\b/gi, m => m.toLowerCase()==="l"?"(L)":"(R)")
            .replace(/\b\w/g, ch => ch.toUpperCase());
  }

  function ensureGeom(id){
    if(!state.controls.geometry[id]) state.controls.geometry[id] = { sx:1.0, sy:1.0, rot_deg:0, tx:0, ty:0 };
    return state.controls.geometry[id];
  }

  function applyStrengthControls(){
    const amt = state.strength;
    const geo = {};
    for(const [id, g] of Object.entries(state.controls.geometry)){
      const sx = 1 + (Number(g.sx || 1) - 1) * amt;
      const sy = 1 + (Number(g.sy || 1) - 1) * amt;
      const rot_deg = Number(g.rot_deg || 0) * amt;
      const tx = Number(g.tx || 0) * amt;
      const ty = Number(g.ty || 0) * amt;
      geo[id] = { sx, sy, rot_deg, tx, ty };
    }
    return { geometry: geo, photo: state.controls.photo || {} };
  }

  function sliderRow(segId, label, key, min, max, step, fmt, onChange){
    const id = `${segId}-${key}`;
    const wrap = document.createElement("div"); wrap.className = "slider";
    const lab = document.createElement("label"); lab.htmlFor = id; lab.textContent = label;
    const rng = document.createElement("input"); rng.type = "range"; rng.id = id; rng.min = String(min); rng.max = String(max); rng.step = String(step);
    rng.value = String(onChange.get());
    const val = document.createElement("div"); val.className = "val"; val.textContent = fmt(onChange.get());
    rng.addEventListener("input", e => { const v = parseFloat(e.target.value); onChange.set(v); val.textContent = fmt(v); scheduleWarp(); });
    wrap.append(lab, rng, val);
    return wrap;
  }

  function segmentCard(seg){
    const {id, label} = seg;
    const card = document.createElement("div"); card.className = "segment-card";
    const title = document.createElement("div"); title.className = "segment-title";
    title.innerHTML = `<span>${label}</span>`;
    const actions = document.createElement("div"); actions.className = "seg-actions";
    const resetBtn = document.createElement("button"); resetBtn.className = "btn mini ghost"; resetBtn.textContent = "Reset";
    resetBtn.addEventListener("click", () => { state.controls.geometry[id] = { sx:1, sy:1, rot_deg:0, tx:0, ty:0 }; buildUI(); scheduleWarp(); });
    actions.append(resetBtn);
    title.append(actions);
    card.append(title);

    const g = ensureGeom(id);
    card.append(
      sliderRow(id, "Scale X", "sx", 0.80, 1.20, 0.01, v=>v.toFixed(2), { get:()=>g.sx, set:v=>g.sx=v }),
      sliderRow(id, "Scale Y", "sy", 0.80, 1.20, 0.01, v=>v.toFixed(2), { get:()=>g.sy, set:v=>g.sy=v }),
      sliderRow(id, "Rotate°", "rot_deg", -15, 15, 1, v=>String(v), { get:()=>g.rot_deg, set:v=>g.rot_deg=v }),
      sliderRow(id, "Offset X", "tx", -50, 50, 1, v=>String(v), { get:()=>g.tx, set:v=>g.tx=v }),
      sliderRow(id, "Offset Y", "ty", -50, 50, 1, v=>String(v), { get:()=>g.ty, set:v=>g.ty=v }),
    );
    return card;
  }

  function groupOf(seg){
    const id = seg.id.toLowerCase();
    if (/(torso|hip|pelvis|body)/.test(id)) return "body";
    if (/(head|neck)/.test(id)) return "body";
    if (/(arm|forearm)/.test(id)) return "arms";
    if (/(hand)/.test(id)) return "ext";
    if (/(thigh|shin|calf|leg)/.test(id)) return "legs";
    if (/(foot)/.test(id)) return "ext";
    return "other";
  }

  function buildGroups(segments){
    const q = els.searchSeg.value.trim().toLowerCase();
    const filt = s => !q || s.id.toLowerCase().includes(q) || s.label.toLowerCase().includes(q);

    const buckets = { body:[], arms:[], legs:[], ext:[], other:[] };
    for(const s of segments){ if(!filt(s)) continue; buckets[groupOf(s)].push(s); }

    const mount = (el, list) => { el.innerHTML=""; list.forEach(s=>el.appendChild(segmentCard(s))); };
    mount(els.groupBody, buckets.body);
    mount(els.groupArms, buckets.arms);
    mount(els.groupLegs, buckets.legs);
    mount(els.groupExt, buckets.ext);
    mount(els.groupOther, buckets.other);
  }

  function buildUI(){
    els.segmentsBanner.textContent = state.segments.length ? `Segments: ${state.segments.length}` : "No segments yet.";
    els.segCount.textContent = state.segments.length ? `${state.segments.length}` : "–";
    els.strengthVal.textContent = `${Math.round(state.strength*100)}%`;
    buildGroups(state.segments);
  }

  const scheduleWarp = debounce(async ()=>{
    if(!state.inputPath) return;
    setStatus("Warping…");
    const scaledControls = applyStrengthControls();
    const body = {
      config: state.configPath,
      input: state.inputPath,
      output: null,
      mode: "warp",
      controls: scaledControls
    };
    try{
      const data = await postProcess(body);
      const outAbs = data.output || "";
      if(outAbs) els.previewImg.src = `${toFileURL(outAbs)}?ts=${Date.now()}`;
      setStatus("Warp complete.");
    }catch(e){
      setStatus("Warp failed.");
      log(`WARP ERROR: ${e}`);
    }
  }, 180);

  async function classify(){
    if(!state.inputPath){ setStatus("Select an input image first."); return; }
    setStatus("Annotating…");
    const body = { config: state.configPath, input: state.inputPath, output: null, mode: "classify" };
    try{
      const data = await postProcess(body);
      const outAbs = data.output || "";
      if(outAbs) els.previewImg.src = `${toFileURL(outAbs)}?ts=${Date.now()}`;
      state.segments = toSegmentsArray(data.segments || data.used_segments);
      state.controls = { geometry:{}, photo:{} };
      buildUI();
      setStatus("Ready.");
    }catch(e){
      setStatus("Annotate failed.");
      log(`ANNOTATE ERROR: ${e}`);
    }
  }

  els.annotateBtn?.addEventListener("click", classify);
  els.resetBtn?.addEventListener("click", ()=>{ state.controls={ geometry:{}, photo:{} }; buildUI(); setStatus("Controls reset."); scheduleWarp(); });
  els.logsBtn?.addEventListener("click", ()=>{ els.logsPane.classList.toggle("hidden"); });
  els.inputPath?.addEventListener("change",(e)=>{ state.inputPath=e.target.value.trim(); if(state.inputPath){ els.previewImg.src=toFileURL(state.inputPath); setStatus("Image loaded. Click Annotate."); } });
  els.searchSeg?.addEventListener("input", ()=> buildUI());
  els.strength?.addEventListener("input", (e)=>{ state.strength = parseFloat(e.target.value); els.strengthVal.textContent = `${Math.round(state.strength*100)}%`; scheduleWarp(); });

  function setZoom(mode){
    state.zoomMode = mode;
    els.viewport.classList.remove("fit","fill","one");
    els.viewport.classList.add(mode);
  }
  els.fitBtn?.addEventListener("click", ()=> setZoom("fit"));
  els.fillBtn?.addEventListener("click", ()=> setZoom("fill"));
  els.oneBtn?.addEventListener("click", ()=> setZoom("one"));

  setZoom("fit");
  if(els.inputPath && els.inputPath.value.trim()){ state.inputPath=els.inputPath.value.trim(); els.previewImg.src=toFileURL(state.inputPath); }
  buildUI();
});
