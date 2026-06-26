#!/usr/bin/env python3
"""
server.py — FastAPI review UI for the POT395 OCR pipeline.

Usage:
    source .venv/bin/activate
    python server.py
    # then open http://localhost:8000

Accepts one or two PNG uploads (page 1, optionally page 2), runs crop_ocr,
returns extracted fields + validation flags in a browser-based review UI.
"""
import base64
import io
import json
import tempfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from crop_ocr import (
    ocr_page, compare_text_fuzzy, normalize_rc, normalize_to_canvas,
    build_extracted, disambiguate_extracted, validate_extracted, escalate,
    CONF_THRESHOLD, field_class,
)
from align_photo import try_align
from field_defs import (
    FIELD_BOXES_P1, FIELD_BOXES_P2, NUMERIC_FIELDS, MONTH_FIELDS, RC_FIELDS,
    DIGIT_COMB_FIELDS, FUZZY_TEXT_FIELDS, CHECKBOX_FIELDS, CANVAS_W, CANVAS_H,
)


def _prepare_page(raw_bytes, page):
    """Load an uploaded page: de-skew a photo via ORB alignment when needed,
    else fall back to a plain resize. Returns (grayscale PIL image, note_or_None)."""
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(raw_bytes))
    if img.size != (CANVAS_W, CANVAS_H):
        aligned, _note = try_align(img, page)
        if aligned is not None:
            return aligned.convert("L"), None          # successfully de-skewed
    return normalize_to_canvas(img)                     # canvas-sized → noop; else resize+warn

app = FastAPI(title="POT395 Review UI")

HTML = r"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="UTF-8">
<title>POT395 — Kontrola extrakcie</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; font-size: 14px; background: #f5f5f5; }
header { background: #1a3a6c; color: #fff; padding: 12px 20px;
         display: flex; align-items: center; gap: 12px; }
header h1 { font-size: 16px; font-weight: 600; }
.badge { background: #e6c354; color: #1a3a6c; border-radius: 4px;
         padding: 2px 8px; font-size: 12px; font-weight: 700; }
.main { display: flex; height: calc(100vh - 50px); }
.left { flex: 1; overflow: auto; background: #333; padding: 12px;
        display: flex; flex-direction: column; gap: 8px; }
.page-tabs { display: flex; gap: 6px; }
.tab { padding: 4px 12px; background: #555; color: #ccc; border: none;
       cursor: pointer; border-radius: 4px; font-size: 12px; }
.tab.active { background: #e6c354; color: #1a3a6c; font-weight: 700; }
.img-wrap { flex: 1; overflow: auto; }
.img-wrap img { max-width: 100%; display: block; }
.right { width: 420px; overflow-y: auto; background: #fff; border-left: 1px solid #ddd; }
.upload-bar { padding: 12px; border-bottom: 1px solid #eee; background: #fafafa; }
.upload-bar h2 { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #333; }
.upload-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.upload-row label { font-size: 12px; color: #666; }
.upload-row input[type=file] { font-size: 12px; max-width: 160px; }
.btn { padding: 6px 14px; border: none; border-radius: 4px; cursor: pointer;
       font-size: 13px; font-weight: 600; }
.btn-primary { background: #1a3a6c; color: #fff; }
.btn-primary:hover { background: #25508a; }
.btn-ok { background: #2e7d32; color: #fff; }
.btn-flag { background: #c62828; color: #fff; }
.results { padding: 0 12px 12px; }
.section { margin-top: 12px; }
.section-title { font-size: 11px; font-weight: 700; color: #888;
                 text-transform: uppercase; letter-spacing: 0.05em;
                 padding: 6px 0 4px; border-bottom: 1px solid #eee; }
.field-row { display: flex; align-items: center; padding: 3px 0;
             gap: 8px; border-bottom: 1px solid #f0f0f0; }
.field-label { width: 120px; font-size: 12px; color: #555; flex-shrink: 0;
               font-family: monospace; }
.field-val { flex: 1; }
.field-val input { width: 100%; border: 1px solid #ddd; border-radius: 3px;
                   padding: 3px 6px; font-size: 13px; font-family: monospace; }
.field-val input.ok { border-color: #81c784; background: #f1f8e9; }
.field-val input.warn { border-color: #e57373; background: #ffebee; }
.month-grid { display: grid; grid-template-columns: repeat(12, 1fr); gap: 3px;
              padding: 6px 0; }
.month-cell { text-align: center; font-size: 11px; }
.month-cell .m-label { color: #999; font-size: 10px; }
.month-cell .m-val { width: 24px; height: 24px; border: 1px solid #ddd;
                     border-radius: 3px; display: flex; align-items: center;
                     justify-content: center; font-size: 14px; font-weight: 700;
                     cursor: pointer; user-select: none; margin: 0 auto; }
.month-cell .m-val.checked { background: #1a3a6c; color: #fff; border-color: #1a3a6c; }
.month-cell .m-val.unchecked { background: #fff; color: #ccc; }
.validation { margin-top: 12px; padding: 8px 10px; border-radius: 6px; font-size: 12px; }
.validation.ok-box { background: #e8f5e9; border: 1px solid #a5d6a7; color: #2e7d32; }
.validation.warn-box { background: #ffebee; border: 1px solid #ef9a9a; color: #c62828; }
.val-issue { margin-top: 4px; padding-left: 8px; border-left: 3px solid #ef9a9a; }
.action-bar { padding: 10px 12px; border-top: 1px solid #eee; background: #fafafa;
              display: flex; gap: 8px; }
.status { padding: 8px 12px; font-size: 12px; color: #555; font-style: italic; }
#spinner { display: none; }
</style>
</head>
<body>
<header>
  <h1>POT395 — Kontrola extrakcie príjmov</h1>
  <span class="badge">SYNTETICKÉ DÁTA</span>
</header>
<div class="main">
  <div class="left" id="leftPane">
    <div class="page-tabs">
      <button class="tab active" id="tabP1" onclick="showPage(1)">Strana 1</button>
      <button class="tab" id="tabP2" onclick="showPage(2)">Strana 2</button>
    </div>
    <div class="img-wrap">
      <img id="imgP1" src="" alt="Nahrajte formulár" style="display:none">
      <img id="imgP2" src="" alt="" style="display:none">
      <div id="noImg" style="color:#888;padding:40px;text-align:center">
        Nahrajte sken formulára a kliknite „Extrahovať"
      </div>
    </div>
  </div>
  <div class="right">
    <div class="upload-bar">
      <h2>Nahratie formulára</h2>
      <div class="upload-row">
        <label>Strana 1:</label>
        <input type="file" id="fileP1" accept="image/*">
        <label>Strana 2:</label>
        <input type="file" id="fileP2" accept="image/*">
      </div>
      <div style="margin-top:8px;display:flex;gap:8px;align-items:center">
        <button class="btn btn-primary" onclick="doExtract()">&#128269; Extrahovať</button>
        <span id="spinner">⏳ Spracovávam…</span>
      </div>
    </div>
    <div class="status" id="statusMsg">Žiaden formulár neextrahovaný.</div>
    <div class="results" id="results" style="display:none">
      <div class="section">
        <div class="section-title">Identifikácia</div>
        <div id="identFields"></div>
      </div>
      <div class="section">
        <div class="section-title">Zamestnanec a adresa</div>
        <div id="employeeFields"></div>
      </div>
      <div class="section">
        <div class="section-title">Príjmy (€)</div>
        <div id="incomeFields"></div>
      </div>
      <div class="section">
        <div class="section-title">Strana 2 — pokračovanie príjmov (€)</div>
        <div id="p2incomeFields"></div>
        <div id="p2monthGrids"></div>
      </div>
      <div class="section">
        <div class="section-title">Daňový bonus na deti</div>
        <div id="bonusFields"></div>
      </div>
      <div class="section">
        <div class="section-title">Zamestnávateľ (III. oddiel)</div>
        <div id="employerFields"></div>
      </div>
      <div id="validationBox"></div>
      <div class="action-bar">
        <button class="btn btn-ok" onclick="doApprove()">✓ Schváliť</button>
        <button class="btn btn-flag" onclick="doFlag()">⚠ Označiť na kontrolu</button>
      </div>
    </div>
  </div>
</div>
<script>
const INCOME_FIELDS = [
  "riadok_01","riadok_01a","riadok_01b",
  "riadok_02","riadok_02a","riadok_02b",
  "riadok_03","riadok_04","riadok_05",
  "riadok_06","riadok_07","riadok_08","riadok_08a","riadok_09"
];
// [key, label, type]  type: text | check
const IDENT_FIELDS = [
  ["rod_cislo","Rodné číslo","text"],
  ["datum_narodenia","Dátum narodenia","text"],
  ["rok","Rok","text"],
  ["oprava","Opravné","check"],
];
const EMPLOYEE_FIELDS = [
  ["meno_zamestnanca","Meno a priezvisko","text"],
  ["titul","Titul","text"],
  ["ulica","Ulica","text"],
  ["supisne_cislo","Súpisné č.","text"],
  ["psc","PSČ","text"],
  ["obec","Obec","text"],
  ["stat","Štát","text"],
  ["danovnik_obmedzena","Obmedzená daň. povinnosť","check"],
];
const P2_INCOME_FIELDS = ["p2_riadok_08a","p2_riadok_09","p2_riadok_10","p2_riadok_11","p2_riadok_12"];
const EMPLOYER_FIELDS = [
  ["zam_dic","DIČ","text"],
  ["zam_obchodne_meno","Obchodné meno","text"],
  ["zam_priezvisko","Priezvisko","text"],
  ["zam_meno","Meno","text"],
  ["zam_titul","Titul","text"],
  ["zam_ulica","Ulica","text"],
  ["zam_supisne_cislo","Súpisné č.","text"],
  ["zam_psc","PSČ","text"],
  ["zam_obec","Obec","text"],
  ["zam_stat","Štát","text"],
  ["vypracoval","Vypracoval(a)","text"],
  ["potvrdenie_datum","Dátum potvrdenia","text"],
];
const BONUS_GRIDS = ["r10","r13"];   // continuation-income month grids
const MONTH_LABELS = ["I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"];
let currentData = null;
let pageImgs = {1: null, 2: null};

function showPage(n) {
  document.getElementById("imgP1").style.display = n===1 ? "block" : "none";
  document.getElementById("imgP2").style.display = n===2 ? "block" : "none";
  document.getElementById("tabP1").className = "tab" + (n===1?" active":"");
  document.getElementById("tabP2").className = "tab" + (n===2?" active":"");
}

async function doExtract() {
  const f1 = document.getElementById("fileP1").files[0];
  if (!f1) { alert("Vyberte aspoň stranu 1."); return; }
  const f2 = document.getElementById("fileP2").files[0];
  const fd = new FormData();
  fd.append("page1", f1);
  if (f2) fd.append("page2", f2);
  document.getElementById("spinner").style.display = "inline";
  document.getElementById("statusMsg").textContent = "Prebieha extrakcia…";
  try {
    const r = await fetch("/api/extract", {method:"POST", body: fd});
    const data = await r.json();
    currentData = data;
    renderResults(data);
    if (data.page1_b64) {
      document.getElementById("imgP1").src = "data:image/png;base64," + data.page1_b64;
      document.getElementById("imgP1").style.display = "block";
      document.getElementById("noImg").style.display = "none";
      pageImgs[1] = data.page1_b64;
    }
    if (data.page2_b64) {
      document.getElementById("imgP2").src = "data:image/png;base64," + data.page2_b64;
      pageImgs[2] = data.page2_b64;
    }
    const n = Object.keys(data.fields).length;
    const vOk = data.validation_issues.length === 0;
    document.getElementById("statusMsg").textContent =
      `Extrahovaných ${n} polí. Validácia: ${vOk ? "OK ✓" : "ZLYHALA ✗"}`;
  } catch(e) {
    document.getElementById("statusMsg").textContent = "Chyba: " + e;
  }
  document.getElementById("spinner").style.display = "none";
}

function confBadge(k, data) {
  // Per-field confidence badge with the flag reason on hover. ⚑ marks a field
  // escalated for human review (low confidence or a failed constraint).
  const c = (data.confidences || {})[k];
  if (c === undefined) return "";
  const thr = (data.thresholds || {})[k];
  const flagged = (data.flagged_fields || []).includes(k);
  const reasons = (data.field_reasons || {})[k] || [];
  const title = (reasons.length ? reasons.join(" • ") : "auto-accepted").replace(/"/g, "'");
  const color = flagged ? "#c62828" : "#2e7d32";
  return `<span class="conf-badge" title="${title}"
    style="margin-left:6px;font-size:11px;color:${color};white-space:nowrap;cursor:help">`
    + `${flagged ? "⚑ " : ""}${Math.round(c*100)}%</span>`;
}

function renderFieldList(containerId, spec, data) {
  const box = document.getElementById(containerId);
  box.innerHTML = "";
  for (const [k, label, type] of spec) {
    const flagged = (data.flagged_fields || []).includes(k);
    const row = document.createElement("div");
    row.className = "field-row";
    if (type === "check") {
      const on = data.fields[k] === true;
      row.innerHTML = `<span class="field-label">${label}</span>
        <div class="field-val"><div class="m-val ${on?'checked':'unchecked'}"
          id="f_${k}" onclick="toggleCheck('${k}')"
          style="margin:0">${on?"×":""}</div>${confBadge(k,data)}</div>`;
    } else {
      const v = data.fields[k] || "";
      row.innerHTML = `<span class="field-label">${label}</span>
        <div class="field-val"><input id="f_${k}" type="text" value="${v}"
          class="${flagged?'warn':'ok'}">${confBadge(k,data)}</div>`;
    }
    box.appendChild(row);
  }
}

function monthGridEl(prefix, label, data) {
  const wrap = document.createElement("div");
  wrap.style.cssText = "margin:4px 0 8px;";
  const vsetky = data.fields[prefix+"_mesiac_vsetky"] === true;
  let grid = `<div class="m-label" style="font-size:11px;margin-bottom:2px">${label}${vsetky?" — [1-12 ×]":""}</div><div class="month-grid">`;
  for (let i=1;i<=12;i++){
    const ck = data.fields[prefix+"_mesiac_"+String(i).padStart(2,"0")]===true;
    grid += `<div class="month-cell"><div class="m-label">${MONTH_LABELS[i-1]}</div>`
          + `<div class="m-val ${ck?'checked':'unchecked'}">${ck?"×":""}</div></div>`;
  }
  wrap.innerHTML = grid + "</div>";
  return wrap;
}

function renderResults(data) {
  document.getElementById("results").style.display = "block";
  renderFieldList("identFields", IDENT_FIELDS, data);
  renderFieldList("employeeFields", EMPLOYEE_FIELDS, data);

  const inc = document.getElementById("incomeFields");
  inc.innerHTML = "";
  for (const k of INCOME_FIELDS) {
    const v = data.fields[k] || "";
    const flagged = (data.flagged_fields || []).includes(k);
    const row = document.createElement("div");
    row.className = "field-row";
    row.innerHTML = `<span class="field-label">${k}</span>
      <div class="field-val"><input id="f_${k}" type="text" value="${v}"
        class="${flagged?'warn':'ok'}">${confBadge(k,data)}</div>`;
    inc.appendChild(row);
  }

  // page-2 continuation income
  const p2inc = document.getElementById("p2incomeFields");
  p2inc.innerHTML = "";
  for (const k of P2_INCOME_FIELDS) {
    const v = data.fields[k] || "";
    const flagged = (data.flagged_fields || []).includes(k);
    const row = document.createElement("div");
    row.className = "field-row";
    row.innerHTML = `<span class="field-label">${k.replace("p2_","")}</span>
      <div class="field-val"><input id="f_${k}" type="text" value="${v}"
        class="${flagged?'warn':'ok'}">${confBadge(k,data)}</div>`;
    p2inc.appendChild(row);
  }
  // page-2 income month grids (r10, r13)
  const pg = document.getElementById("p2monthGrids");
  pg.innerHTML = "";
  for (const pref of BONUS_GRIDS) pg.appendChild(monthGridEl(pref, "Mesiace " + pref, data));

  // child tax bonus (up to 4 children)
  const bf = document.getElementById("bonusFields");
  bf.innerHTML = "";
  for (let c=1; c<=4; c++) {
    const meno = data.fields["dieta"+c+"_meno"] || "";
    const rc = data.fields["dieta"+c+"_rod_cislo"] || "";
    if (!meno && !rc) continue;
    const wrap = document.createElement("div");
    wrap.innerHTML = `<div class="field-row"><span class="field-label">dieťa ${c}</span>
      <div class="field-val"><input type="text" value="${meno}" placeholder="meno"></div>
      <div class="field-val"><input type="text" value="${rc}" placeholder="rodné číslo"></div></div>`;
    wrap.appendChild(monthGridEl("dieta"+c, "Bonus mesiace", data));
    bf.appendChild(wrap);
  }

  renderFieldList("employerFields", EMPLOYER_FIELDS, data);

  const issues = data.validation_issues || [];
  const flaggedN = (data.flagged_fields || []).length;
  const dis = data.disambiguated || [];
  const vbox = document.getElementById("validationBox");
  let html = "";
  if (dis.length) {
    const items = dis.map(d => `<div class="val-issue">↻ ${d}</div>`).join("");
    html += `<div class="validation" style="background:#e3f2fd;border-color:#64b5f6">`
          + `Automaticky zosúladené (nízka istota + ohraničenie):${items}</div>`;
  }
  if (issues.length === 0) {
    html += `<div class="validation ok-box">✓ Validácia extrakcie: aritmetika OK, rodné číslo OK</div>`;
  } else {
    const items = issues.map(i => `<div class="val-issue">⚠ ${i}</div>`).join("");
    html += `<div class="validation warn-box">✗ Validácia extrakcie zlyhala:${items}</div>`;
  }
  html += flaggedN
    ? `<div class="validation warn-box">⚑ ${flaggedN} pol'í označených na kontrolu (nízka istota alebo zlyhané ohraničenie). Najdite ⚑ pri poli; dôvod je v bubline.</div>`
    : `<div class="validation ok-box">✓ Žiadne pole nevyžaduje kontrolu — všetko automaticky prijaté.</div>`;
  vbox.innerHTML = html;
}

function toggleMonth(i) {
  const el = document.getElementById("mc_" + i);
  const on = !el.classList.contains("checked");
  el.className = "m-val " + (on ? "checked" : "unchecked");
  el.textContent = on ? "×" : "";
}

function toggleCheck(k) {
  const el = document.getElementById("f_" + k);
  const on = !el.classList.contains("checked");
  el.className = "m-val " + (on ? "checked" : "unchecked");
  el.textContent = on ? "×" : "";
}

function collectFields() {
  const out = {};
  for (const [k, label, type] of [...IDENT_FIELDS, ...EMPLOYEE_FIELDS]) {
    const el = document.getElementById("f_" + k);
    if (!el) continue;
    out[k] = type === "check" ? el.classList.contains("checked") : el.value;
  }
  for (const k of INCOME_FIELDS) {
    out[k] = document.getElementById("f_" + k)?.value || "";
  }
  for (let i=1; i<=12; i++) {
    const key = "mesiac_" + String(i).padStart(2,"0");
    out[key] = document.getElementById("mc_"+i).classList.contains("checked");
  }
  return out;
}

async function doApprove() {
  const fields = collectFields();
  alert("✓ Schválené.\n\n" + JSON.stringify(fields, null, 2));
}

async function doFlag() {
  alert("⚠ Záznam označený na manuálnu kontrolu.");
}
</script>
</body>
</html>
"""


def _img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML


@app.post("/api/extract")
async def extract(
    page1: UploadFile = File(...),
    page2: UploadFile = File(None),
):
    from PIL import Image

    # De-skew phone photos (ORB alignment) and normalize to the 1241×1755 canvas
    # the field crops assume; a plain resize is the fallback when alignment fails.
    img_p1, warn1 = _prepare_page(await page1.read(), 1)
    img_p2, warn2 = _prepare_page(await page2.read(), 2) if page2 else (None, None)
    size_warnings = [w for w in (warn1, warn2) if w]

    raw_p1 = ocr_page(img_p1, FIELD_BOXES_P1)
    raw_p2 = ocr_page(img_p2, FIELD_BOXES_P2) if img_p2 else {}
    raw_all = {**raw_p1, **raw_p2}

    # Normalize OCR output → final values, reconcile low-confidence digit fields
    # against form constraints, then validate the EXTRACTION (not ground truth).
    fields = build_extracted(raw_all)
    fields, dis_log = disambiguate_extracted(raw_all, fields)
    checks = validate_extracted(fields)
    flagged, reasons = escalate(raw_all, fields, checks)

    # Per-field confidence + a flag threshold for the UI badge.
    confidences = {f: round(raw_all[f]["confidence"], 3) for f in raw_all}
    thresholds = {f: CONF_THRESHOLD[field_class(f)] for f in raw_all}

    issues = list(size_warnings) + [c["msg"] for c in checks]

    return JSONResponse({
        "fields": fields,
        "validation_issues": issues,
        "flagged_fields": flagged,
        "field_reasons": reasons,
        "confidences": confidences,
        "thresholds": thresholds,
        "disambiguated": dis_log,
        "page1_b64": _img_to_b64(img_p1),
        "page2_b64": _img_to_b64(img_p2) if img_p2 else None,
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
