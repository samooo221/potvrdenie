# CLAUDE.md — POT395 extraction pipeline (master doc)

**Project memory for Claude Code. Read this before doing anything in the repo.**
This is the single source of truth: goal, constraints, honest status, roadmap,
hard-won gotchas, and the dead-ends not to repeat. For diagrams of how the app
works, see **`ARCHITECTURE.md`**; for cloud deployment & scaling (sizing for ~100
concurrent users on a GPU host), see **`DEPLOYMENT.md`**, and for the phased rollout
plan + the GPU-vs-CPU-only decision, see **`DEPLOYMENT_PLAN.md`**.

---

## 1. Goal

Extract data from a scanned, hand-filled Slovak income-tax form (*Potvrdenie o
zdaniteľných príjmoch*, form **POT395**) into validated JSON. This is a
**human-in-the-loop assist tool, not unattended automation.** The target audience
is a government IT department (Finančná správa), so **correctness and data
handling matter more than throughput.**

The form is filled in **paličkové písmo** (uppercase block capitals) by hand, or by
typewriter/printer — both are legally accepted fill modes.

---

## 2. NON-NEGOTIABLE CONSTRAINTS

- **Synthetic data only.** Never use, store, commit, or test against a real
  taxpayer's form. All sample images and JSON in this repo are fabricated. Real
  rodné číslo / income data must never enter the codebase or git history.
- **The model never gets the last word.** Every extracted record is checked by
  deterministic validation (form arithmetic + format rules); anything that fails,
  or is low-confidence, is flagged for a human. Do **not** "fix" failing records by
  trusting an LLM.
- **AI lives in perception only; structuring + all maths are deterministic.** OCR
  recognizes fields; `build_extracted` normalizes them with plain Python; form
  arithmetic and format rules validate them. There is **no LLM structuring lane**
  (deliberately dropped — see §5). The only LLM in the system is the scoped
  **text second-check** tier (§4), which touches escalated free-text fields only,
  never numbers, runs **locally**, is grammar-constrained, and is re-validated.

---

## 3. Honest current status

**One-line truth:** A deterministic OCR + rules pipeline for both pages of POT395,
reaching **94.3% (2453/2600)** on 20 synthetic samples it renders itself in
simulated hand-print (paličkové) fonts. The robustness core (confidence,
validation, escalation, gazetteer, scoped LLM second-check) is built; the system
has **never been tested on real handwriting** — that is the one thing that matters
next.

### Phase status

| Phase | Status | Note |
|---|---|---|
| 1 — Confidence capture | ✅ DONE | every field carries a PaddleOCR confidence (min across cells) |
| 2 — Validate the extraction + escalation | ✅ DONE | runs on OCR output (not ground truth); flags low-confidence / constraint-failing fields |
| 3 — Constraint-guided disambiguation | 🟡 PARTIAL | rodné-číslo mod-11 repair done; cross-field arithmetic search NOT implemented |
| Gazetteer tier (closed-set text) | ✅ DONE | štát/titul/obec snap to a register; `obce.txt` is a ~30-name seed, pluggable to the full ~2,900 |
| 6 — Scoped LLM text second-check | ✅ DONE (code) | code-complete + `serve_llm.sh`; **dormant** until a local model runs; safe even with a weak model |
| 4 — Real-handwriting evaluation | ❌ NOT STARTED | `eval_handwriting.py` is built, but **no real pen-filled forms collected** — ZERO real-handwriting numbers exist |
| 5 — ICR (handwriting recognition) | ❌ NOT STARTED | gated on Phase 4's numbers — don't build until they prove which fields need it |
| Two-GPU Vulkan rack / GPU OCR | ❌ NOT STARTED | deployment optimization; the dev box is one laptop, OCR runs on CPU |
| DP-reconciliation demo feature | ❌ NOT STARTED | deferred differentiator |

### The single most important gap: Phase 4

Every accuracy number in this repo is against fonts the code **drew itself** (now
simulated hand-print, earlier printed) — never real pen. Until real pen-filled
forms are photographed and graded, **"94.3%" means "the plumbing works," NOT "it
reads real handwriting."** Phase 4 is blocked only on collecting ~20–30 fake-data
forms in your own paličkové písmo, phone-photographed, run through `align_photo.py`
and `eval_handwriting.py`. Those numbers recalibrate `CONF_THRESHOLD` and decide,
per field, whether ICR (Phase 5) is worth building.

---

## 4. What's built / dormant / deferred

**Built & working:**
- `crop_ocr.py` — OCR pipeline + confidence capture + validate/escalate/disambiguate
  + gazetteer + second-check routing; multi-width name reconstruction.
- `field_defs.py` — field geometry for both pages + type sets
  (`GAZETTEER_FIELDS` / `SEMI_OPEN_FIELDS` / `NAME_FIELDS`, etc.).
- `gazetteer.py` + `data/{stat,titul,obce}.txt` — closed-set snap (obce pluggable).
- `text_second_check.py` — Phase-6 scoped LLM tier.
- `server.py` — FastAPI review UI (confidence badges + flag reasons + name suggestions).
- `eval_handwriting.py` — real per-field accuracy + mean-confidence harness.
- `generate_samples.py` / `render_hand.py` / `make_handwritten.py` — synthetic data.
- `align_photo.py` — ORB+RANSAC photo de-skew.

**Two recognition models, by field type (never mixed):** `PP-OCRv6_medium_rec`
(digits — the validated numeric path) and `latin_PP-OCRv5_mobile_rec` (Slovak free
text with the full diacritic dictionary).

**Phase 6 (text second-check LLM tier) — code-complete, dormant until a model runs.**
`text_second_check.py` talks to a **local** llama-server (`LLAMA_URL`, default
`http://localhost:8080`) via the **`/v1/chat/completions`** endpoint (the chat
template gives instruct models like Gemma better short-text cleanup), grammar-
constrained to Slovak block letters. Fires ONLY on escalated text fields:
gazetteer (closed sets) → LLM CLEAN+adopt for semi-open (ulica / obchodné meno,
re-validated) → LLM SUGGESTION-only for names (never auto-accepted, shown beside the
scan). Numbers never enter. Degrades to gazetteer-plus-flag if the server is down
(verified: identical accuracy with no server). To activate: build llama.cpp + run a
tiny model on :8080 (`bash serve_llm.sh`, smoke-test `python test_phase6.py`).

**Deliberately NOT built / deferred:** `extract_potvrdenie.py` / an LLM *structuring*
lane / GBNF as a *required* stage (dropped on purpose — deterministic structuring of
a fixed comb-boxed form is more correct and more auditable). **Deferred:** Phase-5
ICR and the two-GPU Vulkan rack — validate on REAL pen-filled forms first.

---

## 5. Accuracy numbers (all on synthetic samples)

**Current set — 20 simulated-handwriting samples (`render_hand.py` block-print fonts):**

| Scope | Result | Notes |
|---|---|---|
| Both pages, 20 samples, 130 fields each | **94.3% (2453/2600)** | the honest headline; hand-print is harder than printed |
| Hardest fields: rodné čísla, datum_narodenia, income decimals | **60–75%** | confidence tracks it (low conf, high flagged) → these are the ICR candidates |
| Month grids / checkboxes / occupancy | ~**100%** | occupancy detection robust to messier strokes |
| Gazetteer text (titul/obec/štát + employer) | **95–100%** | snap-to-register holds; misses correctly flagged |
| All 7 constraint breaks + 2 gazetteer-miss | **flagged 9/9** | arithmetic, mod-11, datum⇄RČ, page1⇄page2, PSČ, DIČ, rok, gazetteer — verified in the production path |

**Superseded — earlier printed-monospace runs:** 99.1% (1023/1032) on 8 samples;
98.0% page-1 only. These reflect a clean printed font under mild synthetic noise and
**overstate** real-world capability — treat them as historical.

> Reframe on any high number: it's against a font the code drew itself, graded on
> shared coordinates. It means the *plumbing* works. It says almost nothing about
> real handwriting. No number is informative until measured on real pen-filled paper.

---

## 6. Roadmap (priority order)

The gaps that matter are NOT the original CLAUDE.md target (LLM lane, two-GPU rack).
They are the things that turn "94% on a self-drawn font" into "trustworthy on real
pen-filled forms."

- **Phase 4 — Real-handwriting evaluation (TOP PRIORITY, do this next).** Collect
  20–30 real pen-filled forms (fake data, your own paličkové písmo, phone-photographed),
  align them, type the true values once (`eval_handwriting.py --make-template`), and
  get REAL per-field accuracy + mean confidence. Everything downstream depends on this.
- **Phase 3 (finish) — Constraint-guided disambiguation.** Pull top-k candidates per
  digit cell; for a low-confidence field that fails a constraint, search candidate
  combinations for one that satisfies it; adopt only if a cell was low-confidence.
  HARD RULE: high-confidence + violates constraint → FLAG, never silently correct.
- **Phase 5 — ICR only where Phase 4 proves it's needed.** Likely the gap is letters
  (open-vocabulary child names, titul), not digits. Cheapest-first: isolated-character
  classifier for comb-letter cells → small HTR/VLM only for open-vocabulary fields →
  titul → recognizer + snap-to-known-list. Do NOT build speculatively.
- **Deferred:** two-GPU Vulkan rack + GPU PaddleOCR (a deployment optimization,
  irrelevant to whether it reads real forms); DP-reconciliation demo feature.

---

## 7. Key decisions & dead-ends (don't rediscover these)

**Decisions that stuck:**
- **Two recognition models, by field type.** Digits on PP-OCRv6 (never routed through
  latin); text on latin PP-OCRv5. Mixing regresses the validated numeric path.
- **Gapless comb-text reconstruction** for free text: crop each inked cell's centre,
  paste edge-to-edge, OCR the rebuilt word. Whole-box OCR reads comb dividers as junk.
- **Per-cell digit OCR + programmatic "/"** for rodné číslo (the "/" was systematically misread).
- **Whole-box + structural decimal** for the small page-2 income boxes; per-cell was too
  fragile on faint isolated digits. The 8-digit riadok_12 stays per-cell.
- **Uppercase samples (paličkové) + Slovak-alphabet filter + diacritic-insensitive,
  edit-distance-1 fuzzy match.** Case-insensitive throughout.
- **±1° max synthetic rotation** (row borders bleed into adjacent cells beyond that).
- **`normalize_to_canvas`** (any input size → 1241×1755) + **ORB alignment** for photos.

**Dead-ends / time-wasters:**
- **Guessing coordinates from blind pixel scans** — repeatedly wrong (rodné číslo, name,
  súpisné, datum). The fix every time was to *view* the rendered form. Measure by viewing.
- **Whole-box OCR for rodné číslo** → 0% (the "/").
- **Median filter / autocontrast preprocessing** on faint digits → made it worse.
- **Per-cell OCR on small faint income boxes** → digit-dependent misreads (566→116).
- **TrOCR** was offered, declined, then re-requested as the ICR lane — planned, not built.

---

## 8. The two GPU lanes (mixed NVIDIA+AMD box — keep them separate)

The deploy target ("the rack"): 1× GTX 1050 Ti (4GB, Pascal) + 5× RX 580 (4GB,
Polaris/gfx803), Celeron G3930, Ubuntu Server 24.04. (See `ARCHITECTURE.md` §6.)

- **Vision/OCR lane → CUDA → the NVIDIA card.** gfx803 has no modern ROCm, so all
  PyTorch/PaddleOCR GPU work runs on NVIDIA. CUDA physically cannot see the AMD cards,
  so it lands on the 1050 Ti automatically.
- **Text-cleanup LLM lane → Vulkan/RADV → the RX 580s.** llama.cpp built with
  `GGML_VULKAN=ON`. A tiny model fits on a single RX 580 (4 spare).
- The Celeron's Intel iGPU owns the headless console. Three non-overlapping lanes.

### THE TRAP — Vulkan device enumeration (read before touching GPU indices)

Adding the NVIDIA card changes Vulkan device numbering: NVIDIA's ICD adds a device
and reshuffles indices, so a hardcoded `--device Vulkan1` may now point at a
different card. **Fix:** pin the loader to RADV so the NVIDIA card disappears from
that process's Vulkan view and AMD indices are stable:

    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    RADV_PERFTEST=nogttspill \
    ~/llama.cpp/build/bin/llama-server -m ~/models/<model>.gguf \
      --device Vulkan0 -ngl 99 -c 4096 --port 8080

Always re-run a device list after any hardware change and confirm indices before
trusting them — it's a silent, non-erroring shift, not a crash.

---

## 9. Conventions / paths

- **Setup:** `bash bootstrap.sh` creates the venv (pinned to Python 3.12), installs
  the OCR stack, builds llama.cpp with Vulkan, and optionally fetches a model. Set
  `MODEL_URL` first or drop a `.gguf` into `~/models/` afterward — bootstrap warns
  but doesn't abort, so a missing model is a silent failure at llama-server launch.
- Python venv: `~/potvrdenie/.venv` — **activate before running anything.**
- llama.cpp: `~/llama.cpp/build/bin/`. Models: `~/models/`. OCR model cache:
  `~/.paddlex/official_models/`.
- llama-server listens on :8080; `text_second_check.py` POSTs to
  `http://localhost:8080/v1/chat/completions` and is indifferent to which GPU serves it.
- Hardest recognition target: Slovak diacritics in free-text name/address fields
  (á č ď é í ľ ň ô š ť ž). Numbers are the easy, high-value part — prioritise them.

### Running it

    source .venv/bin/activate
    python crop_ocr.py samples/                       # accuracy harness over the synthetic set
    python crop_ocr.py aligned_p1.png aligned_p2.png  # live mode (no ground truth)
    python server.py                                  # review UI at http://localhost:8000
    python align_photo.py photo.jpg --page 1 --out aligned_p1.png
    python eval_handwriting.py --make-template <dir>  # blank labels file for real-form eval
    python eval_handwriting.py <dir>                  # honest per-field accuracy vs typed labels

---

## 10. Known environment gotchas (don't rediscover these)

- Build fails with a missing shader compiler → install `glslc` for this distro.
- **PaddleOCR full pipeline crashes on the laptop CPU** with
  `ConvertPirAttribute2RuntimeAttribute not support` — an oneDNN/PIR executor bug in
  PaddlePaddle 3.x. **Fix:** use `paddlex.create_model("PP-OCRv6_medium_rec")`
  directly (recognition-only, no detection). Fine for pre-cropped fields anyway. On
  the rack with the NVIDIA GPU the full pipeline works normally.
- **venv requires Python 3.12**, not 3.14 — paddlepaddle has no 3.14 wheel yet.
  `bootstrap.sh` now auto-selects `python3.12`; on the Fedora laptop bare `python3`
  is 3.14 and will build a venv pip can't fill.
- **paddlepaddle default install is CPU-only.** bootstrap installs the CPU build so the
  script always completes. For the CUDA OCR lane on the rack, swap to
  `pip install paddlepaddle-gpu==<version>` matched to your CUDA driver.
- File transfer to the rack over flaky SSH: use single-line
  `echo '<b64>' | base64 -d > file`. Multi-line heredocs corrupt.
- The rack's x16 slot previously had a BIOS "force Gen1" quirk; put the 1050 Ti there
  and set the slot to Gen3/Auto. Re-check Above 4G Decoding + IGD-enabled after any
  card change.

---

## 11. Definition of done (demo)

A box that boots, processes a synthetic scanned form end-to-end, shows the JSON and
validation flags in the UI, and runs the two lanes concurrently (OCR form N+1 on
NVIDIA while the text tier cleans form N on AMD). Report real numbers: N synthetic
forms, per-field accuracy, % flagged.
