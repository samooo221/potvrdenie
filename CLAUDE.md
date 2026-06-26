# CLAUDE.md — potvrdenie extraction pipeline

Project memory for Claude Code. Read this before doing anything in the repo.

## Goal
Extract data from a scanned, hand-filled Slovak income-tax form
(Potvrdenie o zdaniteľných príjmoch, form POT395) into validated JSON.
This is a **human-in-the-loop assist tool**, not unattended automation. The
target audience is a government IT department (Finančná správa), so correctness
and data handling matter more than throughput.

## NON-NEGOTIABLE CONSTRAINTS
- **Synthetic data only.** Never use, store, commit, or test against a real
  taxpayer's form. All sample images and JSON in this repo are fabricated.
  Real rodné číslo / income data must never enter the codebase or git history.
- **The model never gets the last word.** Every extracted record is checked by
  deterministic validation (form arithmetic + format rules) and anything that
  fails is flagged for a human. Do not "fix" failing records by trusting the LLM.
- **Extraction is deterministic.** LLM calls for structuring use temperature
  ~0.1 and a GBNF grammar (or json_schema) that forces valid output shape.

## The two GPU lanes (this is a mixed NVIDIA+AMD box — keep them separate)
The deploy target ("the rack") has 1x GTX 1050 Ti (4GB, Pascal) + 5x RX 580
(4GB, Polaris/gfx803), a Celeron G3930, Ubuntu Server 24.04.

- **Vision/OCR lane → CUDA → the NVIDIA card.** gfx803 has no modern ROCm, so
  all PyTorch/PaddleOCR GPU work runs on NVIDIA. CUDA physically cannot see the
  AMD cards, so it lands on the 1050 Ti automatically.
- **Structuring LLM lane → Vulkan/RADV → the RX 580s.** llama.cpp built with
  GGML_VULKAN=ON. The structuring model (Gemma-3-4B QAT, ~2.5GB Q4) fits on a
  SINGLE RX 580, so 4 of the 5 AMD cards are spare headroom for this task.
- The Celeron's Intel iGPU owns the headless console. Three non-overlapping lanes.

## THE TRAP — Vulkan device enumeration (read before touching GPU indices)
Adding the NVIDIA card changes Vulkan device numbering: NVIDIA's Vulkan ICD adds
a device and reshuffles indices, so a hardcoded `--device Vulkan1` may now point
at a different physical card or at the NVIDIA card.

**Fix:** pin the loader to RADV so the NVIDIA card disappears from that process's
Vulkan view and the AMD indices are stable:

    VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json \
    RADV_PERFTEST=nogttspill \
    ~/llama.cpp/build/bin/llama-server -m ~/models/<model>.gguf \
      --device Vulkan0 -ngl 99 -c 4096 --port 8080

Always re-run a device list after any hardware change and confirm indices before
trusting them. (Same failure class as the dead-CMOS-battery saga: a silent,
non-erroring shift, not a crash.)

## Pipeline architecture
1. Preprocess scan (deskew/denoise/threshold) — OpenCV, CUDA lane.
2. Template-align the scan to the blank POT395 and crop each field by known
   bounding box. The layout is fixed, so per-field crops beat whole-page OCR.
3. Recognize per field: digit OCR on numeric riadky, cell-occupancy detection on
   the 1–12 month grids, text OCR only on the few name/address boxes.
4. Structure: **deterministic, not an LLM.** `build_extracted` (crop_ocr.py)
   normalizes the recognized per-field values into the final JSON. The
   architecture target shifted (see NEXT_PHASE.md): **AI lives in perception;
   structuring + all maths are deterministic.** The LLM structuring lane and
   `extract_potvrdenie.py` are deliberately NOT built — deterministic structuring
   of a fixed comb-boxed form is more correct and more auditable. The ONE scoped
   LLM role that remains planned is a Phase-6 text second-check tier (escalated
   free-text fields only, suggestion-only on names, local model) — still deferred.
5. Confidence + validate + escalate (the robustness core, built): every field
   carries a PaddleOCR-derived confidence (`ocr_crop` returns `(text, score)`,
   aggregated min-across-cells). `validate_extracted` re-derives form arithmetic
   (riadok_03 == riadok_01 − riadok_02), rodné-číslo mod-11, datum⇄RČ, page1⇄page2
   on the EXTRACTED values (not ground truth). `escalate` flags a field when
   confidence < class threshold OR it fails a constraint. `disambiguate_extracted`
   may fix a digit field only by varying its LOW-confidence cells to satisfy a
   constraint — a high-confidence value that violates a constraint is FLAGGED,
   never silently corrected. Closed-set text fields (štát/titul/obec) snap to a
   bundled register via `gazetteer.py` (cannot invent a value; flags on miss).
6. Thin FastAPI review UI (server.py): scan → extracted JSON → per-field
   confidence badge + flag reason + validation panel, side by side.

## Conventions / paths
- **Setup:** `bash bootstrap.sh` creates the venv, builds llama.cpp with Vulkan, and
  optionally downloads the model. Set `MODEL_URL` in the script first, or scp the
  Gemma-3-4B QAT `.gguf` into `~/models/` afterward — bootstrap warns if it's missing
  but doesn't abort, so a missing model is a silent failure at llama-server launch time.
- Python venv: `~/potvrdenie/.venv` (activate before running anything).
- llama.cpp: `~/llama.cpp/build/bin/`. Models: `~/models/`.
- llama-server runs on port 8080; the structuring code just POSTs to
  `http://localhost:8080/completion` and is indifferent to which GPU serves it.
- Hardest recognition target: Slovak diacritics in free-text name/address fields
  (á č ď é í ľ ň ô š ť ž). Numbers are the easy, high-value part — prioritise them.

## Known environment gotchas (don't rediscover these)
- Build fails with a missing shader compiler → install `glslc` for this distro.
- **PaddleOCR full pipeline crashes on the laptop CPU** with `ConvertPirAttribute2RuntimeAttribute not support` — this is an oneDNN/PIR executor bug in PaddlePaddle 3.x. **Fix:** use `paddlex.create_model("PP-OCRv6_medium_rec")` directly (recognition-only, no detection). This is fine for pre-cropped fields anyway. On the rack with the NVIDIA GPU the full pipeline works normally.
- **venv requires Python 3.12**, not 3.14 — paddlepaddle has no 3.14 wheel yet. Use `python3.12 -m venv .venv`.
- **paddlepaddle default install is CPU-only.** bootstrap.sh installs the CPU build so
  the script always completes. For the CUDA OCR lane on the rack, swap it:
  `pip install paddlepaddle-gpu==<version>` matched to your CUDA driver — ask Claude
  Code for the right version if unsure.
- File transfer to the rack over flaky SSH: use single-line
  `echo '<b64>' | base64 -d > file`. Multi-line heredocs corrupt.
- The rack's x16 slot previously had a BIOS "force Gen1" quirk; put the 1050 Ti
  there and set the slot to Gen3/Auto. Re-check Above 4G Decoding + IGD-enabled
  after any card change.

## Implementation status
**Built** (see STATUS.md for the honest detail): `crop_ocr.py` (OCR pipeline +
confidence capture + validate/escalate/disambiguate + gazetteer routing),
`field_defs.py` (per-field geometry + type sets + `GAZETTEER_FIELDS`),
`gazetteer.py` + `data/{stat,titul,obce}.txt` (closed-set snap, obce pluggable),
`server.py` (FastAPI review UI with per-field confidence badges + flag reasons),
`eval_handwriting.py` (real per-field accuracy + mean confidence harness),
`generate_samples.py` / `make_handwritten.py` (synthetic data), `align_photo.py`.
Current: 98.4% on the synthetic printed-font set (1143/1162), ~5.8 fields/form
flagged, confidence reliably catches the genuine misreads.

**Deliberately NOT built** (architecture diverged — this is correct, not missing):
`extract_potvrdenie.py` / LLM structuring / GBNF as a required stage. **Deferred:**
the Phase-6 scoped text-second-check LLM tier (needs a local model), Phase-5 ICR,
and the two-GPU Vulkan rack. Validate on REAL pen-filled forms (eval_handwriting.py)
before building any of those — current accuracy is "plumbing works," not "reads
real handwriting." The next-phase plan lives in `NEXT PHASE.md`.

## Definition of done (demo)
A box that boots, processes a synthetic scanned form end-to-end, shows the JSON
and validation flags in the UI, and runs the two lanes concurrently (OCR form
N+1 on NVIDIA while structuring form N on AMD). Report real numbers: N synthetic
forms, per-field accuracy, % flagged.
