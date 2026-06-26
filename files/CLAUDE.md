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
4. Structure: send the raw field values + labels to local llama-server, which
   emits JSON constrained by a GBNF grammar. (See extract_potvrdenie.py.)
5. Validate: re-derive form arithmetic (e.g. riadok_03 == riadok_01 − riadok_02),
   rodné číslo format + mod-11. Flag failures for human review.
6. Thin FastAPI review UI: scan → extracted JSON → validation flags, side by side.

## Conventions / paths
- Python venv: `~/potvrdenie/.venv` (activate before running anything).
- llama.cpp: `~/llama.cpp/build/bin/`. Models: `~/models/`.
- llama-server runs on port 8080; the structuring code just POSTs to
  `http://localhost:8080/completion` and is indifferent to which GPU serves it.
- Hardest recognition target: Slovak diacritics in free-text name/address fields
  (á č ď é í ľ ň ô š ť ž). Numbers are the easy, high-value part — prioritise them.

## Known environment gotchas (don't rediscover these)
- Build fails with a missing shader compiler → install `glslc` for this distro.
- File transfer to the rack over flaky SSH: use single-line
  `echo '<b64>' | base64 -d > file`. Multi-line heredocs corrupt.
- The rack's x16 slot previously had a BIOS "force Gen1" quirk; put the 1050 Ti
  there and set the slot to Gen3/Auto. Re-check Above 4G Decoding + IGD-enabled
  after any card change.

## Definition of done (demo)
A box that boots, processes a synthetic scanned form end-to-end, shows the JSON
and validation flags in the UI, and runs the two lanes concurrently (OCR form
N+1 on NVIDIA while structuring form N on AMD). Report real numbers: N synthetic
forms, per-field accuracy, % flagged.
