# NEXT_PHASE.md — POT395 pipeline, reprioritized after code review

Context: the deterministic plumbing and field geometry are genuinely solid. The
gaps that matter are NOT the ones in the original CLAUDE.md target (LLM lane,
two-GPU rack). They are the things that turn "99% on a self-drawn printed font"
into "trustworthy on real pen-filled forms." This doc reprioritizes around that.

## What's actually good (don't touch / don't regress)
- Field geometry in `field_defs.py` for both pages — measured, not guessed.
- Two-model routing (PP-OCRv6 digits / latin-PP-OCRv5 text), per-cell digit OCR,
  programmatic "/" for rodné číslo, ink-gating of empty cells.
- The gapless comb-text reconstruction trick — the right fix for comb dividers.
- Deterministic validation logic in `validate_gt` (arithmetic, mod-11, datum⇄RČ
  cross-check, page-1⇄page-2 RČ match). The *logic* is correct.
- `normalize_to_canvas` + ORB `align_photo.py`.

## Two findings that reorder everything

### Finding A — confidence is thrown away (highest leverage)
`ocr_crop` / `ocr_text_crop` return only `rec_text` and discard PaddleOCR's
`rec_score`. The single most important robustness primitive we designed —
per-field confidence → threshold → escalate to a human — is therefore not
implemented anywhere. "Never fail silently" is impossible without this. Fix it
first; it unblocks escalation AND constraint-guided disambiguation.

### Finding B — validation runs on the ground truth, not the extraction
In `process_sample`, `validate_gt(gt)` is called on the **ground-truth dict**,
not on the OCR-extracted values. So the checks currently verify that the
synthetic *generator* produced consistent data — the opposite of the production
job, which is to run those same checks on the *extracted* JSON to catch
recognition errors. In `process_live` (real photos) no validation runs at all.
This inversion is why the broken-sample test, while passing, isn't testing the
thing that matters.

### Plus: bless the divergence
Dropping the LLM structuring lane was the RIGHT call, not a regression. For a
fixed comb-boxed form, deterministic structuring is more correct and more
auditable than an LLM. Do NOT add `extract_potvrdenie.py` / GBNF back as a
required stage. Keep an LLM only as an OPTIONAL future robustness layer for
genuinely messy/variable inputs, if real data ever demands it. Update CLAUDE.md's
target architecture to match: AI lives in perception; structuring + maths are
deterministic.

## The reframe on the 99%
It's against a printed font the code drew itself, graded on shared coordinates.
It means the plumbing works. It says almost nothing about real handwriting. No
number is informative until measured on real pen-filled paper. Treat every
current accuracy figure as "plumbing OK," full stop.

---

## Phased plan (in priority order)

### Phase 1 — Confidence capture (keystone)
- Change `ocr_crop`/`ocr_text_crop` to return `(text, score)`.
- Propagate per-cell scores through `_ocr_income_field`, `_ocr_rc_field`,
  `_ocr_digit_comb`, `_ocr_comb_text`; aggregate to a per-field confidence
  (min across cells is the safe aggregator — a field is only as trustworthy as
  its weakest cell).
- For occupancy fields, derive a confidence from distance to the 0.10 cutoff
  (marks near the threshold are the uncertain ones).
- Output per field: `{value, confidence, low_conf_cells:[...]}`.

### Phase 2 — Validate the EXTRACTION + escalation (the product)
- Add `validate_extracted(extracted)` running the SAME checks as `validate_gt`
  but on OCR output. Keep `validate_gt` only as a generator self-test.
- Escalation rule: a field is FLAGGED if confidence < threshold OR it
  participates in a failed constraint. Everything else auto-accepts.
- Surface flags in `server.py`: the review UI shows accept/flag per field with
  the reason. This is the human-in-the-loop story that defines robustness for a
  tax authority — make it visible.

### Phase 3 — Constraint-guided disambiguation
- Pull top-k candidates per digit cell (not just top-1) from PaddleOCR.
- When a field is low-confidence AND fails a constraint (arithmetic, mod-11,
  datum⇄RČ), search candidate combinations for one that satisfies the
  constraint; adopt it only if a cell was low-confidence.
- HARD RULE: if a cell is high-confidence and still violates a constraint,
  FLAG it — never silently "correct." A confident inconsistency may be a real
  error on the taxpayer's form, which is exactly what you must catch.

### Phase 4 — Real-handwriting evaluation (do this early in parallel)
- Collect 20–30 real pen-filled forms: fake data, your own paličkové písmo,
  phone-photographed, run through `align_photo.py`.
- Build a tiny labeling step (type the true values once) so you get REAL
  per-field accuracy + confidence. This recalibrates everything and tells you
  precisely which fields need ICR. Without it, Phase 5 is guesswork.

### Phase 5 — ICR only where Phase 4 proves it's needed
- Likely: digit cells hold up with the per-cell digit path; the gap is letters,
  especially open-vocabulary (child names, titul).
- Options, cheapest first: isolated-character classifier for comb-letter cells
  (trained on real samples — per-cell labels are trivial); a small HTR/VLM only
  for the open-vocabulary fields; titul → recognizer + snap-to-known-title-list.
- Do NOT build speculatively. The Phase-4 numbers choose the tool per field.

## Explicitly DEFER (not now)
- LLM structuring lane / GBNF / `extract_potvrdenie.py` — deterministic wins.
- Two-GPU Vulkan rack + GPU PaddleOCR — a deployment optimization, not a
  capability gap. Irrelevant to whether it reads real forms. Defer until the
  recognition is proven.
- DP-reconciliation feature — strong demo differentiator, but secondary to
  proving recognition on real paper first.

## Ready-to-paste Claude Code prompts
1. "In crop_ocr.py, change ocr_crop and ocr_text_crop to return (text, score)
    from PaddleOCR's rec_score, and propagate per-cell scores through the
    _ocr_* field readers, aggregating to a per-field confidence using the min
    across cells. Don't change any recognition logic or regress accuracy — just
    capture and surface confidence. Show me the per-field confidence on the
    current samples."
2. "Add validate_extracted(extracted) that runs the same checks as validate_gt
    but on the OCR-extracted values, and wire it into process_live and the
    server so real photos get validated. Keep validate_gt as a generator
    self-test only."
3. "In server.py, flag each field in the review UI when confidence < THRESHOLD
    or it's in a failed constraint, with the reason shown. Auto-accept the rest."
4. (after collecting real samples) "Build an eval that runs the real
    handwritten samples through align + ocr_page, compares to my typed labels,
    and reports per-field accuracy AND mean confidence, so I can see which
    fields actually need ICR."
