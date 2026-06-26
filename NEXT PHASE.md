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

### Plus: bless the divergence (with one scoped LLM exception)
Dropping the LLM structuring lane was the RIGHT call, not a regression. For a
fixed comb-boxed form, deterministic structuring is more correct and more
auditable than an LLM. Do NOT add `extract_potvrdenie.py` / GBNF back as a
*required structuring stage*, and the LLM NEVER touches numbers (amounts, RČ,
DIČ, dates) — those stay deterministic, with arithmetic/mod-11 as the check.

The one place an LLM now earns a defined role is a **text-field second-check
tier** (Phase 6): an escalation-only fallback for the noisy free-text fields,
where there is a language prior to exploit. It is gated behind confidence
(Phase 1) and escalation (Phase 2), always re-validated, suggestion-only on
names, and runs on a small LOCAL model. Update CLAUDE.md's target to: AI lives
in perception; structuring + all maths are deterministic; one local, scoped LLM
tier rescues escalated text fields only.

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

### Phase 6 — Text-field second-check tier (the scoped LLM in robustness)
Applies ONLY to free-text fields, and ONLY on escalation (low confidence from
Phase 1, or a failed gazetteer match). Never to numeric fields — a digit misread
loses the information at perception, so the LLM has nothing to reason from, and
those fields are already covered deterministically. The field tiers:

- **Closed/known sets → gazetteer first, no LLM needed.** štát (tiny list),
  titul (Ing./Mgr./PhD./… finite), obec (Slovak municipality register, ~2,900
  names). Fuzzy-match the noisy OCR against the real list — deterministic,
  auditable, cannot invent a town. This tier alone is a big robustness win and
  can land alongside Phase 2 (it needs no model).
- **Semi-open → gazetteer, then LLM fallback on miss.** ulica, obchodné meno.
  Registers exist but are large/variable; when the fuzzy match fails, the LLM
  cleans the OCR string.
- **Open-vocabulary → LLM SUGGESTION ONLY, never auto-accept.** personal names
  (meno, priezvisko, child names, vypracoval). The name prior helps
  ("Nowák"→"Novák", stitching a split surname), but "correcting" a rare-but-real
  surname into a common one is a worse, invisible error than a flag. Output is
  shown to the human beside the scan as a suggestion; it is never committed.

Rails (non-negotiable):
- Fires only on escalated TEXT fields. Numbers never enter this tier.
- For closed-set fields, constrain LLM output to the known list; for open fields,
  grammar-constrain to a plausible string (GBNF / json_schema).
- Output is RE-VALIDATED (empty name, town not in register → flag, not accept).
- Names are suggestion-only.

Dependency: this lights up the local llama-server lane you deferred — but scoped
to short text-string cleanup, so the model is tiny (1–4B, one RX 580 via Vulkan,
or the laptop) and far lighter than the dropped structuring-LLM idea. It MUST be
local (taxpayer names/addresses — same sovereignty constraint as everything
else). Build it to degrade gracefully: if llama-server is down, fall back to
gazetteer-plus-flag and the pipeline keeps running.

## Explicitly DEFER (not now)
- LLM structuring lane / GBNF as a *required* stage — deterministic wins. (The
  only LLM in scope is the Phase-6 text second-check above.)
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
5. (Phase 6) "Add a text_second_check(field, ocr_string, confidence) for the
    FUZZY_TEXT_FIELDS only, hooked into the FUZZY_TEXT branch of ocr_page so it
    fires only when confidence is low. Tier it: gazetteer fuzzy-match first for
    stat/titul/obec (build the lists), LLM fallback (local llama-server, tiny
    model, grammar-constrained) for ulica/obchodne_meno, and for personal names
    return the LLM output as a human-facing SUGGESTION only — never auto-accept.
    Re-validate every result and flag on failure. Never route numeric fields
    here. Degrade to gazetteer-plus-flag if llama-server is unavailable."
