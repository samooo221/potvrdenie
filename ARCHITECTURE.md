# ARCHITECTURE — POT395 extraction pipeline

How the app works **as built today**, in diagrams. For constraints, status,
roadmap, gotchas and dead-ends, read `CLAUDE.md` (the master doc). This file is
the visual map; `CLAUDE.md` is the operating manual.

**One paragraph:** A scanned/photographed Slovak income-tax form (POT395) comes
in; the page is normalized (and de-skewed if it's a photo); every field is cropped
by fixed coordinates and recognized by a per-type recognizer (digit OCR, ink
occupancy, or text OCR); the recognized values are normalized deterministically,
optionally repaired against form constraints, re-validated by form arithmetic +
format rules, and anything uncertain or inconsistent is **flagged for a human** in
a review UI. **The AI only perceives (OCR). All structuring and maths are
deterministic, and a human — never the model — gets the last word.**

Legend used below: 🟩 built & working · 🟥 not built (target/planned).

---

## 1. End-to-end pipeline

```mermaid
flowchart TD
    A["Scan / phone photo<br/>(page 1, optional page 2)"] --> B["normalize_to_canvas<br/>grayscale → 1241×1755 px"]
    B --> C{"already canvas-sized?"}
    C -- "no — it's a photo" --> D["align_photo.try_align<br/>ORB + RANSAC de-skew to template"]
    C -- "yes" --> E["per-field crop<br/>field_defs.FIELD_BOXES_P1 / P2"]
    D --> E
    E --> F["recognize each field<br/>crop_ocr.ocr_page"]
    F --> G["build_extracted<br/>normalize per field type → field/value dict"]
    G --> H["disambiguate_extracted<br/>constraint-guided repair of LOW-conf digit cells"]
    H --> I["validate_extracted<br/>re-run form arithmetic + format rules on the EXTRACTION"]
    I --> J["escalate<br/>flag if confidence below threshold OR in a failed constraint"]
    J --> K["Review UI — server.py<br/>scan · fields + confidence + ⚑ flags + suggestions"]
    K --> L["Human accepts / corrects<br/>(the model never auto-commits)"]

    classDef built fill:#cdebc5,stroke:#2e7d32,color:#000;
    class A,B,C,D,E,F,G,H,I,J,K,L built;
```

Key point: validation runs on the **extracted** values (to catch OCR errors), not
on the synthetic ground truth. `validate_gt` exists only as a generator self-test.

---

## 2. Per-field recognition routing

`ocr_page` dispatches each field to a recognizer based on its **type set** in
`field_defs.py`. The form is a fixed comb-boxed layout, so per-field crops beat
whole-page OCR.

```mermaid
flowchart TD
    F["field + its box"] --> T{"type set<br/>(field_defs.py)"}
    T -- "MONTH / CHECKBOX" --> O["cell_is_occupied<br/>dark-pixel fraction over 0.10"]
    T -- "income digits" --> N["_ocr_income_field<br/>per-cell PP-OCRv6, ink-gated"]
    T -- "short page-2 income" --> W["_ocr_income_wholebox<br/>padded whole-box + structural decimal"]
    T -- "rodné číslo" --> R["_ocr_rc_field<br/>per-cell digits, slash inserted in code"]
    T -- "digit comb<br/>(datum/rok/psč/DIČ/súpisné)" --> DC["_ocr_digit_comb<br/>per-cell, ink-gated"]
    T -- "name (meno)" --> NM["_ocr_meno_field<br/>multi-width comb reconstruction"]
    T -- "free text<br/>(titul/ulica/obec/štát/…)" --> TX["_ocr_text_field<br/>gapless comb reconstruction"]

    NM --> NS["→ name suggestion tier (§4)"]
    TX --> RT{"closed set?"}
    RT -- "yes" --> GA["→ gazetteer snap (§4)"]
    RT -- "no" --> SO["→ semi-open clean tier (§4)"]

    classDef built fill:#cdebc5,stroke:#2e7d32,color:#000;
    class F,T,O,N,W,R,DC,NM,TX,NS,RT,GA,SO built;
```

**Two recognition models, by field type** (never mixed):
- `PP-OCRv6_medium_rec` — digits (the validated numeric path; ~100% on clean cells).
- `latin_PP-OCRv5_mobile_rec` — Slovak free text, full diacritic dictionary.

**The comb-reconstruction trick** (the breakthrough for free text): crop the
centre of each inked cell and paste them edge-to-edge into one word, then OCR that.
Whole-box OCR reads the printed comb dividers as junk (`Slovenská` →
`Astlolylelnlslkial`). For rodné číslo, the pre-printed `/` is never OCR'd — digits
are read per-cell and the `/` is inserted programmatically.

---

## 3. The robustness core (confidence → repair → validate → escalate)

This is the heart of the product — the human-in-the-loop story for a tax authority.

```mermaid
flowchart TD
    OCR["per-field result<br/>value · confidence (min over cells) · low_conf_cells"] --> EX["build_extracted<br/>→ normalized field/value dict"]
    EX --> DIS{"field has LOW-conf cells<br/>AND fails a constraint?"}
    DIS -- "yes" --> SEARCH["search digit substitutions over<br/>ONLY the low-conf cells"]
    SEARCH --> ADOPT["adopt the assignment that<br/>satisfies the constraint (e.g. mod-11)"]
    DIS -- "no" --> VAL
    ADOPT --> VAL["validate_extracted<br/>re-run ALL form rules on the result"]
    VAL --> ESC{"confidence below class threshold<br/>OR field is in a failed constraint?"}
    ESC -- "yes" --> FLAG["⚑ FLAG for human review<br/>(with the reason)"]
    ESC -- "no" --> ACC["auto-accept"]

    classDef built fill:#cdebc5,stroke:#2e7d32,color:#000;
    classDef flag fill:#f6c2c2,stroke:#c62828,color:#000;
    class OCR,EX,DIS,SEARCH,ADOPT,VAL,ESC,ACC built;
    class FLAG flag;
```

**HARD RULE (non-negotiable):** disambiguation may only vary cells that were
**low-confidence**. A field that is *high-confidence yet violates a constraint* is
**FLAGGED, never silently "corrected"** — a confident inconsistency may be a real
error on the taxpayer's form, which is exactly what a reviewer must see.

**Confidence thresholds** (`crop_ocr.CONF_THRESHOLD`, placeholders until real
handwriting recalibrates them):

| Class | Threshold | Why |
|---|---|---|
| numeric | 0.80 | validated by arithmetic/mod-11, so held to a higher bar |
| text | 0.70 | paličkové písmo is inherently noisier |
| occupancy | 0.30 | only marks near the 0.10 ink cutoff are uncertain |

---

## 4. Deterministic validation constraints

`validate_extracted` (and its self-test twin `validate_gt`) share one body,
`_run_checks`. Each failed check maps back to the field(s) it must flag.

```mermaid
flowchart LR
    V["_run_checks(values)"] --> A["arithmetic<br/>riadok_01 − riadok_02 = riadok_03"]
    V --> B["rodné číslo<br/>format DDDDDD/DDDD + mod-11<br/>(employee + each child)"]
    V --> C["page match<br/>rod_cislo_p2 == rod_cislo"]
    V --> D["datum ⇄ rodné číslo<br/>DOB's YYMMDD = RČ first 6 (month mod 50)"]
    V --> E["DIČ = 10 digits"]
    V --> F["PSČ = 5 digits"]
    V --> G["rok in 2000–2099"]

    classDef built fill:#cdebc5,stroke:#2e7d32,color:#000;
    class V,A,B,C,D,E,F,G built;
```

These fire on the **extracted** values, so a dropped/misread digit that breaks the
form's own arithmetic gets caught and flagged.

---

## 5. Text-field rescue tiers (the only place an LLM lives)

Applies **only** to escalated (low-confidence) **text** fields. Numbers never enter
here — a digit misread loses the information at perception, and numerics are already
covered deterministically.

```mermaid
flowchart TD
    A["escalated text field"] --> B{"field tier<br/>(field_defs sets)"}
    B -- "closed set<br/>titul / obec / štát" --> C["gazetteer fuzzy-match<br/>snap to register, or flag<br/>(CANNOT invent a value)"]
    B -- "semi-open<br/>ulica / obchodné meno" --> D["local LLM CLEAN<br/>adopt ONLY if it re-validates"]
    B -- "personal name<br/>meno / priezvisko / dieťa / vypracoval" --> E["local LLM SUGGESTION only<br/>value is NEVER replaced"]

    C -. "no register hit" .-> F["⚑ flag for human"]
    D -. "fails re-validation" .-> F
    E --> G["shown beside the scan;<br/>human clicks 'použiť' to accept"]

    classDef built fill:#cdebc5,stroke:#2e7d32,color:#000;
    classDef flag fill:#f6c2c2,stroke:#c62828,color:#000;
    class A,B,C,D,E,G built;
    class F flag;
```

Rails: LLM is **LOCAL only** (taxpayer data sovereignty), grammar-constrained to
Slovak block letters, and **degrades gracefully** — if `llama-server` is down,
every text field falls back to gazetteer-plus-flag and the pipeline keeps running
(verified: identical accuracy with no server). `text_second_check.py` talks to
`llama-server` at `LLAMA_URL` (default `http://localhost:8080`) via
`/v1/chat/completions`. **Status:** code-complete but **dormant** until a model is
served — start one with `bash serve_llm.sh`.

---

## 6. Deployment: the two GPU lanes 🟥 (target — not yet built)

Today everything runs on **one laptop, OCR on CPU**. The deploy target ("the rack")
splits work across two physical GPU lanes that never touch each other:

```mermaid
flowchart TB
    subgraph RACK["The rack — Ubuntu 24.04 · Celeron G3930"]
        direction LR
        subgraph NV["CUDA lane → NVIDIA GTX 1050 Ti (4GB)"]
            VIS["Vision / OCR<br/>PaddleOCR + OpenCV preprocess"]
        end
        subgraph AMD["Vulkan / RADV lane → RX 580 ×5 (4GB each)"]
            LLM["Scoped text-cleanup LLM<br/>llama.cpp, GGML_VULKAN=ON<br/>(fits one card; 4 spare)"]
        end
        CON["Intel iGPU → headless console"]
    end

    classDef target fill:#f6c2c2,stroke:#c62828,color:#000;
    class VIS,LLM,CON target;
```

- CUDA physically **cannot see** the AMD cards, so OCR lands on the 1050 Ti
  automatically. gfx803 has no modern ROCm — that's why GPU OCR is CUDA-only.
- **THE TRAP:** adding the NVIDIA card reshuffles Vulkan device indices. Pin the AMD
  lane's loader so NVIDIA disappears from its view and the indices stay stable:
  `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.x86_64.json`. (Full command
  and rationale in `CLAUDE.md`.)
- **Definition of done** runs the lanes concurrently: OCR form N+1 on NVIDIA while
  the text tier cleans form N on AMD.

---

## 7. Module map

```mermaid
flowchart TD
    server["server.py<br/>FastAPI review UI"] --> crop["crop_ocr.py<br/>OCR + normalize + validate + escalate + disambiguate"]
    eval["eval_handwriting.py<br/>real per-field accuracy + mean-confidence harness"] --> crop
    crop --> fd["field_defs.py<br/>field geometry + type sets"]
    crop --> gz["gazetteer.py<br/>closed-set snap (data/*.txt)"]
    crop --> tsc["text_second_check.py<br/>scoped LLM text tier"]
    server --> align["align_photo.py<br/>ORB+RANSAC photo de-skew"]
    eval --> align
    gen["generate_samples.py<br/>spec-driven synthetic forms + ground truth"] --> rh["render_hand.py<br/>paličkové-písmo realism pipeline"]
    mk["make_handwritten.py<br/>one-page handwriting demo"] --> rh
    gen --> fd
    tsc -. "HTTP :8080 (optional)" .-> llama["llama-server (local, dormant)"]

    classDef built fill:#cdebc5,stroke:#2e7d32,color:#000;
    classDef opt fill:#ffe3a3,stroke:#e6a700,color:#000;
    class server,crop,eval,fd,gz,tsc,align,gen,rh,mk built;
    class llama opt;
```

| File | Role |
|---|---|
| `crop_ocr.py` | Core: per-field OCR, `build_extracted`, `validate_extracted`, `escalate`, `disambiguate_extracted`, gazetteer + LLM routing |
| `field_defs.py` | All field bounding boxes (1241×1755 canvas) + the type sets that drive routing |
| `gazetteer.py` | Fuzzy-snap closed-set text to `data/{stat,titul,obce}.txt` (cannot invent a value) |
| `text_second_check.py` | Phase-6 scoped LLM tier (clean / suggest); degrades to flag if no server |
| `align_photo.py` | ORB+RANSAC warp of a phone photo to the canonical canvas |
| `server.py` | FastAPI review UI: scan + fields + confidence badges + flags + name suggestions |
| `eval_handwriting.py` | Honest accuracy harness vs typed labels (the Phase-4 instrument) |
| `generate_samples.py` · `render_hand.py` · `make_handwritten.py` | Synthetic data generation (simulated paličkové písmo) |

---

## 8. Data shapes

Per-field OCR result (from `ocr_page`):

```text
{ "value": <digit/text string  OR  (dark_frac, occupied) for occupancy>,
  "confidence": <float, min over cells>,
  "low_conf_cells": [<indices below CELL_LOW_CONF=0.85>],
  "cells": [(char, score), ...],
  "gazetteer"?: {...}, "suggestion"?: str, "second_check"?: str }
```

`build_extracted` collapses these to the final `{field: value}` dict (occupancy →
bool, income → `"NNNNN.CC"`, rodné číslo → `"DDDDDD/DDDD"`, digit comb → digits,
text → stripped string), which is what validation, escalation and the UI consume.
