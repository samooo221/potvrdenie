#!/usr/bin/env python3
"""
crop_ocr.py — crop per-field regions from POT395 PNGs and run PaddleOCR.

Usage:
    # Synthetic samples directory (sample_NNNN_p1.png + sample_NNNN_p2.png + .json)
    python crop_ocr.py samples/

    # Live aligned photos (no ground truth)
    python crop_ocr.py aligned_p1.png aligned_p2.png
    python crop_ocr.py aligned_p1.png          # page 1 only

Uses paddlex recognition model directly (no detection step) — workaround for
oneDNN/PIR crash in PaddleOCR 3.x detection model on CPU. Pre-cropped fields
don't need detection anyway.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from field_defs import (
    FIELD_BOXES_P1, FIELD_BOXES_P2,
    INCOME_DIGIT_CELLS, _N_DECIMAL,
    NUMERIC_FIELDS, MONTH_FIELDS, TEXT_FIELDS, RC_FIELDS,
    RC_CELLS, _RC_N_LEFT,
    DIGIT_COMB_FIELDS, FUZZY_TEXT_FIELDS, CHECKBOX_FIELDS,
    DIGIT_COMB_CELLS, TEXT_COMB_CELLS, PRIEZVISKO_CELLS, MENO_CELLS,
    WHOLEBOX_INCOME, SLOVAK_ALPHA, CANVAS_W, CANVAS_H,
    GAZETTEER_FIELDS,
)
from gazetteer import gazetteer_match

_SLOVAK_SET = set(SLOVAK_ALPHA)

# --- Confidence / escalation thresholds -------------------------------------
# A single OCR'd cell whose rec_score is below this is "uncertain" and recorded
# in low_conf_cells (and is the only kind of cell constraint-guided
# disambiguation is allowed to alter — see disambiguate_extracted).
CELL_LOW_CONF = 0.85
# Per field-class escalation threshold: a field whose aggregated confidence is
# below its class threshold is FLAGGED for human review. Numeric fields are
# validated by arithmetic/mod-11 so we hold them to a higher bar; free text is
# inherently noisier (paličkové písmo) so a lower bar avoids flagging everything.
# These are placeholders — recalibrate from eval_handwriting.py once real
# pen-filled samples exist.
# Placeholder values, calibrated against the synthetic set so that genuine
# misreads flag while correct fields mostly don't. Recalibrate from
# eval_handwriting.py once real pen-filled samples exist — the occupancy band is
# deliberately narrow (only marks sitting in the 0.07–0.13 dark-fraction
# no-man's-land near the 0.10 cutoff escalate).
CONF_THRESHOLD = {"numeric": 0.80, "text": 0.70, "occupancy": 0.30}


def field_class(field: str) -> str:
    """Coarse class of a field, for picking a confidence threshold."""
    if field in MONTH_FIELDS or field in CHECKBOX_FIELDS:
        return "occupancy"
    if field in TEXT_FIELDS or field in FUZZY_TEXT_FIELDS:
        return "text"
    return "numeric"


_rec_model = None     # PP-OCRv6 — digits (proven 100% on numeric cells)
_text_model = None    # latin_PP-OCRv5 — Slovak free text (full diacritic dictionary)

_CANVAS_RATIO = CANVAS_W / CANVAS_H   # A4 ≈ 0.7071


def normalize_to_canvas(img: Image.Image) -> tuple[Image.Image, str | None]:
    """Grayscale + resize any input to the 1241×1755 canvas the field boxes assume.

    The field crops use fixed pixel coordinates, so an input at a different size
    (e.g. a 2× editor export, or a scan at another DPI) misaligns every crop and
    reads garbage. A full-page form shares the template's A4 aspect ratio, so a
    plain resize is exact. Returns (normalized_img, warning_or_None); the warning
    fires when the input is not A4-proportioned (likely a cropped/rotated photo —
    run align_photo.py first).
    """
    img = img.convert("L")
    warn = None
    if img.size != (CANVAS_W, CANVAS_H):
        ratio = img.width / img.height
        if abs(ratio - _CANVAS_RATIO) / _CANVAS_RATIO > 0.03:
            warn = (f"input {img.width}×{img.height} is not A4-proportioned "
                    f"(ratio {ratio:.3f} vs {_CANVAS_RATIO:.3f}); fields may misalign — "
                    f"for a phone photo run: python align_photo.py <photo> --page 1")
        img = img.resize((CANVAS_W, CANVAS_H), Image.LANCZOS)
    return img, warn


def get_rec_model():
    global _rec_model
    if _rec_model is None:
        from paddlex import create_model
        _rec_model = create_model("PP-OCRv6_medium_rec")
    return _rec_model


def get_text_model():
    """Latin-script recognition model for Slovak free-text fields.

    Its dictionary covers the full Slovak diacritic set (á ä č ď é í ĺ ľ ň ó ô
    ŕ š ť ú ý ž) that PP-OCRv6 cannot emit. Used ONLY for text fields — digits
    stay on PP-OCRv6, which is at 100% and must not be regressed.
    """
    global _text_model
    if _text_model is None:
        from paddlex import create_model
        _text_model = create_model("latin_PP-OCRv5_mobile_rec")
    return _text_model


def cell_is_occupied(pil_crop: Image.Image, cutoff: float = 0.10) -> tuple[float, bool]:
    gray = np.array(pil_crop.convert("L"))
    dark_frac = float(np.sum(gray < 128) / gray.size)
    return dark_frac, dark_frac > cutoff


def ocr_crop(pil_crop: Image.Image) -> tuple[str, float]:
    """OCR a digit crop with PP-OCRv6. Returns (text, rec_score).

    rec_score is PaddleOCR's per-prediction confidence — the keystone primitive
    for the human-in-the-loop escalation path (low score → flag for review).
    Empty result → ("", 0.0).
    """
    rec = get_rec_model()
    arr = np.array(pil_crop.convert("RGB"))
    results = list(rec.predict(arr))
    if not results:
        return "", 0.0
    return results[0].get("rec_text", "").strip(), float(results[0].get("rec_score", 0.0))


def ocr_text_crop(pil_crop: Image.Image) -> tuple[str, float]:
    """OCR a free-text crop with the Latin (Slovak-capable) model.

    Returns (text, rec_score); empty result → ("", 0.0).
    """
    rec = get_text_model()
    arr = np.array(pil_crop.convert("RGB"))
    results = list(rec.predict(arr))
    if not results:
        return "", 0.0
    return results[0].get("rec_text", "").strip(), float(results[0].get("rec_score", 0.0))


def normalize_decimal(s: str) -> str:
    # Canonicalize OCR substitution characters. The decimal separator on the form
    # is a comma (read as "," or fullwidth "，"); treat it as the decimal point.
    s = s.replace("，", ".").replace(",", ".").replace("：", ".").replace(":", ".")
    # Remove internal spaces — OCR sometimes splits "13089.82" as "1308 9.82"
    s = s.replace(" ", "")
    # Extract the last DDDDD.DD pattern (value is right-aligned, so last match wins)
    m = re.findall(r"\d+\.\d{2}", s)
    if m:
        return m[-1]
    # Structural fallback: the form's amounts always have 2 decimal places, so if
    # the separator was lost (faint comma), place it before the last two digits.
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 3:
        return digits[:-2] + "." + digits[-2:]
    return digits


def normalize_rc(s: str) -> str:
    digits_slash = re.sub(r"[^0-9/]", "", s)
    if "/" not in digits_slash and len(digits_slash) == 10:
        digits_slash = digits_slash[:6] + "/" + digits_slash[6:]
    return digits_slash


def _strip_diacritics(s: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def _lev_le1(a: str, b: str) -> bool:
    """True if Levenshtein(a, b) <= 1 — single insert/delete/substitution."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if a == b:
        return True
    # find first mismatch
    i = 0
    while i < min(la, lb) and a[i] == b[i]:
        i += 1
    if la == lb:                     # one substitution
        return a[i + 1:] == b[i + 1:]
    if la < lb:                      # one insertion into a
        return a[i:] == b[i + 1:]
    return a[i + 1:] == b[i:]        # one deletion from a


def _fuzzy_in(word: str, hay: str) -> bool:
    """True if `word` appears in `hay` within edit distance 1 (substring)."""
    if word in hay:
        return True
    n = len(word)
    for L in (n, n - 1, n + 1):      # candidate window lengths
        for s in range(0, len(hay) - L + 1):
            if _lev_le1(word, hay[s:s + L]):
                return True
    return False


def compare_text_fuzzy(gt: str, ocr: str) -> bool:
    """Word-set match after diacritic stripping, tolerating OCR noise per word.

    Comb fields print one letter per cell, so OCR splits words across spaces and
    misreads isolated thin glyphs (l→i/t, ľ→t). Match each GT word against the
    space-stripped OCR allowing one edit (Slovensko≈siovensko, Poľná≈potna) for
    words ≥4 chars; short words use exact/dropped-edge matching to avoid spurious
    hits. A human reviewer reading the scan beside the field absorbs the rest.
    """
    gt_n        = _strip_diacritics(gt)
    ocr_n       = _strip_diacritics(ocr)
    ocr_nospace = ocr_n.replace(" ", "")
    for w in gt_n.split():
        if len(w) < 2:
            continue
        if w in ocr_n or w in ocr_nospace:
            continue
        if w[1:] in ocr_nospace or w[:-1] in ocr_nospace:
            continue
        if len(w) >= 4 and _fuzzy_in(w, ocr_nospace):   # tolerate 1 edit
            continue
        return False
    return True


def _digits_only(s: str) -> str:
    return "".join(c for c in str(s) if c.isdigit())


def compare_field(field: str, gt_val, ocr_raw) -> tuple[bool, str, str]:
    if field in MONTH_FIELDS or field in CHECKBOX_FIELDS:
        dark_frac, occupied = ocr_raw
        match = bool(gt_val) == occupied
        return match, str(gt_val), f"{'X' if occupied else '.'} ({dark_frac*100:.1f}%)"

    gt_str = str(gt_val)

    if field in NUMERIC_FIELDS:
        ocr_norm = normalize_decimal(ocr_raw)
        gt_norm  = normalize_decimal(gt_str)
        return gt_norm == ocr_norm, gt_str, ocr_raw or "(empty)"

    if field in RC_FIELDS:
        ocr_norm = normalize_rc(ocr_raw)
        return gt_str == ocr_norm, gt_str, ocr_raw or "(empty)"

    if field in DIGIT_COMB_FIELDS:
        # exact digit-string match (datum/rok/psc/supisne)
        return _digits_only(gt_str) == _digits_only(ocr_raw), gt_str, ocr_raw or "(empty)"

    # TEXT_FIELDS (name) + FUZZY_TEXT_FIELDS (titul/ulica/obec/stat) — fuzzy match
    match = compare_text_fuzzy(gt_str, ocr_raw or "")
    return match, gt_str, ocr_raw or "(empty)"


def _run_checks(d: dict) -> list[dict]:
    """Deterministic form checks over a value dict. Returns structured issues
    [{"fields": [...], "msg": "..."}] so callers can both display the message and
    map a failed constraint back to the field(s) that must be flagged.

    The SAME body powers two callers with opposite intent: validate_gt (a
    generator self-test on synthetic ground truth) and validate_extracted (the
    production check, run on OCR output to catch recognition errors).
    """
    from decimal import Decimal
    issues: list[dict] = []

    def add(fields, msg):
        issues.append({"fields": list(fields), "msg": msg})

    try:
        r01 = Decimal(d["riadok_01"])
        r02 = Decimal(d["riadok_02"])
        r03 = Decimal(d["riadok_03"])
        if r01 - r02 != r03:
            add(("riadok_01", "riadok_02", "riadok_03"),
                f"arithmetic: r01({r01}) − r02({r02}) = {r01-r02} ≠ r03({r03})")
    except Exception as e:
        add(("riadok_01", "riadok_02", "riadok_03"), f"arithmetic-parse-error: {e}")

    # rodné číslo: format + mod-11 for employee and every (non-empty) child.
    for rcf in sorted(RC_FIELDS):
        rc = d.get(rcf, "")
        if not rc and rcf != "rod_cislo":
            continue                      # blank child row — nothing to validate
        m = re.fullmatch(r"(\d{6})/(\d{4})", rc)
        if not m:
            add((rcf,), f"{rcf} format: {rc!r}")
        elif int(m.group(1) + m.group(2)) % 11 != 0:
            add((rcf,), f"{rcf} mod-11 failed: {rc!r}")

    # Page-2 top rodné číslo must match page-1's (same taxpayer, page matching).
    if d.get("rod_cislo_p2") and d.get("rod_cislo_p2") != d.get("rod_cislo"):
        add(("rod_cislo_p2", "rod_cislo"),
            f"rod_cislo_p2 {d['rod_cislo_p2']!r} ≠ page-1 {d.get('rod_cislo')!r}")

    # DIČ (employer tax ID): exactly 10 digits.
    if d.get("zam_dic"):
        dic = _digits_only(d["zam_dic"])
        if len(dic) != 10:
            add(("zam_dic",), f"zam_dic (need 10 digits): {d['zam_dic']!r}")

    # PSČ: exactly 5 digits
    if "psc" in d:
        psc = _digits_only(d["psc"])
        if len(psc) != 5:
            add(("psc",), f"psc format (need 5 digits): {d['psc']!r}")

    # Rok: plausible assessment year
    if "rok" in d:
        rok = _digits_only(d["rok"])
        year = int(rok) if rok else 0
        # 2-digit suffix means 20YY; 4-digit means full year
        if len(rok) == 2:
            year = 2000 + int(rok)
        if not (2000 <= year <= 2099):
            add(("rok",), f"rok out of range 2000–2099: {d['rok']!r}")

    # Dátum narodenia ⇄ rodné číslo cross-check: DDMMYYYY's YYMMDD must equal
    # the first 6 digits of rod_cislo. Strongest new deterministic check.
    rcm = re.fullmatch(r"(\d{6})/(\d{4})", d.get("rod_cislo", "") or "")
    if "datum_narodenia" in d and rcm:
        dn = _digits_only(d["datum_narodenia"])
        if len(dn) == 8:
            dd, mm, yyyy = dn[0:2], dn[2:4], dn[4:8]
            try:
                from datetime import date
                date(int(yyyy), int(mm), int(dd))   # calendar validity
                rc_yymmdd = rcm.group(1)
                # rod_cislo month may be +50 (female); compare day+year, month mod 50
                rc_yy, rc_mm, rc_dd = rc_yymmdd[0:2], rc_yymmdd[2:4], rc_yymmdd[4:6]
                rc_mm_norm = f"{int(rc_mm) % 50:02d}"
                if (yyyy[2:], mm, dd) != (rc_yy, rc_mm_norm, rc_dd):
                    add(("datum_narodenia", "rod_cislo"),
                        f"datum_narodenia {dd}.{mm}.{yyyy} ≠ rod_cislo {rc_yymmdd}")
            except ValueError:
                add(("datum_narodenia",),
                    f"datum_narodenia invalid date: {d['datum_narodenia']!r}")

    return issues


def validate_gt(gt: dict) -> list[str]:
    """Generator self-test: confirm the SYNTHETIC ground truth is internally
    consistent. NOT the production check — that is validate_extracted."""
    return [c["msg"] for c in _run_checks(gt)]


def validate_extracted(extracted: dict) -> list[dict]:
    """Production check: run the deterministic form rules on OCR-EXTRACTED values
    to catch recognition errors. Returns structured issues (see _run_checks)."""
    return _run_checks(extracted)


def build_extracted(raw_all: dict) -> dict:
    """Turn ocr_page's {field: {value, ...}} into final normalized field values
    {field: value}. Single source of truth for the per-type normalization the
    server UI and the CLI harness both need (occupancy → bool, income →
    decimal, rodné číslo → 'DDDDDD/DDDD', digit comb → digits, text → stripped).
    """
    fields: dict = {}
    for field, res in raw_all.items():
        raw = res["value"]
        if field in MONTH_FIELDS or field in CHECKBOX_FIELDS:
            _dark_frac, occupied = raw
            fields[field] = occupied
        elif field in NUMERIC_FIELDS or field in WHOLEBOX_INCOME:
            fields[field] = normalize_decimal(raw)
        elif field in RC_FIELDS:
            fields[field] = normalize_rc(raw)
        elif field in DIGIT_COMB_FIELDS:
            fields[field] = _digits_only(raw)
        else:
            fields[field] = (raw or "").strip()
    return fields


def escalate(raw_all: dict, extracted: dict, checks: list[dict]) -> tuple[list[str], dict]:
    """Decide which fields a human must review. A field is FLAGGED if its
    confidence is below its class threshold OR it participates in a failed
    constraint; everything else auto-accepts. Returns (sorted_flagged, reasons)
    where reasons maps field → list of human-readable reason strings.
    """
    reasons: dict[str, list[str]] = defaultdict(list)

    for field, res in raw_all.items():
        # A blank field has nothing recognized to be uncertain about — don't flag
        # it on confidence (a faint speck below the ink gate can otherwise zero the
        # score of an unfilled box). A required field left blank is still caught by
        # the validation constraints below (e.g. rodné-číslo format).
        v = extracted.get(field)
        if isinstance(v, str) and not v.strip():
            continue
        thr = CONF_THRESHOLD[field_class(field)]
        if res["confidence"] < thr:
            reasons[field].append(
                f"low confidence {res['confidence']:.2f} < {thr:.2f}")

    # A numeric field that produced no usable digit is unreadable, not just
    # low-confidence — always flag (preserves the original server behaviour).
    for field in NUMERIC_FIELDS:
        v = extracted.get(field)
        if v is not None and (not v or not any(c.isdigit() for c in v)):
            reasons[field].append("no valid decimal extracted")

    for c in checks:
        for f in c["fields"]:
            reasons[f].append(f"constraint: {c['msg']}")

    return sorted(reasons), dict(reasons)


def _reform_value(field: str, digits: str) -> str:
    """Re-insert the structural separator for a digit field's value string."""
    if field in RC_FIELDS and len(digits) == _RC_N_LEFT + 4:
        return digits[:_RC_N_LEFT] + "/" + digits[_RC_N_LEFT:]
    if (field in NUMERIC_FIELDS or field in WHOLEBOX_INCOME) and len(digits) >= _N_DECIMAL:
        return digits[:-_N_DECIMAL] + "." + digits[-_N_DECIMAL:]
    return digits


def disambiguate_extracted(raw_all: dict, extracted: dict) -> tuple[dict, list[str]]:
    """Constraint-guided disambiguation (Phase 3).

    When a digit field fails a constraint AND has low-confidence cells, search
    digit substitutions over ONLY those low-confidence cells for an assignment
    that satisfies the constraint, and adopt it. Currently covers the
    self-contained per-field constraints (rodné-číslo mod-11). Mutates and
    returns a corrected copy of `extracted` plus a log of what changed.

    HARD RULE: a field whose cells are all high-confidence is never altered —
    if it still violates a constraint, that is a real inconsistency to FLAG, not
    silently "correct". Enforced by only ever varying low_conf_cells.
    """
    from itertools import product
    fixed = dict(extracted)
    log: list[str] = []

    for field in sorted(RC_FIELDS):
        res = raw_all.get(field)
        if not res or not res["cells"]:
            continue
        low = res["low_conf_cells"]
        if not low or len(low) > 3:        # nothing uncertain, or search too wide
            continue
        digits = [c for c, _ in res["cells"]]
        if len(digits) != _RC_N_LEFT + 4:
            continue
        # already valid? skip
        cur = "".join(digits)
        if cur.isdigit() and int(cur) % 11 == 0:
            continue
        # search digit assignments over the low-confidence positions only
        found = None
        for combo in product("0123456789", repeat=len(low)):
            trial = list(digits)
            for idx, dch in zip(low, combo):
                trial[idx] = dch
            s = "".join(trial)
            if s.isdigit() and int(s) % 11 == 0:
                found = s
                break
        if found and found != cur:
            new_val = _reform_value(field, found)
            log.append(f"{field}: {_reform_value(field, cur)} → {new_val} "
                       f"(mod-11 via low-conf cells {low})")
            fixed[field] = new_val

    return fixed, log


def _cell_ink(img: Image.Image, cell) -> float:
    """Dark-pixel fraction inside a cell — occupancy test."""
    x1, y1, x2, y2 = cell
    arr = np.array(img.crop((x1, y1, x2, y2)))
    return float(np.sum(arr < 200) / arr.size)


def _ocr_cell(img: Image.Image, cell) -> tuple[str, float]:
    """OCR one digit cell; return (digit-only text, rec_score)."""
    x1, y1, x2, y2 = cell
    text, score = ocr_crop(img.crop((x1, y1, x2, y2)))
    return "".join(c for c in text if c.isdigit()), score


def _field_result(value, cells: list[tuple[str, float]]) -> dict:
    """Bundle a recognized value with per-cell confidence.

    `cells` is the ordered list of (char, rec_score) that produced the digit
    sequence in `value`. Field confidence is the MIN over cells — a field is
    only as trustworthy as its weakest cell. low_conf_cells indexes the cells
    (within `cells`) below CELL_LOW_CONF; those are the only cells that
    constraint-guided disambiguation is permitted to alter.
    """
    scores = [s for _, s in cells]
    conf = min(scores) if scores else 1.0
    low = [i for i, (_, s) in enumerate(cells) if s < CELL_LOW_CONF]
    return {"value": value, "confidence": conf, "low_conf_cells": low, "cells": cells}


def _text_result(value: str, score: float) -> dict:
    """Result bundle for a single whole-string OCR (text / whole-box)."""
    return {"value": value, "confidence": score, "low_conf_cells": [], "cells": []}


def _occupancy_result(dark_frac: float, occupied: bool, cutoff: float = 0.10) -> dict:
    """Result bundle for an ink-occupancy field. Confidence grows with distance
    from the decision cutoff — a mark sitting right on the threshold is the
    uncertain one."""
    conf = min(1.0, abs(dark_frac - cutoff) / cutoff) if cutoff else 1.0
    return {"value": (dark_frac, occupied), "confidence": conf,
            "low_conf_cells": [], "cells": []}


def _ocr_income_wholebox(img: Image.Image, box) -> dict:
    """Whole-box OCR for the small page-2 continuation-income value boxes.

    Per-cell OCR is fragile on these faint isolated digits; reading the whole
    value box (padded + 2× upscaled for the recognizer) gets the digits reliably.
    The decimal point is restored structurally in normalize_decimal.
    """
    x1, y1, x2, y2 = box
    w, h = img.size
    crop = img.crop((max(0, x1 - 4), max(0, y1 - 6), min(w, x2 + 4), min(h, y2 + 6)))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    text, score = ocr_crop(crop)
    return _text_result(text, score)


def _ocr_income_field(img: Image.Image, field: str) -> dict:
    """Per-cell OCR for income digit fields. Returns a _field_result for 'NNNNN.CC'.

    OCRs each 24×36px digit cell individually to avoid confusion from the
    comma gap and empty cells in wide-box crops. Integer cells are skipped
    if empty; decimal cells default to '0' if unreadable.
    """
    cells = INCOME_DIGIT_CELLS[field]
    int_cells = cells[:-_N_DECIMAL]
    dec_cells = cells[-_N_DECIMAL:]

    int_digits = []
    details: list[tuple[str, float]] = []
    for cell in int_cells:
        if _cell_ink(img, cell) > 0.02:  # cell has ink
            d, score = _ocr_cell(img, cell)
            ch = d[-1] if d else "?"
            int_digits.append(ch)
            details.append((ch, score))

    dec_digits = []
    for cell in dec_cells:
        d, score = _ocr_cell(img, cell)
        ch = d[-1] if d else "0"
        dec_digits.append(ch)
        # Only an INKED cell carries recognition uncertainty. A blank cell is
        # confidently blank (→ '0'); counting its noise-level score would drag a
        # correctly-read empty field's confidence down and cause a false flag.
        if _cell_ink(img, cell) > 0.02:
            details.append((ch, score))

    int_part = "".join(int_digits) if int_digits else "0"
    value = int_part + "." + "".join(dec_digits)
    return _field_result(value, details)


def _ocr_rc_field(img: Image.Image, field: str = "rod_cislo") -> dict:
    """Per-cell digit OCR for a rodné číslo field. Returns a _field_result for
    'DDDDDD/DDDD'.

    OCRs each digit cell individually and inserts the slash programmatically
    between the 6th and 7th digit, so the pre-printed '/' is never OCR'd. Works
    for the employee rodné číslo and each child's (RC_CELLS[field]). Empty cells
    (a blank child row) yield an empty value rather than '??????/????'.
    """
    digits = []
    details: list[tuple[str, float]] = []
    for cell in RC_CELLS[field]:
        if _cell_ink(img, cell) <= 0.02:   # blank cell — unfilled row
            continue
        d, score = _ocr_cell(img, cell)
        ch = d[-1] if d else "?"
        digits.append(ch)
        details.append((ch, score))
    if not digits:
        return _field_result("", [])
    value = "".join(digits[:_RC_N_LEFT]) + "/" + "".join(digits[_RC_N_LEFT:])
    return _field_result(value, details)


def _ocr_digit_comb(img: Image.Image, field: str) -> dict:
    """Per-cell digit OCR for an integer comb field (datum/rok/psc/supisne).

    Empty cells are skipped (gated on ink), so partially-filled fields don't
    inject spurious digits. Returns a _field_result for the concatenated digits.
    """
    digits = []
    details: list[tuple[str, float]] = []
    for cell in DIGIT_COMB_CELLS[field]:
        if _cell_ink(img, cell) > 0.02:
            d, score = _ocr_cell(img, cell)
            ch = d[-1] if d else "?"
            digits.append(ch)
            details.append((ch, score))
    return _field_result("".join(digits), details)


def _filter_slovak_text(s: str) -> str:
    """Constrain OCR text to the Slovak alphabet (drop digits/punctuation noise).

    Forms are paličkové písmo — letters only — so anything else the recognizer
    emits is noise ("BratisiavaNIs'" → "BratisiavaNIs"). Keeps letters + single
    spaces. This is the inference-time char-whitelist equivalent for PaddleOCR.
    """
    out = "".join(c if c in _SLOVAK_SET else " " for c in s)
    return " ".join(out.split())


# Centre-crop widths tried when reconstructing a comb word. A fixed narrow crop
# slices the sides off some block-capital glyphs ("ĎURICA"→"JRISA", "TOMÁŠ"→"MAS");
# a wider crop recovers them but can pull in a neighbour's ink on others. So we read
# at several widths and keep the highest-confidence result. The narrow 0.62 stays a
# candidate, so this can only raise (never lower) the best achievable score.
_COMB_INNER_WIDTHS = (0.62, 0.72, 0.82)


def _assemble_comb_strip(img: Image.Image, cells, inner: float):
    """Crop the centre `inner` fraction of each inked cell and paste the tiles
    GAPLESSLY into one strip — the reconstruction that lets the Latin model read a
    comb word ("Krátka") instead of the dividers as junk ("Astlolylelnlslkial").
    Returns the strip, or None if no cell is inked (an empty field)."""
    tiles = []
    for c in cells:
        if _cell_ink(img, c) <= 0.02:
            continue
        x1, y1, x2, y2 = c
        m = int((x2 - x1) * (1 - inner) / 2)
        tiles.append(img.crop((x1 + m, y1, x2 - m, y2)))
    if not tiles:
        return None
    h = max(t.height for t in tiles)
    w = sum(t.width for t in tiles)
    strip = Image.new("L", (w + 8, h + 8), 255)
    x = 4
    for t in tiles:
        strip.paste(t, (x, 4))
        x += t.width
    return strip


def _ocr_comb_text(img: Image.Image, cells, inner: float = 0.62,
                   multiwidth: bool = False) -> tuple[str, float]:
    """Read a free-text comb field by reconstructing the word. Returns (text, score);
    empty (unfilled) field → ("", 1.0).

    Comb fields put one letter per box with vertical tick dividers. Whole-box OCR
    reads those as junk, so we paste the inked-cell centres GAPLESSLY into a word the
    Latin model can read with diacritics intact ("Krátka").

    multiwidth=True reads at several centre-crop widths (_COMB_INNER_WIDTHS) and keeps
    the highest-confidence result. A fixed narrow crop slices the sides off some block
    capitals (ĎURICA→JRISA, conf 0.55); a wider crop recovers them. This is used ONLY
    for the short NAME combs, where it's verified to fix the bad cases with zero
    regression. It is NOT used for long fields (e.g. ulica, 27 cells), where a wide
    crop can pull in a neighbour's ink and a wrong-but-confident read would win.
    """
    widths = _COMB_INNER_WIDTHS if multiwidth else (inner,)
    best_text, best_score = "", -1.0
    for frac in widths:
        strip = _assemble_comb_strip(img, cells, frac)
        if strip is None:
            return "", 1.0      # empty (unfilled) field — not low-confidence
        text, score = ocr_text_crop(strip)
        if score > best_score:
            best_text, best_score = _filter_slovak_text(text), score
    return best_text, best_score


def _ocr_text_field(img: Image.Image, field: str) -> dict:
    """Free-text comb field (titul/ulica/obec/stat) via gapless reconstruction."""
    text, score = _ocr_comb_text(img, TEXT_COMB_CELLS[field])
    return _text_result(text, score)


def _ocr_meno_field(img: Image.Image) -> dict:
    """Surname (Priezvisko comb) + given name (Meno comb), reconstructed words.

    Uses multi-width reconstruction: the name combs are short, and a single fixed
    crop misreads some block-capital glyphs (ĎURICA→JRISA). Verified to fix the bad
    cases with zero regression across the sample set."""
    sur, s_sur = _ocr_comb_text(img, PRIEZVISKO_CELLS, multiwidth=True)
    giv, s_giv = _ocr_comb_text(img, MENO_CELLS, multiwidth=True)
    return _text_result(f"{sur} {giv}".strip(), min(s_sur, s_giv))


def _apply_gazetteer(field: str, res: dict) -> dict:
    """Snap a closed-set text field to its reference list (gazetteer tier).

    On a list hit: adopt the canonical value and treat it as trustworthy (a
    known-register value is auditable, so it clears the review threshold). On a
    miss with a non-empty reading: keep the raw OCR but zero the confidence so it
    is FLAGGED — we never invent a value not in the list. Empty (unfilled) fields
    are left untouched. The match record is kept on `res['gazetteer']` for the UI.
    """
    res = dict(res)
    g = gazetteer_match(field, res["value"])
    res["gazetteer"] = g
    if g["matched"]:
        res["value"] = g["value"]
        res["confidence"] = max(res["confidence"], 0.85)
    elif res["value"].strip():
        res["confidence"] = 0.0          # real reading, not in register → review
    return res


def ocr_page(img: Image.Image, field_boxes: dict) -> dict:
    """Run OCR/occupancy on each field in field_boxes.

    Returns {field: {value, confidence, low_conf_cells, cells}}. `value` is the
    same payload the pipeline used before (digit/text string, or a
    (dark_frac, occupied) tuple for occupancy fields); the extra keys carry the
    per-field confidence used for escalation and disambiguation.
    """
    results = {}
    for field, box in field_boxes.items():
        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))
        if field in MONTH_FIELDS or field in CHECKBOX_FIELDS:
            dark_frac, occupied = cell_is_occupied(crop)
            results[field] = _occupancy_result(dark_frac, occupied)
        elif field in WHOLEBOX_INCOME:
            results[field] = _ocr_income_wholebox(img, box)   # short box → padded whole-box
        elif field in NUMERIC_FIELDS:
            results[field] = _ocr_income_field(img, field)
        elif field in RC_FIELDS:
            results[field] = _ocr_rc_field(img, field)
        elif field in DIGIT_COMB_FIELDS:
            results[field] = _ocr_digit_comb(img, field)
        elif field in TEXT_FIELDS:
            results[field] = _ocr_meno_field(img)
        elif field in FUZZY_TEXT_FIELDS:
            res = _ocr_text_field(img, field)
            if field in GAZETTEER_FIELDS:
                res = _apply_gazetteer(field, res)
            results[field] = res
        else:
            text, score = ocr_crop(crop)
            results[field] = _text_result(text, score)
    return results


def process_sample(png_p1: Path, png_p2: Path | None, json_path: Path) -> dict[str, bool]:
    data = json.loads(json_path.read_text())
    gt   = data["ground_truth"]
    is_broken    = data.get("_is_broken", False)
    broken_fields = set(data.get("_broken_fields", []))

    img_p1, _ = normalize_to_canvas(Image.open(png_p1))
    img_p2 = normalize_to_canvas(Image.open(png_p2))[0] if png_p2 and png_p2.exists() else None

    status = f"[BROKEN: {','.join(data.get('_broken_fields', []))}]" if is_broken else "[valid]"
    print(f"\n{png_p1.name}  {status}")

    raw_p1 = ocr_page(img_p1, FIELD_BOXES_P1)
    raw_p2 = ocr_page(img_p2, FIELD_BOXES_P2) if img_p2 else {}
    raw_all = {**raw_p1, **raw_p2}

    # Production pipeline: normalize → disambiguate low-confidence digit fields
    # against constraints → validate the EXTRACTION → decide which fields escalate.
    extracted = build_extracted(raw_all)
    extracted, dis_log = disambiguate_extracted(raw_all, extracted)
    ex_checks = validate_extracted(extracted)
    flagged, reasons = escalate(raw_all, extracted, ex_checks)
    flagged_set = set(flagged)

    field_results = {}
    for field, res in raw_all.items():
        gt_val = gt.get(field)
        if gt_val is None:
            continue
        raw = res["value"]
        match, gt_disp, ocr_disp = compare_field(field, gt_val, raw)
        field_results[field] = match
        tick = "✓" if match else "✗"
        note = " *" if field in broken_fields else ""
        flag = " ⚑" if field in flagged_set else ""   # authoritative escalate decision
        print(f"  {field:<20} | GT: {gt_disp:<22} | OCR: {ocr_disp:<22} "
              f"| {tick}{note} | conf {res['confidence']:.2f}{flag}")

    # Generator self-test (synthetic data is internally consistent).
    if not is_broken:
        gt_issues = validate_gt(gt)
        if gt_issues:
            print("  [!] GROUND-TRUTH SELF-TEST FAILED:")
            for iss in gt_issues:
                print(f"      {iss}")

    for line in dis_log:
        print(f"  [~] disambiguated {line}")
    if ex_checks:
        print("  [!] EXTRACTION VALIDATION ISSUES:")
        for c in ex_checks:
            print(f"      {c['msg']}")
    if flagged:
        print(f"  [⚑] {len(flagged)} field(s) flagged for review: {', '.join(flagged)}")
    else:
        print("  [✓] extraction: all fields auto-accepted (no flags)")

    return field_results


def print_summary(all_results: list[dict[str, bool]], all_boxes: dict) -> None:
    totals: dict[str, list[bool]] = defaultdict(list)
    for r in all_results:
        for field, match in r.items():
            totals[field].append(match)

    n = len(all_results)
    print(f"\n{'='*62}")
    print(f"ACCURACY SUMMARY  ({n} samples)")
    print(f"{'='*62}")
    print(f"{'Field':<22} {'Correct':>7}  {'Total':>5}  {'Accuracy':>8}")
    print("-" * 52)

    overall_correct = overall_total = 0
    for field in all_boxes:
        vals = totals.get(field, [])
        correct = sum(vals)
        total   = len(vals)
        pct     = 100 * correct / total if total else 0.0
        print(f"  {field:<20} {correct:>7}  {total:>5}  {pct:>7.1f}%")
        overall_correct += correct
        overall_total   += total

    print("-" * 52)
    overall_pct = 100 * overall_correct / overall_total if overall_total else 0.0
    print(f"  {'OVERALL':<20} {overall_correct:>7}  {overall_total:>5}  {overall_pct:>7.1f}%")
    print(f"{'='*62}")


def process_live(paths: list[Path]) -> None:
    """OCR a live aligned photo (no ground truth). Accepts 1 or 2 page paths."""
    img_p1, warn1 = normalize_to_canvas(Image.open(paths[0]))
    img_p2 = normalize_to_canvas(Image.open(paths[1]))[0] if len(paths) > 1 else None

    print(f"\n{paths[0].name}  [live — no ground truth]")
    if warn1:
        print(f"  [!] {warn1}")

    raw_p1 = ocr_page(img_p1, FIELD_BOXES_P1)
    raw_p2 = ocr_page(img_p2, FIELD_BOXES_P2) if img_p2 else {}
    raw_all = {**raw_p1, **raw_p2}

    # Real photos get the SAME production validation + escalation as the UI.
    extracted = build_extracted(raw_all)
    extracted, dis_log = disambiguate_extracted(raw_all, extracted)
    ex_checks = validate_extracted(extracted)
    flagged, _reasons = escalate(raw_all, extracted, ex_checks)
    flagged_set = set(flagged)

    for field, res in raw_all.items():
        conf = res["confidence"]
        flag = " ⚑" if field in flagged_set else ""   # authoritative escalate decision
        if field in MONTH_FIELDS or field in CHECKBOX_FIELDS:
            dark_frac, occupied = res["value"]
            print(f"  {field:<20} | {'X' if occupied else '.'} ({dark_frac*100:.1f}% dark)"
                  f" | conf {conf:.2f}{flag}")
        else:
            print(f"  {field:<20} | {res['value']:<24} | conf {conf:.2f}{flag}")

    for line in dis_log:
        print(f"  [~] disambiguated {line}")
    if ex_checks:
        print("  [!] EXTRACTION VALIDATION ISSUES:")
        for c in ex_checks:
            print(f"      {c['msg']}")
    if flagged:
        print(f"  [⚑] {len(flagged)} field(s) flagged for review: {', '.join(flagged)}")
    else:
        print("  [✓] extraction: all fields auto-accepted (no flags)")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <samples-dir | p1.png [p2.png]>")
        sys.exit(1)

    print("Initializing PaddleOCR recognition model …")
    get_rec_model()
    print("OCR ready.\n")

    # Single or two PNG paths → live mode
    args = [Path(a) for a in sys.argv[1:]]
    if args[0].is_file() and args[0].suffix.lower() == ".png":
        process_live(args)
        return

    # Directory mode — find sample_NNNN_p1.png files
    samples_dir = args[0]
    p1_paths = sorted(samples_dir.glob("*_p1.png"))
    if not p1_paths:
        print(f"No *_p1.png files found in {samples_dir}")
        sys.exit(1)

    print(f"Found {len(p1_paths)} samples in {samples_dir}")
    all_results = []
    for p1 in p1_paths:
        stem = p1.stem.replace("_p1", "")
        p2 = p1.with_name(f"{stem}_p2.png")
        json_path = p1.with_name(f"{stem}.json")
        if not json_path.exists():
            print(f"  WARNING: no JSON for {p1.name}, skipping")
            continue
        results = process_sample(p1, p2 if p2.exists() else None, json_path)
        all_results.append(results)

    all_boxes = {**FIELD_BOXES_P1, **FIELD_BOXES_P2}
    print_summary(all_results, all_boxes)


if __name__ == "__main__":
    main()
