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
)

_SLOVAK_SET = set(SLOVAK_ALPHA)

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


def ocr_crop(pil_crop: Image.Image) -> str:
    rec = get_rec_model()
    arr = np.array(pil_crop.convert("RGB"))
    results = list(rec.predict(arr))
    if not results:
        return ""
    return results[0].get("rec_text", "").strip()


def ocr_text_crop(pil_crop: Image.Image) -> str:
    """OCR a free-text crop with the Latin (Slovak-capable) model."""
    rec = get_text_model()
    arr = np.array(pil_crop.convert("RGB"))
    results = list(rec.predict(arr))
    if not results:
        return ""
    return results[0].get("rec_text", "").strip()


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


def validate_gt(gt: dict) -> list[str]:
    from decimal import Decimal
    issues = []
    try:
        r01 = Decimal(gt["riadok_01"])
        r02 = Decimal(gt["riadok_02"])
        r03 = Decimal(gt["riadok_03"])
        if r01 - r02 != r03:
            issues.append(f"arithmetic: r01({r01}) − r02({r02}) = {r01-r02} ≠ r03({r03})")
    except Exception as e:
        issues.append(f"arithmetic-parse-error: {e}")

    # rodné číslo: format + mod-11 for employee and every (non-empty) child.
    for rcf in sorted(RC_FIELDS):
        rc = gt.get(rcf, "")
        if not rc and rcf != "rod_cislo":
            continue                      # blank child row — nothing to validate
        m = re.fullmatch(r"(\d{6})/(\d{4})", rc)
        if not m:
            issues.append(f"{rcf} format: {rc!r}")
        elif int(m.group(1) + m.group(2)) % 11 != 0:
            issues.append(f"{rcf} mod-11 failed: {rc!r}")

    # Page-2 top rodné číslo must match page-1's (same taxpayer, page matching).
    if gt.get("rod_cislo_p2") and gt.get("rod_cislo_p2") != gt.get("rod_cislo"):
        issues.append(f"rod_cislo_p2 {gt['rod_cislo_p2']!r} ≠ page-1 {gt.get('rod_cislo')!r}")

    # DIČ (employer tax ID): exactly 10 digits.
    if gt.get("zam_dic"):
        dic = _digits_only(gt["zam_dic"])
        if len(dic) != 10:
            issues.append(f"zam_dic (need 10 digits): {gt['zam_dic']!r}")

    # PSČ: exactly 5 digits
    if "psc" in gt:
        psc = _digits_only(gt["psc"])
        if len(psc) != 5:
            issues.append(f"psc format (need 5 digits): {gt['psc']!r}")

    # Rok: plausible assessment year
    if "rok" in gt:
        rok = _digits_only(gt["rok"])
        year = int(rok) if rok else 0
        # 2-digit suffix means 20YY; 4-digit means full year
        if len(rok) == 2:
            year = 2000 + int(rok)
        if not (2000 <= year <= 2099):
            issues.append(f"rok out of range 2000–2099: {gt['rok']!r}")

    # Dátum narodenia ⇄ rodné číslo cross-check: DDMMYYYY's YYMMDD must equal
    # the first 6 digits of rod_cislo. Strongest new deterministic check.
    if "datum_narodenia" in gt and m:
        d = _digits_only(gt["datum_narodenia"])
        if len(d) == 8:
            dd, mm, yyyy = d[0:2], d[2:4], d[4:8]
            try:
                from datetime import date
                date(int(yyyy), int(mm), int(dd))   # calendar validity
                rc_yymmdd = m.group(1)
                # rod_cislo month may be +50 (female); compare day+year, month mod 50
                rc_yy, rc_mm, rc_dd = rc_yymmdd[0:2], rc_yymmdd[2:4], rc_yymmdd[4:6]
                rc_mm_norm = f"{int(rc_mm) % 50:02d}"
                if (yyyy[2:], mm, dd) != (rc_yy, rc_mm_norm, rc_dd):
                    issues.append(
                        f"datum_narodenia {dd}.{mm}.{yyyy} ≠ rod_cislo {rc_yymmdd}")
            except ValueError:
                issues.append(f"datum_narodenia invalid date: {gt['datum_narodenia']!r}")

    return issues


def _cell_ink(img: Image.Image, cell) -> float:
    """Dark-pixel fraction inside a cell — occupancy test."""
    x1, y1, x2, y2 = cell
    arr = np.array(img.crop((x1, y1, x2, y2)))
    return float(np.sum(arr < 200) / arr.size)


def _ocr_cell(img: Image.Image, cell) -> str:
    """OCR one digit cell; return only the digit characters."""
    x1, y1, x2, y2 = cell
    text = ocr_crop(img.crop((x1, y1, x2, y2)))
    return "".join(c for c in text if c.isdigit())


def _ocr_income_wholebox(img: Image.Image, box) -> str:
    """Whole-box OCR for the small page-2 continuation-income value boxes.

    Per-cell OCR is fragile on these faint isolated digits; reading the whole
    value box (padded + 2× upscaled for the recognizer) gets the digits reliably.
    The decimal point is restored structurally in normalize_decimal.
    """
    x1, y1, x2, y2 = box
    w, h = img.size
    crop = img.crop((max(0, x1 - 4), max(0, y1 - 6), min(w, x2 + 4), min(h, y2 + 6)))
    crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    return ocr_crop(crop)


def _ocr_income_field(img: Image.Image, field: str) -> str:
    """Per-cell OCR for income digit fields. Returns 'NNNNN.CC' string.

    OCRs each 24×36px digit cell individually to avoid confusion from the
    comma gap and empty cells in wide-box crops. Integer cells are skipped
    if empty; decimal cells default to '0' if unreadable.
    """
    cells = INCOME_DIGIT_CELLS[field]
    int_cells = cells[:-_N_DECIMAL]
    dec_cells = cells[-_N_DECIMAL:]

    int_digits = []
    for cell in int_cells:
        if _cell_ink(img, cell) > 0.02:  # cell has ink
            d = _ocr_cell(img, cell)
            int_digits.append(d[-1] if d else "?")

    dec_digits = []
    for cell in dec_cells:
        d = _ocr_cell(img, cell)
        dec_digits.append(d[-1] if d else "0")

    int_part = "".join(int_digits) if int_digits else "0"
    return int_part + "." + "".join(dec_digits)


def _ocr_rc_field(img: Image.Image, field: str = "rod_cislo") -> str:
    """Per-cell digit OCR for a rodné číslo field. Returns 'DDDDDD/DDDD'.

    OCRs each digit cell individually and inserts the slash programmatically
    between the 6th and 7th digit, so the pre-printed '/' is never OCR'd. Works
    for the employee rodné číslo and each child's (RC_CELLS[field]). Empty cells
    (a blank child row) yield an empty string rather than '??????/????'.
    """
    digits = []
    for cell in RC_CELLS[field]:
        if _cell_ink(img, cell) <= 0.02:   # blank cell — unfilled row
            continue
        d = _ocr_cell(img, cell)
        digits.append(d[-1] if d else "?")
    if not digits:
        return ""
    return "".join(digits[:_RC_N_LEFT]) + "/" + "".join(digits[_RC_N_LEFT:])


def _ocr_digit_comb(img: Image.Image, field: str) -> str:
    """Per-cell digit OCR for an integer comb field (datum/rok/psc/supisne).

    Empty cells are skipped (gated on ink), so partially-filled fields don't
    inject spurious digits. Returns the concatenated digit string.
    """
    digits = []
    for cell in DIGIT_COMB_CELLS[field]:
        if _cell_ink(img, cell) > 0.02:
            d = _ocr_cell(img, cell)
            digits.append(d[-1] if d else "?")
    return "".join(digits)


def _filter_slovak_text(s: str) -> str:
    """Constrain OCR text to the Slovak alphabet (drop digits/punctuation noise).

    Forms are paličkové písmo — letters only — so anything else the recognizer
    emits is noise ("BratisiavaNIs'" → "BratisiavaNIs"). Keeps letters + single
    spaces. This is the inference-time char-whitelist equivalent for PaddleOCR.
    """
    out = "".join(c if c in _SLOVAK_SET else " " for c in s)
    return " ".join(out.split())


def _ocr_comb_text(img: Image.Image, cells, inner: float = 0.62) -> str:
    """Read a free-text comb field by reconstructing the word.

    Comb fields put one letter per box with vertical tick dividers between them.
    Whole-box OCR reads those dividers and gaps as junk ("Slovenská" →
    "Astlolylelnlslkial"). Instead, crop the centre `inner` fraction of each
    inked cell (excluding the divider edges) and paste the tiles GAPLESSLY into
    one strip, reconstructing a natural word the Latin model can read with its
    Slovak diacritics intact ("Krátka" → "Krátka").
    """
    tiles = []
    for c in cells:
        if _cell_ink(img, c) <= 0.02:
            continue
        x1, y1, x2, y2 = c
        m = int((x2 - x1) * (1 - inner) / 2)
        tiles.append(img.crop((x1 + m, y1, x2 - m, y2)))
    if not tiles:
        return ""
    h = max(t.height for t in tiles)
    w = sum(t.width for t in tiles)
    strip = Image.new("L", (w + 8, h + 8), 255)
    x = 4
    for t in tiles:
        strip.paste(t, (x, 4))
        x += t.width
    return _filter_slovak_text(ocr_text_crop(strip))


def _ocr_text_field(img: Image.Image, field: str) -> str:
    """Free-text comb field (titul/ulica/obec/stat) via gapless reconstruction."""
    return _ocr_comb_text(img, TEXT_COMB_CELLS[field])


def _ocr_meno_field(img: Image.Image) -> str:
    """Surname (Priezvisko comb) + given name (Meno comb), reconstructed words."""
    sur = _ocr_comb_text(img, PRIEZVISKO_CELLS)
    giv = _ocr_comb_text(img, MENO_CELLS)
    return f"{sur} {giv}".strip()


def ocr_page(img: Image.Image, field_boxes: dict) -> dict:
    """Run OCR/occupancy on each field in field_boxes; return {field: raw_result}."""
    results = {}
    for field, box in field_boxes.items():
        x1, y1, x2, y2 = box
        crop = img.crop((x1, y1, x2, y2))
        if field in MONTH_FIELDS or field in CHECKBOX_FIELDS:
            results[field] = cell_is_occupied(crop)
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
            results[field] = _ocr_text_field(img, field)
        else:
            results[field] = ocr_crop(crop)
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

    field_results = {}
    for field, raw in raw_all.items():
        gt_val = gt.get(field)
        if gt_val is None:
            continue
        match, gt_disp, ocr_disp = compare_field(field, gt_val, raw)
        field_results[field] = match
        tick = "✓" if match else "✗"
        note = " *" if field in broken_fields else ""
        print(f"  {field:<20} | GT: {gt_disp:<22} | OCR: {ocr_disp:<22} | {tick}{note}")

    if not is_broken:
        issues = validate_gt(gt)
        if issues:
            print("  [!] VALIDATION ISSUES:")
            for iss in issues:
                print(f"      {iss}")
        else:
            print("  [✓] validation: arithmetic OK, rod_cislo mod-11 OK")

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

    for field, raw in raw_p1.items():
        if field in MONTH_FIELDS:
            dark_frac, occupied = raw
            print(f"  {field:<20} | {'X' if occupied else '.'} ({dark_frac*100:.1f}% dark)")
        else:
            print(f"  {field:<20} | {raw}")

    for field, raw in raw_p2.items():
        dark_frac, occupied = raw
        print(f"  {field:<20} | {'X' if occupied else '.'} ({dark_frac*100:.1f}% dark)")


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
