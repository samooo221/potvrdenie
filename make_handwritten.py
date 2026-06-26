#!/usr/bin/env python3
"""
make_handwritten.py — generate one realistic-looking handwritten sample as handnewsample.png.

Simulates handwriting by:
- Per-character random rotation (±8° text, ±5° digits)
- Baseline wander (characters drift slightly up/down along a line)
- Variable ink darkness (not uniform black — pressure variation)
- Slight ink bleed (per-char blur)
- Heavier scan noise + paper texture

Synthetic data only. No real taxpayer data.
"""
import argparse
import math
import random
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from field_defs import (
    FIELD_BOXES_P1, FIELD_BOXES_P2,
    INCOME_DIGIT_CELLS, _N_DECIMAL,
    NUMERIC_FIELDS, MONTH_FIELDS, RC_FIELDS,
    DATUM_CELLS, ROK_CELLS, PSC_CELLS, SUPISNE_CELLS,
    TITUL_CELLS, ULICA_CELLS, OBEC_CELLS, STAT_CELLS,
    OPRAVA_BOX, DECL_BOX,
)
from generate_samples import datum_from_rc   # reuse DOB-from-rodné-číslo derivation

TEMPLATE_P1 = Path(__file__).parent / "form_template_p1.png"
OUT = Path(__file__).parent / "handnewsample.png"

# --- Comb-cell coordinates measured from form_template_p1.png (1241×1755) ---
# Rodné číslo: top-left header box. 6 digit cells, pre-printed "/", then 4 digit cells.
RC_CELLS_LEFT  = [[c - 15, 200, c + 15, 230] for c in (75, 107, 137, 167, 197, 233)]
RC_CELLS_RIGHT = [[c - 15, 200, c + 15, 230] for c in (282, 317, 347, 377)]
# Priezvisko (surname) comb: 16 cells, x=58→514, 30px pitch, fill y≈314–348.
PRIEZVISKO_CELLS = [[58 + 30 * i, 314, 58 + 30 * (i + 1), 348] for i in range(16)]
# Meno (first name) comb: 10 cells, x=559→859, 30px pitch.
MENO_CELLS = [[559 + 30 * i, 314, 559 + 30 * (i + 1), 348] for i in range(10)]

FONT_CANDIDATES = [
    "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
    "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
]

RNG = random.Random(7)
ALIGN = "right"   # placement of non-euro comb values; overridden by --align


def find_font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_valid_rc(rng: random.Random) -> str:
    yy, mm, dd = 76, 2, 21
    mm_enc = mm
    nnn = 741
    nine = int(f"{yy:02d}{mm_enc:02d}{dd:02d}{nnn:03d}")
    c = (-(nine * 10)) % 11
    return f"{yy:02d}{mm_enc:02d}{dd:02d}/{nnn:03d}{c}"


def draw_char_handwritten(draw: ImageDraw.ImageDraw, img: Image.Image,
                           ch: str, cx: float, cy: float,
                           font: ImageFont.ImageFont, rng: random.Random,
                           rot_range: float = 7.0) -> float:
    """Draw one character with handwriting simulation. Returns character advance width."""
    # Measure char
    bbox = draw.textbbox((0, 0), ch, font=font)
    cw = bbox[2] - bbox[0]
    ch_h = bbox[3] - bbox[1]
    if cw <= 0:
        return 6.0

    # Create small canvas with padding for rotation
    pad = 10
    tile_w = cw + pad * 2
    tile_h = ch_h + pad * 2
    tile = Image.new("L", (tile_w, tile_h), 255)
    td = ImageDraw.Draw(tile)

    # Ink darkness: vary 10-40 (0=black, 255=white) — mimics pen pressure.
    # Capped at 40 (not 55): lighter ink + blur can fade a small glyph's strokes
    # until a "0" loop breaks into a 6-like fragment. A real pen mark stays solid.
    ink = rng.randint(10, 40)
    td.text((pad - bbox[0], pad - bbox[1]), ch, fill=ink, font=font)

    # Slight ink bleed (capped at 0.9 so thin strokes survive on tiny digit cells)
    tile = tile.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.4, 0.9)))

    # Random rotation
    angle = rng.uniform(-rot_range, rot_range)
    tile = tile.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # Baseline wander
    bx = rng.randint(-2, 2)
    by = rng.randint(-3, 3)

    paste_x = int(cx - pad + bx)
    paste_y = int(cy - pad + by)

    # Paste using darkest-wins (min blend)
    region_x1 = max(0, paste_x)
    region_y1 = max(0, paste_y)
    region_x2 = min(img.width, paste_x + tile_w)
    region_y2 = min(img.height, paste_y + tile_h)

    if region_x2 <= region_x1 or region_y2 <= region_y1:
        return float(cw) + rng.uniform(-1, 2)

    tile_x1 = region_x1 - paste_x
    tile_y1 = region_y1 - paste_y
    tile_crop = tile.crop((tile_x1, tile_y1,
                           tile_x1 + (region_x2 - region_x1),
                           tile_y1 + (region_y2 - region_y1)))

    existing = img.crop((region_x1, region_y1, region_x2, region_y2))
    blended = Image.fromarray(np.minimum(np.array(existing), np.array(tile_crop)))
    img.paste(blended, (region_x1, region_y1))

    return float(cw) + rng.uniform(-1, 2)


def draw_text_handwritten(img: Image.Image, text: str,
                           x1: int, y1: int, x2: int, y2: int,
                           font: ImageFont.ImageFont, rng: random.Random) -> None:
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), "M", font=font)
    ch_h = bbox[3] - bbox[1]
    base_y = y1 + (y2 - y1 - ch_h) // 2

    cx = float(x1 + 8)
    for ch in text:
        adv = draw_char_handwritten(draw, img, ch, cx, base_y, font, rng, rot_range=8.0)
        cx += adv
        if cx > x2 - 6:
            break


def draw_digit_in_cell(img: Image.Image, digit: str,
                        cell: list, font: ImageFont.ImageFont,
                        rng: random.Random) -> None:
    draw = ImageDraw.Draw(img)
    x1, y1, x2, y2 = cell
    bbox = draw.textbbox((0, 0), digit, font=font)
    dw = bbox[2] - bbox[0]
    dh = bbox[3] - bbox[1]
    base_x = x1 + (x2 - x1 - dw) / 2
    base_y = y1 + (y2 - y1 - dh) / 2
    draw_char_handwritten(draw, img, digit, base_x, base_y, font, rng, rot_range=5.0)


def draw_chars_in_cells(img: Image.Image, text: str, cells: list,
                         font: ImageFont.ImageFont, rng: random.Random,
                         align: str = "left") -> None:
    """Place each character of `text` centered in comb cells, with alignment.

    align: 'left'   → first char in the leftmost cell
           'right'  → last char in the rightmost cell (numeric convention)
           'middle' → value centered, spare cells split to both sides
    The OCR gates empty cells and reads/reconstructs only inked ones, so any
    placement reads back the same — this exercises that position-independence.
    """
    n = min(len(text), len(cells))
    if align == "right":
        start = len(cells) - n
    elif align == "middle":
        start = (len(cells) - n) // 2
    else:
        start = 0
    for i, ch in enumerate(text[:n]):
        draw_digit_in_cell(img, ch, cells[start + i], font, rng)


def draw_numeric_handwritten(img: Image.Image, value_str: str,
                              field: str, font: ImageFont.ImageFont,
                              rng: random.Random) -> None:
    cells = INCOME_DIGIT_CELLS[field]
    n_dec = _N_DECIMAL
    int_cells = cells[:-n_dec]
    dec_cells = cells[-n_dec:]

    if "." in value_str:
        int_part, dec_part = value_str.split(".", 1)
    else:
        int_part, dec_part = value_str, "00"
    dec_part = (dec_part + "00")[:n_dec]

    int_digits = list(int_part)
    for offset, cell in enumerate(reversed(int_cells)):
        idx = len(int_digits) - 1 - offset
        if idx >= 0:
            draw_digit_in_cell(img, int_digits[idx], cell, font, rng)

    for ch, cell in zip(dec_part, dec_cells):
        draw_digit_in_cell(img, ch, cell, font, rng)


def draw_cross(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
               rng: random.Random) -> None:
    draw = ImageDraw.Draw(img)
    ink = rng.randint(10, 50)
    jx = rng.randint(-2, 2)
    jy = rng.randint(-2, 2)
    draw.line([(x1+3+jx, y1+3+jy), (x2-3+jx, y2-3+jy)], fill=ink, width=2)
    draw.line([(x2-3+jx, y1+3+jy), (x1+3+jx, y2-3+jy)], fill=ink, width=2)


def add_paper_texture(img: Image.Image, rng: random.Random) -> Image.Image:
    arr = np.array(img, dtype=np.float32)

    # Paper yellowing/aging — slight uniform darkening towards edges
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    vignette = 1.0 - 0.04 * ((xx - cx) ** 2 / cx ** 2 + (yy - cy) ** 2 / cy ** 2)
    vignette = np.clip(vignette, 0.92, 1.0)
    arr = arr * vignette

    # Fine paper grain noise
    np_rng2 = np.random.default_rng(rng.randint(0, 2**31))
    noise = np_rng2.random((h, w)) * 14 - 7
    arr = np.clip(arr + noise, 0, 255)

    # Salt-and-pepper scan dust
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    arr[np_rng.random((h, w)) < 0.004] = 0
    arr[np_rng.random((h, w)) < 0.006] = 255

    # Slight scan rotation
    result = Image.fromarray(arr.astype(np.uint8))
    angle = rng.uniform(-0.6, 0.6)
    result = result.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # Mild scan blur (not as sharp as a digital render)
    result = result.filter(ImageFilter.GaussianBlur(radius=0.5))

    return result


def main() -> None:
    global ALIGN
    ap = argparse.ArgumentParser(description="Generate a handwritten-style POT395 page-1 sample.")
    ap.add_argument("--align", choices=["left", "middle", "right"], default="right",
                    help="placement of non-euro comb values (euro/income stay right-aligned)")
    ap.add_argument("--out", default=str(OUT), help="output PNG path")
    args = ap.parse_args()
    ALIGN = args.align
    out_path = Path(args.out)

    # Fixed synthetic values — plausible Slovak employee, seed 7
    rc = make_valid_rc(RNG)

    r01a = Decimal("18450.00")
    r01b = Decimal("320.50")
    r01  = r01a + r01b   # 18770.50

    r02a = (r01a * Decimal("0.134")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r02b = Decimal("0.00")
    r02  = r02a + r02b   # ~2472.30

    r03 = r01 - r02      # taxable base

    r04 = (r03 * Decimal("0.19")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r05 = Decimal("0.00")
    r06 = Decimal("142.80")
    r07 = Decimal("0.00")
    r08 = Decimal("142.80")
    r08a = Decimal("0.00")
    r09  = Decimal("142.80")

    months = [True, True, True, True, True, True, True, True, True, True, True, True]

    # Date of birth derived from rodné číslo so the cross-check passes.
    datum = datum_from_rc(rc)

    # Text fields UPPERCASE — paličkové písmo (block capitals), as on real forms.
    gt = {
        "meno_zamestnanca": "NOVÁK JÁN",
        "rod_cislo": rc,
        "datum_narodenia": datum,
        "rok": "25",
        "oprava": False,
        "titul": "ING",
        "ulica": "HLAVNÁ",
        "supisne_cislo": "1234",
        "psc": "81101",
        "obec": "BRATISLAVA",
        "stat": "SLOVENSKO",
        "danovnik_obmedzena": True,
        "riadok_01":  str(r01),
        "riadok_01a": str(r01a),
        "riadok_01b": str(r01b),
        "riadok_02":  str(r02),
        "riadok_02a": str(r02a),
        "riadok_02b": str(r02b),
        "riadok_03":  str(r03),
        "riadok_04":  str(r04),
        "riadok_05":  str(r05),
        "riadok_06":  str(r06),
        "riadok_07":  str(r07),
        "riadok_08":  str(r08),
        "riadok_08a": str(r08a),
        "riadok_09":  str(r09),
    }
    for i, occ in enumerate(months, 1):
        gt[f"mesiac_{i:02d}"] = occ

    font = find_font(21)
    print(f"Using font: {font.path if hasattr(font, 'path') else 'default'}")

    img = Image.open(TEMPLATE_P1).convert("L")

    # meno_zamestnanca: Priezvisko (surname) and Meno (first name) are comb fields
    # with one white box per letter. Split "Novák Ján" and place each part letter-by-letter
    # into its own column of cells.
    meno_full = gt["meno_zamestnanca"]
    parts = meno_full.split(" ", 1)
    surname   = parts[0]
    firstname = parts[1] if len(parts) > 1 else ""
    draw_chars_in_cells(img, surname,   PRIEZVISKO_CELLS, font, RNG, align=ALIGN)
    if firstname:
        draw_chars_in_cells(img, firstname, MENO_CELLS, font, RNG, align=ALIGN)

    # rod_cislo: top-left header box. 6 digits left of the pre-printed "/", 4 digits right.
    # Positional (slash-split) so it always fills both groups — alignment N/A.
    rc_digits = gt["rod_cislo"].replace("/", "")   # "7602217414"
    draw_chars_in_cells(img, rc_digits[:6], RC_CELLS_LEFT,  font, RNG)
    draw_chars_in_cells(img, rc_digits[6:], RC_CELLS_RIGHT, font, RNG)

    # --- Header + address fields ---  (non-euro fields use the chosen alignment;
    # datum/rok/psc fill all their cells so alignment has no visible effect there)
    draw_chars_in_cells(img, gt["datum_narodenia"], DATUM_CELLS, font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["rok"],             ROK_CELLS,   font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["psc"],             PSC_CELLS,   font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["supisne_cislo"],   SUPISNE_CELLS, font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["titul"],           TITUL_CELLS, font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["ulica"],           ULICA_CELLS, font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["obec"],            OBEC_CELLS,  font, RNG, align=ALIGN)
    draw_chars_in_cells(img, gt["stat"],            STAT_CELLS,  font, RNG, align=ALIGN)
    if gt["oprava"]:
        draw_cross(img, *OPRAVA_BOX, RNG)
    if gt["danovnik_obmedzena"]:
        draw_cross(img, *DECL_BOX, RNG)

    # --- Income fields ---
    for key in NUMERIC_FIELDS:
        val = gt.get(key)
        if val is None:
            continue
        draw_numeric_handwritten(img, str(val), key, font, RNG)

    # --- Page 2 month cells (draw on page 1 image for single-page output) ---
    # We only have page 1 here; months would be on page 2 — skip for this output
    # (the user asked for a single handnewsample.png, so page 1 with all income data)

    img = add_paper_texture(img, RNG)
    img.save(out_path)
    print(f"Saved {out_path}  (align={ALIGN}, {img.width}×{img.height} px)")
    print()
    print("Values written:")
    for k, v in gt.items():
        if not k.startswith("mesiac"):
            print(f"  {k:<20} {v}")


if __name__ == "__main__":
    main()
