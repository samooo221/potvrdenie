#!/usr/bin/env python3
"""
generate_samples.py — synthetic POT395 form sample generator (real form template).

Usage:
    python generate_samples.py --n 8 --out samples/ --break-frac 0.25

Produces per sample:
    samples/sample_NNNN_p1.png  — page 1 (real blank form + synthetic handwriting)
    samples/sample_NNNN_p2.png  — page 2 (real blank form + month checkmarks)
    samples/sample_NNNN.json    — ground truth + field box coordinates

Synthetic data only. Never use real taxpayer data.
"""
import argparse
import json
import math
import random
import re
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from field_defs import (
    FIELD_BOXES_P1, FIELD_BOXES_P2,
    INCOME_DIGIT_CELLS, _N_DECIMAL,
    NUMERIC_FIELDS, MONTH_FIELDS, TEXT_FIELDS, RC_FIELDS,
    RC_CELLS, PRIEZVISKO_CELLS, MENO_CELLS,
    DIGIT_COMB_FIELDS, FUZZY_TEXT_FIELDS, CHECKBOX_FIELDS,
    DIGIT_COMB_CELLS, TEXT_COMB_CELLS, CHECKBOX_BOXES,
)

TEMPLATE_P1 = Path(__file__).parent / "form_template_p1.png"
TEMPLATE_P2 = Path(__file__).parent / "form_template_p2.png"

FONT_CANDIDATES = [
    "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
]

FIRST_NAMES = ["Ján", "Mária", "Peter", "Anna", "Tomáš", "Zuzana",
               "Martin", "Jana", "Michal", "Eva", "Ľubomír", "Katarína"]
LAST_NAMES  = ["Novák", "Horváth", "Kováč", "Varga", "Szabó", "Tóth",
               "Lukáč", "Blaho", "Šimko", "Krajčí", "Ďurica", "Žiak"]
TITULY      = ["", "", "", "Ing", "Mgr", "JUDr", "Bc", "PhD"]
ULICE       = ["Hlavná", "Štúrova", "Mierová", "Lipová", "Záhradná", "Školská",
               "Nová", "Dlhá", "Krátka", "Slnečná", "Poľná", "Brezová"]
OBCE        = ["Bratislava", "Košice", "Prešov", "Žilina", "Nitra", "Trnava",
               "Martin", "Trenčín", "Poprad", "Levice", "Čadca", "Michalovce"]


def make_address(rng: random.Random) -> dict:
    """Synthetic Slovak address — diacritic-rich to exercise the Latin model."""
    return {
        "titul": rng.choice(TITULY),
        "ulica": rng.choice(ULICE),
        "supisne_cislo": str(rng.randint(1, 4999)),
        "psc": f"{rng.randint(10,99)}{rng.randint(0,9)}{rng.randint(10,99)}",  # 5 digits
        "obec": rng.choice(OBCE),
        "stat": "Slovensko",      # fits the 10-cell Štát comb
    }


def datum_from_rc(rc: str) -> str:
    """Date of birth (DDMMYYYY) derived from a rodné číslo 'YYMMDD/NNNC'."""
    yy, mm, dd = int(rc[0:2]), int(rc[2:4]) % 50, int(rc[4:6])
    year = 1900 + yy if yy >= 30 else 2000 + yy
    return f"{dd:02d}{mm:02d}{year:04d}"


FIRMY = ["TECHNOSK", "ALFA STAV", "MONTÁŽE PLUS", "DREVOVÝROBA",
         "KOVOSLUŽBA", "GASTRO CENTRUM", "LOGISTIKA SK", "ELEKTRO MORAVA"]


def make_employer(rng: random.Random, full: bool = False) -> dict:
    """Synthetic III. ODDIEL employer (a právnická osoba / company).

    full=True also fills the fyzická-osoba name fields (not realistic — a real
    form is one or the other — but useful for a complete test of every field).
    """
    addr = make_address(rng)
    return {
        "zam_dic": str(rng.randint(1000000000, 9999999999)),   # 10-digit DIČ
        "zam_priezvisko": rng.choice(LAST_NAMES).upper() if full else "",
        "zam_meno": rng.choice(FIRST_NAMES).upper() if full else "",
        "zam_titul": "ING" if full else "",
        "zam_obchodne_meno": (rng.choice(FIRMY) + " SRO").upper(),
        "zam_ulica": addr["ulica"].upper(),
        "zam_supisne_cislo": str(rng.randint(1, 4999)),
        "zam_psc": f"{rng.randint(10,99)}{rng.randint(0,9)}{rng.randint(10,99)}",
        "zam_obec": addr["obec"].upper(),
        "zam_stat": "SLOVENSKO",
        "vypracoval": (rng.choice(LAST_NAMES) + " " + rng.choice(FIRST_NAMES)).upper(),
        "potvrdenie_datum": f"{rng.randint(1,28):02d}0125",      # DD.01.2025
    }


def make_page2_extras(rng: random.Random, full: bool = False) -> dict:
    """Page-2 II. ODDIEL continuation income, month grids, child-bonus table."""
    d = {
        "p2_riadok_08a": str(make_numeric(rng, lo=0, hi=999)),
        "p2_riadok_09":  str(make_numeric(rng, lo=0, hi=999)),
        "p2_riadok_10":  str(make_numeric(rng, lo=0, hi=9999)),
        "p2_riadok_11":  str(make_numeric(rng, lo=0, hi=9999)),
        "p2_riadok_12":  str(make_numeric(rng, lo=0, hi=99999)),
    }
    # Month grids for riadok 10 & 13 (per-month booleans; master left blank).
    for pref in ("r10", "r13"):
        d[f"{pref}_mesiac_vsetky"] = False
        for i in range(1, 13):
            d[f"{pref}_mesiac_{i:02d}"] = True if full else rng.random() < 0.5
    # Child tax-bonus table: all 4 if full, else 1–3 filled.
    n_children = 4 if full else rng.randint(1, 3)
    for c in range(1, 5):
        filled = c <= n_children
        # short names so they fit the 11-cell box
        d[f"dieta{c}_meno"] = (rng.choice(LAST_NAMES)[:5].upper() if filled else "")
        d[f"dieta{c}_rod_cislo"] = make_valid_rc(rng) if filled else ""
        d[f"dieta{c}_mesiac_vsetky"] = False
        for i in range(1, 13):
            d[f"dieta{c}_mesiac_{i:02d}"] = filled and (True if full else rng.random() < 0.6)
    return d


def find_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_valid_rc(rng: random.Random) -> str:
    yy = rng.randint(50, 99)
    mm = rng.randint(1, 12)
    dd = rng.randint(1, 28)
    mm_enc = mm + 50 if rng.random() < 0.5 else mm
    while True:
        nnn = rng.randint(1, 999)
        nine = int(f"{yy:02d}{mm_enc:02d}{dd:02d}{nnn:03d}")
        c = (-(nine * 10)) % 11
        if c != 10:
            break
    return f"{yy:02d}{mm_enc:02d}{dd:02d}/{nnn:03d}{c}"


def make_invalid_rc(rng: random.Random) -> str:
    valid = make_valid_rc(rng)
    body, check = valid[:-1], int(valid[-1])
    bad = (check + rng.randint(1, 9)) % 10
    return body + str(bad)


def make_numeric(rng: random.Random, lo: int = 1000, hi: int = 99999) -> Decimal:
    whole = rng.randint(lo, hi)
    cents = rng.randint(0, 99)
    return Decimal(f"{whole}.{cents:02d}")


def make_ground_truth(rng: random.Random, break_type: str | None,
                      full: bool = False) -> tuple[dict, list[str]]:
    # r01 = r01a + r01b  (total income = main + small exempt supplement)
    r01a = make_numeric(rng, lo=5000, hi=80000)
    r01b_whole = rng.randint(0, 3000)
    r01b = Decimal(f"{r01b_whole}.{rng.randint(0, 99):02d}")
    r01  = r01a + r01b

    # r02 = r02a + r02b  (total mandatory contributions)
    # Employee mandatory contributions: ~13.4% of r01a (health 4% + social 9.4%)
    r02a = (r01a * Decimal("0.134")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r02b = Decimal("0.00")
    r02  = r02a + r02b

    r03_correct = r01 - r02   # taxable base

    # r04–r09: plausible but not cross-validated against each other in this MVP
    r04 = (r03_correct * Decimal("0.19")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r05 = Decimal("0.00")
    # riadok_06/08/08a/09 have only 3 integer cells (max 999.99)
    # riadok_07 has 4 integer cells (max 9999.99)
    r06  = make_numeric(rng, lo=0, hi=999)
    r07  = Decimal("0.00")
    r08  = make_numeric(rng, lo=0, hi=999)
    r08a = Decimal("0.00")
    r09  = make_numeric(rng, lo=0, hi=999)

    broken_fields = []

    if break_type in ("arithmetic", "both"):
        r03 = make_numeric(rng)
        while r03 == r03_correct:
            r03 = make_numeric(rng)
        broken_fields.append("riadok_03")
    else:
        r03 = r03_correct

    if break_type in ("rod_cislo", "both"):
        rc = make_invalid_rc(rng)
        broken_fields.append("rod_cislo")
    else:
        rc = make_valid_rc(rng)

    name = rng.choice(LAST_NAMES) + " " + rng.choice(FIRST_NAMES)

    addr = make_address(rng)

    # Dátum narodenia derived from rodné číslo so the cross-check passes for
    # valid samples. The "datum" break mode desyncs it to exercise the check.
    if break_type == "datum":
        datum = datum_from_rc(make_valid_rc(rng))  # unrelated DOB
        broken_fields.append("datum_narodenia")
    else:
        datum = datum_from_rc(rc if re.fullmatch(r"\d{6}/\d{4}", rc) else make_valid_rc(rng))

    # Text fields are UPPERCASE — real forms are filled in paličkové písmo
    # (block capitals). Matching stays case-insensitive (compare_text_fuzzy).
    gt = {
        "meno_zamestnanca": name.upper(),
        "rod_cislo": rc,
        "rod_cislo_p2": rc,          # page-2 top repeats the same rodné číslo
        "datum_narodenia": datum,
        "rok": "25",                       # 2-digit suffix → assessment year 2025
        "oprava": True if full else rng.random() < 0.15,
        "titul": "ING" if full else addr["titul"].upper(),
        "ulica": addr["ulica"].upper(),
        "supisne_cislo": addr["supisne_cislo"],
        "psc": addr["psc"],
        "obec": addr["obec"].upper(),
        "stat": addr["stat"].upper(),
        "danovnik_obmedzena": True if full else rng.random() < 0.3,
        **make_employer(rng, full),
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
        **make_page2_extras(rng, full),
    }
    return gt, broken_fields


def _draw_char_in_cell(draw: ImageDraw.ImageDraw, ch: str,
                       cell: list, font: ImageFont.ImageFont,
                       rng: random.Random) -> None:
    """Draw a single character centered inside a digit cell box."""
    x1, y1, x2, y2 = cell
    tw = draw.textlength(ch, font=font)
    _, _, _, th = draw.textbbox((0, 0), ch, font=font)
    base_x = x1 + (x2 - x1 - tw) / 2
    base_y = y1 + (y2 - y1 - th) / 2
    jx = rng.randint(-1, 1)
    jy = rng.randint(-1, 1)
    draw.text((base_x + jx, base_y + jy), ch, fill=0, font=font)


def _draw_numeric_in_cells(draw: ImageDraw.ImageDraw, value_str: str,
                            cells: list, font: ImageFont.ImageFont,
                            rng: random.Random) -> None:
    """Place each digit of value_str into its individual digit cell.

    The last _N_DECIMAL cells receive the cents digits (after the decimal point).
    Integer digits are right-aligned into the remaining cells.
    """
    n_dec = _N_DECIMAL
    int_cells = cells[:-n_dec]
    dec_cells = cells[-n_dec:]

    if "." in value_str:
        int_part, dec_part = value_str.split(".", 1)
    else:
        int_part, dec_part = value_str, "00"

    dec_part = (dec_part + "00")[:n_dec]  # ensure exactly n_dec digits

    # Right-align integer digits: fill from the rightmost integer cell backwards
    int_digits = list(int_part)
    for offset, cell in enumerate(reversed(int_cells)):
        idx = len(int_digits) - 1 - offset
        if idx >= 0:
            _draw_char_in_cell(draw, int_digits[idx], cell, font, rng)

    # Decimal digits go left-to-right in decimal cells
    for ch, cell in zip(dec_part, dec_cells):
        _draw_char_in_cell(draw, ch, cell, font, rng)


def draw_values(img: Image.Image, draw: ImageDraw.ImageDraw,
                gt: dict, field_boxes: dict, font: ImageFont.ImageFont,
                rng: random.Random) -> None:
    """Render synthetic handwritten values into the given field boxes."""
    for key, box in field_boxes.items():
        val = gt.get(key)
        if val is None:
            continue
        x1, y1, x2, y2 = box

        if key in MONTH_FIELDS or key in CHECKBOX_FIELDS:
            if val:
                jx = rng.randint(-2, 2)
                jy = rng.randint(-2, 2)
                draw.line([(x1+3+jx, y1+3+jy), (x2-3+jx, y2-3+jy)], fill=0, width=2)
                draw.line([(x2-3+jx, y1+3+jy), (x1+3+jx, y2-3+jy)], fill=0, width=2)
        elif key in NUMERIC_FIELDS:
            _draw_numeric_in_cells(draw, str(val), INCOME_DIGIT_CELLS[key], font, rng)
        elif key in RC_FIELDS:
            # rodné číslo (employee or child): 10 digits, one per cell; "/" pre-printed.
            digits = str(val).replace("/", "")
            for ch, cell in zip(digits, RC_CELLS[key]):
                _draw_char_in_cell(draw, ch, cell, font, rng)
        elif key in DIGIT_COMB_FIELDS:
            # datum/rok/psc/supisne: digits RIGHT-aligned — last digit in the
            # rightmost cell, filling leftward (form convention; leaves any
            # spare cells blank on the LEFT).
            for ch, cell in zip(reversed(str(val)), reversed(DIGIT_COMB_CELLS[key])):
                _draw_char_in_cell(draw, ch, cell, font, rng)
        elif key in FUZZY_TEXT_FIELDS:
            # titul/ulica/obec/stat: one letter per comb cell (may overflow = truncated).
            for ch, cell in zip(str(val), TEXT_COMB_CELLS[key]):
                if ch == " ":
                    continue
                _draw_char_in_cell(draw, ch, cell, font, rng)
        elif key == "meno_zamestnanca":
            # surname → Priezvisko comb, first name → Meno comb, one letter per cell.
            parts = str(val).split(" ", 1)
            surname = parts[0]
            given = parts[1] if len(parts) > 1 else ""
            for ch, cell in zip(surname, PRIEZVISKO_CELLS):
                _draw_char_in_cell(draw, ch, cell, font, rng)
            for ch, cell in zip(given, MENO_CELLS):
                _draw_char_in_cell(draw, ch, cell, font, rng)
        else:
            text = str(val)
            _, _, _, th = draw.textbbox((0, 0), "M", font=font)
            base_y = y1 + (y2 - y1 - th) // 2
            base_x = x1 + 8
            cx = base_x
            for ch in text:
                jx = rng.randint(-2, 2)
                jy = rng.randint(-2, 2)
                draw.text((cx + jx, base_y + jy), ch, fill=0, font=font)
                cx += draw.textlength(ch, font=font)


def add_scan_noise(img: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.uniform(-1.0, 1.0)   # >±1° shifts row borders 6+px into adjacent cells
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))
    arr = np.array(img)
    mask_pepper = rng.random((arr.shape[0], arr.shape[1])) < 0.01
    mask_salt   = rng.random((arr.shape[0], arr.shape[1])) < 0.01
    arr[mask_pepper] = 0
    arr[mask_salt]   = 255
    return Image.fromarray(arr)


def generate_sample(idx: int, break_type: str | None,
                    rng: random.Random, font_fill: ImageFont.ImageFont,
                    full: bool = False
                    ) -> tuple[Image.Image, Image.Image, dict]:
    gt, broken_fields = make_ground_truth(rng, break_type, full)

    p1 = Image.open(TEMPLATE_P1).convert("L")
    draw1 = ImageDraw.Draw(p1)
    draw_values(p1, draw1, gt, FIELD_BOXES_P1, font_fill, rng)
    p1 = add_scan_noise(p1, rng)

    p2 = Image.open(TEMPLATE_P2).convert("L")
    draw2 = ImageDraw.Draw(p2)
    draw_values(p2, draw2, gt, FIELD_BOXES_P2, font_fill, rng)
    p2 = add_scan_noise(p2, rng)

    data = {
        "sample_id": f"sample_{idx:04d}",
        "_is_broken": break_type is not None,
        "_broken_fields": broken_fields,
        "_field_boxes_p1": FIELD_BOXES_P1,
        "_field_boxes_p2": FIELD_BOXES_P2,
        "ground_truth": gt,
    }
    return p1, p2, data


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic POT395 form samples.")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--out", type=str, default="samples/")
    parser.add_argument("--break-frac", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true",
                        help="emit ONE fully-filled sample (every field populated) for testing")
    args = parser.parse_args()

    for tmpl in [TEMPLATE_P1, TEMPLATE_P2]:
        if not tmpl.exists():
            raise FileNotFoundError(
                f"Template not found: {tmpl}\n"
                "Run: pdftoppm -r 150 -png newestpotvrdenietemplate.pdf /tmp/pot395_page\n"
                "     cp /tmp/pot395_page-1.png form_template_p1.png\n"
                "     cp /tmp/pot395_page-2.png form_template_p2.png"
            )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    class _HybridRng(random.Random):
        def random(self, shape=None):
            if shape is not None:
                return np_rng.random(shape)
            return super().random()
    rng.__class__ = _HybridRng

    font_fill = find_font(20)

    if args.full:
        p1, p2, data = generate_sample(1, None, rng, font_fill, full=True)
        stem = "sample_full"
        p1.save(out_dir / f"{stem}_p1.png")
        p2.save(out_dir / f"{stem}_p2.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Fully-filled sample → {out_dir}/{stem}_p1.png + _p2.png + .json")
        return

    n_broken = math.ceil(args.n * args.break_frac)
    break_types = ["arithmetic", "rod_cislo", "datum", "both"]
    configs = [break_types[i % len(break_types)] for i in range(n_broken)] + [None] * (args.n - n_broken)
    rng.shuffle(configs)

    print(f"Generating {args.n} samples ({n_broken} broken) → {out_dir}/")
    for idx, break_type in enumerate(configs, 1):
        p1, p2, data = generate_sample(idx, break_type, rng, font_fill)
        stem = f"sample_{idx:04d}"
        p1.save(out_dir / f"{stem}_p1.png")
        p2.save(out_dir / f"{stem}_p2.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        status = f"[BROKEN: {','.join(data['_broken_fields'])}]" if data["_is_broken"] else "[valid]"
        print(f"  {stem}  {status}")

    print("Done.")


if __name__ == "__main__":
    main()
