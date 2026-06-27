#!/usr/bin/env python3
"""make_handwritten.py — one realistic hand-printed sample as handnewsample.png.

A debug/eyeball tool: renders page 1 with a fixed synthetic employee using the
SHARED realism pipeline in render_hand.py (the same one generate_samples.py uses),
so there is a single source of truth for stroke/ink/paper simulation.

Synthetic data only. No real taxpayer data.
"""
import argparse
import random
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from PIL import Image

import render_hand as rh
from generate_samples import datum_from_rc, make_valid_rc
from field_defs import (
    INCOME_DIGIT_CELLS, NUMERIC_FIELDS, RC_DIGIT_CELLS,
    PRIEZVISKO_CELLS, MENO_CELLS, DATUM_CELLS, ROK_CELLS, PSC_CELLS,
    SUPISNE_CELLS, TITUL_CELLS, ULICA_CELLS, OBEC_CELLS, STAT_CELLS,
    OPRAVA_BOX, DECL_BOX,
)

TEMPLATE_P1 = Path(__file__).parent / "form_template_p1.png"
OUT = Path(__file__).parent / "handnewsample.png"


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a hand-printed POT395 page-1 sample.")
    ap.add_argument("--align", choices=["left", "middle", "right"], default="right")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rng = random.Random(7)
    pal_pool, _printed, note = rh.build_profiles()
    profile = pal_pool[2 % len(pal_pool)]          # a full-coverage, neat hand
    print(f"Render: {note}  (writer={profile.name})")

    rc = make_valid_rc(rng, sex="M")

    r01a, r01b = Decimal("18450.00"), Decimal("320.50")
    r01 = r01a + r01b
    r02a = (r01a * Decimal("0.134")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r02 = r02a + Decimal("0.00")
    r03 = r01 - r02
    r04 = (r03 * Decimal("0.19")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)

    gt = {
        "meno_zamestnanca": "NOVÁK JÁN",
        "rod_cislo": rc,
        "datum_narodenia": datum_from_rc(rc),
        "rok": "25",
        "titul": "ING",
        "ulica": "HLAVNÁ",
        "supisne_cislo": "1234",
        "psc": "81101",
        "obec": "BRATISLAVA",
        "stat": "SLOVENSKO",
        "riadok_01": str(r01), "riadok_01a": str(r01a), "riadok_01b": str(r01b),
        "riadok_02": str(r02), "riadok_02a": str(r02a), "riadok_02b": "0.00",
        "riadok_03": str(r03), "riadok_04": str(r04), "riadok_05": "0.00",
        "riadok_06": "142.80", "riadok_07": "0.00", "riadok_08": "142.80",
        "riadok_08a": "0.00", "riadok_09": "142.80",
    }

    img = Image.open(TEMPLATE_P1).convert("L")

    parts = gt["meno_zamestnanca"].split(" ", 1)
    rh.draw_chars_in_cells(img, parts[0], PRIEZVISKO_CELLS, profile, rng, align=args.align)
    if len(parts) > 1:
        rh.draw_chars_in_cells(img, parts[1], MENO_CELLS, profile, rng, align=args.align)

    rh.draw_chars_in_cells(img, rc.replace("/", ""), RC_DIGIT_CELLS, profile, rng, align="left")
    rh.draw_chars_in_cells(img, gt["datum_narodenia"], DATUM_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["rok"], ROK_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["psc"], PSC_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["supisne_cislo"], SUPISNE_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["titul"], TITUL_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["ulica"], ULICA_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["obec"], OBEC_CELLS, profile, rng, align=args.align)
    rh.draw_chars_in_cells(img, gt["stat"], STAT_CELLS, profile, rng, align=args.align)

    for key in NUMERIC_FIELDS:
        if gt.get(key) is not None and key in INCOME_DIGIT_CELLS:
            rh.draw_numeric(img, str(gt[key]), key, profile, rng)

    img = rh.add_paper_texture(img, profile, rng)
    img.save(args.out)
    print(f"Saved {args.out}  ({img.width}×{img.height} px)")


if __name__ == "__main__":
    main()
