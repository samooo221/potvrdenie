#!/usr/bin/env python3
"""generate_samples.py — synthetic POT395 form sample generator (real form template).

Produces a SPEC-DRIVEN set of samples that (a) render as realistic paličkové písmo
(hand-printed block capitals — the way POT395 is filled by hand) plus a few
typewriter/printer samples, and (b) collectively exercise every deterministic
validation / escalation code path. Rendering realism lives in render_hand.py.

Usage:
    python generate_samples.py --out samples/ --seed 42        # all 20 specs
    python generate_samples.py --out samples/ --n 5            # first 5 specs
    python generate_samples.py --full                          # legacy: one full sample

Per sample:
    samples/sample_NNNN_p1.png  — page 1 (blank form + synthetic fill)
    samples/sample_NNNN_p2.png  — page 2
    samples/sample_NNNN.json    — ground truth + field boxes + break/gazetteer markers

HONESTY: these are SIMULATED block letters. They are a far better visual proxy and
a much stronger validation stress-test than a printed monospace, but they do NOT
satisfy Phase 4's real intent — an honest accuracy number on REAL pen-on-paper
forms. A self-rendered image still shares the generator's stroke idea and sits on
the exact field_defs coordinates, so it cannot reveal the recognizer's true failure
modes on genuine handwriting. Real-handwriting validation stays open (label real
forms via eval_handwriting.py's <stem>.labels.json path).

Synthetic data only. Never use real taxpayer data.
"""
import argparse
import json
import random
import re
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

from PIL import Image

import render_hand as rh
from gazetteer import _norm as _gznorm, gazetteer_match
from field_defs import (
    FIELD_BOXES_P1, FIELD_BOXES_P2,
    INCOME_DIGIT_CELLS, NUMERIC_FIELDS, MONTH_FIELDS, RC_FIELDS,
    RC_CELLS, PRIEZVISKO_CELLS, MENO_CELLS,
    DIGIT_COMB_FIELDS, FUZZY_TEXT_FIELDS, CHECKBOX_FIELDS,
    DIGIT_COMB_CELLS, TEXT_COMB_CELLS,
)

TEMPLATE_P1 = Path(__file__).parent / "form_template_p1.png"
TEMPLATE_P2 = Path(__file__).parent / "form_template_p2.png"
_DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Synthetic data pools (100% fabricated)
# ---------------------------------------------------------------------------
# Three diacritic-density tiers so samples span the easy → hard recognition range.
FIRST_NAMES_PLAIN = ["JAN", "PETER", "MARTIN", "TOMAS", "MICHAL", "MAREK", "LUKAS", "JOZEF"]
LAST_NAMES_PLAIN  = ["NOVAK", "HORAK", "BLAHO", "SOKOL", "POLAK", "URBAN", "MRAZ", "SIMON"]

FIRST_NAMES_LIGHT = ["JÁN", "MÁRIA", "PETER", "ANNA", "TOMÁŠ", "ZUZANA", "EVA", "JANA"]
LAST_NAMES_LIGHT  = ["NOVÁK", "HORVÁTH", "KOVÁČ", "VARGA", "TÓTH", "LUKÁČ", "ŠIMKO", "ŽIAK"]

FIRST_NAMES_HEAVY = ["ĽUBOMÍR", "ŽOFIA", "ONDREJ", "SOŇA", "ĽUDOVÍT", "BOHDANA", "MÚČKA", "ŠTEFÁNIA"]
LAST_NAMES_HEAVY  = ["ĎURČOVIČ", "ĽUPTÁK", "HRÍBKOVÁ", "ŤAPÁK", "ŽIŠKO", "ŠŤASTNÝ", "NÔTOVÁ", "ČAČKO"]

ULICE = ["HLAVNÁ", "ŠTÚROVA", "MIEROVÁ", "LIPOVÁ", "ZÁHRADNÁ", "ŠKOLSKÁ",
         "NOVÁ", "DLHÁ", "KRÁTKA", "SLNEČNÁ", "POĽNÁ", "BREZOVÁ"]

FIRMY = ["TECHNOSK", "ALFA STAV", "MONTÁŽE PLUS", "DREVOVÝROBA",
         "KOVOSLUŽBA", "GASTRO CENTRUM", "LOGISTIKA SK", "ELEKTRO MORAVA"]


def _load_register(fname: str) -> list[str]:
    p = _DATA / fname
    out = []
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


# Valid (in-register) closed-set values so a "valid" sample truly snaps to the
# gazetteer. Filtered to what fits the form's comb cells (obec 19, štát 10, titul 4).
OBCE_REG = [o.upper() for o in _load_register("obce.txt")
            if len(o.replace(" ", "")) <= 18] or ["BRATISLAVA"]
VALID_TITUL = sorted({_gznorm(t).upper() for t in _load_register("titul.txt")
                      if 0 < len(_gznorm(t)) <= 4}) or ["ING", "MGR", "BC"]

# Out-of-register values for gazetteer-MISS samples. Filtered at load time to
# entries that fit the comb cells AND genuinely do NOT snap to the register — Slovak
# country names cluster on the "-sko" suffix, so a naive pick (FÍNSKO→Írsko,
# NÓRSKO→Poľsko) silently matches. This guarantees a real miss → confidence-flag.
def _miss_pool(field: str, candidates: list[str], max_len: int) -> list[str]:
    out = [c for c in candidates
           if len(c.replace(" ", "")) <= max_len and not gazetteer_match(field, c)["matched"]]
    if not out:
        raise RuntimeError(f"no non-snapping miss values for {field}: {candidates}")
    return out

OBCE_MISS  = _miss_pool("obec", ["HÔRKA", "DRÁHOVCE", "STRÁŽKY", "BOHDANOVCE", "ZÁRIEČIE"], 18)
STAT_MISS  = _miss_pool("stat", ["KANADA", "MEXIKO", "EGYPT", "BRAZÍLIA", "JAPONSKO", "TURECKO"], 10)
TITUL_MISS = _miss_pool("titul", ["MBA", "DBA", "LLM"], 4)


def _name_pools(style: str):
    if style == "heavy":
        return LAST_NAMES_HEAVY, FIRST_NAMES_HEAVY
    if style == "light":
        return LAST_NAMES_LIGHT, FIRST_NAMES_LIGHT
    return LAST_NAMES_PLAIN, FIRST_NAMES_PLAIN


# ---------------------------------------------------------------------------
# Rodné číslo / numbers
# ---------------------------------------------------------------------------
def datum_from_rc(rc: str) -> str:
    """Date of birth (DDMMYYYY) derived from a rodné číslo 'YYMMDD/NNNC'."""
    yy, mm, dd = int(rc[0:2]), int(rc[2:4]) % 50, int(rc[4:6])
    year = 1900 + yy if yy >= 30 else 2000 + yy
    return f"{dd:02d}{mm:02d}{year:04d}"


def make_valid_rc(rng: random.Random, sex: str | None = None) -> str:
    """Valid rodné číslo. sex='F' forces the +50 month encoding, 'M' forbids it."""
    yy = rng.randint(50, 99)
    mm = rng.randint(1, 12)
    dd = rng.randint(1, 28)
    if sex == "F":
        mm_enc = mm + 50
    elif sex == "M":
        mm_enc = mm
    else:
        mm_enc = mm + 50 if rng.random() < 0.5 else mm
    while True:
        nnn = rng.randint(1, 999)
        nine = int(f"{yy:02d}{mm_enc:02d}{dd:02d}{nnn:03d}")
        c = (-(nine * 10)) % 11
        if c != 10:
            break
    return f"{yy:02d}{mm_enc:02d}{dd:02d}/{nnn:03d}{c}"


def make_invalid_rc(rng: random.Random, sex: str | None = None) -> str:
    """Valid format, wrong mod-11 check digit (isolates the mod-11 constraint)."""
    valid = make_valid_rc(rng, sex)
    body, check = valid[:-1], int(valid[-1])
    bad = (check + rng.randint(1, 9)) % 10
    return body + str(bad)


def make_numeric(rng: random.Random, lo: int = 1000, hi: int = 99999) -> Decimal:
    return Decimal(f"{rng.randint(lo, hi)}.{rng.randint(0, 99):02d}")


def _make_psc(rng: random.Random, n: int = 5) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def _make_dic(rng: random.Random, n: int = 10) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


# ---------------------------------------------------------------------------
# Income rows (honour the per-sample income profile; keep r01=r01a+r01b etc.)
# ---------------------------------------------------------------------------
def _compute_income(rng: random.Random, profile: str) -> dict:
    if profile == "large":
        r01a = make_numeric(rng, lo=1_000_000, hi=40_000_000)   # fills 7-8 integer cells
        r01b = Decimal(f"{rng.randint(0, 9999)}.{rng.randint(0, 99):02d}")
    elif profile == "small":
        r01a = make_numeric(rng, lo=100, hi=999)
        r01b = Decimal(f"{rng.randint(0, 99)}.{rng.randint(0, 99):02d}")
    else:                                                        # typical / zero / mix
        r01a = make_numeric(rng, lo=5000, hi=80000)
        r01b = Decimal(f"{rng.randint(0, 3000)}.{rng.randint(0, 99):02d}")

    if profile == "zero":
        r01b = Decimal("0.00")

    r01 = r01a + r01b
    r02a = (r01a * Decimal("0.134")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r02b = Decimal("0.00")
    r02 = r02a + r02b
    r03 = r01 - r02
    r04 = (r03 * Decimal("0.19")).quantize(Decimal("0.01"), rounding=ROUND_DOWN)
    r05 = Decimal("0.00")

    if profile == "zero":
        r06 = r07 = r08 = r08a = r09 = Decimal("0.00")
    else:
        # riadok_06/08/08a/09 have only 3 integer cells (≤999.99); 07 has 4 (≤9999.99)
        r06 = make_numeric(rng, lo=0, hi=999)
        r07 = Decimal("0.00")
        r08 = make_numeric(rng, lo=0, hi=999)
        r08a = Decimal("0.00")
        r09 = make_numeric(rng, lo=0, hi=999)

    return dict(r01=r01, r01a=r01a, r01b=r01b, r02=r02, r02a=r02a, r02b=r02b,
                r03=r03, r04=r04, r05=r05, r06=r06, r07=r07, r08=r08, r08a=r08a, r09=r09)


# ---------------------------------------------------------------------------
# Employer (III. ODDIEL) + page-2 continuation (II. ODDIEL) + child bonus table
# ---------------------------------------------------------------------------
def _make_employer(rng: random.Random, kind: str, names_style: str,
                   full: bool = False, bad_dic: bool = False) -> dict:
    sur, giv = _name_pools(names_style)
    is_company = (kind == "company")
    return {
        "zam_dic": _make_dic(rng, n=9 if bad_dic else 10),
        "zam_priezvisko": (rng.choice(sur) if (not is_company or full) else ""),
        "zam_meno":       (rng.choice(giv) if (not is_company or full) else ""),
        "zam_titul":      ("ING" if (not is_company or full) else ""),
        "zam_obchodne_meno": ((rng.choice(FIRMY) + " SRO") if (is_company or full) else ""),
        "zam_ulica": rng.choice(ULICE),
        "zam_supisne_cislo": str(rng.randint(1, 4999)),
        "zam_psc": _make_psc(rng),
        "zam_obec": rng.choice(OBCE_REG),
        "zam_stat": "SLOVENSKO",
        "vypracoval": rng.choice(sur) + " " + rng.choice(giv),
        "potvrdenie_datum": f"{rng.randint(1, 28):02d}0125",     # DD.01.2025
    }


def _month_pattern(rng: random.Random, mode: str):
    if mode == "master":
        return True, [False] * 12
    if mode == "all":
        return False, [True] * 12
    if mode == "none":
        return False, [False] * 12
    return False, [rng.random() < 0.5 for _ in range(12)]         # partial


def _make_page2_extras(rng: random.Random, spec: dict) -> dict:
    profile = spec["income"]
    d = {
        "p2_riadok_08a": str(make_numeric(rng, lo=0, hi=999)),
        "p2_riadok_09":  str(make_numeric(rng, lo=0, hi=999)),
        "p2_riadok_10":  str(make_numeric(rng, lo=0, hi=9999)),
        "p2_riadok_11":  str(make_numeric(rng, lo=0, hi=9999)),
        "p2_riadok_12":  str(make_numeric(rng, lo=1_000_000, hi=90_000_000)
                             if profile in ("large", "mix") else make_numeric(rng, lo=0, hi=99999)),
    }
    for pref in ("r10", "r13"):
        vsetky, months = _month_pattern(rng, spec["months"])
        d[f"{pref}_mesiac_vsetky"] = vsetky
        for i in range(1, 13):
            d[f"{pref}_mesiac_{i:02d}"] = months[i - 1]

    sur, _giv = _name_pools(spec["names"])
    n_children = spec["children"]
    for c in range(1, 5):
        filled = c <= n_children
        d[f"dieta{c}_meno"] = (rng.choice(sur)[:9] if filled else "")
        d[f"dieta{c}_rod_cislo"] = make_valid_rc(rng) if filled else ""
        d[f"dieta{c}_mesiac_vsetky"] = False
        for i in range(1, 13):
            d[f"dieta{c}_mesiac_{i:02d}"] = filled and (
                True if spec["months"] == "all" else rng.random() < 0.6)
    return d


# ---------------------------------------------------------------------------
# Ground truth (spec-driven)
# ---------------------------------------------------------------------------
def make_ground_truth(rng: random.Random, spec: dict) -> tuple[dict, list[str], list[str]]:
    sex = spec["sex"]
    brk = spec.get("brk")
    gaz = spec.get("gaz", [])
    full = spec.get("full", False)
    empty_opt = spec.get("empty_optionals", False)

    broken_fields: list[str] = []
    gaz_miss: list[str] = []

    inc = _compute_income(rng, spec["income"])

    # --- rodné číslo (employee) ---
    if brk == "mod11":
        rc = make_invalid_rc(rng, sex)
        broken_fields.append("rod_cislo")
    else:
        rc = make_valid_rc(rng, sex)

    # --- page-2 top rodné číslo (page matching) ---
    if brk == "p2_rc":
        rc_p2 = make_valid_rc(rng, sex)
        while rc_p2 == rc:
            rc_p2 = make_valid_rc(rng, sex)
        broken_fields.append("rod_cislo_p2")
    else:
        rc_p2 = rc

    # --- dátum narodenia (cross-check with rod_cislo) ---
    if brk == "datum":
        datum = datum_from_rc(make_valid_rc(rng, sex))       # unrelated DOB
        broken_fields.append("datum_narodenia")
    else:
        datum = datum_from_rc(rc if re.fullmatch(r"\d{6}/\d{4}", rc) else make_valid_rc(rng, sex))

    # --- taxable-base arithmetic (r01 − r02 == r03) ---
    if brk == "arithmetic":
        r03 = make_numeric(rng)
        while r03 == inc["r03"]:
            r03 = make_numeric(rng)
        broken_fields.append("riadok_03")
    else:
        r03 = inc["r03"]

    # --- assessment year range ---
    if brk == "rok_range":
        rok = "5"                                            # malformed (len≠2) → out of range
        broken_fields.append("rok")
    else:
        rok = "25"

    # --- PSČ length ---
    if brk == "psc_len":
        psc = _make_psc(rng, n=4)
        broken_fields.append("psc")
    else:
        psc = _make_psc(rng, n=5)

    # --- names + address text (closed sets, with optional gazetteer miss) ---
    sur, giv = _name_pools(spec["names"])
    name = rng.choice(sur) + " " + rng.choice(giv)

    titul = "" if empty_opt else ("" if rng.random() < 0.3 else rng.choice(VALID_TITUL))
    if "titul" in gaz:
        titul = rng.choice(TITUL_MISS); gaz_miss.append("titul")
    obec = rng.choice(OBCE_REG)
    if "obec" in gaz:
        obec = rng.choice(OBCE_MISS); gaz_miss.append("obec")
    stat = "SLOVENSKO"
    if "stat" in gaz:
        stat = rng.choice(STAT_MISS); gaz_miss.append("stat")

    emp = _make_employer(rng, spec["employer"], spec["names"], full=full,
                         bad_dic=(brk == "dic_len"))
    if brk == "dic_len":
        broken_fields.append("zam_dic")
    if empty_opt:
        emp["vypracoval"] = ""

    gt = {
        "meno_zamestnanca": name,
        "rod_cislo": rc,
        "rod_cislo_p2": rc_p2,
        "datum_narodenia": datum,
        "rok": rok,
        "oprava": bool(spec["oprava"]),
        "titul": titul,
        "ulica": rng.choice(ULICE),
        "supisne_cislo": "" if empty_opt else str(rng.randint(1, 4999)),
        "psc": psc,
        "obec": obec,
        "stat": stat,
        "danovnik_obmedzena": bool(spec["obmedz"]),
        **emp,
        "riadok_01":  str(inc["r01"]),
        "riadok_01a": str(inc["r01a"]),
        "riadok_01b": str(inc["r01b"]),
        "riadok_02":  str(inc["r02"]),
        "riadok_02a": str(inc["r02a"]),
        "riadok_02b": str(inc["r02b"]),
        "riadok_03":  str(r03),
        "riadok_04":  str(inc["r04"]),
        "riadok_05":  str(inc["r05"]),
        "riadok_06":  str(inc["r06"]),
        "riadok_07":  str(inc["r07"]),
        "riadok_08":  str(inc["r08"]),
        "riadok_08a": str(inc["r08a"]),
        "riadok_09":  str(inc["r09"]),
        **_make_page2_extras(rng, spec),
    }
    return gt, broken_fields, gaz_miss


# ---------------------------------------------------------------------------
# Drawing — route each field class to the shared render_hand pipeline
# ---------------------------------------------------------------------------
def draw_values(img: Image.Image, gt: dict, field_boxes: dict,
                profile: rh.WriterProfile, rng: random.Random) -> None:
    for key, box in field_boxes.items():
        val = gt.get(key)
        if val is None:
            continue
        x1, y1, x2, y2 = box

        if key in MONTH_FIELDS or key in CHECKBOX_FIELDS:
            if val:
                rh.draw_cross(img, x1, y1, x2, y2, profile, rng)
        elif key in NUMERIC_FIELDS:
            rh.draw_numeric(img, str(val), key, profile, rng)
        elif key in RC_FIELDS:
            digits = str(val).replace("/", "")
            rh.draw_chars_in_cells(img, digits, RC_CELLS[key], profile, rng, align="left")
        elif key in DIGIT_COMB_FIELDS:
            # datum/rok/psc/supisne/DIČ: digits RIGHT-aligned (form convention).
            rh.draw_chars_in_cells(img, str(val), DIGIT_COMB_CELLS[key], profile, rng, align="right")
        elif key in FUZZY_TEXT_FIELDS:
            rh.draw_chars_in_cells(img, str(val), TEXT_COMB_CELLS[key], profile, rng, align="left")
        elif key == "meno_zamestnanca":
            parts = str(val).split(" ", 1)
            rh.draw_chars_in_cells(img, parts[0], PRIEZVISKO_CELLS, profile, rng, align="left")
            if len(parts) > 1:
                rh.draw_chars_in_cells(img, parts[1], MENO_CELLS, profile, rng, align="left")
        else:
            rh.draw_text(img, str(val), x1, y1, x2, y2, profile, rng)


def generate_sample(idx: int, spec: dict, rng: random.Random,
                    pal_pool: list, pr_pool: list
                    ) -> tuple[Image.Image, Image.Image, dict]:
    gt, broken_fields, gaz_miss = make_ground_truth(rng, spec)

    pool = pr_pool if spec["style"] == "printed" else pal_pool
    profile = pool[idx % len(pool)]

    p1 = Image.open(TEMPLATE_P1).convert("L")
    draw_values(p1, gt, FIELD_BOXES_P1, profile, rng)
    p1 = rh.add_paper_texture(p1, profile, rng)

    p2 = Image.open(TEMPLATE_P2).convert("L")
    draw_values(p2, gt, FIELD_BOXES_P2, profile, rng)
    p2 = rh.add_paper_texture(p2, profile, rng)

    data = {
        "sample_id": f"sample_{idx:04d}",
        "_is_broken": bool(broken_fields),
        "_broken_fields": broken_fields,
        "_gazetteer_miss": gaz_miss,
        "_style": spec["style"],
        "_writer": profile.name,
        "_field_boxes_p1": FIELD_BOXES_P1,
        "_field_boxes_p2": FIELD_BOXES_P2,
        "ground_truth": gt,
    }
    return p1, p2, data


# ---------------------------------------------------------------------------
# The 20-sample coverage matrix (13 valid + 7 broken; ~15 paličkové / 5 printed)
# ---------------------------------------------------------------------------
def _spec(children, sex, months, oprava, obmedz, employer, income, names, style,
          brk=None, gaz=(), full=False, empty_optionals=False) -> dict:
    return dict(children=children, sex=sex, months=months, oprava=oprava,
                obmedz=obmedz, employer=employer, income=income, names=names,
                style=style, brk=brk, gaz=list(gaz), full=full,
                empty_optionals=empty_optionals)


SAMPLE_SPECS = [
    _spec(0, "M", "master",  False, False, "company", "typical", "plain", "palickove"),
    _spec(1, "F", "partial", True,  False, "company", "zero",    "light", "palickove"),
    _spec(2, "M", "none",    False, True,  "person",  "typical", "plain", "printed"),
    _spec(3, "F", "partial", False, False, "company", "large",   "heavy", "palickove"),
    _spec(0, "M", "master",  True,  True,  "person",  "typical", "plain", "palickove", brk="arithmetic"),
    _spec(4, "F", "all",     False, False, "company", "large",   "heavy", "palickove"),
    _spec(1, "M", "partial", False, True,  "company", "zero",    "plain", "printed"),
    _spec(2, "F", "partial", True,  False, "person",  "typical", "light", "palickove", brk="mod11"),
    _spec(0, "M", "none",    False, False, "company", "typical", "plain", "palickove", empty_optionals=True),
    _spec(3, "F", "partial", False, True,  "company", "typical", "heavy", "palickove"),
    _spec(1, "M", "master",  True,  False, "person",  "typical", "plain", "palickove", brk="datum"),
    _spec(2, "M", "partial", False, False, "company", "typical", "light", "printed",   gaz=("obec",)),
    _spec(0, "F", "none",    False, True,  "company", "small",   "plain", "printed"),
    _spec(4, "M", "all",     True,  False, "company", "typical", "heavy", "palickove", brk="p2_rc"),
    _spec(2, "F", "partial", False, False, "person",  "typical", "light", "palickove", gaz=("stat", "titul")),
    _spec(1, "M", "master",  False, True,  "company", "typical", "plain", "palickove", brk="psc_len"),
    _spec(3, "F", "partial", True,  False, "company", "large",   "heavy", "palickove"),
    _spec(0, "M", "none",    False, False, "company", "typical", "plain", "printed",   brk="dic_len"),
    _spec(2, "M", "partial", False, True,  "person",  "typical", "light", "palickove"),
    _spec(4, "F", "all",     True,  True,  "company", "mix",     "heavy", "palickove", brk="rok_range", full=True),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic POT395 form samples.")
    parser.add_argument("--n", type=int, default=0, help="limit to first N specs (0 = all 20)")
    parser.add_argument("--out", type=str, default="samples/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--break-frac", type=float, default=0.0, help="(ignored; specs are explicit)")
    parser.add_argument("--full", action="store_true",
                        help="legacy: emit ONE fully-filled sample (sample_full)")
    args = parser.parse_args()

    for tmpl in (TEMPLATE_P1, TEMPLATE_P2):
        if not tmpl.exists():
            raise FileNotFoundError(f"Template not found: {tmpl}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    pal_pool, pr_pool, note = rh.build_profiles()
    print(f"Render: {note}")

    if args.full:
        spec = SAMPLE_SPECS[-1]                              # the fully-populated spec
        p1, p2, data = generate_sample(1, spec, rng, pal_pool, pr_pool)
        stem = "sample_full"
        p1.save(out_dir / f"{stem}_p1.png")
        p2.save(out_dir / f"{stem}_p2.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Fully-filled sample → {out_dir}/{stem}_*.png + .json")
        return

    specs = SAMPLE_SPECS[:args.n] if args.n else SAMPLE_SPECS
    n_broken = sum(1 for s in specs if s["brk"])
    n_printed = sum(1 for s in specs if s["style"] == "printed")
    print(f"Generating {len(specs)} samples ({n_broken} broken, {n_printed} printed) → {out_dir}/")

    for idx, spec in enumerate(specs, 1):
        p1, p2, data = generate_sample(idx, spec, rng, pal_pool, pr_pool)
        stem = f"sample_{idx:04d}"
        p1.save(out_dir / f"{stem}_p1.png")
        p2.save(out_dir / f"{stem}_p2.png")
        (out_dir / f"{stem}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tags = []
        if data["_broken_fields"]:
            tags.append(f"BROKEN:{','.join(data['_broken_fields'])}")
        if data["_gazetteer_miss"]:
            tags.append(f"GAZ-MISS:{','.join(data['_gazetteer_miss'])}")
        status = f"[{' '.join(tags)}]" if tags else "[valid]"
        print(f"  {stem}  {spec['style']:9} {status}")

    print("Done.")


if __name__ == "__main__":
    main()
