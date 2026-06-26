"""
field_defs.py — canonical field bounding boxes for the real POT395 form.

Measured from form_template_p1.png and form_template_p2.png rendered at 150 DPI
(1241×1755 px per page) by direct pixel analysis of the white cell regions.

Format: [x1, y1, x2, y2] — top-left inclusive, bottom-right exclusive.
"""

CANVAS_W, CANVAS_H = 1241, 1755

# ---------------------------------------------------------------------------
# Income field digit cells — each cell is one digit position (24px wide × 36px tall)
# The form has 10 possible x positions per row; rows 06-09 use only the rightmost ones.
# Positions 9 and 10 are the decimal (cents) cells, separated from 1-8 by a comma gap
# printed in the form at x≈992-1028.
#
# Cell column x extents (same for all income rows):
#   c01:(758,782)  c02:(788,812)  c03:(818,842)  c04:(848,872)
#   c05:(878,902)  c06:(908,932)  c07:(938,962)  c08:(968,992)
#   [comma gap x=992-1028]
#   c09:(1028,1052) c10:(1058,1082)
# ---------------------------------------------------------------------------

_CX = [
    (758, 782), (788, 812), (818, 842), (848, 872),   # c01-c04
    (878, 902), (908, 932), (938, 962), (968, 992),   # c05-c08
    (1028, 1052), (1058, 1082),                        # c09-c10 (decimal)
]
_N_DECIMAL = 2  # last 2 cells are always decimal (cents)


def _cells(col_indices, y1, y2):
    """Build list of [x1,y1,x2,y2] for the given 1-based column indices."""
    return [[_CX[i-1][0], y1, _CX[i-1][1], y2] for i in col_indices]


# Per-row digit cell lists.  Each list is left-to-right; last 2 entries are decimal.
# All rows have y extents matching the actual white-cell pixel bounds.
INCOME_DIGIT_CELLS = {
    "riadok_01":  _cells([1,2,3,4,5,6,7,8,9,10], 639, 675),   # 8+2
    "riadok_01a": _cells([1,2,3,4,5,6,7,8,9,10], 693, 729),
    "riadok_01b": _cells([1,2,3,4,5,6,7,8,9,10], 746, 782),
    "riadok_02":  _cells([1,2,3,4,5,6,7,8,9,10], 799, 835),
    "riadok_02a": _cells([1,2,3,4,5,6,7,8,9,10], 852, 888),
    "riadok_02b": _cells([1,2,3,4,5,6,7,8,9,10], 906, 942),
    "riadok_03":  _cells([1,2,3,4,5,6,7,8,9,10], 959, 995),
    "riadok_04":  _cells([1,2,3,4,5,6,7,8,9,10], 1012, 1048),
    "riadok_05":  _cells([1,2,3,4,5,6,7,8,9,10], 1065, 1101),
    "riadok_06":  _cells([6,7,8,9,10],            1123, 1159),  # 3+2
    "riadok_07":  _cells([5,6,7,8,9,10],          1176, 1212),  # 4+2
    "riadok_08":  _cells([6,7,8,9,10],            1230, 1266),  # 3+2
    "riadok_08a": _cells([6,7,8,9,10],            1283, 1319),  # 3+2
    "riadok_09":  _cells([6,7,8,9,10],            1336, 1372),  # 3+2
}


def _income_ocr_box(field_name, n_decimal=_N_DECIMAL):
    """Return the OCR bounding box covering all digit cells for a field.

    Spans from the leftmost to rightmost cell, INCLUDING the comma gap,
    so PaddleOCR gets the full visual context of the number.
    """
    cells = INCOME_DIGIT_CELLS[field_name]
    x1 = cells[0][0]
    x2 = cells[-1][2]
    y1 = cells[0][1]
    y2 = cells[-1][3]
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Employee identity comb fields (top of page 1)
#
# Rodné číslo lives in the TOP-LEFT header box (NOT the form body). It is a comb
# of 6 digit cells, a pre-printed "/", then 4 digit cells. Per-cell digit OCR
# (like the income rows) avoids the "/" misread that made whole-box OCR fail.
#
# Priezvisko (surname) and Meno (first name) are separate comb fields, each one
# white box per letter, divided by a vertical border. OCR crops each section as
# a word and combines "surname firstname".
# ---------------------------------------------------------------------------
RC_DIGIT_CELLS = [
    [60, 198, 90, 232], [92, 198, 122, 232], [122, 198, 152, 232],
    [152, 198, 182, 232], [182, 198, 212, 232], [214, 198, 250, 232],   # 6 left of "/"
    [264, 198, 300, 232], [302, 198, 332, 232],
    [332, 198, 362, 232], [364, 198, 396, 232],                          # 4 right of "/"
]
_RC_N_LEFT = 6  # digits before the pre-printed slash

# Name comb cells — one box per letter, 30px pitch, fill band y≈314–348.
PRIEZVISKO_CELLS = [[58 + 30 * i, 314, 58 + 30 * (i + 1), 348] for i in range(16)]
MENO_CELLS       = [[559 + 30 * i, 314, 559 + 30 * (i + 1), 348] for i in range(10)]

# OCR sub-boxes spanning each name section (Priezvisko / Meno), excluding Titul.
MENO_SURNAME_BOX = [58, 312, 514, 350]
MENO_GIVEN_BOX   = [559, 312, 859, 350]


def _comb_cells(x0, n, y1, y2, pitch=30):
    """Build n cells of `pitch` px starting at x0 — matches the form's comb fields."""
    return [[x0 + pitch * i, y1, x0 + pitch * (i + 1), y2] for i in range(n)]


# ---------------------------------------------------------------------------
# Additional page-1 fields (measured from form_template_p1.png, 1241×1755)
# ---------------------------------------------------------------------------

# --- Header digit combs ---
# Dátum narodenia: DD MM YYYY, 8 digit cells with printed group separators.
DATUM_CELLS = [
    [413, 200, 436, 230], [443, 200, 465, 230],                        # day
    [488, 200, 511, 230], [518, 200, 540, 230],                        # month
    [562, 200, 584, 230], [591, 200, 614, 230],
    [621, 200, 643, 230], [650, 200, 673, 230],                        # year (real boxes, clear of margin text)
]
# Rok: assessment year — "20" is pre-printed; two writable cells for the suffix.
ROK_CELLS = [[931, 186, 961, 212], [964, 186, 992, 212]]

# --- Address digit combs ---
PSC_CELLS = _comb_cells(60, 5, 498, 522)                                # postal code (5)
SUPISNE_CELLS = _comb_cells(962, 7, 422, 452)                           # house number (7; box y=420-453)

# --- Free-text comb fields: per-letter cells (for drawing) + span box (for OCR) ---
TITUL_CELLS = _comb_cells(1069, 4, 314, 348)
ULICA_CELLS = _comb_cells(60, 27, 432, 462)
OBEC_CELLS  = _comb_cells(257, 19, 498, 522)
STAT_CELLS  = _comb_cells(872, 10, 498, 522)

TITUL_BOX = [1069, 312, 1181, 350]
ULICA_BOX = [60, 430, 872, 464]
OBEC_BOX  = [257, 496, 826, 524]
STAT_BOX  = [872, 496, 1171, 524]

# --- Checkboxes (occupancy) ---
OPRAVA_BOX = [1056, 180, 1084, 202]
DECL_BOX   = [847, 543, 868, 563]

# Per-cell lists for the digit-comb fields, keyed by field name.
DIGIT_COMB_CELLS = {
    "datum_narodenia": DATUM_CELLS,
    "rok":             ROK_CELLS,
    "psc":             PSC_CELLS,
    "supisne_cislo":   SUPISNE_CELLS,
}
# Per-letter cells for text fields (drawing) + their OCR span boxes.
TEXT_COMB_CELLS = {
    "titul": TITUL_CELLS,
    "ulica": ULICA_CELLS,
    "obec":  OBEC_CELLS,
    "stat":  STAT_CELLS,
}
TEXT_FIELD_BOXES = {
    "titul": TITUL_BOX,
    "ulica": ULICA_BOX,
    "obec":  OBEC_BOX,
    "stat":  STAT_BOX,
}
CHECKBOX_BOXES = {
    "oprava":             OPRAVA_BOX,
    "danovnik_obmedzena": DECL_BOX,
}

# ---------------------------------------------------------------------------
# Page 1: employee identity + 14 income fields
# Income field boxes span all digit cells (including the comma gap) so that
# PaddleOCR sees the structured digit grid with printed comma separator.
# ---------------------------------------------------------------------------
def _digit_comb_box(cells):
    return [cells[0][0], cells[0][1], cells[-1][2], cells[-1][3]]


FIELD_BOXES_P1 = {
    # Header identity
    "rod_cislo":        [58, 196, 398, 234],   # top-left header box (6 + "/" + 4 digits)
    "datum_narodenia":  _digit_comb_box(DATUM_CELLS),
    "rok":              _digit_comb_box(ROK_CELLS),
    "oprava":           OPRAVA_BOX,

    # Employee section
    "meno_zamestnanca": [58, 312, 859, 350],   # Priezvisko + Meno combs (excludes Titul)
    "titul":            TITUL_BOX,
    "ulica":            ULICA_BOX,
    "supisne_cislo":    _digit_comb_box(SUPISNE_CELLS),
    "psc":              _digit_comb_box(PSC_CELLS),
    "obec":             OBEC_BOX,
    "stat":             STAT_BOX,
    "danovnik_obmedzena": DECL_BOX,

    # Income fields — bounding box of all digit cells per row
    **{name: _income_ocr_box(name) for name in INCOME_DIGIT_CELLS},
}

# ===========================================================================
# PAGE 2 — III. ODDIEL employer (same column layout as the page-1 employee
# block, 30px pitch, shifted down). Measured from form_template_p2.png.
# ===========================================================================
ZAM_DIC_CELLS     = _comb_cells(68, 10, 942, 974)     # DIČ (10 digits, no slash)
ZAM_PRIEZ_CELLS   = _comb_cells(68, 16, 1014, 1046)   # Priezvisko
ZAM_MENO_CELLS    = _comb_cells(562, 10, 1014, 1046)  # Meno
ZAM_TITUL_CELLS   = _comb_cells(1069, 4, 1014, 1046)  # Titul
ZAM_OBCHOD_CELLS  = _comb_cells(68, 37, 1090, 1122)   # Obchodné meno (právnická osoba)
ZAM_ULICA_CELLS   = _comb_cells(68, 28, 1170, 1202)   # Ulica
ZAM_SUPISNE_CELLS = _comb_cells(938, 8, 1166, 1198)   # Súpisné/orientačné číslo (raised to centre)
ZAM_PSC_CELLS     = _comb_cells(68, 5, 1252, 1284)    # PSČ
ZAM_OBEC_CELLS    = _comb_cells(233, 20, 1252, 1284)  # Obec
ZAM_STAT_CELLS    = _comb_cells(848, 11, 1252, 1284)  # Štát
VYPRACOVAL_CELLS  = _comb_cells(70, 37, 1330, 1362)   # Potvrdenie vypracoval(a)
POTVRDENIE_DATUM_CELLS = [                              # DD MM (printed 20) YY
    [153, 1437, 176, 1465], [182, 1437, 205, 1465],   # day
    [228, 1437, 251, 1465], [257, 1437, 280, 1465],   # month
    [360, 1437, 383, 1465], [390, 1437, 413, 1465],   # year suffix
]

# Register page-2 employer fields into the global type dicts.
DIGIT_COMB_CELLS.update({
    "zam_dic":          ZAM_DIC_CELLS,
    "zam_supisne_cislo": ZAM_SUPISNE_CELLS,
    "zam_psc":          ZAM_PSC_CELLS,
    "potvrdenie_datum": POTVRDENIE_DATUM_CELLS,
})
TEXT_COMB_CELLS.update({
    "zam_priezvisko":    ZAM_PRIEZ_CELLS,
    "zam_meno":          ZAM_MENO_CELLS,
    "zam_titul":         ZAM_TITUL_CELLS,
    "zam_obchodne_meno": ZAM_OBCHOD_CELLS,
    "zam_ulica":         ZAM_ULICA_CELLS,
    "zam_obec":          ZAM_OBEC_CELLS,
    "zam_stat":          ZAM_STAT_CELLS,
    "vypracoval":        VYPRACOVAL_CELLS,
})

_P2_EMPLOYER_BOXES = {
    name: _digit_comb_box(cells) for name, cells in {
        "zam_dic": ZAM_DIC_CELLS, "zam_supisne_cislo": ZAM_SUPISNE_CELLS,
        "zam_psc": ZAM_PSC_CELLS, "potvrdenie_datum": POTVRDENIE_DATUM_CELLS,
        "zam_priezvisko": ZAM_PRIEZ_CELLS, "zam_meno": ZAM_MENO_CELLS,
        "zam_titul": ZAM_TITUL_CELLS, "zam_obchodne_meno": ZAM_OBCHOD_CELLS,
        "zam_ulica": ZAM_ULICA_CELLS, "zam_obec": ZAM_OBEC_CELLS,
        "zam_stat": ZAM_STAT_CELLS, "vypracoval": VYPRACOVAL_CELLS,
    }.items()
}


# ===========================================================================
# PAGE 2 — II. ODDIEL continuation income (decimal combs, reuse _CX columns)
# ===========================================================================
INCOME_DIGIT_CELLS.update({
    "p2_riadok_08a": _cells([6, 7, 8, 9, 10],        122, 157),   # 3 int + 2 dec
    "p2_riadok_09":  _cells([6, 7, 8, 9, 10],        175, 210),
    "p2_riadok_10":  _cells([5, 6, 7, 8, 9, 10],     228, 263),   # 4 int + 2 dec
    "p2_riadok_11":  _cells([5, 6, 7, 8, 9, 10],     339, 374),
    "p2_riadok_12":  _cells([1,2,3,4,5,6,7,8,9,10],  393, 428),   # 8 int + 2 dec
})

# --- Month grids: a "1-12" master box + 12 month boxes ---
_MON_MASTER = (734, 758)
_MON_X = [(784, 806), (817, 839), (850, 872), (883, 905), (916, 938), (949, 971),
          (982, 1004), (1015, 1037), (1048, 1070), (1082, 1104), (1115, 1137), (1148, 1170)]


def _month_grid(prefix, y1, y2):
    """{prefix}_mesiac_vsetky (master) + {prefix}_mesiac_01..12 occupancy boxes."""
    g = {f"{prefix}_mesiac_vsetky": [_MON_MASTER[0], y1, _MON_MASTER[1], y2]}
    for i, (mx1, mx2) in enumerate(_MON_X, 1):
        g[f"{prefix}_mesiac_{i:02d}"] = [mx1, y1, mx2, y2]
    return g


_P2_MONTH_GRIDS = {
    **_month_grid("r10", 296, 316),
    **_month_grid("r13", 464, 484),
}

# ===========================================================================
# PAGE 2 — riadok 14: child tax-bonus table (4 rows: name | rodné číslo | months)
# ===========================================================================
_BONUS_RC_X = [(392, 416), (421, 445), (452, 476), (481, 505), (512, 536), (541, 565),
               (602, 626), (631, 655), (661, 685), (691, 715)]   # 6 left "/" 4 right
_BONUS_ROW_Y = [576, 636, 696, 756]   # top of each child row's fill band

DIETA_RC_CELLS = {}      # per-child rodné číslo cells (6 + 4)
_P2_BONUS_NAME_CELLS = {}
_P2_BONUS_MONTHS = {}
for _i, _by in enumerate(_BONUS_ROW_Y, 1):
    DIETA_RC_CELLS[f"dieta{_i}_rod_cislo"] = [[x1, _by, x2, _by + 26] for x1, x2 in _BONUS_RC_X]
    _P2_BONUS_NAME_CELLS[f"dieta{_i}_meno"] = _comb_cells(64, 11, _by, _by + 26, pitch=28)
    _P2_BONUS_MONTHS.update(_month_grid(f"dieta{_i}", _by + 18, _by + 38))

TEXT_COMB_CELLS.update(_P2_BONUS_NAME_CELLS)

# Page-2 top "Rodné číslo" (repeats the employee's, for page matching): 6 + "/" + 4.
RC_P2_TOP_CELLS = (
    [[x1, 70, x2, 100] for x1, x2 in
     [(465, 488), (495, 518), (526, 548), (555, 578), (586, 608), (615, 638)]]
    + [[x1, 70, x2, 100] for x1, x2 in
       [(675, 698), (705, 728), (735, 758), (765, 788)]]
)

# Multiple rodné-číslo fields now (employee + page-2 top + 4 children).
RC_CELLS = {"rod_cislo": RC_DIGIT_CELLS, "rod_cislo_p2": RC_P2_TOP_CELLS, **DIETA_RC_CELLS}


def _income_box(field):
    cells = INCOME_DIGIT_CELLS[field]
    return [cells[0][0], cells[0][1], cells[-1][2], cells[-1][3]]


_P2_INCOME_BOXES = {f: _income_box(f) for f in
                    ["p2_riadok_08a", "p2_riadok_09", "p2_riadok_10",
                     "p2_riadok_11", "p2_riadok_12"]}
_P2_BONUS_BOXES = {f: _digit_comb_box(c) for f, c in _P2_BONUS_NAME_CELLS.items()}
_P2_BONUS_RC_BOXES = {f: _digit_comb_box(c) for f, c in DIETA_RC_CELLS.items()}

FIELD_BOXES_P2 = {
    "rod_cislo_p2": _digit_comb_box(RC_P2_TOP_CELLS),
    **_P2_INCOME_BOXES,
    **_P2_MONTH_GRIDS,
    **_P2_BONUS_BOXES,
    **_P2_BONUS_RC_BOXES,
    **_P2_BONUS_MONTHS,
    **_P2_EMPLOYER_BOXES,
}

# ---------------------------------------------------------------------------
# Field type sets
# ---------------------------------------------------------------------------
NUMERIC_FIELDS    = set(INCOME_DIGIT_CELLS.keys())   # income rows (decimal)
# The short page-2 continuation income boxes (≤4 integer digits) have faint
# isolated digits that per-cell OCR misreads; whole-box OCR reads them reliably.
# riadok_12 is an 8-digit box and reads fine per-cell (like page-1 income), so it
# stays on the per-cell path.
WHOLEBOX_INCOME   = {"p2_riadok_08a", "p2_riadok_09", "p2_riadok_10", "p2_riadok_11"}
MONTH_FIELDS      = {k for k in FIELD_BOXES_P2 if "_mesiac_" in k}  # real p2 month grids
TEXT_FIELDS       = {"meno_zamestnanca"}             # name (Latin-model word OCR)
RC_FIELDS         = set(RC_CELLS.keys())             # rod_cislo + 4 child rodné čísla

DIGIT_COMB_FIELDS = set(DIGIT_COMB_CELLS.keys())     # datum/rok/psc/supisne/DIČ (integer, per-cell)
FUZZY_TEXT_FIELDS = set(TEXT_COMB_CELLS.keys())      # titul/ulica/obec/stat + employer text
CHECKBOX_FIELDS   = set(CHECKBOX_BOXES.keys())       # oprava/danovnik_obmedzena (occupancy)

# Slovak alphabet (lower+upper, with diacritics). Forms are filled in paličkové
# písmo (block capitals); text-field OCR output is constrained to this set so
# digit/punctuation noise is dropped (ICR "allowed character set" best practice).
SLOVAK_ALPHA = (
    "aábäcčdďeéfghiíjklĺľmnňoóôpqrŕsštťuúvwxyýzž"
    "AÁBÄCČDĎEÉFGHIÍJKLĹĽMNŇOÓÔPQRŔSŠTŤUÚVWXYÝZŽ"
)
