#!/usr/bin/env python3
"""render_hand.py — shared "handwriting / print" rendering for synthetic POT395 samples.

Extracted from make_handwritten.py so BOTH make_handwritten.py and
generate_samples.py share one realism pipeline instead of two drifting copies.

What it adds over the old printed-monospace draw layer:

- WriterProfile — a per-"writer" bundle of a font + physical parameters (ink-band,
  slant, baseline wander, ink bleed, scan skew) and a fill *mode*:
    * "palickove" — hand-printed BLOCK CAPITALS, the hard real case (POT395 must be
      filled "paličkovým písmom", i.e. disconnected block letters).
    * "printed"   — typewriter/printer output (the other legally-accepted fill mode):
      a monospace face with near-zero jitter.
- Per-glyph diacritic fallback — any glyph the chosen font lacks (e.g. some hand
  fonts have no uppercase Č Ď Ľ Ň Ť) is drawn from a diacritic-complete fallback
  face, so Slovak letters á č ď é í ľ ň ô š ť ž never render as .notdef "tofu".
- Ink as a GRAYSCALE level. The whole pipeline — and eval_handwriting.py — is
  grayscale ("L"), and a real scan converts blue ink to a mid-gray anyway. So pen
  colour (black vs dark-blue) is modelled as a darker/lighter ink BAND, not RGB
  (which the eval's convert("L") would discard). A lighter band == blue-ink look.

Synthetic data only. No real taxpayer data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from field_defs import INCOME_DIGIT_CELLS, _N_DECIMAL

# --- The 11 Slovak diacritic glyphs that must always render (upper + lower) ----
SLOVAK_DIACRITICS = "áčďéíľňôšťž" + "ÁČĎÉÍĽŇÔŠŤŽ"

HAND_FONT_DIR = Path(__file__).parent / "fonts"

# Diacritic-complete faces used as the per-glyph fallback (sans) and as the
# typewriter/printer primary (mono). First existing wins; verified at import.
_FALLBACK_CANDIDATES = [
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_PRINTED_CANDIDATES = [
    "/usr/share/fonts/liberation-mono-fonts/LiberationMono-Regular.ttf",
    "/usr/share/fonts/google-noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/adwaita-mono-fonts/AdwaitaMono-Regular.ttf",
    "/usr/share/fonts/google-carlito-fonts/Carlito-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]


def _first_existing(paths: list[str]) -> str | None:
    return next((p for p in paths if Path(p).exists()), None)


@lru_cache(maxsize=64)
def _coverage_set(font_path: str) -> frozenset[int]:
    """Set of Unicode codepoints a font actually maps (its best cmap).

    Used for the per-glyph fallback. Reading the cmap is reliable, unlike probing
    a rendered glyph for emptiness (many fonts draw a .notdef *box* that has a
    bounding box and would fool a pixel check).
    """
    try:
        from fontTools.ttLib import TTFont
        return frozenset(TTFont(font_path, fontNumber=0).getBestCmap().keys())
    except Exception:
        return frozenset()


@lru_cache(maxsize=64)
def _load(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


@dataclass
class WriterProfile:
    """One synthetic 'writer' (or machine): a font + how messy its strokes are."""
    name: str
    primary_path: str
    fallback_path: str
    size: int = 21
    mode: str = "palickove"          # "palickove" | "printed"
    ink_lo: int = 10                 # ink darkness band (0=black .. 255=white);
    ink_hi: int = 40                 #   a lighter band models blue ink / light hand
    rot_text: float = 8.0
    rot_digit: float = 5.0
    wander_x: int = 2
    wander_y: int = 3
    advance_jitter: float = 2.0
    blur_lo: float = 0.4
    blur_hi: float = 0.9
    scan_skew: float = 0.6
    scan_blur: float = 0.5

    @property
    def primary(self) -> ImageFont.FreeTypeFont:
        return _load(self.primary_path, self.size)

    @property
    def fallback(self) -> ImageFont.FreeTypeFont:
        return _load(self.fallback_path, self.size)

    def font_for(self, ch: str) -> ImageFont.FreeTypeFont:
        """Primary font, or the diacritic-complete fallback for glyphs it lacks."""
        if ch == " " or ord(ch) in _coverage_set(self.primary_path):
            return self.primary
        return self.fallback


# ---------------------------------------------------------------------------
# Core glyph renderer
# ---------------------------------------------------------------------------
def draw_char(img: Image.Image, ch: str, cx: float, cy: float,
              profile: WriterProfile, rng: random.Random,
              rot_range: float) -> float:
    """Draw one character with handwriting simulation. Returns advance width."""
    font = profile.font_for(ch)
    bbox = font.getbbox(ch)
    cw = bbox[2] - bbox[0]
    ch_h = bbox[3] - bbox[1]
    if cw <= 0:                       # space / zero-width — advance only, no ink
        return 6.0

    pad = 10
    tile_w = cw + pad * 2
    tile_h = ch_h + pad * 2
    tile = Image.new("L", (tile_w, tile_h), 255)
    td = ImageDraw.Draw(tile)

    # Ink darkness: pen pressure + (modelled) ink colour, as a gray level.
    ink = rng.randint(profile.ink_lo, profile.ink_hi)
    td.text((pad - bbox[0], pad - bbox[1]), ch, fill=ink, font=font)

    # Slight ink bleed (capped so thin strokes survive on tiny digit cells).
    blur = rng.uniform(profile.blur_lo, profile.blur_hi)
    if blur > 0.01:
        tile = tile.filter(ImageFilter.GaussianBlur(radius=blur))

    # Per-character rotation.
    if rot_range > 0.01:
        angle = rng.uniform(-rot_range, rot_range)
        tile = tile.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)

    # Baseline wander.
    bx = rng.randint(-profile.wander_x, profile.wander_x) if profile.wander_x else 0
    by = rng.randint(-profile.wander_y, profile.wander_y) if profile.wander_y else 0

    paste_x = int(cx - pad + bx)
    paste_y = int(cy - pad + by)

    region_x1 = max(0, paste_x)
    region_y1 = max(0, paste_y)
    region_x2 = min(img.width, paste_x + tile_w)
    region_y2 = min(img.height, paste_y + tile_h)
    if region_x2 <= region_x1 or region_y2 <= region_y1:
        return float(cw) + rng.uniform(-1, profile.advance_jitter)

    tile_x1 = region_x1 - paste_x
    tile_y1 = region_y1 - paste_y
    tile_crop = tile.crop((tile_x1, tile_y1,
                           tile_x1 + (region_x2 - region_x1),
                           tile_y1 + (region_y2 - region_y1)))

    existing = img.crop((region_x1, region_y1, region_x2, region_y2))
    blended = Image.fromarray(np.minimum(np.array(existing), np.array(tile_crop)))
    img.paste(blended, (region_x1, region_y1))

    return float(cw) + rng.uniform(-1, profile.advance_jitter)


# ---------------------------------------------------------------------------
# Placement helpers (free text, comb cells, numeric rows, checkboxes)
# ---------------------------------------------------------------------------
def draw_text(img: Image.Image, text: str, x1: int, y1: int, x2: int, y2: int,
              profile: WriterProfile, rng: random.Random) -> None:
    """Flowing free text inside a box (left-aligned, baseline-centred)."""
    m = profile.primary.getbbox("M")
    ch_h = m[3] - m[1]
    base_y = y1 + (y2 - y1 - ch_h) // 2
    cx = float(x1 + 8)
    for ch in text:
        cx += draw_char(img, ch, cx, base_y, profile, rng, rot_range=profile.rot_text)
        if cx > x2 - 6:
            break


def draw_digit_in_cell(img: Image.Image, ch: str, cell: list,
                       profile: WriterProfile, rng: random.Random) -> None:
    """One character centred inside a single comb cell."""
    x1, y1, x2, y2 = cell
    font = profile.font_for(ch)
    bbox = font.getbbox(ch)
    dw = bbox[2] - bbox[0]
    dh = bbox[3] - bbox[1]
    if dw <= 0:                       # space — leave the cell blank
        return
    base_x = x1 + (x2 - x1 - dw) / 2
    base_y = y1 + (y2 - y1 - dh) / 2
    draw_char(img, ch, base_x, base_y, profile, rng, rot_range=profile.rot_digit)


def draw_chars_in_cells(img: Image.Image, text: str, cells: list,
                        profile: WriterProfile, rng: random.Random,
                        align: str = "left") -> None:
    """Place each character of `text` centred in comb cells, with alignment.

    align: 'left'  → first char in the leftmost cell
           'right' → last char in the rightmost cell (numeric / right-fill combs)
    A space consumes a cell but draws nothing (matches the form's letter combs).
    """
    n = min(len(text), len(cells))
    if align == "right":
        start = len(cells) - n
    elif align == "middle":
        start = (len(cells) - n) // 2
    else:
        start = 0
    for i, ch in enumerate(text[:n]):
        draw_digit_in_cell(img, ch, cells[start + i], profile, rng)


def draw_numeric(img: Image.Image, value_str: str, field_name: str,
                 profile: WriterProfile, rng: random.Random) -> None:
    """Right-align an income value into its per-row digit cells (last 2 = cents)."""
    cells = INCOME_DIGIT_CELLS[field_name]
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
            draw_digit_in_cell(img, int_digits[idx], cell, profile, rng)

    for ch, cell in zip(dec_part, dec_cells):
        draw_digit_in_cell(img, ch, cell, profile, rng)


def draw_cross(img: Image.Image, x1: int, y1: int, x2: int, y2: int,
               profile: WriterProfile, rng: random.Random) -> None:
    """An 'X' occupancy mark in a month/checkbox cell."""
    draw = ImageDraw.Draw(img)
    ink = rng.randint(profile.ink_lo, min(255, profile.ink_hi + 10))
    jx = rng.randint(-profile.wander_x, profile.wander_x) if profile.wander_x else 0
    jy = rng.randint(-profile.wander_x, profile.wander_x) if profile.wander_x else 0
    draw.line([(x1 + 3 + jx, y1 + 3 + jy), (x2 - 3 + jx, y2 - 3 + jy)], fill=ink, width=2)
    draw.line([(x2 - 3 + jx, y1 + 3 + jy), (x1 + 3 + jx, y2 - 3 + jy)], fill=ink, width=2)


def add_paper_texture(img: Image.Image, profile: WriterProfile,
                      rng: random.Random) -> Image.Image:
    """Vignette + grain + salt-pepper dust + scan skew/blur (the 'scanned' look)."""
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape

    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    vignette = 1.0 - 0.04 * ((xx - cx) ** 2 / cx ** 2 + (yy - cy) ** 2 / cy ** 2)
    vignette = np.clip(vignette, 0.92, 1.0)
    arr = arr * vignette

    np_rng2 = np.random.default_rng(rng.randint(0, 2**31))
    arr = np.clip(arr + (np_rng2.random((h, w)) * 14 - 7), 0, 255)

    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    arr[np_rng.random((h, w)) < 0.004] = 0
    arr[np_rng.random((h, w)) < 0.006] = 255

    result = Image.fromarray(arr.astype(np.uint8))
    if profile.scan_skew > 0.01:
        result = result.rotate(rng.uniform(-profile.scan_skew, profile.scan_skew),
                               resample=Image.BICUBIC, expand=False, fillcolor=255)
    if profile.scan_blur > 0.01:
        result = result.filter(ImageFilter.GaussianBlur(radius=profile.scan_blur))
    return result


# ---------------------------------------------------------------------------
# Writer-profile pools
# ---------------------------------------------------------------------------
# Physical-style presets layered onto whichever hand fonts are available. Mixing
# fonts × presets makes the 20 samples look like different people. The lighter
# ink bands model blue ink / a light hand; darker bands model a heavy presser.
_HAND_PRESETS = [
    dict(name="heavy",  ink_lo=8,  ink_hi=24, rot_text=6.0,  rot_digit=4.0, wander_x=2, wander_y=2, blur_lo=0.4, blur_hi=0.8),
    dict(name="light",  ink_lo=30, ink_hi=62, rot_text=9.0,  rot_digit=6.0, wander_x=2, wander_y=3, blur_lo=0.5, blur_hi=1.0),  # blue-ink look
    dict(name="neat",   ink_lo=12, ink_hi=34, rot_text=5.0,  rot_digit=3.0, wander_x=1, wander_y=2, blur_lo=0.4, blur_hi=0.7),
    dict(name="messy",  ink_lo=10, ink_hi=46, rot_text=10.0, rot_digit=7.0, wander_x=2, wander_y=4, blur_lo=0.5, blur_hi=0.9),
    dict(name="bluepen",ink_lo=40, ink_hi=78, rot_text=8.0,  rot_digit=5.0, wander_x=2, wander_y=3, blur_lo=0.5, blur_hi=0.9),
]

_PRINTED_PRESETS = [
    dict(name="typewriter", ink_lo=5,  ink_hi=18, rot_text=0.4, rot_digit=0.4, wander_x=0, wander_y=1, advance_jitter=0.4, blur_lo=0.2, blur_hi=0.4, scan_skew=0.4),
    dict(name="laser",      ink_lo=8,  ink_hi=22, rot_text=0.3, rot_digit=0.3, wander_x=0, wander_y=0, advance_jitter=0.3, blur_lo=0.1, blur_hi=0.3, scan_skew=0.3),
]


def discover_hand_fonts() -> list[str]:
    """Downloaded block-print TTFs that cover the basic A–Z + digits."""
    if not HAND_FONT_DIR.is_dir():
        return []
    out = []
    for p in sorted(HAND_FONT_DIR.glob("*.ttf")):
        cov = _coverage_set(str(p))
        if cov and all(ord(c) in cov for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"):
            out.append(str(p))
    return out


def build_profiles(size: int = 21) -> tuple[list[WriterProfile], list[WriterProfile], str]:
    """(palickove_profiles, printed_profiles, note).

    Hybrid sourcing: use the downloaded block-print fonts for paličkové; if none
    qualify, degrade to the offline jitter pipeline on the mono face (still real
    diacritics via the fallback). `note` records which path was taken.
    """
    fallback = _first_existing(_FALLBACK_CANDIDATES)
    printed_font = _first_existing(_PRINTED_CANDIDATES)
    if fallback is None:
        fallback = printed_font            # last resort: mono is diacritic-complete
    if not _coverage_set(fallback or "").issuperset(ord(c) for c in SLOVAK_DIACRITICS):
        # fallback must be diacritic-complete or the whole guarantee breaks
        for cand in _FALLBACK_CANDIDATES + _PRINTED_CANDIDATES:
            if Path(cand).exists() and _coverage_set(cand).issuperset(ord(c) for c in SLOVAK_DIACRITICS):
                fallback = cand
                break

    hand_fonts = discover_hand_fonts()

    if hand_fonts:
        note = f"palickove: {len(hand_fonts)} hand font(s); fallback={Path(fallback).name}"
        palickove = []
        for i, preset in enumerate(_HAND_PRESETS):
            fp = hand_fonts[i % len(hand_fonts)]
            palickove.append(WriterProfile(
                name=f"hand-{preset['name']}-{Path(fp).stem}",
                primary_path=fp, fallback_path=fallback, size=size,
                mode="palickove", **{k: v for k, v in preset.items() if k != "name"}))
    else:
        # OFFLINE FALLBACK: no usable hand fonts → jitter the mono face.
        note = f"palickove UNAVAILABLE — offline jitter on {Path(printed_font).name}"
        palickove = [WriterProfile(
            name=f"jitter-{preset['name']}",
            primary_path=printed_font, fallback_path=fallback, size=size,
            mode="palickove", **{k: v for k, v in preset.items() if k != "name"})
            for preset in _HAND_PRESETS]

    printed = [WriterProfile(
        name=f"print-{preset['name']}",
        primary_path=printed_font, fallback_path=fallback, size=size, mode="printed",
        **{k: v for k, v in preset.items() if k != "name"})
        for preset in _PRINTED_PRESETS]

    return palickove, printed, note
