#!/usr/bin/env python3
"""
gazetteer.py — snap noisy OCR of closed-set text fields to a known reference list.

For fields whose value MUST come from a finite, knowable set (štát, titul, obec),
fuzzy-matching the noisy OCR string against the real list is deterministic,
auditable, and — crucially — cannot invent a value. This is the no-LLM tier of
the text second-check: a big robustness win that needs no model.

Contract: gazetteer_match(field, ocr_string) -> {value, matched, score}.
  - matched=True  → `value` is the canonical list entry; `score` is similarity.
  - matched=False → `value` is the original OCR string (flag it downstream).

Lists live in data/<name>.txt, one canonical entry per line ('#' / blank ignored).
obce.txt is optional and pluggable: absent → obec degrades to flag-on-miss.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"

# field → reference-list filename (under data/)
_FIELD_LIST = {
    "titul": "titul.txt",
    "zam_titul": "titul.txt",
    "stat": "stat.txt",
    "zam_stat": "stat.txt",
    "obec": "obce.txt",
    "zam_obec": "obce.txt",
}

# Accept a match when normalized edit distance ≤ this fraction of the longer
# string (plus a 1-char floor). Tight enough not to snap unrelated words,
# loose enough to absorb paličkové-písmo OCR noise (l→i/t, ä→a, dropped tick).
_MAX_DIST_RATIO = 0.34

_cache: dict[str, list[str]] = {}


def _strip_diacritics(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def _norm(s: str) -> str:
    """Comparison key: diacritics stripped, lowercased, alnum only."""
    return "".join(c for c in _strip_diacritics(s) if c.isalnum())


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _load_list(filename: str) -> list[str]:
    if filename in _cache:
        return _cache[filename]
    path = _DATA_DIR / filename
    entries: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    _cache[filename] = entries
    return entries


def is_supported(field: str) -> bool:
    return field in _FIELD_LIST


def gazetteer_match(field: str, ocr_string: str) -> dict:
    """Fuzzy-match `ocr_string` for `field` against its reference list.

    Returns {"value", "matched", "score"}. On any miss (empty input, no list,
    nothing close enough) returns the original string with matched=False — the
    caller flags it. Never fabricates a value not present in the list.
    """
    raw = (ocr_string or "").strip()
    miss = {"value": raw, "matched": False, "score": 0.0}

    entries = _load_list(_FIELD_LIST[field]) if field in _FIELD_LIST else []
    if not raw or not entries:
        return miss

    q = _norm(raw)
    if not q:
        return miss

    best_entry, best_dist = None, None
    for entry in entries:
        d = _levenshtein(q, _norm(entry))
        if best_dist is None or d < best_dist:
            best_entry, best_dist = entry, d
            if d == 0:
                break

    longest = max(len(q), len(_norm(best_entry)))
    tolerance = max(1, int(longest * _MAX_DIST_RATIO))
    if best_dist is not None and best_dist <= tolerance:
        score = 1.0 - best_dist / max(1, longest)
        return {"value": best_entry, "matched": True, "score": round(score, 3)}
    return miss


if __name__ == "__main__":
    # Tiny self-test / demo.
    for f, s in [("stat", "Slovenska republlka"), ("titul", "lng."),
                 ("obec", "Bratisiava"), ("obec", "Xyzville"), ("stat", "")]:
        print(f"{f:6} {s!r:24} -> {gazetteer_match(f, s)}")
