#!/usr/bin/env python3
"""
text_second_check.py — Phase-6 scoped LLM tier for noisy free-text fields.

The ONE place an LLM earns a role in this otherwise-deterministic pipeline: a
second-check for escalated (low-confidence) TEXT fields only. Numbers never enter
here — a digit misread loses the information at perception, so the LLM has nothing
to reason from, and numerics are already covered by arithmetic/mod-11.

Two tiers (the closed-set gazetteer tier lives in gazetteer.py and runs first):
  - SEMI_OPEN (ulica, obchodné meno): the LLM CLEANS the OCR string; the cleaned
    value is adopted by the caller only if it re-validates.
  - NAME (personal names): the LLM output is a human-facing SUGGESTION only — the
    caller never replaces the OCR value with it.

Non-negotiables honoured here:
  - LOCAL only (taxpayer names/addresses — data sovereignty). Talks to a local
    llama.cpp server; no network egress.
  - Degrades gracefully: if the server is down/unreachable, every call returns the
    OCR string unchanged with source="unavailable" so the pipeline keeps running
    (gazetteer-plus-flag behaviour).
  - Output is grammar-constrained to Slovak block letters, then re-validated.

Config via env: LLAMA_URL (default http://localhost:8080).
"""
from __future__ import annotations

import difflib
import json
import os
import unicodedata
import urllib.request

LLAMA_URL = os.environ.get("LLAMA_URL", "http://localhost:8080").rstrip("/")

# GBNF: 1–2 words of Slovak UPPERCASE block letters (paličkové písmo), single
# space. Constrains the model to a plausible name/street shape — no digits,
# punctuation, or lowercase. Bounded to 2 words: the form fields here are a
# surname+given or a 1–2 word street, and a tight bound stops a model from
# rambling into a multi-word hallucination ("TOTH KOSTOLAN TAINKO ĎURČAŠÍN").
_SLOVAK_GBNF = (
    'root ::= word (" " word)?\n'
    "word ::= letter letter? letter? letter? letter? letter? letter? letter? "
    "letter? letter? letter? letter? letter? letter? letter?\n"
    'letter ::= [A-Z] | "Á" | "Ä" | "Č" | "Ď" | "É" | "Í" | "Ĺ" | "Ľ" | "Ň" '
    '| "Ó" | "Ô" | "Ŕ" | "Š" | "Ť" | "Ú" | "Ý" | "Ž"\n'
)

# Cached health probe — one cheap check per process, so a down server costs ~one
# refused connection, not a per-field timeout.
_health = {"checked": False, "ok": False}


def reset_health_cache() -> None:
    """Force the next llm_available() to re-probe (tests / after starting a server)."""
    _health["checked"] = False


def llm_available(timeout: float = 1.5) -> bool:
    if _health["checked"]:
        return _health["ok"]
    ok = False
    try:
        req = urllib.request.Request(LLAMA_URL + "/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ok = (getattr(r, "status", r.getcode()) == 200)
    except Exception:
        ok = False
    _health["checked"], _health["ok"] = True, ok
    return ok


def _prompt(field: str, ocr_string: str, is_name: bool) -> str:
    if is_name:
        kind = "personal name"
    elif "obchodne" in field:
        kind = "company (business) name"
    else:
        kind = "street name"
    return (f'OCR misread a hand-printed Slovak {kind} as "{ocr_string}". '
            f"Reply with ONLY the most likely correct {kind} in Slovak UPPERCASE "
            f"block letters, no quotes, no extra words.")


def _llm_clean(field: str, ocr_string: str, is_name: bool, timeout: float = 20.0):
    """Ask the local server for a cleaned string. Uses the OpenAI-style
    /v1/chat/completions endpoint so the loaded model's OWN chat template is applied
    automatically (instruct models like Gemma underperform on a raw /completion
    prompt) — keeps this model-agnostic. Grammar-constrained to Slovak block letters.
    Returns the string, or None on unavailability / any error (→ graceful degrade)."""
    if not llm_available():
        return None
    body = json.dumps({
        "messages": [{"role": "user", "content": _prompt(field, ocr_string, is_name)}],
        "temperature": 0.0,          # greedy = the single most likely correction
        "n_predict": 16,
        "grammar": _SLOVAK_GBNF,     # llama.cpp honours grammar on the chat endpoint
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            LLAMA_URL + "/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        text = (data["choices"][0]["message"]["content"] or "").strip()
        return text or None
    except Exception:
        return None


def _norm(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.upper())
                   if unicodedata.category(c) != "Mn" and c.isalnum())


def _revalidate(field: str, value: str, ocr: str) -> bool:
    """Deterministic re-validation of an LLM-adopted value (semi-open tier) — the
    model never gets the last word. The adopted string must be:
      1. non-empty, only Slovak block letters + spaces;
      2. not wildly longer than the OCR (catches repetition loops);
      3. an actual CLEANUP of the OCR, not an invention — it has to resemble the
         input (diacritic-insensitive similarity above a floor). A model that
         hallucinates an unrelated string fails here and the OCR value is kept+flagged.
    """
    from field_defs import SLOVAK_ALPHA
    allowed = set(SLOVAK_ALPHA) | {" "}
    v = value.strip()
    if not v or not all(c in allowed for c in v):
        return False
    if len(v) > max(len(ocr) + 4, int(len(ocr) * 1.6)):      # repetition / runaway
        return False
    return _similarity(v, ocr) >= 0.5                          # resembles the OCR


def _similarity(a: str, b: str) -> float:
    """Diacritic-insensitive 0–1 similarity between two strings."""
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


# A name SUGGESTION is only worth showing if it plausibly relates to the OCR.
# Looser than the adoption threshold (the OCR is noisier on hard names), but it
# suppresses wild hallucinations ("NOVAK PETF" → "HORVÁTH SLOVÁK ĎURČAN").
_NAME_SUGGEST_MIN = 0.4


def text_second_check(field: str, ocr_string: str, is_name: bool) -> dict:
    """Run the scoped LLM second-check for one escalated text field.

    Returns {value, suggestion, source}:
      - source="llm-suggestion": NAME tier — value unchanged, suggestion = LLM output.
      - source="llm-clean":      SEMI_OPEN tier — value replaced by re-validated LLM
                                 output (adopted).
      - source="ocr":            LLM ran but its output didn't re-validate / was empty.
      - source="unavailable":    server down — degrade to the OCR value (caller flags).
    """
    raw = (ocr_string or "").strip()
    if not raw:                       # nothing to reason from
        return {"value": raw, "suggestion": None, "source": "ocr"}
    if not llm_available():
        return {"value": raw, "suggestion": None, "source": "unavailable"}

    cleaned = _llm_clean(field, raw, is_name)
    if not cleaned:
        return {"value": raw, "suggestion": None, "source": "ocr"}

    if is_name:
        # Suggestion only — NEVER replace a real value with the model's guess. And
        # only surface it if it plausibly relates to the OCR, so a hallucinated
        # unrelated name isn't shown to the reviewer.
        if _similarity(cleaned, raw) < _NAME_SUGGEST_MIN:
            return {"value": raw, "suggestion": None, "source": "ocr"}
        return {"value": raw, "suggestion": cleaned, "source": "llm-suggestion"}

    # Semi-open: adopt the cleaned string iff it re-validates against the OCR.
    if _revalidate(field, cleaned, raw):
        return {"value": cleaned, "suggestion": None, "source": "llm-clean"}
    return {"value": raw, "suggestion": None, "source": "ocr"}


if __name__ == "__main__":
    print(f"LLAMA_URL = {LLAMA_URL}")
    print(f"llm_available() = {llm_available()}")
    for fld, s, nm in [("meno_zamestnanca", "JRISA MAS", True),
                       ("ulica", "SHOROVAL", False),
                       ("zam_obchodne_meno", "FIRMA SRO", False)]:
        print(f"  {fld:<18} {s!r:14} -> {text_second_check(fld, s, nm)}")
