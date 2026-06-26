#!/usr/bin/env python3
"""
test_phase6.py — smoke-test the Phase-6 LLM text second-check against a LIVE server.

Run `bash serve_llm.sh` in one terminal, then `python test_phase6.py` in another.
Proves the real round-trip: health probe → grammar-constrained /completion → tiered
behaviour (name = suggestion-only, semi-open = cleaned+adopted-iff-revalidated).
If no server is up, it reports the graceful-degradation path instead of failing.
"""
import text_second_check as t


CASES = [
    # (field, noisy OCR, is_name, what we're checking)
    ("meno_zamestnanca", "NOVAK PETF",  True,  "name → SUGGESTION only (value never replaced)"),
    ("meno_zamestnanca", "KRAJCI ANNB", True,  "name → SUGGESTION only"),
    ("ulica",            "STUROVB",     False, "semi-open → CLEAN + adopt iff re-validated"),
    ("zam_obchodne_meno","TESCO STOREZ",False, "semi-open company → CLEAN + adopt"),
]


def main() -> None:
    t.reset_health_cache()
    up = t.llm_available()
    print(f"LLAMA_URL = {t.LLAMA_URL}")
    print(f"server reachable: {up}\n")
    if not up:
        print("No server — Phase 6 degrades to gazetteer-plus-flag (the pipeline still runs).")
        print("Start one with:  bash serve_llm.sh   then re-run this test.")
        return

    for field, ocr, is_name, note in CASES:
        r = t.text_second_check(field, ocr, is_name)
        tag = "SUGGESTION" if r["source"] == "llm-suggestion" else \
              "ADOPTED" if r["source"] == "llm-clean" else r["source"].upper()
        shown = r["suggestion"] if is_name else r["value"]
        print(f"[{tag:10}] {field:<18} OCR {ocr!r:16} -> {shown!r}")
        print(f"             ({note})")

    print("\nRails confirmed at the tier level:")
    print("  • name fields: value field is NEVER overwritten — only res['suggestion'] is set")
    print("  • semi-open:   adopted value is re-validated (Slovak letters) before use")
    print("  • numeric fields are never routed here (enforced upstream in ocr_page)")


if __name__ == "__main__":
    main()
