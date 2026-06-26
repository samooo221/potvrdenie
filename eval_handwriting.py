#!/usr/bin/env python3
"""
eval_handwriting.py — REAL per-field accuracy + confidence on pen-filled forms.

The headline 99% in STATUS.md is against a printed font the code drew itself, on
shared coordinates. It proves the plumbing, not handwriting. This harness is the
instrument that produces an HONEST number on real paper: it aligns each photo the
same way the server does, runs the live OCR pipeline, and compares to typed labels
you provide once. It reports, per field, accuracy AND mean confidence — so you can
see precisely which fields the recognizer actually struggles with and whether the
confidence signal tracks the errors. Those numbers recalibrate CONF_THRESHOLD and
decide, per field, whether ICR (Phase 5) is worth building.

Layout (a directory):
    <stem>_p1.png          page 1 photo (phone photo is fine — it gets aligned)
    <stem>_p2.png          page 2 photo (optional)
    <stem>.labels.json     {field: true_value, ...}  (type the truth ONCE)

Synthetic-data-only still applies: label fabricated forms, never a real taxpayer's.

Usage:
    python eval_handwriting.py <dir>
    python eval_handwriting.py samples/    # works on the synthetic set too
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from align_photo import try_align
from crop_ocr import (
    ocr_page, build_extracted, compare_field, field_class, CONF_THRESHOLD,
    normalize_to_canvas, disambiguate_extracted, validate_extracted, escalate,
)
from field_defs import FIELD_BOXES_P1, FIELD_BOXES_P2, CANVAS_W, CANVAS_H


def _prepare(path: Path, page: int) -> Image.Image:
    """Same load path as the server: de-skew a photo, else resize to canvas."""
    img = Image.open(path)
    if img.size != (CANVAS_W, CANVAS_H):
        aligned, _note = try_align(img, page)
        if aligned is not None:
            return aligned.convert("L")
    return normalize_to_canvas(img)[0]


def _load_labels(stem_json: Path) -> dict:
    """Accept either a flat {field: value} dict, or {"labels": {...}}, or the
    synthetic sample format {"ground_truth": {...}}."""
    data = json.loads(stem_json.read_text())
    if "labels" in data:
        return data["labels"]
    if "ground_truth" in data:
        return data["ground_truth"]
    return data


def evaluate(directory: Path) -> None:
    p1_paths = sorted(directory.glob("*_p1.png"))
    if not p1_paths:
        print(f"No *_p1.png in {directory}. Expected <stem>_p1.png + <stem>.labels.json")
        return

    correct = defaultdict(int)
    total = defaultdict(int)
    conf_sum = defaultdict(float)
    conf_n = defaultdict(int)
    flagged_count = defaultdict(int)
    n_forms = 0

    print(f"Found {len(p1_paths)} form(s) in {directory}\n")
    for p1 in p1_paths:
        stem = p1.stem.replace("_p1", "")
        labels_path = p1.with_name(f"{stem}.labels.json")
        if not labels_path.exists():
            labels_path = p1.with_name(f"{stem}.json")   # synthetic fallback
        if not labels_path.exists():
            print(f"  [skip] {p1.name}: no labels file")
            continue
        labels = _load_labels(labels_path)
        p2 = p1.with_name(f"{stem}_p2.png")

        img_p1 = _prepare(p1, 1)
        img_p2 = _prepare(p2, 2) if p2.exists() else None
        raw = {**ocr_page(img_p1, FIELD_BOXES_P1),
               **(ocr_page(img_p2, FIELD_BOXES_P2) if img_p2 else {})}

        extracted = build_extracted(raw)
        extracted, _log = disambiguate_extracted(raw, extracted)
        checks = validate_extracted(extracted)
        flagged, _reasons = escalate(raw, extracted, checks)
        flagged_set = set(flagged)
        n_forms += 1

        for field, res in raw.items():
            if field not in labels:
                continue
            match, _gt, _ocr = compare_field(field, labels[field], res["value"])
            total[field] += 1
            correct[field] += int(match)
            conf_sum[field] += res["confidence"]
            conf_n[field] += 1
            if field in flagged_set:
                flagged_count[field] += 1

    if not n_forms:
        print("No labeled forms evaluated.")
        return

    # Report — sort worst-accuracy first so the fields needing ICR surface at top.
    rows = []
    for field in total:
        acc = correct[field] / total[field]
        mean_conf = conf_sum[field] / conf_n[field] if conf_n[field] else 0.0
        rows.append((acc, field, correct[field], total[field], mean_conf,
                     flagged_count[field]))
    rows.sort(key=lambda r: (r[0], r[4]))

    print(f"{'='*78}")
    print(f"REAL-HANDWRITING EVAL  ({n_forms} forms)")
    print(f"{'='*78}")
    print(f"{'Field':<22} {'Acc':>7} {'Correct':>9} {'MeanConf':>9} {'Flagged':>8}")
    print("-" * 78)
    oc = ot = 0
    flagged_correct_caught = flagged_total = 0
    for acc, field, c, t, mc, fl in rows:
        mark = "  <-- low" if acc < 0.95 else ""
        print(f"  {field:<20} {acc*100:>6.1f}% {c:>4}/{t:<4} {mc:>8.2f} {fl:>7}{mark}")
        oc += c; ot += t
        flagged_total += fl
        flagged_correct_caught += (fl if acc < 1.0 else 0)
    print("-" * 78)
    print(f"  {'OVERALL':<20} {100*oc/ot:>6.1f}% {oc:>4}/{ot:<4}")
    print(f"\n  {flagged_total} field-instances flagged for review across all forms.")
    print(f"  (A field flagged AND wrong = a caught error; flagged AND right = a")
    print(f"   false alarm. Tune CONF_THRESHOLD in crop_ocr.py from these numbers.)")
    print(f"{'='*78}")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <dir-of-forms>")
        sys.exit(1)
    print("Initializing OCR models …")
    from crop_ocr import get_rec_model
    get_rec_model()
    evaluate(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
