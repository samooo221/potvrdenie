#!/usr/bin/env python3
"""
align_photo.py — warp a phone/scanner photo of the POT395 form to the
1241×1755 canvas that crop_ocr.py expects.

Usage:
    python align_photo.py photo.jpg [--page 1] [--out aligned_p1.png] [--debug]

Run for each page separately, then pipe into crop_ocr.py:
    python align_photo.py front.jpg --page 1 --out aligned_p1.png
    python align_photo.py back.jpg  --page 2 --out aligned_p2.png
    python crop_ocr.py aligned_p1.png aligned_p2.png

Tips for a good photo:
    - Lay the form flat on a DARK surface (dark table, black folder).
    - Even overhead lighting — avoid shadows across the form.
    - Camera directly above, ±30° from perpendicular is fine.
    - Include a small border of the dark surface around the form.
    - 5–12 MP is plenty; huge phone photos slow feature matching.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from field_defs import CANVAS_W, CANVAS_H

TEMPLATE_PATHS = {
    1: Path(__file__).parent / "form_template_p1.png",
    2: Path(__file__).parent / "form_template_p2.png",
}
MIN_GOOD_MATCHES = 30


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def detect_and_match(template_gray: np.ndarray,
                     photo_gray: np.ndarray,
                     n_features: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    orb = cv2.ORB_create(nfeatures=n_features)
    kp_t, des_t = orb.detectAndCompute(template_gray, None)
    kp_p, des_p = orb.detectAndCompute(photo_gray, None)

    if des_t is None or des_p is None or len(des_t) < 10 or len(des_p) < 10:
        raise RuntimeError("Too few keypoints. See lighting tips in the script header.")

    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = bf.knnMatch(des_t, des_p, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    if len(good) < MIN_GOOD_MATCHES:
        raise RuntimeError(
            f"Only {len(good)} good matches (need {MIN_GOOD_MATCHES}). "
            "Try: better lighting, darker background, camera more perpendicular."
        )

    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good])
    pts_p = np.float32([kp_p[m.trainIdx].pt for m in good])
    return pts_t, pts_p


def compute_homography(pts_t: np.ndarray, pts_p: np.ndarray) -> np.ndarray:
    H, mask = cv2.findHomography(pts_p, pts_t, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
    print(f"  Homography: {inliers} inliers from {len(pts_p)} matches")
    if H is None or inliers < 10:
        raise RuntimeError(
            f"Homography failed ({inliers} inliers). "
            "Try a flatter camera angle or better lighting."
        )
    return H


def warp(photo_gray: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(photo_gray, H, (CANVAS_W, CANVAS_H),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=255)


def try_align(pil_img, page: int):
    """Align a PIL photo to the page-`page` template via ORB+RANSAC.

    Returns (aligned PIL 'L' image, note) on success, or (None, reason) when the
    image can't be reliably aligned (too few matches/inliers — e.g. a blurry or
    cropped photo). Used by the server to de-skew phone photos before OCR; on
    failure the caller falls back to a plain resize.
    """
    tmpl = TEMPLATE_PATHS.get(page)
    if not tmpl or not tmpl.exists():
        return None, "no template"
    photo = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    h, w = photo.shape
    if max(h, w) > 3000:                       # cap feature-matching cost
        s = 3000 / max(h, w)
        photo = cv2.resize(photo, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    try:
        pts_t, pts_p = detect_and_match(load_gray(tmpl), photo)
        H = compute_homography(pts_t, pts_p)
    except RuntimeError as e:
        return None, str(e)
    from PIL import Image
    return Image.fromarray(warp(photo, H)), "aligned (ORB+RANSAC)"


def save_debug(template_gray, photo_gray, kp_t_pts, kp_p_pts, warped, out_path: Path) -> None:
    h1, w1 = template_gray.shape
    h2, w2 = photo_gray.shape
    vis_h = max(h1, h2)
    vis   = np.full((vis_h, w1 + w2 + 10), 200, dtype=np.uint8)
    vis[:h1, :w1]    = template_gray
    vis[:h2, w1+10:] = photo_gray
    for pt1, pt2 in zip(kp_t_pts[:50], kp_p_pts[:50]):
        x1, y1 = int(pt1[0]), int(pt1[1])
        x2, y2 = int(pt2[0]) + w1 + 10, int(pt2[1])
        cv2.line(vis, (x1, y1), (x2, y2), 80, 1)
        cv2.circle(vis, (x1, y1), 3, 0, -1)
        cv2.circle(vis, (x2, y2), 3, 0, -1)
    scale = 800 / vis.shape[1]
    vis_small = cv2.resize(vis, (800, int(vis.shape[0] * scale)))
    debug_m = out_path.with_stem(out_path.stem + "_debug_matches").with_suffix(".png")
    debug_w = out_path.with_stem(out_path.stem + "_debug_warped").with_suffix(".png")
    cv2.imwrite(str(debug_m), vis_small)
    cv2.imwrite(str(debug_w), warped)
    print(f"  Debug: {debug_m}, {debug_w}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Align POT395 photo to canonical canvas.")
    parser.add_argument("photo", help="Path to form photo (JPG/PNG)")
    parser.add_argument("--page", type=int, default=1, choices=[1, 2],
                        help="Which form page this photo is (default: 1)")
    parser.add_argument("--out", default=None, help="Output PNG path (default: aligned_pN.png)")
    parser.add_argument("--debug", action="store_true", help="Save match/warp debug images")
    parser.add_argument("--template", default=None, help="Override template PNG path")
    args = parser.parse_args()

    photo_path    = Path(args.photo)
    out_path      = Path(args.out) if args.out else Path(f"aligned_p{args.page}.png")
    template_path = Path(args.template) if args.template else TEMPLATE_PATHS[args.page]

    if not template_path.exists():
        print(f"ERROR: template not found: {template_path}")
        sys.exit(1)

    print(f"Template (page {args.page}): {template_path}")
    template_gray = load_gray(template_path)

    print(f"Photo: {photo_path}")
    photo_gray = load_gray(photo_path)
    print(f"  Photo size: {photo_gray.shape[1]}×{photo_gray.shape[0]} px")

    max_side = 3000
    h, w = photo_gray.shape
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        photo_gray = cv2.resize(photo_gray, (int(w*scale), int(h*scale)),
                                interpolation=cv2.INTER_AREA)
        print(f"  Scaled to: {photo_gray.shape[1]}×{photo_gray.shape[0]} px")

    print("Detecting and matching features …")
    try:
        pts_t, pts_p = detect_and_match(template_gray, photo_gray)
    except RuntimeError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)
    print(f"  {len(pts_t)} good matches after ratio test")

    print("Computing homography …")
    try:
        H = compute_homography(pts_t, pts_p)
    except RuntimeError as e:
        print(f"\nFAIL: {e}\n")
        sys.exit(1)

    print("Warping …")
    warped = warp(photo_gray, H)

    if args.debug:
        save_debug(template_gray, photo_gray, pts_t, pts_p, warped, out_path)

    Image.fromarray(warped).save(out_path)
    print(f"\nAligned image saved: {out_path}  ({CANVAS_W}×{CANVAS_H} px)")
    print(f"\nNext step:")
    print(f"  source .venv/bin/activate")
    print(f"  python crop_ocr.py {out_path}")


if __name__ == "__main__":
    main()
