#!/usr/bin/env python3
"""Receipt image preprocessing helpers for OCR and Donut inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def estimate_skew_angle(gray: np.ndarray) -> float:
    """Estimate small document skew from dark-pixel line geometry."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(30, min(gray.shape) // 5),
        minLineLength=max(40, gray.shape[1] // 4),
        maxLineGap=12,
    )
    if lines is None:
        return 0.0
    angles = []
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line)
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -15.0 <= angle <= 15.0:
            angles.append(float(angle))
    return float(np.median(angles)) if angles else 0.0


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 0.2:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_receipt_image(
    image: Image.Image,
    target_width: int = 1200,
    binarize: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """Return (enhanced_color, grayscale_or_binary) receipt variants.

    The color variant is safer for Donut. The grayscale/binary variant is
    useful for OCR engines and debugging. The input image is never modified.
    """
    bgr = _pil_to_bgr(image)
    height, width = bgr.shape[:2]
    if width < target_width:
        scale = target_width / width
        bgr = cv2.resize(
            bgr,
            (target_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    angle = estimate_skew_angle(gray)
    bgr = rotate_image(bgr, angle)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=5, templateWindowSize=7, searchWindowSize=21)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enhanced_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if binarize:
        output_gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
    else:
        output_gray = gray
    return _bgr_to_pil(enhanced_color), Image.fromarray(output_gray)


def crop_receipt_item_region(
    image: Image.Image,
    top_fraction: float = 0.28,
    bottom_fraction: float = 0.78,
) -> Image.Image:
    """Remove common receipt header/footer noise while keeping the item list."""
    width, height = image.size
    top = max(0, min(height - 1, int(height * top_fraction)))
    bottom = max(top + 1, min(height, int(height * bottom_fraction)))
    return image.crop((0, top, width, bottom))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess a receipt image")
    parser.add_argument("--input", required=True)
    parser.add_argument("--color-output", required=True)
    parser.add_argument("--gray-output", required=True)
    parser.add_argument("--target-width", type=int, default=1200)
    parser.add_argument("--binarize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.input)
    color, gray = preprocess_receipt_image(
        image,
        target_width=args.target_width,
        binarize=args.binarize,
    )
    Path(args.color_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.gray_output).parent.mkdir(parents=True, exist_ok=True)
    color.save(args.color_output)
    gray.save(args.gray_output)
    print(f"Enhanced color image written to: {args.color_output}")
    print(f"Grayscale/OCR image written to: {args.gray_output}")


if __name__ == "__main__":
    main()
