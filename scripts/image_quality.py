#!/usr/bin/env python3
"""Pre-OCR image quality checks for receipt photos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityResult:
    status: str
    score: float
    reasons: List[str]
    guidance: List[str]


def assess_receipt_image(
    image: Image.Image,
    min_width: int = 500,
    min_height: int = 500,
    blur_threshold: float = 45.0,
    contrast_threshold: float = 22.0,
) -> QualityResult:
    """Assess whether a receipt image is suitable for OCR/model inference."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_score = float(gray.std())
    brightness = float(gray.mean())

    reasons: List[str] = []
    guidance: List[str] = []
    if width < min_width or height < min_height:
        reasons.append("image resolution is too low")
        guidance.append("Move closer and fill the frame with the receipt.")
    if blur_score < blur_threshold:
        reasons.append("image is blurry or out of focus")
        guidance.append("Hold the phone steady and tap the receipt to focus.")
    if contrast_score < contrast_threshold:
        reasons.append("receipt text has low contrast")
        guidance.append("Use brighter, even lighting and avoid glare.")
    if brightness < 35:
        reasons.append("image is too dark")
        guidance.append("Turn on more light and avoid casting a shadow.")
    elif brightness > 225:
        reasons.append("image is overexposed")
        guidance.append("Reduce direct light or avoid using flash on glossy paper.")

    penalties = 0.0
    penalties += 0.35 if width < min_width or height < min_height else 0.0
    penalties += 0.35 if blur_score < blur_threshold else 0.0
    penalties += 0.2 if contrast_score < contrast_threshold else 0.0
    penalties += 0.1 if brightness < 35 or brightness > 225 else 0.0
    score = round(max(0.0, 1.0 - penalties), 3)

    if len(reasons) >= 2 or width < min_width // 2 or height < min_height // 2:
        status = "retake_photo"
    elif reasons:
        status = "review_quality"
    else:
        status = "ready"
    return QualityResult(status, score, reasons, guidance)
