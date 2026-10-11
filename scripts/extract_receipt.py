#!/usr/bin/env python3
"""Run the SplitSnap receipt extraction workflow on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from data.normalize_dataset import normalize_receipt_record
from data.repair_receipt import repair_receipt
from data.postprocess_receipt import remove_likely_venue_items
from image_preprocessing import crop_receipt_item_region, preprocess_receipt_image
from image_quality import assess_receipt_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a receipt into SplitSnap JSON")
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-dir", default="models/receipt_model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-width", type=int, default=1200)
    parser.add_argument("--min-quality", choices=["ready", "review_quality"], default="review_quality")
    parser.add_argument("--output", default="")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_model_json(text: str) -> Dict[str, Any] | None:
    cleaned = text
    if "<s>" in cleaned:
        cleaned = cleaned.split("<s>", 1)[1]
    if "</s>" in cleaned:
        cleaned = cleaned.split("</s>", 1)[0]
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
        cleaned, _ = remove_likely_venue_items(normalize_receipt_record(payload))
        return cleaned
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    image_path = Path(args.input)
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    image = Image.open(image_path).convert("RGB")
    quality = assess_receipt_image(image)
    result: Dict[str, Any] = {
        "input": str(image_path),
        "quality": {
            "status": quality.status,
            "score": quality.score,
            "reasons": quality.reasons,
            "guidance": quality.guidance,
        },
    }
    if quality.status == "retake_photo":
        result["status"] = "retake_photo"
        print(json.dumps(result, indent=2))
        return

    model_image = crop_receipt_item_region(image)
    processor = DonutProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(args.device).eval()
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    inputs = processor(images=model_image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(args.device)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=args.mixed_precision and args.device.startswith("cuda"),
        ):
            generated = model.generate(
                pixel_values,
                max_length=512,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                num_beams=1,
            )
    raw_text = processor.batch_decode(generated, skip_special_tokens=False)[0]
    receipt = parse_model_json(raw_text)
    result["status"] = "extracted"
    result["raw_output"] = raw_text
    if receipt is None:
        result["status"] = "manual_review"
        result["warnings"] = ["The model output was not valid SplitSnap JSON."]
    else:
        cleaned_receipt, postprocess_warnings = remove_likely_venue_items(receipt)
        repaired = repair_receipt(cleaned_receipt)
        result["receipt"] = repaired["receipt"]
        result["repair"] = {
            "changes": repaired["changes"],
            "warnings": postprocess_warnings + repaired["warnings"],
            "consistent": repaired["consistent"],
        }
        if quality.status == "review_quality":
            result["warnings"] = ["Image quality may reduce extraction accuracy."]
    output = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"Extraction written to: {output_path}")
    else:
        print(output)


if __name__ == "__main__":
    main()
