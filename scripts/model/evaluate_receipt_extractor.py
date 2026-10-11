#!/usr/bin/env python3
"""Evaluate a trained Donut checkpoint on held-out CORD and SROIE data."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from normalize_dataset import normalize_receipt_record
from repair_receipt import repair_receipt
from postprocess_receipt import remove_likely_venue_items
from validate_receipt import validate_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Donut receipt extraction quality")
    parser.add_argument("--model-dir", default="models/receipt_model")
    parser.add_argument("--test-manifest", default="data/processed/test.jsonl")
    parser.add_argument(
        "--source",
        choices=["all", "cord", "sroie"],
        default="all",
        help="Evaluate all sources or one source only",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="data/processed/donut_evaluation.json")
    parser.add_argument("--mixed-precision", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def read_manifest(path: str) -> List[Dict[str, Any]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Test manifest does not exist: {manifest_path}")
    rows: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "image" not in row or "receipt" not in row:
                raise ValueError(f"Invalid test manifest row at {manifest_path}:{line_number}")
            image_path = Path(row["image"])
            if not image_path.is_absolute():
                row["image"] = str((manifest_path.parent / "images" / image_path).resolve())
            rows.append(row)
    return rows


def image_from_row(row: Dict[str, Any]) -> Image.Image:
    return Image.open(row["image"]).convert("RGB")


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default


def target_to_receipt(row: Dict[str, Any]) -> Dict[str, Any]:
    return normalize_receipt_record(row["receipt"])


def parse_prediction(text: str) -> Dict[str, Any] | None:
    cleaned = text.strip()
    if "<s>" in cleaned:
        cleaned = cleaned.split("<s>", 1)[1]
    if "</s>" in cleaned:
        cleaned = cleaned.split("</s>", 1)[0]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None

    try:
        cleaned, _ = remove_likely_venue_items(normalize_receipt_record(payload))
        return cleaned
    except (TypeError, ValueError):
        return None


def number_error(predicted: float, expected: float) -> float:
    return abs(predicted - expected)


def compare(predicted: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    pred_items = predicted["items"]
    true_items = expected["items"]
    pred_totals = [predicted["subtotal"], predicted["grand_total"]]
    true_totals = [expected["subtotal"], expected["grand_total"]]
    predicted_validation = validate_receipt(predicted)
    repaired_prediction = repair_receipt(predicted)
    expected_validation = validate_receipt(expected)
    return {
        "items_exact": pred_items == true_items,
        "subtotal_exact": math.isclose(predicted["subtotal"], expected["subtotal"], abs_tol=0.01),
        "grand_total_exact": math.isclose(
            predicted["grand_total"], expected["grand_total"], abs_tol=0.01
        ),
        "subtotal_abs_error": number_error(predicted["subtotal"], expected["subtotal"]),
        "grand_total_abs_error": number_error(predicted["grand_total"], expected["grand_total"]),
        "receipt_exact": predicted == expected,
        "predicted_totals": pred_totals,
        "expected_totals": true_totals,
        "predicted_consistent": predicted_validation["receipt_consistent"],
        "repair_made": repaired_prediction["repaired"],
        "repair_consistent": repaired_prediction["consistent"],
        "repair_warning_count": len(repaired_prediction["warnings"]),
        "expected_consistent": expected_validation["receipt_consistent"],
    }


def evaluate_source(
    model: VisionEncoderDecoderModel,
    processor: DonutProcessor,
    rows: Sequence[Dict[str, Any]],
    device: str,
    mixed_precision: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    results: List[Dict[str, Any]] = []
    parse_failures = 0
    model.eval()
    for row in rows:
        image = image_from_row(row)
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(device)
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=mixed_precision and device.startswith("cuda"),
            ):
                generated = model.generate(
                    pixel_values,
                    max_length=512,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                )
        text = processor.batch_decode(generated, skip_special_tokens=False)[0]
        predicted = parse_prediction(text)
        expected = target_to_receipt(row)
        if predicted is None:
            parse_failures += 1
            continue
        results.append({"source": row.get("source", "unknown"), **compare(predicted, expected)})
    return results, parse_failures


def summarize(results: Sequence[Dict[str, Any]], parse_failures: int) -> Dict[str, Any]:
    count = len(results)
    total = count + parse_failures
    if not count:
        return {"samples": total, "parsed": 0, "parse_rate": 0.0}
    return {
        "samples": total,
        "parsed": count,
        "parse_failures": parse_failures,
        "parse_rate": count / total if total else 0.0,
        "receipt_exact_rate": sum(r["receipt_exact"] for r in results) / count,
        "items_exact_rate": sum(r["items_exact"] for r in results) / count,
        "subtotal_accuracy": sum(r["subtotal_exact"] for r in results) / count,
        "grand_total_accuracy": sum(r["grand_total_exact"] for r in results) / count,
        "subtotal_mae": sum(r["subtotal_abs_error"] for r in results) / count,
        "grand_total_mae": sum(r["grand_total_abs_error"] for r in results) / count,
        "predicted_consistency_rate": sum(r["predicted_consistent"] for r in results) / count,
        "repair_consistency_rate": sum(r["repair_consistent"] for r in results) / count,
        "repair_rate": sum(r["repair_made"] for r in results) / count,
        "repair_warning_rate": sum(bool(r["repair_warning_count"]) for r in results) / count,
        "expected_consistency_rate": sum(r["expected_consistent"] for r in results) / count,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    processor = DonutProcessor.from_pretrained(args.model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_dir).to(args.device)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id

    rows = read_manifest(args.test_manifest)
    if args.source != "all":
        rows = [row for row in rows if row.get("source") == args.source]
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    all_results: List[Dict[str, Any]] = []
    source_summaries: Dict[str, Any] = {}
    for source in sorted({row.get("source", "unknown") for row in rows}):
        source_rows = [row for row in rows if row.get("source", "unknown") == source]
        results, failures = evaluate_source(
            model, processor, source_rows, args.device, args.mixed_precision
        )
        all_results.extend(results)
        source_summaries[source] = summarize(results, failures)
        print(f"{source}: {source_summaries[source]}")

    summary = {
        "model_dir": str(Path(args.model_dir)),
        "device": args.device,
        "sources": source_summaries,
        "overall": summarize(
            all_results,
            sum(
                source_summaries[source].get("parse_failures", 0)
                for source in source_summaries
            ),
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluation written to: {output_path}")


if __name__ == "__main__":
    main()
