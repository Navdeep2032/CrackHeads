#!/usr/bin/env python3
"""FastAPI service for SplitSnap extraction and bill splitting."""

from __future__ import annotations

import io
import os
import math
from functools import lru_cache
from typing import Any, Dict, List

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from transformers import DonutProcessor, VisionEncoderDecoderModel

try:
    from billing.explain_split import explain_split
    from billing.split_bill import split_bill
    from data.repair_receipt import repair_receipt
    from data.postprocess_receipt import recover_items_from_malformed_output, remove_likely_venue_items
    from data.normalize_dataset import normalize_receipt_record
    from image_preprocessing import crop_receipt_item_region, preprocess_receipt_image
    from image_quality import assess_receipt_image
except ModuleNotFoundError:
    from scripts.billing.explain_split import explain_split
    from scripts.billing.split_bill import split_bill
    from scripts.data.repair_receipt import repair_receipt
    from scripts.data.postprocess_receipt import recover_items_from_malformed_output, remove_likely_venue_items
    from scripts.data.normalize_dataset import normalize_receipt_record
    from scripts.image_preprocessing import crop_receipt_item_region, preprocess_receipt_image
    from scripts.image_quality import assess_receipt_image


MODEL_DIR = os.getenv("SPLITSNAP_MODEL_DIR", "models/receipt_model")
DEVICE = os.getenv("SPLITSNAP_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="SplitSnap API", version="0.1.0")
cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "SPLITSNAP_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def load_model():
    processor = DonutProcessor.from_pretrained(MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    return processor, model


class SplitRequest(BaseModel):
    receipt: Dict[str, Any]
    people: List[str] = Field(min_length=1)
    assignments: Dict[int, Any] = Field(default_factory=dict)


def parse_model_json(text: str) -> Dict[str, Any] | None:
    cleaned = text.split("<s>", 1)[-1].split("</s>", 1)[0]
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        payload, _ = __import__("json").JSONDecoder().raw_decode(cleaned[start:])
        cleaned, _ = remove_likely_venue_items(normalize_receipt_record(payload))
        return cleaned
    except (ValueError, TypeError):
        return None


def item_confidences(generated: Any, processor: DonutProcessor) -> List[float]:
    """Estimate confidence per JSON item from generated-token probabilities."""
    scores = getattr(generated, "scores", None)
    sequences = getattr(generated, "sequences", None)
    if not scores or sequences is None:
        return []
    sequence = sequences[0].detach().cpu().tolist()
    token_ids = sequence[-len(scores):]
    token_texts = [
        processor.tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        for token_id in token_ids
    ]
    cumulative_lengths: List[int] = []
    cumulative = ""
    for token_text in token_texts:
        cumulative += token_text
        cumulative_lengths.append(len(cumulative))
    full_text = cumulative
    items_start = full_text.find('"items"')
    if items_start < 0:
        return []
    array_start = full_text.find("[", items_start)
    if array_start < 0:
        return []
    spans: List[tuple[int, int]] = []
    depth = 0
    object_start = -1
    for index in range(array_start + 1, len(full_text)):
        character = full_text[index]
        if character == "{":
            if depth == 0:
                object_start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and object_start >= 0:
                spans.append((object_start, index + 1))
                object_start = -1
    probabilities = []
    for logits, token_id in zip(scores, token_ids):
        probabilities.append(float(torch.softmax(logits[0], dim=-1)[token_id].item()))
    confidences: List[float] = []
    for start, end in spans:
        selected = [
            probabilities[index]
            for index, character_end in enumerate(cumulative_lengths)
            if character_end > start and (index == 0 or cumulative_lengths[index - 1] < end)
        ]
        if not selected:
            confidences.append(0.0)
            continue
        mean_log_probability = sum(math.log(max(value, 1e-8)) for value in selected) / len(selected)
        confidences.append(round(math.exp(mean_log_probability) * 100, 1))
    return confidences


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE, "model_dir": MODEL_DIR}


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "SplitSnap API",
        "status": "ok",
        "app": "GET /app",
        "endpoints": {
            "health": "GET /health",
            "extract": "POST /extract",
            "split": "POST /split",
            "docs": "GET /docs",
        },
    }


@app.get("/app", response_class=HTMLResponse)
def web_app() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SplitSnap</title>
  <style>
    :root { font-family: system-ui, sans-serif; color: #18202a; background: #f4f6f8; }
    body { max-width: 760px; margin: 0 auto; padding: 16px; }
    main { background: white; border-radius: 16px; padding: 20px; box-shadow: 0 4px 20px #0001; }
    h1 { margin: 0; } h2 { margin-bottom: 8px; font-size: 1.1rem; }
    button { background: #166534; color: white; border: 0; border-radius: 10px; padding: 12px 16px; font-weight: 700; cursor: pointer; }
    button.secondary { background: #e5e7eb; color: #18202a; } button:disabled { opacity: .5; cursor: not-allowed; }
    input[type=file], input[type=text], input[type=number] { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; }
    input[type=file] { margin: 12px 0 16px; } label { display: block; font-size: .85rem; font-weight: 650; margin-bottom: 4px; }
    .row { display: grid; grid-template-columns: 1fr 70px 100px 100px 40px; gap: 6px; align-items: end; margin: 8px 0; }
    .charge-row { display: grid; grid-template-columns: 1fr 120px 40px; gap: 6px; align-items: end; margin: 8px 0; }
    .status { padding: 12px; border-radius: 10px; margin: 12px 0; } .ok { background: #dcfce7; } .warn { background: #fef3c7; } .error { background: #fee2e2; }
    .card { border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; margin-top: 14px; }
    .hidden { display: none; } .muted { color: #64748b; font-size: .9rem; } .people { display: flex; gap: 8px; flex-wrap: wrap; margin: 8px 0; }
    .person { background: #eff6ff; border-radius: 8px; padding: 8px; } .result { white-space: pre-wrap; background: #f0f2f4; padding: 12px; border-radius: 10px; margin-top: 8px; }
    @media (max-width: 600px) { .row { grid-template-columns: 1fr 60px 90px 90px 34px; font-size: .9rem; } body { padding: 8px; } main { padding: 14px; } }
  </style>
</head>
<body>
<main>
  <h1>SplitSnap</h1>
  <p class="muted">Scan, correct, assign, and split your receipt.</p>
  <form id="form">
    <input id="file" type="file" accept="image/*" capture="environment" required>
    <button type="submit">Extract receipt</button>
  </form>
  <div id="status"></div>
  <section id="editor" class="card hidden">
    <h2>Review receipt</h2>
    <p class="muted">Correct anything highlighted before splitting.</p>
    <div id="items"></div>
    <button type="button" class="secondary" id="addItem">Add item</button>
    <h2>Charges</h2>
    <div id="charges"></div>
    <button type="button" class="secondary" id="addCharge">Add charge</button>
    <div class="row" style="grid-template-columns:1fr 100px">
      <label>Subtotal<input id="subtotal" type="number" step="0.01"></label>
      <label>Grand total<input id="grandTotal" type="number" step="0.01"></label>
    </div>
    <h2>People</h2>
    <input id="people" type="text" placeholder="Names separated by commas">
    <div id="assignments"></div>
    <button type="button" id="split" disabled>Calculate split</button>
  </section>
  <section id="splitResult" class="card hidden">
    <h2>Split summary</h2>
    <div id="summary"></div>
    <div id="peopleResult"></div>
  </section>
</main>
<script>
const form = document.getElementById("form");
const statusBox = document.getElementById("status");
const editor = document.getElementById("editor");
const itemsBox = document.getElementById("items");
const chargesBox = document.getElementById("charges");
const assignmentsBox = document.getElementById("assignments");
let receipt = null;
const money = value => Number(value || 0).toFixed(2);
function itemRow(item = {}) {
  const row = document.createElement("div"); row.className = "row item-row";
  row.innerHTML = `<input class="name" placeholder="Item name" value="${item.name || ""}">
    <input class="qty" type="number" min="0" step="1" placeholder="Qty" value="${item.qty || 1}">
    <input class="unit" type="number" min="0" step="0.01" placeholder="Unit" value="${item.unit_price || 0}">
    <input class="total" type="number" step="0.01" placeholder="Total" value="${item.total || 0}">
    <button type="button" class="secondary remove">×</button>`;
  row.querySelector(".remove").onclick = () => { row.remove(); renderAssignments(); };
  ["name","qty","unit","total"].forEach(cls => row.querySelector("." + cls).oninput = renderAssignments);
  itemsBox.appendChild(row);
}
function chargeRow(charge = {}) {
  const row = document.createElement("div"); row.className = "charge-row";
  row.innerHTML = `<input class="type" placeholder="Charge type" value="${charge.type || ""}">
    <input class="amount" type="number" step="0.01" placeholder="Amount" value="${charge.amount || 0}">
    <button type="button" class="secondary remove">×</button>`;
  row.querySelector(".remove").onclick = () => row.remove();
  chargesBox.appendChild(row);
}
function renderAssignments() {
  const names = document.getElementById("people").value.split(",").map(x => x.trim()).filter(Boolean);
  assignmentsBox.innerHTML = "";
  [...itemsBox.querySelectorAll(".item-row")].forEach((row, index) => {
    const label = document.createElement("label"); label.textContent = `Assign item ${index + 1}`;
    const select = document.createElement("select"); select.multiple = true; select.style.width = "100%";
    names.forEach(name => { const option = document.createElement("option"); option.value = name; option.textContent = name; select.appendChild(option); });
    label.appendChild(select); assignmentsBox.appendChild(label);
  });
  document.getElementById("split").disabled = !names.length || !itemsBox.children.length;
}
function readReceipt() {
  return {
    items: [...itemsBox.querySelectorAll(".item-row")].map(row => ({
      name: row.querySelector(".name").value, qty: Number(row.querySelector(".qty").value || 0),
      unit_price: Number(row.querySelector(".unit").value || 0), total: Number(row.querySelector(".total").value || 0)
    })),
    charges: [...chargesBox.querySelectorAll(".charge-row")].map(row => ({
      type: row.querySelector(".type").value, amount: Number(row.querySelector(".amount").value || 0)
    })),
    subtotal: Number(document.getElementById("subtotal").value || 0),
    grand_total: Number(document.getElementById("grandTotal").value || 0)
  };
}
document.getElementById("people").oninput = renderAssignments;
document.getElementById("addItem").onclick = () => { itemRow(); renderAssignments(); };
document.getElementById("addCharge").onclick = () => chargeRow();
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("file").files[0];
  if (!file) return;
  statusBox.className = "status";
  statusBox.textContent = "Checking and extracting receipt...";
  editor.classList.add("hidden");
  const data = new FormData();
  data.append("file", file);
  try {
    const response = await fetch("/extract", { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Extraction failed");
    const retake = payload.status === "retake_photo";
    if (!retake && !payload.receipt) {
      const warnings = (payload.warnings || []).join(" ");
      throw new Error(warnings || "The model could not parse this receipt. Please try a clearer photo.");
    }
    statusBox.className = "status " + (retake ? "warn" : "ok");
    statusBox.textContent = retake
      ? "Please take the photo again: " + (payload.quality.guidance || []).join(" ")
      : "Extraction complete. Review the result before splitting.";
    if (!retake) {
      receipt = payload.receipt; itemsBox.innerHTML = ""; chargesBox.innerHTML = "";
      receipt.items.forEach(itemRow); receipt.charges.forEach(chargeRow);
      document.getElementById("subtotal").value = money(receipt.subtotal);
      document.getElementById("grandTotal").value = money(receipt.grand_total);
      editor.classList.remove("hidden"); renderAssignments();
    }
  } catch (error) {
    statusBox.className = "status error";
    statusBox.textContent = error.message;
  }
});
document.getElementById("split").onclick = async () => {
  const names = document.getElementById("people").value.split(",").map(x => x.trim()).filter(Boolean);
  const assignments = {};
  [...assignmentsBox.querySelectorAll("select")].forEach((select, index) => assignments[index] = [...select.selectedOptions].map(option => option.value));
  try {
    const response = await fetch("/split", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({receipt: readReceipt(), people: names, assignments})});
    const payload = await response.json(); if (!response.ok) throw new Error(payload.detail || "Split failed");
    document.getElementById("summary").textContent = payload.summary;
    document.getElementById("peopleResult").innerHTML = payload.people.map(person => `<div class="person"><strong>${person.name}: ${person.amount}</strong><div class="result">${person.explanation}</div></div>`).join("");
    document.getElementById("splitResult").classList.remove("hidden");
  } catch (error) { statusBox.className = "status error"; statusBox.textContent = error.message; }
};
</script>
</body>
</html>"""


@app.post("/extract")
async def extract(file: UploadFile = File(...)) -> Dict[str, Any]:
    payload = await file.read()
    try:
        image = Image.open(io.BytesIO(payload)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from error

    quality = assess_receipt_image(image)
    result: Dict[str, Any] = {
        "quality": {
            "status": quality.status,
            "score": quality.score,
            "reasons": quality.reasons,
            "guidance": quality.guidance,
        }
    }
    if quality.status == "retake_photo":
        result["status"] = "retake_photo"
        return result

    # The checkpoints were trained on original CORD images. Keep preprocessing
    # available for future OCR fallback, but avoid an inference-time
    # distribution shift for Donut.
    processor, model = load_model()
    images = [
        crop_receipt_item_region(image),
        crop_receipt_item_region(image, 0.16, 0.88),
        image,
    ]
    generated = None
    raw_output = ""
    receipt = None
    for model_image in images:
        encoded = processor(images=model_image, return_tensors="pt")
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=DEVICE.startswith("cuda"),
            ):
                candidate = model.generate(
                    encoded.pixel_values.to(DEVICE),
                    max_length=512,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    num_beams=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
        candidate_text = processor.batch_decode(candidate.sequences, skip_special_tokens=False)[0]
        candidate_receipt = parse_model_json(candidate_text)
        if candidate_receipt is not None:
            generated = candidate
            raw_output = candidate_text
            receipt = candidate_receipt
            break
        raw_output = candidate_text
    recovery_warnings: List[str] = []
    if receipt is None:
        receipt, recovery_warnings = recover_items_from_malformed_output(raw_output)
    result["raw_output"] = raw_output
    if receipt is None:
        result["status"] = "manual_review"
        result["warnings"] = ["Model output was not valid SplitSnap JSON."]
        return result

    confidences = item_confidences(generated, processor) if generated is not None else []
    for index, item in enumerate(receipt["items"]):
        if index < len(confidences):
            item["confidence"] = confidences[index]
    cleaned_receipt, postprocess_warnings = remove_likely_venue_items(receipt)
    repaired = repair_receipt(cleaned_receipt)
    result["status"] = "extracted"
    result["receipt"] = repaired["receipt"]
    result["repair"] = {
        "changes": repaired["changes"],
        "warnings": recovery_warnings + postprocess_warnings + repaired["warnings"],
        "consistent": repaired["consistent"],
    }
    return result


@app.post("/split")
def split(request: SplitRequest) -> Dict[str, Any]:
    try:
        receipt = normalize_receipt_record(request.receipt)
        split_result = split_bill(receipt, request.people, request.assignments)
        return explain_split(split_result, receipt)
    except (TypeError, ValueError, KeyError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
