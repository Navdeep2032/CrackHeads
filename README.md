# SplitSnap

SplitSnap is a receipt extraction and bill-splitting application built for the Track 1 challenge.

## Current status

This repository now includes the first two steps of the project plan:
- standardized receipt JSON schema
- normalization script for raw receipt records
- deterministic dataset split and augmentation pipeline
- dependency baseline for model training and the web app

## Step 1: dataset schema and data pipeline

The dataset pipeline is defined around a unified receipt schema:

```json
{
  "items": [
    {
      "name": "Paneer Butter Masala",
      "qty": 1,
      "unit_price": 220,
      "total": 220
    }
  ],
  "charges": [
    { "type": "CGST", "amount": 18.5 },
    { "type": "Service Charge", "amount": 40.0 },
    { "type": "Discount", "amount": -50.0 }
  ],
  "subtotal": 740,
  "grand_total": 767
}
```

The normalization script lives at:
- `scripts/data/normalize_dataset.py`

The schema definition lives at:
- `config/receipt_schema.json`

## Usage

```bash
python scripts/data/normalize_dataset.py --input data/raw/example_receipt.json --output data/processed/example_receipt_normalized.json
```

## Step 2: dataset preparation

See [data/raw/README.md](data/raw/README.md) for the expected CORD, SROIE,
and held-out Indian-bill layout. After converting source annotations to the
`records.jsonl` format, run:

Download CORD using:

```powershell
python scripts\data\download_datasets.py --cord
```

SROIE must be downloaded from an official organizer-provided archive URL:

```powershell
python scripts\data\download_datasets.py --sroie-url "PASTE_THE_OFFICIAL_ZIP_URL_HERE"
```

```bash
python scripts/data/prepare_dataset.py --config config/dataset_config.json
```

If the command reports missing paths, the datasets have not been populated
yet. This command intentionally does not invent records or silently skip a
dataset. Download CORD and SROIE from their official sources, convert their
annotations to the documented JSONL format, and place them under `data/raw`.

The command writes:
- `data/processed/train.jsonl`
- `data/processed/validation.jsonl`
- `data/processed/test.jsonl`
- `data/processed/train_augmented.jsonl`
- augmented images under `data/processed/augmented_images`

The split uses seed `42` and keeps augmentation restricted to training records.
The Indian-bill set is optional during initial preparation and remains a
separate source so its generalization performance can be measured honestly.

## Step 3: convert raw parquet sources into records.jsonl

After the raw CORD and SROIE parquet files are downloaded, convert them into the
`records.jsonl` format expected by the project pipeline:

```powershell
python scripts\data\convert_parquet_to_records.py --source all --root .
```

Use `--clean-images` when re-running after an earlier conversion, so stale
generated PNGs are removed and every Parquet shard gets a unique image name:

```powershell
python scripts\data\convert_parquet_to_records.py --source all --root . --clean-images
```

This writes:
- `data/raw/cord/images/*.png`
- `data/raw/cord/records.jsonl`
- `data/raw/sroie/images/*.png`
- `data/raw/sroie/records.jsonl`

The generated receipt objects are normalized to the shared SplitSnap schema before
being used by the deterministic train/validation/test split.

## Step 4: baseline fine-tuning

The baseline extraction model trains against the unified SplitSnap target from
the prepared manifests. This keeps CORD and SROIE output structure consistent.

Run the smoke test:

```powershell
python scripts\model\train_receipt_extractor.py --epochs 1 --max-samples 10 --batch-size 1 --output-dir models\receipt_extractor_smoke_test --device cuda --image-height 640 --image-width 480 --mixed-precision
```

For a full but still lightweight run:

```powershell
python scripts\model\train_receipt_extractor.py --epochs 3 --batch-size 1 --output-dir models\receipt_extractor_experiment --device cuda --image-height 640 --image-width 480 --mixed-precision
```

Use `--max-samples` for a bounded experiment. Without it, the command uses all
prepared training and validation records.

Training now saves the checkpoint with the lowest validation loss rather than
always keeping the final epoch. The selected epoch and loss are printed when
training finishes.

## Receipt arithmetic validation

Before using extracted values for bill splitting, validate the arithmetic:

```powershell
python -c "from scripts.data.validate_receipt import validate_receipt; print(validate_receipt({'items': [], 'charges': [], 'subtotal': 0, 'grand_total': 0}))"
```

The evaluator reports whether predictions are internally consistent:
`items_consistent`, `subtotal_consistent`, and `grand_total_consistent`.
Inconsistent predictions should be shown for user correction rather than
silently passed to the split calculation.

The conservative repair helper is available at
`scripts/data/repair_receipt.py`. It fixes item totals only when quantity and
unit price provide an unambiguous calculation, and otherwise returns warnings.
The evaluator also reports `repair_consistency_rate` and `repair_warning_rate`.

## Bill splitting

The deterministic splitter is in `scripts/billing/split_bill.py`. It accepts
item-to-person assignments, divides shared items in cents, allocates charges
proportionally to each person's assigned item spend, and reports warnings for
unassigned items or totals that do not reconcile.

Use `scripts/billing/explain_split.py` to produce a plain-language breakdown
for each person, including assigned items, shared charges, total owed, and any
repair or reconciliation warnings.

## API service

Start the FastAPI service from the project root:

```powershell
python -m uvicorn scripts.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` - service and device status
- - `GET /app` - mobile-friendly receipt upload, correction, assignment, and split page
- `POST /extract` - upload a receipt image and receive quality-gated extraction
- `POST /split` - submit a receipt, people, and item assignments for a
  rounding-safe split with explanations

Set `SPLITSNAP_MODEL_DIR` or `SPLITSNAP_DEVICE` as environment variables when
using a different checkpoint or inference device.

Open the browser interface at:

```text
http://127.0.0.1:8000/app
```

## Vercel React frontend

The deployable React frontend is in [frontend/](frontend/). It uses the
FastAPI service through `VITE_API_URL`.

Local development:

```powershell
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

For Vercel, import the repository, set the project root to `frontend`, and add
the environment variable `VITE_API_URL` pointing to the deployed FastAPI
backend. The backend must allow the Vercel domain in its CORS configuration.
Set `SPLITSNAP_CORS_ORIGINS` on the backend to a comma-separated list such as
`https://your-app.vercel.app,http://localhost:5173,http://127.0.0.1:5173`.

## Photo quality gate

Before running receipt extraction, call
`scripts/image_quality.py:assess_receipt_image`. It returns:

- `ready`: run extraction
- `review_quality`: show a quality warning and allow the user to continue
- `retake_photo`: prompt the user to capture the receipt again

The gate checks resolution, blur, contrast, and brightness, and provides
capture guidance. It is a pre-OCR signal, not a replacement for model
confidence or receipt arithmetic validation.

## Image preprocessing

Use [scripts/image_preprocessing.py](scripts/image_preprocessing.py) after the
quality gate to create OCR-friendly variants:

```powershell
python scripts\image_preprocessing.py `
  --input receipt.jpg `
  --color-output data\processed\receipt_enhanced.png `
  --gray-output data\processed\receipt_ocr.png `
  --binarize
```

The pipeline upscales small images, estimates and corrects small skew, reduces
noise, enhances local contrast, and optionally creates an adaptive-threshold
black-and-white variant. Keep the original photo and prefer the enhanced color
variant for Donut; use the grayscale/binary variant for traditional OCR or
diagnostics.

## Integrated extraction workflow

Run the complete quality-gated extraction flow with:

```powershell
python scripts\extract_receipt.py `
  --input receipt.jpg `
  --model-dir models\receipt_model `
  --device cuda `
  --output data\processed\receipt_extraction.json
```

The command returns `retake_photo` for clearly unusable images, otherwise it
preprocesses the photo, runs Donut, parses the SplitSnap JSON, and applies
conservative arithmetic repair. A result with repair warnings or
`consistent: false` must be shown for user confirmation before bill splitting.

Extraction post-processing also filters obvious venue/header terms such as
`Hotel`, `Restaurant`, `Cafe`, and `Kitchen` when they appear as item names.
Each removal is returned as a warning for review; this heuristic does not
replace user confirmation.

For item-extraction experiments, train only on CORD because the downloaded
SROIE annotations do not contain line-item labels:

```powershell
python scripts\model\train_receipt_extractor.py --source cord --epochs 3 --batch-size 1 --output-dir models\cord_experiment --device cuda --image-height 640 --image-width 480 --mixed-precision
```

## Synthetic Indian restaurant augmentation

Synthetic receipts can improve vocabulary and layout coverage for Indian
restaurant use cases, but they must only be added to training. Keep real CORD,
SROIE, and Indian receipts in validation/test sets.

Generate deterministic labeled receipt images:

```powershell
python scripts\data\generate_synthetic_receipts.py --count 1000 --seed 42 --output-dir data\raw\synthetic_indian
```

The synthetic generator now renders a full scene, automatically detects and
rectifies the receipt, and saves the cropped receipt image for training. It
also adds rotation, perspective warp, blur, noise, brightness/contrast
variation, and JPEG artifacts so the model sees harder photo-like examples.

Fine-tune with the generated records appended to the real training manifest:

```powershell
python scripts\model\train_receipt_extractor.py --source cord --synthetic-manifest data\raw\synthetic_indian\records.jsonl --epochs 3 --batch-size 1 --output-dir models\cord_indian_experiment --device cuda --image-height 640 --image-width 480 --mixed-precision
```

Use `--max-synthetic-samples` to prevent synthetic data from dominating the
real CORD examples. For example, 300 synthetic records plus real CORD:

```powershell
python scripts\model\train_receipt_extractor.py --source cord --synthetic-manifest data\raw\synthetic_indian\records.jsonl --max-synthetic-samples 300 --epochs 3 --batch-size 1 --output-dir models\receipt_model --device cuda --image-height 640 --image-width 480 --mixed-precision
```

Synthetic data is not a substitute for real Indian receipt photos. Add a
separate held-out real Indian set before claiming deployment accuracy.

## Next steps

1. Download and convert CORD and SROIE annotations
2. Populate the held-out Indian-bill records
3. Run the dataset preparation script
4. Fine-tune the baseline Donut model
5. Expand the target schema to the final SplitSnap JSON format
