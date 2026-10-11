# 🍽️ SplitSnap: Local Setup Guide 

SplitSnap is an AI-powered receipt extraction and fair bill-splitting web application designed for group dining and Indian receipts. It uses a fine-tuned vision-language model (Donut) to extract line items, taxes, and totals directly from receipt photos, validates arithmetic consistency, and calculates mathematically exact, proportional splits.

---

## How to Run SplitSnap Locally

### 1. Prerequisites
- **Python**: 3.10 to 3.12 (or 3.14)
- **Node.js**: v18 or higher (with `npm`)
- **GPU (Optional)**: NVIDIA GPU with CUDA for faster inference (runs on CPU automatically if CUDA is unavailable).

---

### 2. Backend Setup (FastAPI & PyTorch)

1. **Open a terminal in the project root directory**:
   ```powershell
   cd D:\CrackHeads
   ```

2. **Create and activate a Python virtual environment**:
   * Windows (PowerShell):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * macOS / Linux:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the FastAPI backend server**:
   ```bash
   python -m uvicorn scripts.app:app --host 127.0.0.1 --port 8000
   ```

5. **Verify Backend Health**:
   Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or check the interactive API docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

### 3. Frontend Setup (React + Vite + TypeScript)

1. **Open a new terminal window** and navigate to the `frontend` directory:
   ```powershell
   cd D:\CrackHeads\frontend
   ```

2. **Install frontend dependencies**:
   ```bash
   npm install
   ```

3. **Start the Vite development server**:
   ```bash
   npm run dev
   ```

4. **Access the Web App**:
   Open your browser at:
   👉 **[http://localhost:5173](http://localhost:5173)**

---

### 4. Running Extraction via Command Line (CLI)

You can also run receipt extraction directly without starting the web server:
```powershell
python scripts\extract_receipt.py --input path\to\receipt.jpg --output extraction_result.json
```

---
