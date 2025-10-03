Project backend for ExoAI

Quick setup (Windows PowerShell):

1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies

# PyTorch on Windows: pick the right command for your CUDA or CPU setup from https://pytorch.org/get-started/locally/

# Example (CPU-only):

pip install torch --index-url https://download.pytorch.org/whl/cpu

Then install the rest:

```powershell
pip install -r requirements.txt
```

3. Run tests

```powershell
cd app
pytest -q
```

4. Run the API (development)

```powershell
cd app
uvicorn main:app --reload --port 8000
```

Notes:

- If you plan to use GPU, install the matching torch package from the PyTorch website.
- ONNX and ONNX Runtime are optional and used only in export tests.
