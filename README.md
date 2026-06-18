# Benchmark — OpenRouter / OpenAI VLM experiments

This repository contains scripts and data for running and evaluating vision-language model (VLM) benchmarks using OpenRouter and OpenAI APIs.

**Quick summary:**
- **Install:** create a Python 3.12 virtual environment and install `requirements.txt`.
- **Run example scripts:** `ingopenrouteur.py` (image prompt + Responses API) and `listmodelopenrouteur.py` (list OpenRouter models).
- **Benchmarks:** organized under `benchmarkvlm/` with `data/`, `gen/`, `eval/`, and result folders.

**Requirements**: See [requirements.txt](requirements.txt) for exact package versions.

**Setup**

1. Create and activate a virtual environment:

```bash
python3 -m venv venv_benchmark
source venv_benchmark/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set API keys in environment (example):

```bash
export OPENAI_API_KEY="sk..."
export OPENROUTER_API_KEY="or_..."
```

Note: some scripts use `openai` and others use `openrouter` packages; both keys may be required depending on the script.

**Usage examples**

- Run the image prompt example (demonstrates sending an image + text prompt):

```bash
python ingopenrouteur.py
```

- List available OpenRouter models:

```bash
python listmodelopenrouteur.py
```

- Run benchmark generation or evaluation scripts (examples):

```bash
python benchmarkvlm/gen/test1gen.py
python benchmarkvlm/eval/test1eval.py
```

Adjust the script names or test numbers to run different benchmark cases.

**Repository structure**

- `ingopenrouteur.py` — example that encodes an image to base64 and calls a Responses-style API.
- `listmodelopenrouteur.py` — example script that lists models via the OpenRouter client.
- `requirements.txt` — pinned Python dependencies.
- `venv_benchmark/` — (optional) committed virtual environment in this workspace (already present here).
- `benchmarkvlm/`
  - `data/` — datasets used by benchmarks (`datatest1,2,3`, `datatest4,5`, ...)
  - `gen/` — generation scripts (e.g. `test1gen.py`)
  - `eval/` — evaluation scripts (e.g. `test1eval.py`)
  - `resulteval/` — evaluation outputs organized by backend (e.g. `chat/`, `claude/`)
  - `resultgen/` — generation output files

**How results are organized**

- Generated outputs: `benchmarkvlm/resultgen/` (CSV/JSON runs)
- Evaluation outputs: `benchmarkvlm/resulteval/` with subfolders per backend (`chat`, `claude`, ...)

**Notes & recommendations**

- Keep API keys out of the repository — use environment variables or an external secrets manager.
- Consider adding a lightweight runner script (e.g. `run_benchmark.py`) to parameterize tests and ensure reproducible runs.
- If you want CI for reproducible benchmarking, add a small `tox` or GitHub Actions workflow that installs the pinned `requirements.txt` and runs a subset of the gen/eval scripts.

**Next steps I can help with**

- Add a `run_benchmark.py` orchestrator.
- Add a minimal `README` examples section showing how to run a single end-to-end test and reproduce results.
- Create a `.env.example` showing required environment variables.

---
Generated from repository inspection on 2026-06-17.
