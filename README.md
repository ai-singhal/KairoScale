# KairoScale

KairoScale is an ML training optimization pipeline that runs:

1. `Profile` your training script
2. `Analyze` bottlenecks and generate optimization candidates
3. `Validate` candidates against a control run
4. `Report` speed/cost/stability results


## What You Need

- Python 3.10+
- `pip`
- A training repo with an entry script (default: `train.py`)

For cloud GPU runs:
- Modal account + Modal CLI configured (`modal setup`)

For LLM-backed analysis providers:
- `ANTHROPIC_API_KEY` (provider `claude`) or
- `OPENAI_API_KEY` (provider `openai`)

## Quick Setup (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs this package with all integrations and dev tooling.

## Environment Variables

Create `.env` (or export directly in shell):

```bash
# For Modal sandbox runs
export MODAL_TOKEN_ID=your_modal_token_id
export MODAL_TOKEN_SECRET=your_modal_token_secret

# Pick one (or both) for LLM analysis
export ANTHROPIC_API_KEY=your_anthropic_key
export OPENAI_API_KEY=your_openai_key

# Optional: pre-existing Modal vLLM endpoint
export MODAL_VLLM_URL=https://<workspace>--KairoScale-vllm-serve.modal.run
```

`.env.example` includes Modal + OpenAI placeholders.

## Run KairoScale

### Option A: Streamlit Command Center (fastest demo path)

```bash
streamlit run KairoScale/ui/app.py
```

Use this for live demos and hackathon judging flow.

### Option B: CLI Full Pipeline

#### Local mode (no Modal required)

```bash
KairoScale run /abs/path/to/your/repo \
  --local \
  --provider heuristic \
  --objective-profile latency \
  --entry train.py
```

Good for quick iteration when cloud GPUs or API keys are unavailable.

#### Modal mode (cloud GPU)

```bash
KairoScale run /abs/path/to/your/repo \
  --provider openai \
  --objective-profile balanced \
  --gpu a100-80gb \
  --entry train.py
```

Notes:
- Omit `--local` to run in Modal sandboxes.
- `--provider` can be `claude`, `openai`, `modal`, or `heuristic`.
- If `--provider modal` is used and `MODAL_VLLM_URL` is not set, KairoScale auto-deploys `modal_app.py`.

## Deploy Winning Config (Production-style run)

After a pipeline run, config JSONs are exported to `KairoScale_configs/`.

Deploy one config:

```bash
KairoScale deploy KairoScale_configs/opt-001.json \
  --repo /abs/path/to/your/repo \
  --gpu a100-80gb \
  --entry train.py
```

Run deploy locally instead of Modal:

```bash
KairoScale deploy KairoScale_configs/opt-001.json \
  --repo /abs/path/to/your/repo \
  --local \
  --entry train.py
```

Optional deploy flags:
- `--steps 1000` (sets `TRAIN_STEPS` in wrapper env)
- `--timeout 3600`
- `--python-bin /path/to/python` (local mode)

## Typical Outputs

- `KairoScale_report.md` (final report)
- `KairoScale_configs/*.json` (candidate and combo configs)
- Local/Modal artifact directories with profiling and validation outputs

## Useful Commands

```bash
KairoScale --help
KairoScale run --help
KairoScale deploy --help
pytest -q
```

## Tools Used

Core tools/services used in this project:

- Python + Click (CLI)
- Streamlit (UI command center)
- Modal (GPU sandbox execution + optional hosted vLLM endpoint)
- PyTorch profiler + runtime instrumentation (profiling)
- Anthropic/OpenAI-compatible providers for optimization generation
- PyYAML (config loading)
- Pytest (test suite)