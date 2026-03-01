# AutoProfile

GPUnity is an ML training optimization pipeline that profiles a training job, generates optimization candidates, validates them, and reports cost/performance tradeoffs.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Streamlit Command Center

```bash
streamlit run gpunity/ui/app.py
```

The dashboard executes the full GPUnity flow in-process:

- profile
- analyze
- validate
- report

Then it renders:

- cost × time decision surface
- time-first recommendation under cost cap
- 1–2 alternatives
- diverged configuration warnings
- run summary and artifact paths

## CLI (optional)

```bash
gpunity run /path/to/repo --local --provider heuristic --objective-profile latency
```