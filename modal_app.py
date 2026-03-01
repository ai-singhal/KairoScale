"""GPUnity vLLM inference server on Modal.

Deploys a Qwen3-8B model with OpenAI-compatible API and tool-use support.
Used by gpunity's Modal provider for the agent analysis loop.

Deploy:
    modal deploy modal_app.py

The endpoint URL will be printed after deployment. Set it in .env:
    MODAL_VLLM_URL=https://<your-workspace>--gpunity-vllm-serve.modal.run
"""

from __future__ import annotations

import modal

MODEL_NAME = "Qwen/Qwen3-8B"
GPU_TYPE = "A10G"

app = modal.App("gpunity-vllm")

# Persistent volume for model weights — avoids re-download on cold start
model_volume = modal.Volume.from_name("gpunity-model-weights", create_if_missing=True)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.8.0",
        "torch>=2.4.0",
    )
)


@app.function(
    image=vllm_image,
    gpu=GPU_TYPE,
    volumes={"/model-weights": model_volume},
    timeout=600,
    min_containers=1,
)
@modal.web_server(port=8000, startup_timeout=300)
def serve():
    import subprocess

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--download-dir", "/model-weights",
        "--port", "8000",
        "--host", "0.0.0.0",
        "--dtype", "float16",
        "--max-model-len", "8192",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]
    subprocess.Popen(cmd)
