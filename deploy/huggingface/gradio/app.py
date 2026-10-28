"""Hugging Face Spaces entry point (Gradio SDK): serve the Sentinel public demo on port 7860.

The Space runs `python app.py`. The demo is the regular FastAPI app in demo mode (see
sentinel/demo.py) with the prebuilt dashboard at /app/. A small Gradio page is mounted at
/gradio, because Gradio-SDK (and ZeroGPU) Spaces expect a Gradio app.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The Space installs requirements.txt before its files exist, so Sentinel is not installed
# as a package; it is imported from src/ (it finds models and params.yaml relative to it).
sys.path.insert(0, str(HERE / "src"))
os.environ.setdefault("SENTINEL_DEMO", "1")
os.environ.setdefault(
    "POSTGRES_URL", f"sqlite:///{HERE / 'data' / 'demo' / 'sentinel_demo.sqlite'}"
)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# ZeroGPU: `spaces` must be imported before torch. It is effect-free on CPU hardware.
import spaces  # noqa: E402


@spaces.GPU
def gpu_probe() -> str:
    """Never called: Sentinel's models run on CPU. ZeroGPU expects one GPU function."""
    return "cpu"


import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from sentinel.services.api import create_app  # noqa: E402

with gr.Blocks(title="Sentinel") as landing:
    gr.Markdown(
        "# Sentinel\n\nA Security Operations Center that runs on a laptop: ML detection of "
        "network attacks and phishing, an LLM copilot that writes cited incident reports, "
        "and the MLOps that keeps both honest.\n\n"
        "**[Open the dashboard](/app/)** · "
        "[Code and full results on GitHub](https://github.com/sreevidhuyarra/sentinel)"
    )

# No server-side rendering: on Spaces it would start a Node server on the Space's only port
# (7860), the one this app must listen on. The landing page is static anyway.
app = gr.mount_gradio_app(create_app(), landing, path="/gradio", ssr_mode=False)


def zerogpu_startup() -> None:
    """Register with the ZeroGPU controller, as gr.Blocks.launch does; without the report the
    Space is stopped soon after it starts. `spaces.zero.startup` exists only on ZeroGPU."""
    startup = getattr(getattr(spaces, "zero", None), "startup", None)
    if startup is not None:
        startup()
        print("ZeroGPU startup reported", flush=True)


if __name__ == "__main__":
    zerogpu_startup()
    # Listen where the Space tells Gradio apps to (the same variables gr.Blocks.launch reads).
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    # Port settings only (never secrets), so the Space logs show where the app should listen.
    ports = {k: v for k, v in os.environ.items() if "PORT" in k or k.startswith("GRADIO_SERVER")}
    print(f"Sentinel demo listening on {host}:{port} (environment: {ports})", flush=True)
    uvicorn.run(app, host=host, port=port)
