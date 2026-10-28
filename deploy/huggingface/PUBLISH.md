# Publishing the public demo on Hugging Face Spaces

The demo runs as a **Gradio** Space on free hardware (CPU Basic, or ZeroGPU where CPU Basic is
not offered). It serves the dashboard from a SQLite snapshot and scores emails live; it
needs no secrets. A Docker variant (`deploy/huggingface/docker`) exists for accounts where
Docker Spaces are available.

## 1. Build the Space folder

```bash
docker compose up -d --wait                  # the snapshot is read from Postgres
uv run sentinel demo snapshot                # -> data/demo/sentinel_demo.sqlite (~23 MB)
cd frontend && npm run build && cd ..        # the Space serves the built dashboard
uv run python scripts/stage_hf_space.py      # -> build/hf-space (~320 MB)
```

The overview's throughput chart in the demo shows a series Prometheus recorded during a real
replay, saved as `data/demo/throughput.json` (git-ignored, like the snapshot): query
`sum(rate(sentinel_flows_scored_total[1m]))` over a replay window at a 15 s step and store
`{"recorded", "note", "points": [{"t", "flows_per_s"}]}`. Without the file the chart says
there is no data.

`deploy/huggingface/gradio/requirements.txt` comes from `uv.lock` via
`scripts/hf_requirements.py` (see its docstring); regenerate it when dependencies change.

Optional local check, installing the way the Space does:

```bash
docker build -f deploy/huggingface/gradio/test.Dockerfile -t sentinel-gradio-test build/hf-space
docker run --rm -p 127.0.0.1:7860:7860 sentinel-gradio-test     # http://localhost:7860/app/
```

## 2. Create the Space (once)

1. Sign in at https://huggingface.co and open **New Space**.
2. Name it (for example `sentinel`). Choose the **Gradio** SDK, **Blank** template, the free
   hardware on offer (**CPU Basic**, otherwise **ZeroGPU**) and **Public** visibility.
   ZeroGPU hosting on a free account needs a verified email and an account older than 30
   days.
3. Create an access token with **write** permission (Settings → Access Tokens).

## 3. Push the folder

With Git and Git LFS (the model and database are stored with LFS):

```bash
git lfs install
git clone https://huggingface.co/spaces/<your-username>/<space-name> hf-space-repo
robocopy build\hf-space hf-space-repo /E       # macOS/Linux: cp -r build/hf-space/. hf-space-repo/
cd hf-space-repo
git add .
git commit -m "Sentinel public demo"
git push                                       # user name + the write token as password
```

The Space installs the requirements (about 10–15 minutes the first time) and starts. The
dashboard is at `https://<your-username>-<space-name>.hf.space/app/` (the Space page shows a
small Gradio landing page with the same link).

## Updating

Re-run step 1, copy the folder over the clone again, commit and push. The Space rebuilds.

## What the demo turns off

Analyst feedback, live copilot investigations (stored reports still open), and live evasion
runs. Scoring endpoints allow 20 requests a minute per visitor. See `src/sentinel/demo.py`.
