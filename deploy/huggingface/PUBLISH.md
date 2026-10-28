# Publishing the public demo on Hugging Face Spaces

The demo is a Docker Space: free CPU hardware (2 vCPU, 16 GB RAM) is enough. It serves the
dashboard from a SQLite snapshot and scores emails live; it needs no secrets.

## 1. Build the Space folder

```bash
docker compose up -d --wait                  # the snapshot is read from Postgres
uv run sentinel demo snapshot                # -> data/demo/sentinel_demo.sqlite (~23 MB)
uv run python scripts/stage_hf_space.py      # -> build/hf-space (~320 MB)
```

Optional local check, exactly as the Space will run it:

```bash
docker build -t sentinel-hf-demo build/hf-space
docker run --rm -p 7860:7860 sentinel-hf-demo      # http://localhost:7860/app/
```

## 2. Create the Space (once)

1. Sign in at https://huggingface.co and open **New Space**.
2. Name it (for example `sentinel`), choose **Docker** as the SDK with the **Blank** template,
   **CPU basic** (free) hardware, and **Public** visibility.
3. Create an access token with **write** permission (Settings → Access Tokens).

## 3. Push the folder

With Git and Git LFS (the model and database are stored with LFS):

```bash
git lfs install
git clone https://huggingface.co/spaces/<your-username>/sentinel hf-space-repo
cp -r build/hf-space/. hf-space-repo/          # on Windows: robocopy build\hf-space hf-space-repo /E
cd hf-space-repo
git add .
git commit -m "Sentinel public demo"
git push                                       # user name + the write token as password
```

The Space builds the image (about 10 minutes the first time) and starts. The dashboard is
at `https://<your-username>-sentinel.hf.space/app/`.

## Updating

Re-run step 1, copy the folder over the clone again, commit and push. The Space rebuilds.

## What the demo turns off

Analyst feedback, live copilot investigations (stored reports still open), and live evasion
runs. Scoring endpoints allow 20 requests a minute per visitor. See `src/sentinel/demo.py`.
