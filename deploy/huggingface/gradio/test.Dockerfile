# Local stand-in for how Hugging Face builds a Gradio Space, to test build/hf-space before
# pushing (not part of the Space):
#   docker build -f deploy/huggingface/gradio/test.Dockerfile -t sentinel-gradio-test build/hf-space
#   docker run --rm -p 127.0.0.1:7860:7860 sentinel-gradio-test
FROM python:3.12.12
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app
# Like the Space: requirements are installed before the rest of the files are copied in.
COPY --chown=user requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt "torch<=2.13.0" "gradio[oauth,mcp]==6.31.0" \
    "uvicorn>=0.14.0" "websockets>=10.4" spaces==0.51.3
COPY --chown=user . .
EXPOSE 7860
CMD ["python", "app.py"]
