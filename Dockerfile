# Backend image for the move API (server.py). Works on Hugging Face Spaces
# (Docker SDK, app_port 7860) and any other container host.
FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch first (no CUDA -> far smaller image), then the serving deps.
# requests/tqdm are only needed for training, so they're omitted here.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir python-chess fastapi "uvicorn[standard]" numpy

# App code + the trained model (models/ is gitignored locally; see DEPLOY.md
# for how to include the .pt file when pushing to the Space).
COPY server.py engine.py model.py encoding.py ./
COPY models/ ./models/

# HF Spaces routes external traffic to app_port (7860). Keep caches writable.
ENV HOME=/app PORT=7860
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
