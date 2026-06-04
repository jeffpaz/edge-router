FROM python:3.11-slim

# Create a non-root user before installing anything
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── Layer 1: dependencies (cached unless requirements.txt changes) ────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Layer 2: application source ───────────────────────────────────────────────
COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000

# --workers 1     One process per container; avoids competing for memory with
#                 Ollama on the Jetson's unified address space.
# --log-level warning  The app emits structured JSON logs itself; suppress
#                      uvicorn's own info-level chatter.
# --no-access-log  Redundant with the app's per-request JSON logging.
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "warning", \
     "--no-access-log"]
