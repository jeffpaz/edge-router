#!/usr/bin/env bash
# deploy.sh — build and start the edge-router container on Jetson
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST_IP="192.168.11.50"

# ── 1. Guard: .env must exist ─────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo "No .env file found — creating one from .env.example."
    cp .env.example .env
    echo ""
    echo "  Fill in your API keys, then re-run this script:"
    echo "    nano .env"
    echo "    ./deploy.sh"
    echo ""
    exit 1
fi

# ── 2. Tear down any running instance ────────────────────────────────────────
echo "==> Stopping existing container..."
docker compose down

# ── 3. Fresh build (no layer cache) ──────────────────────────────────────────
echo "==> Building image (no cache)..."
docker compose build --no-cache

# ── 4. Start detached ────────────────────────────────────────────────────────
echo "==> Starting edge-router..."
docker compose up -d

# ── 5. Connect ollama container to edge-net for DNS-based access ──────────────
# This step enables http://ollama:11434 as an alternative to host.docker.internal.
# Safe to run repeatedly — fails silently if already connected or container absent.
if docker inspect ollama > /dev/null 2>&1; then
    if docker network connect edge-net ollama 2>/dev/null; then
        echo "==> ollama connected to edge-net"
        echo "    http://ollama:11434 is now available inside edge-router"
    else
        echo "==> ollama already connected to edge-net"
    fi
else
    echo "==> ollama container not found — Ollama will be reached via host.docker.internal"
fi

# ── 6. Wait briefly for the container to be healthy ──────────────────────────
echo "==> Waiting for health check..."
for i in $(seq 1 10); do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        echo "    Container is healthy."
        break
    fi
    echo "    Attempt ${i}/10 — retrying in 2s..."
    sleep 2
done

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "┌─────────────────────────────────────────────────────────────────────┐"
echo "│  edge-router running on http://${HOST_IP}:8000                  │"
echo "└─────────────────────────────────────────────────────────────────────┘"
echo ""
echo "  Test query (coding skill → Claude):"
echo "    curl -s -X POST http://${HOST_IP}:8000/query \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"query\": \"Write a Python function to sort a list\"}' \\"
echo "         | python3 -m json.tool"
echo ""
echo "  Force cloud (skip local model):"
echo "    curl -s -X POST http://${HOST_IP}:8000/query \\"
echo "         -H 'Content-Type: application/json' \\"
echo "         -d '{\"query\": \"Latest AI news\", \"force_cloud\": true}' \\"
echo "         | python3 -m json.tool"
echo ""
echo "  Health:  curl http://${HOST_IP}:8000/health"
echo "  Stats:   curl http://${HOST_IP}:8000/stats"
echo ""
echo "==> Tailing logs (Ctrl+C to exit without stopping the container)..."
echo ""
docker compose logs -f edge-router
