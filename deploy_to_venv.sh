#!/usr/bin/env bash
# Deploy DAG-aware changes from this repo to the motivation venv site-packages.
# Run this after editing any of the tracked files, then restart the vLLM server.

set -euo pipefail

REPO="/home/cyf/dag-aware-vllm-0.19.1/vllm"
SITE="/home/cyf/dag-aware-vllm/experiments/motivation/.venv/lib/python3.10/site-packages/vllm"

echo "==> Deploying DAG-aware changes to site-packages..."

# New files (DAG registry + API router)
cp "$REPO/dag_registry.py"   "$SITE/dag_registry.py"
cp "$REPO/dag_api_router.py" "$SITE/dag_api_router.py"

# Modified existing files
cp "$REPO/entrypoints/openai/chat_completion/protocol.py" \
   "$SITE/entrypoints/openai/chat_completion/protocol.py"
cp "$REPO/entrypoints/openai/chat_completion/serving.py" \
   "$SITE/entrypoints/openai/chat_completion/serving.py"
cp "$REPO/entrypoints/openai/api_server.py" \
   "$SITE/entrypoints/openai/api_server.py"

# Clear stale .pyc caches for the changed modules
find "$SITE/entrypoints/openai/" -name "*.pyc" \
  \( -name "protocol*" -o -name "serving*" -o -name "api_server*" \) -delete 2>/dev/null || true
find "$SITE" -name "dag_registry*" -name "*.pyc" -delete 2>/dev/null || true
find "$SITE" -name "dag_api_router*" -name "*.pyc" -delete 2>/dev/null || true

echo "==> Done. Restart vLLM server to pick up changes."
