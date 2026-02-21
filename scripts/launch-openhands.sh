#!/usr/bin/env bash
# launch-openhands.sh — Start OpenHands for the ROSE project
#
# Uses the officially recommended approach from:
# https://docs.openhands.dev/openhands/usage/run-openhands/local-setup#linux
#
# Usage:
#   ./scripts/launch-openhands.sh          # Use uv (recommended)
#   ./scripts/launch-openhands.sh docker   # Use Docker directly
#
# Prerequisites:
#   - Docker Engine running (native, not Docker Desktop)
#   - Local LLM server running at http://localhost:8555/v1
#
# The script will install uv and openhands if needed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

# Local LLM server endpoint (as seen from inside the container)
LLM_BASE_URL="${LLM_BASE_URL:-http://host.docker.internal:8555/v1}"

# Model identifier
LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"

# API key (local servers typically don't need one, but OpenHands requires a value)
LLM_API_KEY="${LLM_API_KEY:-not-needed}"

# OpenHands version
OPENHANDS_VERSION="${OPENHANDS_VERSION:-1.4}"

# Port for the web UI
OPENHANDS_PORT="${OPENHANDS_PORT:-3000}"

# Project directory (this repo)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Launch method: "uv" (default) or "docker"
METHOD="${1:-uv}"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "🔍 Pre-flight checks..."

if ! command -v docker &>/dev/null; then
    echo "❌ Docker is not installed or not in PATH."
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    echo "❌ Docker daemon is not running."
    echo "   Try: sudo systemctl start docker"
    exit 1
fi

# Ensure we're using the native Docker Engine, not Docker Desktop
DOCKER_CONTEXT=$(docker context show 2>/dev/null || echo "unknown")
if [[ "${DOCKER_CONTEXT}" == "desktop-linux" ]]; then
    echo "⚠️  Switching from Docker Desktop to native Docker Engine..."
    docker context use default
    if ! docker info &>/dev/null 2>&1; then
        echo "❌ Native Docker Engine is not running."
        echo "   Try: sudo systemctl start docker"
        exit 1
    fi
fi

echo "  ✅ Docker is available (context: $(docker context show 2>/dev/null))"

# Check if LLM server is reachable from the host
HOST_LLM_URL="$(echo "${LLM_BASE_URL}" | sed 's|host.docker.internal|localhost|')"
if command -v curl &>/dev/null; then
    if curl --silent --max-time 3 "${HOST_LLM_URL}/models" >/dev/null 2>&1; then
        echo "  ✅ LLM server is reachable at ${HOST_LLM_URL}"
    else
        echo "  ⚠️  LLM server not reachable at ${HOST_LLM_URL}"
        echo "     Make sure your local LLM server is running before using OpenHands."
    fi
fi

echo ""

# ---------------------------------------------------------------------------
# Show startup prompt
# ---------------------------------------------------------------------------

show_prompt() {
    local PROMPT_FILE="${PROJECT_DIR}/scripts/startup-prompt.txt"
    if [[ -f "${PROMPT_FILE}" ]]; then
        echo ""
        echo "📋 Paste this into the OpenHands chat to start:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        cat "${PROMPT_FILE}"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "💡 The prompt is also at: scripts/startup-prompt.txt"
        # Copy to clipboard if possible
        if command -v xclip &>/dev/null; then
            cat "${PROMPT_FILE}" | xclip -selection clipboard
            echo "📎 Copied to clipboard!"
        elif command -v xsel &>/dev/null; then
            cat "${PROMPT_FILE}" | xsel --clipboard
            echo "📎 Copied to clipboard!"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if [[ "${METHOD}" == "uv" ]]; then
    # -----------------------------------------------------------------------
    # Option 1: uv (Recommended by OpenHands docs)
    # -----------------------------------------------------------------------

    # Install uv if needed
    if ! command -v uv &>/dev/null; then
        echo "📦 Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi

    # Install openhands if needed
    if ! uv tool list 2>/dev/null | grep -q openhands; then
        echo "📦 Installing openhands via uv..."
        uv tool install openhands --python 3.12
    fi

    echo "🚀 Launching OpenHands (via uv)..."
    echo "   UI:        http://localhost:${OPENHANDS_PORT}"
    echo "   LLM URL:   ${LLM_BASE_URL}  (inside container)"
    echo "   LLM Model: ${LLM_MODEL}"
    echo "   Workspace: ${PROJECT_DIR}"
    echo ""
    echo "   Configure in the UI Settings (LLM tab, enable Advanced):"
    echo "     • Custom Model → ${LLM_MODEL}"
    echo "     • Base URL     → ${LLM_BASE_URL}"
    echo "     • API Key      → ${LLM_API_KEY}"
    echo ""

    show_prompt

    # Launch from the project directory so --mount-cwd picks it up
    cd "${PROJECT_DIR}"
    openhands serve --mount-cwd

elif [[ "${METHOD}" == "docker" ]]; then
    # -----------------------------------------------------------------------
    # Option 2: Docker directly (from official docs)
    # -----------------------------------------------------------------------

    echo "🚀 Launching OpenHands (via Docker)..."
    echo "   Version:   ${OPENHANDS_VERSION}"
    echo "   UI:        http://localhost:${OPENHANDS_PORT}"
    echo "   LLM URL:   ${LLM_BASE_URL}  (inside container)"
    echo "   LLM Model: ${LLM_MODEL}"
    echo "   Workspace: ${PROJECT_DIR}"
    echo ""
    echo "   Configure in the UI Settings (LLM tab, enable Advanced):"
    echo "     • Custom Model → ${LLM_MODEL}"
    echo "     • Base URL     → ${LLM_BASE_URL}"
    echo "     • API Key      → ${LLM_API_KEY}"
    echo ""

    # Stop any existing container
    docker stop openhands-app 2>/dev/null || true
    docker rm openhands-app 2>/dev/null || true

    # Migrate state directory if needed (per docs: v0.44+ uses ~/.openhands)
    if [[ -d "$HOME/.openhands-state" ]] && [[ ! -d "$HOME/.openhands" ]]; then
        echo "   Migrating state: ~/.openhands-state → ~/.openhands"
        mv "$HOME/.openhands-state" "$HOME/.openhands"
    fi
    mkdir -p "$HOME/.openhands"

    show_prompt

    docker run -it --rm --pull=always \
        --name openhands-app \
        --add-host host.docker.internal:host-gateway \
        -e LOG_ALL_EVENTS=true \
        -v /var/run/docker.sock:/var/run/docker.sock \
        -v "$HOME/.openhands:/.openhands" \
        -v "${PROJECT_DIR}:/opt/workspace_base" \
        -p "${OPENHANDS_PORT}:3000" \
        "docker.openhands.dev/openhands/openhands:${OPENHANDS_VERSION}"

else
    echo "❌ Unknown method: ${METHOD}"
    echo "   Usage: $0 [uv|docker]"
    exit 1
fi