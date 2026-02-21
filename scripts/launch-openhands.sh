#!/usr/bin/env bash
# launch-openhands.sh — Start OpenHands for the ROSE project
#
# Usage:
#   ./scripts/launch-openhands.sh
#
# Prerequisites:
#   - Docker Engine running
#   - Local LLM server running at http://localhost:8555/v1
#   - openhands installed:  uv tool install openhands --python 3.12

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

LLM_BASE_URL="${LLM_BASE_URL:-http://host.docker.internal:8555/v1}"
LLM_MODEL="${LLM_MODEL:-openai/gpt-oss-120b}"
LLM_API_KEY="${LLM_API_KEY:-not-needed}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

echo "🔍 Pre-flight checks..."

if ! command -v docker &>/dev/null; then
    echo "❌ Docker is not installed."; exit 1
fi
if ! docker info &>/dev/null 2>&1; then
    echo "❌ Docker daemon is not running.  Try: sudo systemctl start docker"; exit 1
fi
if ! command -v openhands &>/dev/null; then
    echo "❌ openhands not found.  Install with: uv tool install openhands --python 3.12"; exit 1
fi

# Switch away from Docker Desktop if needed
DOCKER_CTX=$(docker context show 2>/dev/null || echo "unknown")
if [[ "${DOCKER_CTX}" == "desktop-linux" ]]; then
    echo "⚠️  Switching to native Docker Engine..."
    docker context use default
fi
echo "  ✅ Docker OK (context: $(docker context show 2>/dev/null))"

# Check LLM reachability
HOST_URL="$(echo "${LLM_BASE_URL}" | sed 's|host.docker.internal|localhost|')"
if curl --silent --max-time 3 "${HOST_URL}/models" >/dev/null 2>&1; then
    echo "  ✅ LLM server reachable at ${HOST_URL}"
else
    echo "  ⚠️  LLM server not reachable at ${HOST_URL}"
fi

# ---------------------------------------------------------------------------
# Expose localhost LLM to Docker containers via socat (no sudo required)
# ---------------------------------------------------------------------------
# Containers resolve host.docker.internal → docker bridge IP (172.17.0.1).
# If the LLM only listens on 127.0.0.1, containers can't reach it.
# socat forwards 172.17.0.1:PORT → 127.0.0.1:PORT in userspace.

LLM_PORT=$(echo "${LLM_BASE_URL}" | grep -oP ':\K[0-9]+(?=/)')
DOCKER_BRIDGE_IP=$(ip -4 addr show docker0 2>/dev/null | grep -oP 'inet \K[\d.]+')

if [[ -n "${LLM_PORT}" && -n "${DOCKER_BRIDGE_IP}" ]]; then
    # Check if something already listens on the bridge IP (socat or LLM itself)
    if ss -tlnp 2>/dev/null | grep -q "${DOCKER_BRIDGE_IP}:${LLM_PORT}"; then
        echo "  ✅ LLM already reachable on ${DOCKER_BRIDGE_IP}:${LLM_PORT}"
    elif command -v socat &>/dev/null; then
        echo "  🔧 Starting socat forwarder: ${DOCKER_BRIDGE_IP}:${LLM_PORT} → 127.0.0.1:${LLM_PORT}"
        socat "TCP-LISTEN:${LLM_PORT},bind=${DOCKER_BRIDGE_IP},reuseaddr,fork" \
              "TCP:127.0.0.1:${LLM_PORT}" &
        SOCAT_PID=$!
        # Give it a moment to bind
        sleep 0.3
        if kill -0 "${SOCAT_PID}" 2>/dev/null; then
            echo "  ✅ socat forwarder running (PID ${SOCAT_PID})"
            # Clean up socat when the script exits
            trap "kill ${SOCAT_PID} 2>/dev/null" EXIT
        else
            echo "  ⚠️  socat failed to start — containers may not reach LLM"
        fi
    else
        echo "  ⚠️  socat not found. Install it (apt install socat) or bind your LLM to 0.0.0.0"
    fi
fi


echo ""

# ---------------------------------------------------------------------------
# Write ~/.openhands/settings.json  (skips the setup wizard)
# ---------------------------------------------------------------------------

SETTINGS_DIR="$HOME/.openhands"
mkdir -p "${SETTINGS_DIR}"

cat > "${SETTINGS_DIR}/settings.json" <<EOF
{
  "language": "en",
  "agent": "CodeActAgent",
  "llm_model": "${LLM_MODEL}",
  "llm_api_key": "${LLM_API_KEY}",
  "llm_base_url": "${LLM_BASE_URL}",
  "enable_default_condenser": true,
  "v1_enabled": false
}
EOF
echo "  ✅ ${SETTINGS_DIR}/settings.json"

# ---------------------------------------------------------------------------
# Write ~/.openhands/config.toml  (disable native tool calling for local LLM)
# ---------------------------------------------------------------------------

cat > "${SETTINGS_DIR}/config.toml" <<'TOML'
[llm]
native_tool_calling = false
TOML
echo "  ✅ ${SETTINGS_DIR}/config.toml"
echo ""

# ---------------------------------------------------------------------------
# Show the startup prompt
# ---------------------------------------------------------------------------

PROMPT_FILE="${PROJECT_DIR}/scripts/startup-prompt.txt"
if [[ -f "${PROMPT_FILE}" ]]; then
    echo "📋 To start a conversation:"
    echo "   1. Click  \"+ New Conversation\""
    echo "   2. Choose \"Start from scratch\""
    echo "   3. Paste the prompt below into the chat"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    cat "${PROMPT_FILE}"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    # Copy to clipboard if possible
    if command -v xclip &>/dev/null; then
        xclip -selection clipboard < "${PROMPT_FILE}"
        echo "📎 Copied to clipboard!"
    elif command -v xsel &>/dev/null; then
        xsel --clipboard < "${PROMPT_FILE}"
        echo "📎 Copied to clipboard!"
    fi
    echo ""
fi

# ---------------------------------------------------------------------------
# Launch:  cd into the project and use --mount-cwd
# ---------------------------------------------------------------------------

# Pre-create directories that OpenHands writes inside /workspace.
# The sandbox user UID may differ from the host user, so make them world-writable.
# Both are already in .gitignore.
mkdir -p "${PROJECT_DIR}/conversations" "${PROJECT_DIR}/bash_events"
chmod 777 "${PROJECT_DIR}/conversations" "${PROJECT_DIR}/bash_events"

echo "🚀 Launching OpenHands..."
echo "   UI:        http://localhost:3000"
echo "   LLM:       ${LLM_MODEL} @ ${LLM_BASE_URL}"
echo "   Workspace: ${PROJECT_DIR}  →  /workspace (inside sandbox)"
echo ""

cd "${PROJECT_DIR}"
exec openhands serve --mount-cwd
