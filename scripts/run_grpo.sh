#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B-Instruct-2507}"
CONFIG="${CONFIG:-configs/grpo.yaml}"
PORT="${PORT:-8000}"
SMOKE="${SMOKE:-0}"
MAX_STEPS="${MAX_STEPS:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
SFT_ADAPTER="${SFT_ADAPTER:-}"
DATA_PATH="${DATA_PATH:-}"
RESUME="${RESUME:-}"
TRAIN_GPU="${TRAIN_GPU:-0}"
ROLLOUT_GPU="${ROLLOUT_GPU:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.72}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
export PORT

# The vLLM server and the trainer communicate over loopback.  Some desktop or
# proxy setups proxy 127.0.0.1 even when localhost is already excluded, which
# makes both the readiness check and GRPOTrainer wait for a server that is
# actually healthy.  Keep both spellings because different HTTP clients consult
# different variants of the variable.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

SERVER_ARGS=(
  --model "$MODEL_ID"
  --port "$PORT"
  --dtype bfloat16
  --max_model_len 2048
  --gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
)
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  SERVER_ARGS+=(--enforce_eager)
fi

# vLLM creates EngineCore/resource-tracker child processes. Put the entire
# server tree in its own session/process group so no descendant can keep the
# caller's stdout pipe open after training has finished.
CUDA_VISIBLE_DEVICES="$ROLLOUT_GPU" setsid trl vllm-serve "${SERVER_ARGS[@]}" &
SERVER_PID=$!
SERVER_PGID=$SERVER_PID

cleanup_server() {
  local exit_code=$?
  trap - EXIT INT TERM
  echo "Stopping vLLM process group ${SERVER_PGID}..."
  kill -TERM -- "-${SERVER_PGID}" 2>/dev/null || true

  # Uvicorn gives its worker up to 10 seconds to exit. Allow a little more,
  # then kill any orphan/resource-tracker still holding stdout or GPU memory.
  local attempt
  for ((attempt = 0; attempt < 30; attempt++)); do
    if ! pgrep -g "$SERVER_PGID" >/dev/null 2>&1; then
      break
    fi
    sleep 0.5
  done
  if pgrep -g "$SERVER_PGID" >/dev/null 2>&1; then
    echo "Force-stopping remaining vLLM child processes..."
    kill -KILL -- "-${SERVER_PGID}" 2>/dev/null || true
  fi
  wait "$SERVER_PID" 2>/dev/null || true
  echo "vLLM process group stopped."
  return "$exit_code"
}

trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

python - <<'PY'
import os
import time
import urllib.request

# TRL registers the route with a trailing slash.  Disable proxies explicitly in
# addition to NO_PROXY so this check is independent of the user's proxy setup.
url = f"http://127.0.0.1:{os.environ.get('PORT', '8000')}/health/"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for _ in range(120):
    try:
        with opener.open(url, timeout=2) as response:
            if response.status == 200:
                print(f"vLLM server is healthy: {url}", flush=True)
                break
    except Exception:
        time.sleep(2)
else:
    raise SystemExit("vLLM server did not become healthy within 4 minutes")
PY

EXTRA_ARGS=()
if [[ "$SMOKE" == "1" ]]; then
  EXTRA_ARGS+=(--smoke)
fi
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA_ARGS+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  EXTRA_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$SFT_ADAPTER" ]]; then
  EXTRA_ARGS+=(--sft-adapter "$SFT_ADAPTER")
fi
if [[ -n "$DATA_PATH" ]]; then
  EXTRA_ARGS+=(--data "$DATA_PATH")
fi
if [[ -n "$RESUME" ]]; then
  EXTRA_ARGS+=(--resume "$RESUME")
fi
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" python scripts/train_grpo.py --config "$CONFIG" "${EXTRA_ARGS[@]}"

