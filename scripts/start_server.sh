#!/usr/bin/env bash
#
# start_server.sh — launch llama-server for the main coding profile
# (Qwen3.5-4B, Q4_K_M, text-only), tuned for a 4GB-VRAM / low-RAM laptop.
#
# Flags explained:
#   -ngl 999              offload every layer the model has to GPU (clamped to
#                          the model's real layer count — Qwen3.5-4B Q4_K_M at
#                          ~2.74GB fits entirely in a 4GB card)
#   --cache-type-k/v q4_0  quantize the KV cache so more context fits in the
#                          VRAM left over after the model weights
#   --ctx-size             context window; pass as $1, default 49152. On the
#                          reference hardware (GTX 1650, 4GB VRAM) this model
#                          was tested under real generation up to 57344 (3.6GB
#                          used) before 65536 reliably OOM-crashed mid-inference
#                          (not just at load). 49152 leaves a safer ~600MB
#                          headroom margin than 57344's ~480MB. Your card may
#                          differ — push the number up and re-run to re-check.
#   --parallel 1           llama-server defaults to splitting --ctx-size across
#                          4 parallel request "slots" (so --ctx-size 8192 would
#                          actually give any one conversation only 2048 tokens).
#                          We only ever have one aider session talking to this
#                          server, so pin to 1 slot and give it the full window.
#
# Usage: ./start_server.sh [ctx-size]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=./lib_log.sh
source "$SCRIPT_DIR/lib_log.sh"

LLAMA_SERVER="$ROOT_DIR/llama.cpp/build/bin/llama-server"
MODEL="$ROOT_DIR/models/Qwen3.5-4B-Q4_K_M.gguf"
CTX_SIZE="${1:-49152}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/server-ctx${CTX_SIZE}.log"

mkdir -p "$LOG_DIR"

require_file "$LLAMA_SERVER" "run scripts/build_llama_cpp.sh first"
require_file "$MODEL" "run scripts/download_models.sh first"

section "Starting llama-server"
log "Model:    $MODEL ($(human_bytes "$(stat -c%s "$MODEL")"))"
log "Context:  $CTX_SIZE tokens (KV cache quantized to q4_0)"
log "Endpoint: http://127.0.0.1:8080/v1 (OpenAI-compatible)"
log "Logging to: $LOG_FILE"
log "Watch 'nvidia-smi' and 'free -h' in another terminal to confirm this fits."

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --ctx-size "$CTX_SIZE" \
  --parallel 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8080 \
  --jinja \
  2>&1 | tee "$LOG_FILE"
