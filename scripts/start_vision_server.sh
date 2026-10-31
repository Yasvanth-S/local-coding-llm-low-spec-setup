#!/usr/bin/env bash
#
# start_vision_server.sh — launch llama-server for the optional vision/PDF
# profile (Qwen3.5-2B + mmproj-F16), kept separate from the main coding
# profile so the smaller model/vision tower leaves far more VRAM for KV
# cache (see README for the memory-budget table and why).
#
# llama.cpp's multimodal support takes images, not raw PDFs — render PDF
# pages to images first (e.g. `pdftoppm`) before sending them in.
#
# Usage: ./start_vision_server.sh [ctx-size]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=./lib_log.sh
source "$SCRIPT_DIR/lib_log.sh"

LLAMA_SERVER="$ROOT_DIR/llama.cpp/build/bin/llama-server"
MODEL="$ROOT_DIR/models/Qwen3.5-2B-Q4_K_M.gguf"
MMPROJ="$ROOT_DIR/models/mmproj-F16.gguf"
CTX_SIZE="${1:-16384}"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/vision-server-ctx${CTX_SIZE}.log"

mkdir -p "$LOG_DIR"

require_file "$LLAMA_SERVER" "run scripts/build_llama_cpp.sh first"
require_file "$MODEL" "run WITH_VISION=1 scripts/download_models.sh first"
require_file "$MMPROJ" "run WITH_VISION=1 scripts/download_models.sh first"

section "Starting vision llama-server"
log "Model:    $MODEL ($(human_bytes "$(stat -c%s "$MODEL")"))"
log "mmproj:   $MMPROJ ($(human_bytes "$(stat -c%s "$MMPROJ")"))"
log "Context:  $CTX_SIZE tokens (KV cache quantized to q4_0)"
log "Endpoint: http://127.0.0.1:8081/v1 (OpenAI-compatible, separate port from"
log "          the main coding server so both can run at once if VRAM allows)"
log "Logging to: $LOG_FILE"

exec "$LLAMA_SERVER" \
  --model "$MODEL" \
  --mmproj "$MMPROJ" \
  --ctx-size "$CTX_SIZE" \
  --parallel 1 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -ngl 999 \
  --host 127.0.0.1 \
  --port 8081 \
  2>&1 | tee "$LOG_FILE"
