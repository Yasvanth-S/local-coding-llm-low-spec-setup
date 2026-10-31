#!/usr/bin/env bash
#
# build_llama_cpp.sh — clone and build llama.cpp with CUDA support.
#
# Why -j2 by default: this setup targets low-RAM machines (as little as ~6.6GB
# total). Letting cmake use every CPU thread means that many gcc/nvcc processes
# run at once, each holding its own working set in RAM — on a box this tight that
# reliably tips into heavy swapping or an OOM kill partway through the CUDA build.
# Override with BUILD_JOBS=N if your machine has more headroom.
#
# Requires: git, cmake, make, gcc/g++, nvcc (CUDA toolkit).
# Usage: ./build_llama_cpp.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=./lib_log.sh
source "$SCRIPT_DIR/lib_log.sh"

LLAMA_DIR="$ROOT_DIR/llama.cpp"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/build.log"
BUILD_JOBS="${BUILD_JOBS:-2}"

mkdir -p "$LOG_DIR"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

require_cmd git
require_cmd cmake
require_cmd nvcc

section "Preflight"
log "Build jobs: $BUILD_JOBS (override with BUILD_JOBS=N)"
log "RAM:  $(free -h | awk '/^Mem:/ {print $7" available of "$2}')"
log "Disk: $(df -h "$ROOT_DIR" | awk 'NR==2 {print $4" free on "$6}')"
log "CUDA: $(nvcc --version | grep release)"

section "Fetching llama.cpp source"
if [ ! -d "$LLAMA_DIR" ]; then
  log "Cloning ggml-org/llama.cpp into $LLAMA_DIR"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
else
  log "Repo already present at $LLAMA_DIR — pulling latest"
  git -C "$LLAMA_DIR" pull --ff-only
fi
log "Source at commit: $(git -C "$LLAMA_DIR" rev-parse --short HEAD)"

section "Configuring (cmake, GGML_CUDA=ON)"
START_CFG=$(date +%s)
cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
log "Configure finished in $(( $(date +%s) - START_CFG ))s"

section "Building (llama-server, llama-cli) with $BUILD_JOBS job(s)"
log "This step compiles CUDA kernels and is the slow part — can take 15-45+ minutes"
log "on a 6-core laptop CPU. Progress is streamed below and saved to $LOG_FILE."
START_BUILD=$(date +%s)
cmake --build "$LLAMA_DIR/build" --config Release -j"$BUILD_JOBS" --target llama-server llama-cli
log "Build finished in $(( $(date +%s) - START_BUILD ))s"

section "Done"
BIN_DIR="$LLAMA_DIR/build/bin"
for bin in llama-server llama-cli; do
  if [ -x "$BIN_DIR/$bin" ]; then
    log "OK: $BIN_DIR/$bin ($(human_bytes "$(stat -c%s "$BIN_DIR/$bin")"))"
  else
    die "expected binary missing: $BIN_DIR/$bin"
  fi
done
log "Full build log saved to: $LOG_FILE"
