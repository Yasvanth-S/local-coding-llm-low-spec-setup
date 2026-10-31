#!/usr/bin/env bash
#
# download_models.sh — fetch the GGUF model files this setup uses into models/
# (gitignored — weights are multi-GB and don't belong in git).
#
# Main coding profile (always fetched): Qwen3.5-4B, Q4_K_M quant, text-only.
# Optional vision/PDF profile (WITH_VISION=1): Qwen3.5-2B Q4_K_M + its mmproj.
#
# Uses the `hf` CLI (from huggingface_hub[cli]) for resumable, progress-reported
# downloads; falls back to curl if `hf` isn't on PATH.
#
# Usage:
#   ./download_models.sh                 # main coding model only
#   WITH_VISION=1 ./download_models.sh    # + vision/PDF profile model
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=./lib_log.sh
source "$SCRIPT_DIR/lib_log.sh"

MODELS_DIR="$ROOT_DIR/models"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/download.log"
VENV_HF="$ROOT_DIR/.venv/bin/hf"

mkdir -p "$MODELS_DIR" "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

section "Preflight"
log "Disk free: $(df -h "$MODELS_DIR" | awk 'NR==2 {print $4}') at $MODELS_DIR"

fetch() {
  local repo="$1" file="$2"
  local dest="$MODELS_DIR/$file"

  if [ -f "$dest" ]; then
    log "Already have $file ($(human_bytes "$(stat -c%s "$dest")")) — skipping"
    return
  fi

  log "Downloading $repo/$file ..."
  local start
  start=$(date +%s)
  if [ -x "$VENV_HF" ]; then
    "$VENV_HF" download "$repo" "$file" --local-dir "$MODELS_DIR"
  else
    require_cmd curl
    curl -L --fail --progress-bar -o "$dest" \
      "https://huggingface.co/${repo}/resolve/main/${file}"
  fi
  log "Finished $file in $(( $(date +%s) - start ))s, size $(human_bytes "$(stat -c%s "$dest")")"
}

section "Main coding profile: Qwen3.5-4B (text-only)"
fetch "unsloth/Qwen3.5-4B-GGUF" "Qwen3.5-4B-Q4_K_M.gguf"

if [ "${WITH_VISION:-0}" = "1" ]; then
  section "Optional vision/PDF profile: Qwen3.5-2B + mmproj"
  fetch "unsloth/Qwen3.5-2B-GGUF" "Qwen3.5-2B-Q4_K_M.gguf"
  fetch "unsloth/Qwen3.5-2B-GGUF" "mmproj-F16.gguf"
else
  log "WITH_VISION not set — skipping optional vision/PDF model (run with WITH_VISION=1 to fetch it)"
fi

section "Done"
log "Models in $MODELS_DIR:"
ls -lh "$MODELS_DIR" | tail -n +2 | awk '{print "  "$9" ("$5")"}'
