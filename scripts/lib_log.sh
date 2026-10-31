#!/usr/bin/env bash
# Shared logging helpers for scripts/*.sh.
# Source this file, then call log/section/die. Pass a LOG_FILE path to also tee output.

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

section() {
  printf '\n==== %s ====\n' "$*"
}

die() {
  log "ERROR: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command '$1' not found on PATH"
}

require_file() {
  [ -f "$1" ] || die "required file not found: $1 ($2)"
}

human_bytes() {
  numfmt --to=iec-i --suffix=B "$1" 2>/dev/null || echo "${1}B"
}
