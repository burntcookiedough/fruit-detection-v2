#!/usr/bin/env bash
# ============================================================
# Fruit Detector v2 — Runner script (Linux / macOS)
#
# Description:
#   Loads .env, dispatches on FRUIT_RUN_MODE or CLI args,
#   auto-detects installed package vs source fallback.
#
# Usage:
#   ./scripts/run.sh train --epochs 10
#   ./scripts/run.sh infer --image photo.jpg
#   ./scripts/run.sh                    # dispatches on FRUIT_RUN_MODE
#   ./scripts/run.sh --help
#
# Environment:
#   FRUIT_RUN_MODE  — auto-dispatch mode (train|image|infer|webcam|verify|export|analyze)
#   PYTHON_BIN      — Python interpreter (default: python3)
#
# Exit codes:
#   0  — success
#   1  — general error
#   2  — Python not found
#   3  — unknown FRUIT_RUN_MODE
# ============================================================
set -Eeuo pipefail

# ── Constants ────────────────────────────────────────────────
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
readonly VALID_MODES="train image infer webcam verify export analyze"

# ── Logging ──────────────────────────────────────────────────
_log_info()  { printf '\033[0;36m[run.sh]\033[0m %s\n' "$*"; }
_log_warn()  { printf '\033[0;33m[run.sh]\033[0m %s\n' "$*" >&2; }
_log_error() { printf '\033[0;31m[run.sh]\033[0m %s\n' "$*" >&2; }

# ── Error trap ───────────────────────────────────────────────
_on_error() {
    local exit_code=$?
    _log_error "Command failed at line $1 (exit code $exit_code)"
    exit "$exit_code"
}
trap '_on_error $LINENO' ERR

# ── Load .env ────────────────────────────────────────────────
_load_dotenv() {
    local env_file="$ROOT_DIR/.env"
    if [[ -f "$env_file" ]]; then
        set -a
        # shellcheck disable=SC1090
        . "$env_file"
        set +a
    fi
}

# ── Validate Python ──────────────────────────────────────────
_require_python() {
    local python_bin="${PYTHON_BIN:-python3}"

    if ! command -v "$python_bin" &>/dev/null; then
        _log_error "Python not found. Set PYTHON_BIN or install Python 3.10+."
        exit 2
    fi

    printf '%s' "$python_bin"
}

# ── Resolve mode → CLI args ──────────────────────────────────
_resolve_mode_to_args() {
    local mode="$1"

    case "$mode" in
        train)         printf 'train'   ;;
        image|infer)   printf 'infer'   ;;
        webcam)        printf 'webcam'  ;;
        verify)        printf 'verify'  ;;
        export)        printf 'export'  ;;
        analyze)       printf 'analyze' ;;
        *)
            _log_error "Unknown FRUIT_RUN_MODE: $mode"
            _log_error "Valid modes: $VALID_MODES"
            exit 3
            ;;
    esac
}

# ── Execute via installed package or source fallback ─────────
_exec_fruit_detect() {
    local python_bin="$1"
    shift

    # Prefer installed entry point
    if "$python_bin" -c "import fruit_detector" 2>/dev/null; then
        exec fruit-detect "$@"
    fi

    # Fallback: run from source tree
    _log_warn "Package not installed — running from source"
    export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
    exec "$python_bin" -m fruit_detector.cli "$@"
}

# ── Main ─────────────────────────────────────────────────────
main() {
    cd -- "$ROOT_DIR"
    _load_dotenv

    local python_bin
    python_bin="$(_require_python)"

    # Auto-dispatch on FRUIT_RUN_MODE when no args given
    if [[ $# -eq 0 ]]; then
        local mode="${FRUIT_RUN_MODE:-}"
        if [[ -n "$mode" ]]; then
            local resolved
            resolved="$(_resolve_mode_to_args "$mode")"
            _log_info "FRUIT_RUN_MODE=$mode → fruit-detect $resolved"
            set -- "$resolved"
        else
            set -- "--help"
        fi
    fi

    _exec_fruit_detect "$python_bin" "$@"
}

main "$@"
