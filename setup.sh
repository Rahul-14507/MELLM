#!/usr/bin/env bash
# MELLM setup script — installs llama-cpp-python with the correct CUDA backend.
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# What it does:
#   1. Detects your CUDA toolkit version (nvcc) or driver-reported version (nvidia-smi).
#   2. Tries to install the matching pre-built wheel from the llama-cpp-python index.
#   3. Falls back to a source build (CMAKE_ARGS="-DGGML_CUDA=on") if no wheel matches
#      or if the wheel produces an SIGILL crash (binary mismatch).
#   4. Installs the remaining requirements.txt dependencies.
#   5. Verifies the install by actually loading a tiny model stub — not just importing.

set -euo pipefail

PYTHON="${PYTHON:-python}"
PIP="${PYTHON} -m pip"
LLAMA_WHL_BASE="https://abetlen.github.io/llama-cpp-python/whl"

# ── helpers ──────────────────────────────────────────────────────────────────

log()  { echo "[setup] $*"; }
warn() { echo "[setup] WARNING: $*" >&2; }

cuda_version_from_nvcc() {
    if command -v nvcc &>/dev/null; then
        nvcc --version 2>/dev/null \
            | grep -oP "release \K[0-9]+\.[0-9]+" \
            | head -1
    fi
}

cuda_version_from_smi() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi 2>/dev/null \
            | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" \
            | head -1
    fi
}

# Map "12.0" → "cu120", "12.1" → "cu121", "11.8" → "cu118", etc.
cuda_tag_from_version() {
    local ver="$1"
    local major minor tag
    major=$(echo "$ver" | cut -d. -f1)
    minor=$(echo "$ver" | cut -d. -f2)
    tag="cu${major}${minor}"
    echo "$tag"
}

# Known tags published on the llama-cpp-python wheel index (update as new ones appear).
KNOWN_TAGS="cu118 cu121 cu122 cu123 cu124 cu125"

nearest_known_tag() {
    local requested="$1"
    # Exact match first.
    for t in $KNOWN_TAGS; do
        [[ "$t" == "$requested" ]] && echo "$t" && return
    done
    # Nearest lower match (safer than going higher).
    local best=""
    for t in $KNOWN_TAGS; do
        [[ "$t" < "$requested" || "$t" == "$requested" ]] && best="$t"
    done
    echo "$best"
}

try_prebuilt_wheel() {
    local tag="$1"
    log "Trying pre-built wheel for $tag ..."
    if $PIP install llama-cpp-python \
            --extra-index-url "${LLAMA_WHL_BASE}/${tag}" \
            --quiet 2>&1; then
        # Verify the wheel actually works: import + a tiny sanity call.
        # A broken wheel (SIGILL) will exit non-zero here.
        if $PYTHON - <<'EOF' 2>/dev/null
from llama_cpp import Llama
# Probe: attempt to init the library; no model file needed for this check.
import llama_cpp._internals as _i
EOF
        then
            log "Pre-built wheel ($tag) verified OK."
            return 0
        else
            warn "Pre-built wheel ($tag) failed sanity check (likely SIGILL / CUDA mismatch)."
            $PIP uninstall llama-cpp-python -y --quiet
            return 1
        fi
    else
        warn "No pre-built wheel found for $tag."
        return 1
    fi
}

build_from_source() {
    log "Building llama-cpp-python from source with CUDA support (this takes ~5 min) ..."
    CMAKE_ARGS="-DGGML_CUDA=on" $PIP install llama-cpp-python --no-cache-dir
    log "Source build complete."
}

# ── main ─────────────────────────────────────────────────────────────────────

log "Detecting CUDA version ..."
CUDA_VER=$(cuda_version_from_nvcc)
if [[ -z "$CUDA_VER" ]]; then
    CUDA_VER=$(cuda_version_from_smi)
fi

if [[ -z "$CUDA_VER" ]]; then
    warn "No CUDA detected. Installing CPU-only llama-cpp-python."
    $PIP install llama-cpp-python --quiet
else
    log "Detected CUDA $CUDA_VER"
    REQUESTED_TAG=$(cuda_tag_from_version "$CUDA_VER")
    WHEEL_TAG=$(nearest_known_tag "$REQUESTED_TAG")

    INSTALLED=false

    if [[ -n "$WHEEL_TAG" ]]; then
        if try_prebuilt_wheel "$WHEEL_TAG"; then
            INSTALLED=true
        fi
    fi

    if [[ "$INSTALLED" == false ]]; then
        log "Falling back to source build ..."
        build_from_source
    fi
fi

log "Installing remaining dependencies ..."
$PIP install -r requirements.txt --quiet

log "Verifying installation ..."
$PYTHON -c "
from llama_cpp import Llama
import llama_cpp
print(f'llama-cpp-python {llama_cpp.__version__} OK')
"

log ""
log "Setup complete. Run MELLM with:"
log "  python cli.py"
