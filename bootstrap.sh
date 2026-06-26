#!/usr/bin/env bash
#
# bootstrap.sh — mixed-GPU-box setup for the potvrdenie pipeline.
#
# Sets up TWO independent GPU lanes that never touch each other:
#   - CUDA lane  (NVIDIA: laptop GPU now / GTX 1050 Ti on the rack later) -> vision/OCR
#   - Vulkan lane (RADV: the 5x RX 580 on the rack)                       -> structuring LLM
#
# Runs on both your Fedora laptop (dnf) and the Ubuntu 24.04 rack (apt).
# It is deliberately FORGIVING about system packages: if a package name is
# wrong for your distro it WARNS and continues, then tells you to let Claude
# Code fix that one line on your actual machine. Don't fight package names by
# hand — that's exactly the 10% the agent is good at.
#
# Usage:  bash bootstrap.sh
#
set -uo pipefail

PROJ="${HOME}/potvrdenie"
VENV="${PROJ}/.venv"
LLAMA_DIR="${HOME}/llama.cpp"
MODEL_DIR="${HOME}/models"

# >>> set this to the exact GGUF URL of the model you serve (Gemma-3-4B QAT).
# Leave empty to skip the download and copy the model over from the rack yourself.
MODEL_URL=""

say()  { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$*"; }
ok()   { printf '\033[1;32m[ok] %s\033[0m\n' "$*"; }

# --- detect package manager -------------------------------------------------
if   command -v apt-get >/dev/null 2>&1; then PKG=apt
elif command -v dnf     >/dev/null 2>&1; then PKG=dnf
else warn "No apt or dnf found — install system deps manually, then re-run."; PKG=none
fi
say "Package manager: ${PKG}"

# best-effort install: never aborts the whole script on one bad name
pkg_install() {
  for p in "$@"; do
    if [ "$PKG" = apt ]; then sudo apt-get install -y "$p" 2>/dev/null && ok "apt: $p" || warn "apt: could not install '$p' (ask Claude Code for the right name)"
    elif [ "$PKG" = dnf ]; then sudo dnf install -y "$p" 2>/dev/null && ok "dnf: $p" || warn "dnf: could not install '$p' (ask Claude Code for the right name)"
    fi
  done
}

# --- 1. system dependencies -------------------------------------------------
say "Installing build tools + Vulkan toolchain"
[ "$PKG" = apt ] && sudo apt-get update -y
pkg_install git cmake curl python3 python3-pip
[ "$PKG" = apt ] && pkg_install build-essential python3-venv libvulkan-dev vulkan-tools glslc glslang-tools
[ "$PKG" = dnf ] && pkg_install gcc gcc-c++ make python3-virtualenv vulkan-loader-devel vulkan-headers vulkan-tools glslc glslang

# --- 2. python project + CUDA vision stack ---------------------------------
say "Creating Python venv and installing the CUDA vision stack"
mkdir -p "$PROJ"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip

# torch: the default Linux wheel is CUDA-enabled and bundles its own CUDA
# runtime, so you do NOT need the system CUDA toolkit — only the NVIDIA driver.
# If you need a specific CUDA version for the 1050 Ti, swap to the pinned index:
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install torch torchvision

# OCR + glue. paddlepaddle (CPU) is installed so the script ALWAYS completes;
# for GPU OCR on the NVIDIA card, swap to a paddlepaddle-gpu build matched to
# your CUDA — let Claude Code pick the exact version for your driver.
pip install paddleocr paddlepaddle
pip install requests fastapi "uvicorn[standard]" pillow numpy opencv-python-headless

# Pre-fetch the two OCR recognition models so first run isn't a silent network
# dependency: PP-OCRv6 (digits) + latin_PP-OCRv5 (Slovak free text w/ diacritics).
# On the offline rack these land in ~/.paddlex/official_models and must be staged
# like the Gemma .gguf — a missing model is otherwise a silent failure at run time.
say "Pre-fetching OCR recognition models"
python - <<'PY' && ok "OCR models cached" || warn "OCR model prefetch failed — needs network; stage ~/.paddlex/official_models on the rack."
from paddlex import create_model
for m in ("PP-OCRv6_medium_rec", "latin_PP-OCRv5_mobile_rec"):
    create_model(m)
PY

ok "Python env ready at ${VENV}"

# --- 3. Vulkan llama.cpp build ---------------------------------------------
say "Building llama.cpp with the Vulkan backend"
if [ ! -d "$LLAMA_DIR" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cmake -S "$LLAMA_DIR" -B "${LLAMA_DIR}/build" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
  && cmake --build "${LLAMA_DIR}/build" --config Release -j"$(nproc)" \
  && ok "llama-server built at ${LLAMA_DIR}/build/bin/llama-server" \
  || warn "llama.cpp Vulkan build failed — usual cause is a missing shader compiler (glslc). Ask Claude Code to resolve it for ${PKG}."

# --- 4. model (optional) ----------------------------------------------------
mkdir -p "$MODEL_DIR"
if [ -n "$MODEL_URL" ]; then
  say "Downloading model"
  curl -L "$MODEL_URL" -o "${MODEL_DIR}/$(basename "$MODEL_URL")" && ok "model in ${MODEL_DIR}"
else
  warn "MODEL_URL empty — copy your Gemma-3-4B QAT .gguf into ${MODEL_DIR} (or scp it from the rack)."
fi

# --- 4b. PHASE-6 LLM tier (text second-check) ------------------------------
# The ONLY LLM in the pipeline (structuring is deterministic). It cleans/suggests
# for escalated TEXT fields only and needs just a TINY model — CPU is fine for the
# dev box. Two ways to get a llama-server here:
#   FAST  : a prebuilt release binary (no compiler needed)  -> set PHASE6_PREBUILT=1
#   FULL  : the Vulkan build in section 3 (for the rack's RX 580s)
# Default model is Gemma-3-4B-it (≈2.5GB Q4) — markedly better Slovak suggestions
# than a 1–2B model, still small enough for CPU. For a lighter footprint swap in a
# tiny model, e.g. Qwen2.5-1.5B-Instruct-Q4_K_M (≈1GB). serve_llm.sh auto-serves
# the LARGEST .gguf in ~/models, so keep only the one you want as default.
PHASE6_PREBUILT="${PHASE6_PREBUILT:-0}"
PHASE6_MODEL_URL="${PHASE6_MODEL_URL:-https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf}"

if [ "$PHASE6_PREBUILT" = "1" ]; then
  say "Phase-6: fetching a prebuilt llama.cpp binary (no compile)"
  PB_DIR="${HOME}/llama_prebuilt"; mkdir -p "$PB_DIR"
  if ! ls "$PB_DIR"/*/llama-server >/dev/null 2>&1; then
    PB_URL="$(curl -s https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
      | grep -oE '"browser_download_url": *"[^"]*bin-ubuntu-x64[^"]*"' | grep -oE 'https[^"]*' | head -1)"
    if [ -n "$PB_URL" ]; then
      curl -L "$PB_URL" -o "${PB_DIR}/llama.tgz" && tar xzf "${PB_DIR}/llama.tgz" -C "$PB_DIR" \
        && ok "prebuilt llama-server in ${PB_DIR}" \
        || warn "prebuilt fetch/extract failed — fall back to the Vulkan build (section 3)."
    else
      warn "could not resolve a prebuilt asset URL — use the Vulkan build instead."
    fi
  else ok "prebuilt llama-server already present"; fi
fi

if [ -n "$PHASE6_MODEL_URL" ]; then
  say "Phase-6: downloading the tiny text-cleanup model"
  P6_OUT="${MODEL_DIR}/$(basename "$PHASE6_MODEL_URL")"
  if [ ! -f "$P6_OUT" ]; then
    curl -L "$PHASE6_MODEL_URL" -o "$P6_OUT" && ok "phase-6 model: $P6_OUT" \
      || warn "phase-6 model download failed (needs network)."
  else ok "phase-6 model already present: $P6_OUT"; fi
  echo ">>> Start the Phase-6 server (CPU):   bash ${PROJ}/serve_llm.sh"
  echo ">>> ...or GPU-offloaded:              NGL=99 bash ${PROJ}/serve_llm.sh"
  echo ">>> Smoke-test once it's up:          python ${PROJ}/test_phase6.py"
fi

# --- 5. DEVICE MAP — the whole point on a mixed box ------------------------
# Confirms the lanes are separate BEFORE you trust any hardcoded GPU indices.
say "DEVICE MAP"

echo "--- CUDA lane (NVIDIA, for vision/OCR) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L || warn "nvidia-smi present but no GPU listed"
else
  warn "nvidia-smi not found — install the NVIDIA driver (needed for the 1050 Ti / laptop GPU)."
fi

echo
echo "--- Vulkan lane: ALL devices the loader sees (no pin) ---"
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName" || warn "vulkaninfo ran but printed no devices"
else
  warn "vulkaninfo not found (vulkan-tools)."
fi

echo
echo "--- Vulkan lane: RADV-ONLY view (the rack's real launch env) ---"
RADV_ICD="/usr/share/vulkan/icd.d/radeon_icd.x86_64.json"
[ -f "$RADV_ICD" ] || RADV_ICD="/usr/share/vulkan/icd.d/radeon_icd.json"
if [ -f "$RADV_ICD" ]; then
  VK_DRIVER_FILES="$RADV_ICD" vulkaninfo --summary 2>/dev/null | grep -E "deviceName" \
    || warn "Pinned to RADV but saw no AMD devices (expected on the laptop — no RX 580s here)."
  echo
  echo ">>> On the rack, launch llama-server with this env set so the NVIDIA"
  echo ">>> card vanishes from the Vulkan list and your Vulkan0..4 mean AMD only:"
  echo ">>>   VK_DRIVER_FILES=${RADV_ICD} RADV_PERFTEST=nogttspill \\"
  echo ">>>   ${LLAMA_DIR}/build/bin/llama-server -m ${MODEL_DIR}/<model>.gguf \\"
  echo ">>>     --device Vulkan0 -ngl 99 -c 4096 --port 8080"
else
  warn "No RADV ICD here (normal on the laptop). It'll exist on the rack once mesa/amdgpu is installed."
fi

say "Done. Re-read the DEVICE MAP above and confirm three things on the rack:"
echo "  1. nvidia-smi lists the 1050 Ti  (CUDA lane up)"
echo "  2. the no-pin Vulkan list shows NVIDIA + the AMD cards"
echo "  3. the RADV-only list shows ONLY the AMD cards, in a stable order"
