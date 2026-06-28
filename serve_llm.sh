#!/usr/bin/env bash
#
# serve_llm.sh — launch the LOCAL llama-server for the Phase-6 text second-check tier.
#
# Phase 6 only needs a TINY model doing short Slovak text cleanup (name/ulica/
# obchodné meno), so CPU is fine for the dev box. text_second_check.py POSTs to
# http://localhost:8080/v1/chat/completions with a GBNF grammar (the chat endpoint
# applies the model's own chat template); this serves that endpoint.
#
# Env overrides:
#   LLM_MODEL      path to the .gguf            (default ~/models/*Qwen2.5-1.5B*.gguf)
#   LLAMA_SERVER   path to the llama-server bin (default: built or prebuilt copy)
#   PORT           listen port                  (default 8080 — matches LLAMA_URL)
#   NGL            GPU layers to offload         (default 0 = CPU; 99 = full GPU)
#
# Rack note: to pin the Vulkan loader to the AMD cards (see CLAUDE.md "THE TRAP"),
# launch with VK_DRIVER_FILES=<radeon_icd.json> NGL=99 ./serve_llm.sh
set -uo pipefail

PORT="${PORT:-8080}"
NGL="${NGL:-0}"

# Locate the model: explicit LLM_MODEL, else the LARGEST gguf in ~/models
# (bigger ≈ better suggestions — picks Gemma-3-4B over a tiny model automatically).
MODEL="${LLM_MODEL:-}"
if [ -z "$MODEL" ]; then
  MODEL="$(ls -S "$HOME"/models/*.gguf 2>/dev/null | head -1)"
fi

# Locate llama-server: a Vulkan/CUDA build, else a downloaded prebuilt copy.
BIN="${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}"
[ -x "$BIN" ] || BIN="$(ls "$HOME"/llama_prebuilt/*/llama-server 2>/dev/null | head -1)"

if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
  echo "[serve_llm] no model .gguf found — set LLM_MODEL or drop one in ~/models/"; exit 1
fi
if [ -z "${BIN:-}" ] || [ ! -x "$BIN" ]; then
  echo "[serve_llm] llama-server not found — build it (bootstrap.sh) or set LLAMA_SERVER"; exit 1
fi

echo "[serve_llm] model : $MODEL"
echo "[serve_llm] binary: $BIN"
echo "[serve_llm] http  : http://localhost:${PORT}  (ngl=${NGL})"
# LD_LIBRARY_PATH lets the prebuilt binary find its bundled libllama/libggml .so's.
exec env LD_LIBRARY_PATH="$(dirname "$BIN")" "$BIN" \
  -m "$MODEL" --host 127.0.0.1 --port "$PORT" -c 2048 -ngl "$NGL" --no-webui
