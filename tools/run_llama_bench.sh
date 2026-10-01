#!/usr/bin/env bash
set -e

LLAMA_DIR="$HOME/.local/llama-bench/llama-b11320"
if [ ! -d "$LLAMA_DIR" ]; then
    echo "Error: llama-bench directory not found at $LLAMA_DIR" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$LLAMA_DIR:$LD_LIBRARY_PATH"
exec "$LLAMA_DIR/llama-bench" "$@"
