#!/bin/bash
set -e

echo "Received arguments: $@"

if [ $# -ne 2 ]; then
  echo "Usage: docker run ... <input_dir> <output_json>"
  exit 1
fi

INPUT_DIR="$1"
OUTPUT_JSON="$2"

OUTPUT_DIR="$(dirname "$OUTPUT_JSON")"
DEBUG_DIR="$OUTPUT_DIR/debug_output"
DEBUG_JSON="$DEBUG_DIR/detailed_output.json"
mkdir -p "$DEBUG_DIR"

echo "[Entrypoint] Running persona_analyzer.py on $INPUT_DIR -> $OUTPUT_JSON and $DEBUG_JSON"
python persona_analyzer.py --input "$INPUT_DIR" --output "$OUTPUT_JSON" --debug_output "$DEBUG_JSON" --enhanced

echo "[Entrypoint] Done. Output: $OUTPUT_JSON, Debug: $DEBUG_JSON"