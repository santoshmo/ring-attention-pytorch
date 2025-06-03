#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="/home/ubuntu/.llama/checkpoints/Llama3.2-1B"
K=16
M=1024
ROWS=8
SEED=42
PROMPTS_FILE="prompts.txt"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

while IFS= read -r prompt || [ -n "$prompt" ]; do
  # skip empty lines
  [[ -z "$prompt" ]] && continue

  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TS] Starting recovery for prompt: $prompt"

  # make a safe filename for log
  SAFE_NAME=$(echo "$prompt" | tr -cs '[:alnum:]' '_' | cut -c1-50)
  LOG_FILE="$LOG_DIR/${SAFE_NAME}.log"

  # run with unbuffered output, tee to log file for quick inspection
  python -u -m ring_attention_pytorch.count_min_sketch.llama_layer_recovery \
    --model_path "$MODEL_PATH" \
    --prompt "$prompt" \
    --k "$K" --m "$M" --rows "$ROWS" --seed "$SEED" 2>&1 | tee "$LOG_FILE"

  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[$TS] Finished recovery for prompt: $prompt (log: $LOG_FILE)"
done < "$PROMPTS_FILE"
# End Generation Here
```
