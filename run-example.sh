#! /bin/bash
set -euo pipefail

PROFILE=${1:-"false"}

if [ "$PROFILE" = "true" ]; then
    EXEC="pyinstrument -r html -o profile.html -t"
    # "python -m cProfile -o profile.prof"
else
    EXEC="python"
fi

MAX_SAMPLES=10
NUM_TASKS=2
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
# Split Model into two parts
MODEL_NAME=$(echo $MODEL | cut -d'/' -f2)

SUITE="helm"
BENCHMARK="mmlu"
# Quote the task string to prevent bash from interpreting the |
TASK="${SUITE}|${BENCHMARK}"
DTYPE="bfloat16"
LAUNCHER_TYPE="accelerate"
NUM_FEW_SHOT_K=5
TRUNCATE_FEW_SHOT=True
USE_CHAT_TEMPLATE=False
OVERRIDE_BATCH_SIZE=0 # 0 means use the default batch size
LOG_LEVEL="INFO"

SAVE_DIR="./eval_results"
SAVE_DIR=$SAVE_DIR/$SUITE/$BENCHMARK/$MODEL_NAME/$LAUNCHER_TYPE
mkdir -p $SAVE_DIR
LOG_NAME=$SAVE_DIR/run.log

CMD="$EXEC example.py \
    --task \"$TASK\" \
    --model \"$MODEL\" \
    --dtype \"$DTYPE\" \
    --launcher_type \"$LAUNCHER_TYPE\" \
    --override_batch_size 0 \
    --max_samples \"$MAX_SAMPLES\" \
    --num_tasks \"$NUM_TASKS\" \
    --num_few_shot_k \"$NUM_FEW_SHOT_K\" \
    --truncate_few_shot \"$TRUNCATE_FEW_SHOT\" \
    --log_level \"$LOG_LEVEL\" \
    --save_dir \"$SAVE_DIR\""

if [ "$USE_CHAT_TEMPLATE" = "True" ]; then
    CMD="$CMD --use_chat_template"
fi

echo $CMD
eval $CMD 2>&1 | tee $LOG_NAME
