#! /bin/bash
set -euo pipefail

PROFILE=${1:-"false"}

if [ "$PROFILE" = "--profile" ]; then
    #EXEC="pyinstrument -r html -t --color-o profile.html"
#    EXEC="python -m cProfile -o profile.prof"
    EXEC="scalene --no-browser --cli" #--reduced-profile  --no-browser
else
    EXEC="python"
fi

MAX_SAMPLES=10
NUM_TASKS=2
MODEL="meta-llama/Llama-3.2-1B-Instruct"

# Split Model into two parts
MODEL_NAME=$(echo $MODEL | cut -d'/' -f2)

SUITE="original"
BENCHMARK="mmlu:high_school_geography"
# Quote the task string to prevent bash from interpreting the |
TASK="${SUITE}|${BENCHMARK}"
DTYPE="bfloat16"
LAUNCHER_TYPE="accelerate"
NUM_FEW_SHOT_K=0
TRUNCATE_FEW_SHOT=False
USE_CHAT_TEMPLATE=False
OVERRIDE_BATCH_SIZE=0 # 0 means use the default batch size
LOG_LEVEL="INFO"
PUSH_TO_HUB=True
HUB_ARGS="--hub_results_org=jeromeku"

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

if [ "$PUSH_TO_HUB" = "True" ]; then
    CMD="$CMD --push_to_hub $HUB_ARGS"
fi

if [ "$USE_CHAT_TEMPLATE" = "True" ]; then
    CMD="$CMD --use_chat_template"
fi

echo $CMD
eval $CMD 2>&1 | tee $LOG_NAME
