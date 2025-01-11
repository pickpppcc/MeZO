#!/bin/bash

TASK=${TASK:-SST-2}
K=${K:-16}
SEED=${SEED:-42}
BS=${BS:-8}
LR=${LR:-1e-5}
STEP=${STEP:-1000}
EVAL_STEP=${EVAL_STEP:-100}
MODEL=${MODEL:-roberta-large}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"
PRECISION=${PRECISION:-"fp32"}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

echo "TASK: $TASK"
echo "K: $K"
echo "Seed: $SEED"
echo "BS: $BS"
echo "LR: $LR"
echo "Step: $STEP; Eval step: $EVAL_STEP"
echo "Precision: $PRECISION"

GR_TAG=seed$SEED-bs$BS-lr$LR-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft}
TAG=${TAG:-${TASK}-k${K}-${MODEL_NAME}-${PRECISION}-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

mkdir -p "log/$MODEL_NAME" 
mkdir -p "result/$TAG"

# 记录开始时间
start_time=$(date +%s)

if [ -f "curves/$MODEL_NAME/$TASK-$TAG-acc.jpg" ]; then
    echo "Curve file curves/$MODEL_NAME/$TAG.jpg already exists. Skipping..."
else
    TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K PRECISION=$PRECISION\
        bash run_fewshot.sh --per_device_train_batch_size $BS --learning_rate $LR --eval_steps $EVAL_STEP \
        $@ &> "log/$MODEL_NAME/$TAG-v0.log"
fi

# 记录结束时间
end_time=$(date +%s)
training_time=$((end_time - start_time))

# 获取内存使用情况
memory_usage=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')

# 输出结果到文件
echo "Training Time: $training_time seconds" > "result/$TAG/training_metrics.txt"
echo "Memory Usage: $memory_usage MB" >> "result/$TAG/training_metrics.txt"
echo "Precision: $PRECISION" >> "result/$TAG/training_metrics.txt"
echo "Training Method: $EXTRA_TAG" >> "result/$TAG/training_metrics.txt"