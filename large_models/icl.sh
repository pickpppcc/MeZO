MODEL=${MODEL:-facebook/opt-13b}
MODEL_NAME=(${MODEL//\// })
MODEL_NAME="${MODEL_NAME[-1]}"

if [ "$BF" == "fp16" ]; then
    BF_ARGS="--load_float16"
fi

TAG=$BF

if [ -f "curves/$MODEL_NAME/$TASK-$TAG-acc.jpg" ]; then
    echo "Curve file curves/$MODEL_NAME/$TASK-$TAG.jpg already exists. Skipping..."
else
    python run_train.py --model_name $MODEL --task_name $TASK --output_dir result/tmp --tag icl --num_train 32 --num_eval 1000 --max_steps 100 $BF_ARGS --verbose "$@" &> "log/$MODEL_NAME/zeroshot-$TASK-$TAG.log"
fi
