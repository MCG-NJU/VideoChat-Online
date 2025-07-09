export HF_DATASETS_OFFLINE=1
MASTER_PORT=$((18000 + $RANDOM % 100))

CKPT_PATH=/path_to_your/Qwen2-VL-7B-Instruct
MODEL_NAME=qwen2_vl
CONV_TEMPLATE=qwen_2
MAX_NUM_FRAMES=64
NUM_GPUS=8



TASK=ovbench_window32

TASK_SUFFIX="${TASK//,/_}"
echo $TASK_SUFFIX
JOB_NAME=$(basename $0)_$(date +"%Y%m%d_%H%M%S")

srun -p video5 \
    --quotatype=spot \
    --job-name=${JOB_NAME} \
    --ntasks=1 \
    --gres=gpu:${NUM_GPUS} \
    --ntasks-per-node=1 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    accelerate launch --num_processes ${NUM_GPUS} --main_process_port 10078 -m lmms_eval \
        --model ${MODEL_NAME} \
        --model_args pretrained=$CKPT_PATH,conv_template=$CONV_TEMPLATE,max_frames_num=$MAX_NUM_FRAMES \
        --tasks $TASK \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $TASK_SUFFIX \
        --output_path ./eval/log_result/${JOB_NAME}_f${MAX_NUM_FRAMES} \
        2>&1 | tee ./eval/log/${JOB_NAME}_f${MAX_NUM_FRAMES}.log

