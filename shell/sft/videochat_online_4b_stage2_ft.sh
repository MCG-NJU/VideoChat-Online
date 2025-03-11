set -x

export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

#export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

# number of gpus: 128
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 1024
# epoch: 1
NNODE=4
NUM_GPUS=8
NUM_CPU=128

GPUS=$((NNODE * NUM_GPUS))
BATCH_SIZE=${BATCH_SIZE:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

MASTER_NODE='localhost'
OUTPUT_DIR='work_dirs/videochat_online_stage2'
NODE_RANK=${NODE_RANK:-0}
wandb offline
#if [ \$NODE_RANK -eq 0 ]; then wandb offline; fi;


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_NODE} \
    --master_port=10096 \
    --node_rank=${NODE_RANK} \
    internvl/train/internvl_chat_finetune_online.py \
      --model_name_or_path "OpenGVLab/InternVL2-4B" \
      --local_rank ${NODE_RANK} \
      --conv_style "phi3-chat" \
      --output_dir ${OUTPUT_DIR} \
      --meta_path "anno_data/finetune_online.json" "anno_data/finetune_videochat2.json" "anno_data/finetune_image.json" \
      --overwrite_output_dir True \
      --force_image_size 448 \
      --max_dynamic_patch 12 \
      --down_sample_ratio 0.5 \
      --avg_pooling_down_sample_ratio 1 \
      --reverse_memory_sample_ratio 8 4 1 \
      --drop_path_rate 0.1 \
      --freeze_llm False \
      --freeze_mlp False \
      --freeze_backbone True \
      --vision_select_layer -1 \
      --dataloader_num_workers 8 \
      --bf16 True \
      --num_train_epochs 1 \
      --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
      --gradient_accumulation_steps ${GRADIENT_ACC} \
      --evaluation_strategy "no" \
      --save_strategy "steps" \
      --save_steps 200 \
      --save_total_limit 3 \
      --learning_rate 1e-4 \
      --weight_decay 0.05 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --max_seq_length 32768 \
      --do_train True \
      --grad_checkpoint True \
      --group_by_length True \
      --dynamic_image_size True \
      --use_thumbnail True \
      --ps_version 'v2' \
      --max_num_frame 64 \
      --sampling_method fps1 \
      --report_to "wandb" \
      2>&1 | tee -a "${OUTPUT_DIR}/training.log"