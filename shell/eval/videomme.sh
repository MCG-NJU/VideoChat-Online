set -x

export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=4
NUM_GPUS=8
NUM_CPU=128
MASTER_NODE=${MASTER_NODE:-'localhost'}
MODEL_DIR='work_dirs/VideoChatOnline_Stage1'
SLIDE_WINDOW=1800
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_NODE} \
    --master_port=10096 \
    --node_rank=${NODE_RANK} \
    eval/evaluate_videomme.py \
        --checkpoint ${MODEL_DIR} \
        --num_segments ${SLIDE_WINDOW} \
        --long_memory_bank 24 \
        --mid_memory_bank 24 \
        --short_memory_bank 144 \
        --out-dir work_dirs/videomme_stage1 \
        > "videochatonline_eval_videomme_${NODE_RANK}_${SLIDE_WINDOW}.log" 2>&1