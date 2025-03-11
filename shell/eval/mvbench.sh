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
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=10085
MODEL_DIR='work_dirs/VideoChatOnline_Stage2'
SLIDE_WINDOW=1800
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_NODE} \
    --master_port=${MASTER_PORT} \
    --node_rank=${NODE_RANK} \
    eval/evaluate_mvbench.py \
        --checkpoint ${MODEL_DIR} \
        --num_segments ${SLIDE_WINDOW} \
        --long_memory_bank 24 \
        --mid_memory_bank 24 \
        --short_memory_bank 144 \
        --out-dir work_dirs/mvbench \
        > "videochatonline_eval_mvbench_${NODE_RANK}_${SLIDE_WINDOW}.log" 2>&1
