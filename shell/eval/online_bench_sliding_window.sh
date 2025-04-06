set -x

export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

MODEL_DIR='MCG-NJU/VideoChatOnline-4B'
if [ ! -d "$MODEL_DIR" ]; then
  mkdir -p "$MODEL_DIR"
fi

NNODE=1
NUM_GPUS=8
NUM_CPU=128
MASTER_NODE='localhost'
MASTER_PORT=10085
FPS=2
slide_window=32
# If using multiple GPUs for evaluation, the following method is recommended to ensure load balancing between GPUs with different durations of data
dataset_infos=(
    "ovbench_action /path/to/AVA_RAW /OVBench/jsons/ovbench_action.json"
    "ovbench_event /path/to/COIN&HiREST /OVBench/jsons/ovbench_event.json"
    "ovbench_object /path/to/Others /OVBench/jsons/ovbench_object.json"
)
# otherwise:
# dataset_infos=(
#     "ovbench /path/to/OVBench /OVBench/ovbench.json" #
# )

# Create the output directory
out_dir="${MODEL_DIR}/slide_window_${slide_window}_${dataset_name}"
mkdir -p "$out_dir"

# Iterate over the MODEL_DIR list
for dataset_info in "${datasets[@]}"; do
    # Split dataset information
    dataset_name=$(echo $dataset_info | cut -d' ' -f1)
    data_root=$(echo $dataset_info | cut -d' ' -f2)
    anno_root=$(echo $dataset_info | cut -d' ' -f3)

    torchrun \
        --nnodes=${NNODE} \
        --nproc_per_node=${NUM_GPUS} \
        --master_addr=${MASTER_NODE} \
        --master_port=${MASTER_PORT} \
        --node_rank=${NODE_RANK} \
        eval/evaluate_online_sliding_window.py \
        --dataset ${dataset_name} \
        --data_root ${data_root} \
        --anno_root ${anno_root} \
        --checkpoint ${MODEL_DIR} \
        --fps ${FPS} \
        --num_segments 64 \
        --slide_window ${SLIDE_WINDOW} \
        --time \
        --out-dir ${out_dir} \
        > "${out_dir}/eval_online_${dataset_name}_${NODE_RANK}_1000_${SLIDE_WINDOW}.log" 2>&1
done


