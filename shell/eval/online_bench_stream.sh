set -x

export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"


NNODE=1
NUM_GPUS=8
NUM_CPU=128
MASTER_NODE=${MASTER_NODE:-'localhost'}
NODE_RANK=${NODE_RANK:-0}
MASTER_PORT=10085
FPS=8

# If using multiple GPUs for evaluation, the following method is recommended to ensure load balancing between GPUs with different durations of data
dataset_infos=(
    "ovbench_action /workspace/data/hzp/OVBench /workspace/data/hzp/VideoChat-Online/ovbench_action.json"
    "ovbench_event /workspace/data/hzp/OVBench /workspace/data/hzp/VideoChat-Online/ovbench_event.json"
    "ovbench_object /workspace/data/hzp/OVBench /workspace/data/hzp/VideoChat-Online/ovbench_object.json"
)
# otherwise:
# dataset_infos=(
#     "ovbench /workspace/data/hzp/OVBench /workspace/data/hzp/VideoChat-Online/ovbench.json" #
# )

NODE_RANK=${NODE_RANK:-0}
memory_bank="2 2 12"
MODEL_DIR="work_dirs/VideoChatOnline_Stage2"
out_dir="work_dirs/ovbench"
mkdir -p "$out_dir"

for dataset_info in "${dataset_infos[@]}"; do
    dataset_name=$(echo "$dataset_info" | cut -d' ' -f1)
    data_root=$(echo "$dataset_info" | cut -d' ' -f2)
    anno_root=$(echo "$dataset_info" | cut -d' ' -f3)
    long_memory_bank=$(echo $memory_bank | cut -d' ' -f1)
    mid_memory_bank=$(echo $memory_bank | cut -d' ' -f2)
    short_memory_bank=$(echo $memory_bank | cut -d' ' -f3)

    torchrun \
        --nnodes=${NNODE} \
        --nproc_per_node=${NUM_GPUS} \
        --master_addr=${MASTER_NODE} \
        --master_port=${MASTER_PORT} \
        --node_rank=${NODE_RANK} \
        eval/evaluate_online_stream_single.py \
        --dataset ${dataset_name} \
        --num-workers 4 \
        --long_memory_bank ${long_memory_bank} \
        --mid_memory_bank ${mid_memory_bank} \
        --short_memory_bank ${short_memory_bank} \
        --data_root ${data_root} \
        --anno_root ${anno_root} \
        --checkpoint ${MODEL_DIR} \
        --fps ${FPS} \
        --time \
        --num_segments 2000 \
        --out-dir "${out_dir}" \
        > "${out_dir}/ovbench_eval_$(basename ${anno_root} .json).log" 2>&1
done


csv_files=(
    "${out_dir}/ovbench_action_result_final.csv"
    "${out_dir}/ovbench_event_result_final.csv"
    "${out_dir}/ovbench_object_result_final.csv"
)

args=()
for file in "${csv_files[@]}"; do
    args+=(<(head -n3 "$file"))
done

# 合并文件并输出
paste -d ',' "${args[@]}" > "${out_dir}/ovbench_result_final.csv"