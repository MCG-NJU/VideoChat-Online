set -x

export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

MODEL_DIR='work_dirs/slow_fast/online_offline_image'
if [ ! -d "$MODEL_DIR" ]; then
  mkdir -p "$MODEL_DIR"
fi

NNODE=5
NUM_GPUS=8
NUM_CPU=128
MASTER_NODE='10.13.149.182'
MASTER_PORT=10085
FPS=2

# 定义 MODEL_DIR 列表
MODEL_DIR_LIST=(
    "work_dirs/slow_fast/online_offline_image"
    #"work_dirs/InternVL2-1B"
    #"work_dirs/InternVL2-2B"
    #"work_dirs/ablation/finetune_online"
    #"work_dirs/ablation/finetune_online_wo_sl"
    #"work_dirs/ablation/finetune_online_wo_tg"
    #"work_dirs/ablation/finetune_online_wo_dvc"
    #"work_dirs/ablation/finetune_online_wo_stad"
    #"work_dirs/ablation/finetune_online_wo_track"
    #"work_dirs/ablation/finetune_online_wo_image"
)

# 定义数据集列表
datasets=( 
    "ava /workspace/data/datasets/AVA_Actions/raw/trainval /workspace/data/hzp/InternVL2/bench/AVA.json"
    "event /workspace/data/datasets /workspace/data/hzp/InternVL2/bench/CH_merge.json"
    "tao /workspace/data/hzp/Data/TAO/frames /workspace/data/hzp/InternVL2/bench/TAO.json"
)

# 定义滑动窗口参数
slide_windows=(32)

# 遍历 MODEL_DIR 列表
for MODEL_DIR in "${MODEL_DIR_LIST[@]}"; do
    # 循环遍历数据集列表
    for dataset_info in "${datasets[@]}"; do
        # 拆分数据集信息
        dataset_name=$(echo $dataset_info | cut -d' ' -f1)
        data_root=$(echo $dataset_info | cut -d' ' -f2)
        anno_root=$(echo $dataset_info | cut -d' ' -f3)

        # 循环遍历滑动窗口参数
        for SLIDE_WINDOW in "${slide_windows[@]}"; do
            # 创建输出目录
            out_dir="${MODEL_DIR}/ours_slide_window_${SLIDE_WINDOW}_new_tao"
            mkdir -p "$out_dir"

            # 执行 torchrun 命令
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
    done
done


