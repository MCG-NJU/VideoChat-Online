import time
import torch
from transformers import AutoTokenizer, AutoConfig
from internvl.model.videochat_online import VideoChatOnline_IT
from decord import VideoReader, cpu
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from internvl.model.videochat_online.modeling_internvl_chat import InternVLChatModel
from internvl.model.videochat_online.modeling_videochat_online import VideoChatOnline_Stream


# Configuration and model loading
model_name = "work_dirs/online_offline_image"
config = AutoConfig.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, add_eos_token=False, trust_remote_code=True, use_fast=False
)
model = VideoChatOnline_Stream.from_pretrained(
    model_name,
    config=config,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.to(torch.bfloat16).to(f"cuda:{0}").eval()


# Video preprocessing function
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transform


def load_video(video_path, fps=1, input_size=448):
    vr = VideoReader(video_path, ctx=cpu(0))
    max_frame = len(vr) - 1
    frame_indices = np.arange(0, max_frame, int(vr.get_avg_fps() / fps))
    frames = []
    original_frames = []  # 用于存储原始帧
    transform = build_transform(input_size)

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
        original_frames.append(np.array(img))  # 保存原始帧
        frames.append(transform(img))  # 保存归一化后的帧

    pixel_values = torch.stack(frames)  # 归一化后的帧
    original_frames = np.stack(original_frames)  # 原始帧
    timestamps = frame_indices / vr.get_avg_fps()  # 时间戳

    return pixel_values, original_frames, timestamps


# History synchronizer class
class HistorySynchronizer:
    def __init__(self):
        self.history = []
        self.frame_count = 0

    def set_history(self, history):
        self.history = history

    def get_history(self):
        return self.history

    def clean_history_item(self, item):
        return item.strip()  # Placeholder: implement your clean-up logic

    def get_clean_history(self):
        return [[self.clean_history_item(item[0]), item[1]] for item in self.history]

    def reset(self):
        self.history = []
        self.frame_count = 0


# Global history synchronizer
history_synchronizer = HistorySynchronizer()


def generate_answer_with_metrics(question, video_frame_data):
    # 记录提问开始时间
    question_start_time = time.perf_counter()

    # 编码问题文本，计算输入的 token 数量
    input_tokens = tokenizer.encode(question, return_tensors="pt").size(1)

    # 生成模型的回答，设置 max_new_tokens=1 计算 TTFT
    generation_config_first_token = dict(max_new_tokens=1, do_sample=False)

    pixel_values = video_frame_data.to(model.device).to(model.dtype)
    #video_prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(pixel_values))])
    video_prefix = [f"Frame{i+1}: <image>\n" for i in range(len(pixel_values))]
    question = question
    # 记录生成第一个 token 的时间（TTFT）
    llm_start_time = time.perf_counter()
    llm_message_first_token, history_first_token = model.chat(
        tokenizer,
        pixel_values,
        question,
        generation_config_first_token,
        history=history_synchronizer.get_history(),
        return_history=True,
        verbose=True,
        timestamps=video_prefix,
    )
    llm_end_time = time.perf_counter()

    # 记录第一个 token 生成的时间（TTFT）
    #TTFT = llm_end_time - llm_start_time
#
    ## 获取生成的第一个 token
    #first_token = llm_message_first_token
    #first_token_tokens = tokenizer.encode(first_token, return_tensors="pt").size(1)
#
    ## 生成剩余的 tokens，设置 max_new_tokens=256 来计算每个 token 的生成时间（TPOT）
    #remaining_tokens = 256
    #generation_config_remaining_tokens = dict(
    #    max_new_tokens=remaining_tokens, do_sample=False
    #)
#
    #llm_start_time = time.perf_counter()
    #llm_message_remaining_tokens, history_remaining_tokens = model.chat(
    #    tokenizer,
    #    pixel_values,
    #    question,
    #    generation_config_remaining_tokens,
    #    history=history_first_token,
    #    return_history=True,
    #    verbose=True,
    #    #timestamps=video_prefix,
    #)
    #llm_end_time = time.perf_counter()
#
    ## 记录剩余 tokens 生成的总时间
    #total_generation_time = llm_end_time - llm_start_time
#
    ## 计算每个 token 的时间（不含首个 token），即 Time Per Output Token (TPOT)
    #TPOT = (
    #    (total_generation_time - TTFT)
    #    / (tokenizer.encode(llm_message_remaining_tokens, return_tensors="pt").size(1))
    #    if remaining_tokens > 0
    #    else 0
    #)
#
    ## 计算 Latency（理论延迟）
    #latency = TTFT + TPOT * (remaining_tokens)
#
    ## 计算每秒生成的 token 数量 (TPS)
    #TPS = remaining_tokens / latency if latency > 0 else 0
#
    ## 输出性能指标
    #print(
    #    f"Total Generation Time (excluding first token): {total_generation_time:.4f} seconds"
    #)
    #total_token = tokenizer.encode(
    #    llm_message_remaining_tokens, return_tensors="pt"
    #).size(1)
    #print(f"Time To First Token (TTFT): {TTFT:.4f} seconds")
    #print(f"Time Per Output Token (TPOT): {TPOT:.6f} / {total_token} seconds/token")
    #print(f"Latency (Total): {latency:.4f} seconds")
    #print(f"Tokens Per Second (TPS): {TPS:.4f} tokens/second")
#
    ## 更新历史记录
    #history_synchronizer.set_history(history_remaining_tokens)

    # 返回清理后的历史记录
    return history_synchronizer.get_clean_history()

model.long_bank = 2
model.mid_bank = 2
model.short_bank = 12
# You can call this function like:
question = "What happened in the video? Describe in detail."
video_frame_data = torch.randn(
    3000, 3, 448, 448
)  # Example tensor representing video frames
generate_answer_with_metrics(question, video_frame_data)
