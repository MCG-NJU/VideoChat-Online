import re
import threading
import gradio as gr
from sympy import content
import torch
from PIL import Image
import numpy as np
import decord
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from internvl.model.videochat_online import (
    VideoChatOnline_IT,
    VideoChatOnline_Stream,
    InternVLChatConfig,
)

# Configuration and model loading
model_name = "work_dirs/VideoChatOnline_Stage2"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, add_eos_token=False, trust_remote_code=True, use_fast=False
)
config = InternVLChatConfig.from_pretrained(
        model_name, trust_remote_code=True
    )
model = (
        VideoChatOnline_IT.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
)
prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
model.system_message = prompt
model.to(torch.bfloat16).to(f"cuda:{0}").eval()


generation_config = dict(
        max_new_tokens=256, 
        do_sample=False, 
        num_beams=1, 
        temperature=0.95
    )
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
        self.chat_history = []
        self.frame_count = 0

    def set_history(self, history):
        self.history = history

    def get_history(self):
        return self.history
    
    def get_chat_history(self):
        return self.chat_history

    def clean_history_item(self, item):
        # 使用正则表达式移除 "FrameX: <image>" 形式的文本
        return re.sub(r"Frame\d+: <image>", "", item).strip()

    #def get_clean_history(self):
    #    """
    #    返回对模型透明的历史记录，移除 FrameX: <image> 这样的字样。
    #    """
#
    #    return [[self.clean_history_item(item[0]), item[1]] for item in self.history]

    def update(self, new_msg):
        new_msg["content"] = self.clean_history_item(new_msg["content"])
        new_msg = gr.ChatMessage(role=new_msg["role"], content=new_msg["content"])
        if self.chat_history:
            self.chat_history.append(new_msg)
        else:
            self.chat_history = [new_msg]

    def reset(self):
        self.history = []
        self.chat_history = []
        self.frame_count = 0


# Global history synchronizer
history_synchronizer = HistorySynchronizer()


def generate_answer(question, video_frame_data):
    video_prefix = "".join(
        [
            f"Frame{history_synchronizer.frame_count+i+1}: <image>\n"
            for i in range(len(video_frame_data[history_synchronizer.frame_count :]))
        ]
    )
    history_synchronizer.frame_count = len(video_frame_data)
    full_question = video_prefix + question
    

    pixel_values = video_frame_data.to(model.device).to(model.dtype)

    # 添加用户问题到历史并立即显示
    history_synchronizer.update({"role": "user", "content": question})
    current_chat_history = history_synchronizer.get_chat_history()
    
    # 添加临时"Thinking..."消息到前端显示（不保存到真实历史）
    temp_chat = current_chat_history.copy()
    temp_chat.append(gr.ChatMessage(role="assistant", content="Generating..."))
    yield temp_chat  # 第一次返回：用户问题 + Thinking...

    # 生成回答（使用真实历史，不含临时消息）
    llm_start_time = time.perf_counter()
    llm_message, history = model.chat(
        tokenizer,
        pixel_values,
        full_question,
        generation_config,
        history=history_synchronizer.get_history(),  # 真实历史
        return_history=True,
        verbose=False,
    )
    llm_end_time = time.perf_counter()
    print("LLM Latency:", llm_end_time - llm_start_time)

    # 更新真实历史记录
    history_synchronizer.set_history(history)
    history_synchronizer.update({"role": "assistant", "content": llm_message})

    # 返回最终结果（用户问题 + 真实回答）
    yield history_synchronizer.get_chat_history()




# Global state for pause/resume
pause_event = threading.Event()
pause_event.set()  # Start with video playing


def start_chat(video_path, frame_interval, current_history):
    if not video_path:
        raise gr.Error("Please upload a video file.")

    # Load video and get frames
    pixel_values, original_frames, timestamps = load_video(
        video_path, fps=1 / frame_interval
    )

    # Keep the existing chat history if there is one
    if current_history:
        history = current_history
    else:
        history_synchronizer.reset()
        history = history_synchronizer.get_chat_history()

    # Iterate through frames
    for idx, (frame, original_frame, timestamp) in enumerate(
        zip(pixel_values, original_frames, timestamps)
    ):
        if not pause_event.is_set():
            pause_event.wait()  # Pause processing

        # Get the latest chat history before yielding
        current_chat_history = history_synchronizer.get_chat_history()
        
        # Display current frame, time, and maintain chat history
        yield timestamp, original_frame, pixel_values[: idx + 1], current_chat_history

        # Simulate frame delay
        time.sleep(frame_interval)

    # End of video - use the latest chat history
    yield timestamps[-1], original_frames[-1], pixel_values, history_synchronizer.get_chat_history()


def toggle_pause():
    if pause_event.is_set():
        pause_event.clear()  # Pause processing
        return "Resume Video", history_synchronizer.get_chat_history()
    else:
        pause_event.set()  # Resume processing
        return "Pause Video", history_synchronizer.get_chat_history()


def stop_chat():
    pause_event.clear()  # Stop processing
    history_synchronizer.reset()  # Reset history
    return 0, None, None, []  # Return empty history


# Gradio UI layout
def build_ui():
    with gr.Blocks() as demo:
        # Previous state definitions remain the same
        pixel_values_state = gr.State()

        # Previous markdown and accordion sections remain the same
        
        with gr.Row():
            with gr.Column():
                gr_frame_display = gr.Image(
                    label="Current Model Input Frame", interactive=False
                )
                gr_time_display = gr.Number(label="Current Video Time", value=0)
                with gr.Row():
                    gr_pause_button = gr.Button("Pause Video")
                    gr_stop_button = gr.Button("Stop Video", variant="stop")

            with gr.Column():
                gr_chat_interface = gr.Chatbot(type="messages", label="Chat History")
                gr_question_input = gr.Textbox(label="Ask your question. Click enter to send the message")
                gr_question_input.submit(
                    generate_answer,
                    inputs=[gr_question_input, pixel_values_state],
                    outputs=gr_chat_interface,
                    queue=True,
                )

        gr_frame_interval = gr.Slider(
            minimum=0.1,
            maximum=10,
            step=0.1,
            value=0.5,
            interactive=True,
            label="Frame Interval (sec)",
        )

        gr_start_button = gr.Button("Start Chat")

        # Start chat function connection remains the same
        gr_start_button.click(
            start_chat,
            inputs=[
                gr.Video(label="Upload Video"),
                gr_frame_interval,
                gr_chat_interface,
            ],
            outputs=[
                gr_time_display,
                gr_frame_display,
                pixel_values_state,
                gr_chat_interface,
            ],
        )

        # Modified pause button connection
        gr_pause_button.click(
            toggle_pause,
            inputs=[],
            outputs=[gr_pause_button, gr_chat_interface]  # Add chat interface to outputs
        )

        # Stop button connection remains the same
        gr_stop_button.click(
            stop_chat,
            inputs=[],
            outputs=[
                gr_time_display,
                gr_frame_display,
                pixel_values_state,
                gr_chat_interface,
            ],
        )

    return demo



# Run the interface
demo = build_ui()
demo.launch(debug=True)
