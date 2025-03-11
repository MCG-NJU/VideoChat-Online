import re
import threading
import gradio as gr
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
)

# Configuration and model loading
model_name = "work_dirs/online_offline_image"
config = AutoConfig.from_pretrained("OpenGVLab/InternVL2-4B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, add_eos_token=False, trust_remote_code=True, use_fast=False
)
model = VideoChatOnline_IT.from_pretrained(
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
        # 使用正则表达式移除 "FrameX: <image>" 形式的文本
        return re.sub(r"Frame\d+: <image>", "", item).strip()

    def get_clean_history(self):
        """
        返回对模型透明的历史记录，移除 FrameX: <image> 这样的字样。
        """

        return [[self.clean_history_item(item[0]), item[1]] for item in self.history]

    def reset(self):
        self.history = []
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
    generation_config = dict(
        max_new_tokens=256, do_sample=True, num_beams=5, temperature=0.95
    )

    pixel_values = video_frame_data.to(model.device).to(model.dtype)

    # Use model's chat method if available
    llm_start_time = time.perf_counter()
    llm_message, history = model.chat(
        tokenizer,
        pixel_values,
        full_question,
        generation_config,
        history=history_synchronizer.get_history(),  # 使用当前历史记录
        return_history=True,
        verbose=True,
    )
    llm_end_time = time.perf_counter()
    print("LLM Latency:", llm_end_time - llm_start_time)

    # Update history
    history_synchronizer.set_history(history)

    # Format the response for gr.Chatbot
    return history_synchronizer.get_clean_history()  # 返回完整的聊天历史


# Global state for pause/resume
pause_event = threading.Event()
pause_event.set()  # Start with video playing


def start_chat(video_path, frame_interval, history):
    if not video_path:
        raise gr.Error("Please upload a video file.")

    # Load video and get frames
    pixel_values, original_frames, timestamps = load_video(
        video_path, fps=1 / frame_interval
    )

    # Reset history
    history_synchronizer.reset()
    history = history_synchronizer.get_clean_history()

    # Iterate through frames
    for idx, (frame, original_frame, timestamp) in enumerate(
        zip(pixel_values, original_frames, timestamps)
    ):
        if not pause_event.is_set():
            pause_event.wait()  # Pause processing

        # Display current frame and time
        yield timestamp, original_frame, pixel_values[: idx + 1], history

        # Simulate frame delay
        time.sleep(frame_interval)

    # End of video
    yield timestamps[-1], original_frames[-1], pixel_values, history


def toggle_pause():
    if pause_event.is_set():
        pause_event.clear()  # Pause processing
        return "Resume Video"
    else:
        pause_event.set()  # Resume processing
        return "Pause Video"


def stop_chat():
    pause_event.clear()  # Stop processing
    history_synchronizer.reset()  # Reset history
    return 0, None, None, []  # Return empty history


# Gradio UI layout
def build_ui():

    with gr.Blocks() as demo:
        # State to store pixel_values
        pixel_values_state = gr.State()

        # Title and description
        gr.Markdown(
            """
            # VideoChat Online Demo
            This interface allows you to upload a video, play/pause it, and ask questions about the video content.
            The model will generate answers based on the frames up to the paused time.
            """
        )

        # Instructions
        with gr.Accordion("How to Use", open=False):
            gr.Markdown(
                """
                ### Steps:
                1. **Upload a Video**: Click the "Upload Video" button to select a video file.
                2. **Set Frame Interval**: Adjust the slider to control the frame rate (frames per second).
                3. **Start Chat**: Click the "Start Chat" button to begin video playback.
                4. **Pause/Resume**: Use the "Pause Video" button to pause the video and ask questions.
                5. **Ask Questions**: Type your question in the text box and press Enter to get an answer.
                6. **Stop Chat**: Click the "Stop Video" button to reset the interface.
                """
            )

        # Main interface
        with gr.Row():
            with gr.Column():
                # Current frame display
                gr_frame_display = gr.Image(
                    label="Current Model Input Frame", interactive=False
                )
                # Current video time display
                gr_time_display = gr.Number(label="Current Video Time", value=0)
                # Pause and Stop buttons
                with gr.Row():
                    gr_pause_button = gr.Button("Pause Video")
                    gr_stop_button = gr.Button("Stop Video", variant="stop")

            with gr.Column():
                # Chat interface
                gr_chat_interface = gr.Chatbot(label="Chat History")
                gr_question_input = gr.Textbox(label="Ask your question")
                gr_question_input.submit(
                    generate_answer,
                    inputs=[
                        gr_question_input,
                        pixel_values_state,
                    ],  # 使用 pixel_values_state
                    outputs=gr_chat_interface,
                )

        # Frame interval control
        gr_frame_interval = gr.Slider(
            minimum=0.1,
            maximum=10,
            step=0.1,
            value=1,
            interactive=True,
            label="Frame Interval (sec)",
        )

        # Start button
        gr_start_button = gr.Button("Start Chat")

        # Start chat function
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

        # Pause button logic
        gr_pause_button.click(toggle_pause, inputs=[], outputs=gr_pause_button)

        # Stop button logic
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
