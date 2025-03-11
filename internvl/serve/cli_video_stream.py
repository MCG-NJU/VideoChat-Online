import argparse
import logging
import threading
import torch
import numpy as np
import os
from multiprocessing import Process, Queue, Manager
from decord import VideoReader, cpu
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from PIL import Image
import numpy as np
import re
import json
import io
import sys
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt


from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm

import decord
import time
import torch
import json
from tqdm import tqdm
import multiprocessing as mp
from torch.utils.data import Subset
import os
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from internvl.model.videochat_online import (
    VideoChatOnline_IT,
    VideoChatOnline_Stream,
)

decord.bridge.set_bridge("torch")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array(
        [
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ]
    )
    return frame_indices


def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(
        bound, fps, max_frame, first_idx=0, num_segments=num_segments
    )
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy()).convert("RGB")
        img = dynamic_preprocess(
            img, image_size=input_size, use_thumbnail=True, max_num=max_num
        )
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    secs = [round(ids / fps, 1) for ids in frame_indices]
    return pixel_values, num_patches_list, secs


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=12, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class VideoStreamManager:
    def __init__(
        self,
        video_path,
        model,
        tokenizer,
        num_segments=32,
        input_size=448,
        target_fps=1,
    ):
        self.video_path = video_path
        self.model = model
        self.tokenizer = tokenizer
        self.num_segments = num_segments
        self.input_size = input_size
        self.target_fps = target_fps
        self.question_queue = Queue()  # Queue to handle user input questions
        self.response_queue = Queue()  # Queue to handle model responses
        self.history = []

    def frame_generator(self, queue):
        vr = VideoReader(self.video_path, ctx=cpu(0))
        original_fps = float(vr.get_avg_fps())
        frame_indices = self.get_frame_indices(len(vr), original_fps)

        for idx in frame_indices:
            img = Image.fromarray(vr[idx].numpy()).convert("RGB")
            queue.put(img)
            # Wait to simulate fps, check if a question was received in this time
            time.sleep(1 / self.target_fps)

    def get_frame_indices(self, max_frame, original_fps):
        frame_interval = int(original_fps / self.target_fps)
        if frame_interval < 1:
            frame_interval = 1  # Ensure minimum frame interval of 1
        return list(range(0, max_frame, frame_interval))

    def process_frames(self, queue, output_queue):
        while True:
            img = queue.get()
            if img is None:
                output_queue.put(None)
                break

            processed_images = self.dynamic_preprocess(img)
            pixel_values = (
                torch.stack([self.transform_image(tile) for tile in processed_images])
                .cpu()
                .numpy()
            )
            output_queue.put(pixel_values)

    def transform_image(self, image):
        transform = T.Compose(
            [
                T.Resize((self.input_size, self.input_size)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform(image)

    def dynamic_preprocess(self, image):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_width, target_height = self.input_size, self.input_size
        resized_img = image.resize((target_width, target_height))
        return [resized_img.crop((0, 0, self.input_size, self.input_size))]

    def ask_question(self):
        while True:
            question = input("Enter a question (type Enter Button to quit): ")
            if question.lower() == "\n":
                break
            self.question_queue.put(question)  # Put the question in the queue

    def inference(self, output_queue, model):
        pixel_values = None
        while True:
            # Get the next frame of processed pixel values
            pixel_value = output_queue.get()

            if pixel_value is None:
                break

            if pixel_values is None:
                pixel_values = torch.from_numpy(pixel_value).to(
                    dtype=self.model.dtype, device=self.model.device
                )
            else:
                pixel_values = torch.concat(
                    [
                        pixel_values,
                        torch.from_numpy(pixel_value).to(
                            dtype=self.model.dtype, device=self.model.device
                        ),
                    ],
                    dim=0,
                )

            # Wait for user input or continue processing
            try:
                question = (
                    self.question_queue.get()
                )  # Non-blocking check for a new question
            except:
                question = None

            if question:
                # Process the question
                video_prefix = "".join(
                    [f"Frame{i+1}: <image>\n" for i in range(len(pixel_values))]
                )
                full_question = video_prefix + question
                generation_config = dict(
                    max_new_tokens=256, do_sample=True, num_beams=5, temperature=0.95
                )

                llm_start_time = time.perf_counter()
                print(pixel_values.shape)
                llm_message, self.history = model.chat(
                    self.tokenizer,
                    pixel_values,
                    full_question,
                    generation_config,
                    history=self.history,
                    return_history=True,
                    verbose=True,
                )
                llm_end_time = time.perf_counter()

                print("llm_latency", llm_end_time - llm_start_time)
                print("Response:", llm_message)
                self.history.append((full_question, llm_message))

            # Wait to simulate the FPS
            time.sleep(1 / self.target_fps)


# Main Program
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Online Video Stream Demo")
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file"
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="Model name or path"
    )
    parser.add_argument("--fps", type=int, default=1, help="play fps")
    args = parser.parse_args()

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        "OpenGVLab/InternVL2-4B", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    model = VideoChatOnline_IT.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(torch.bfloat16).to(f"cuda:{7}").eval()
    model.system_message = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"

    # Initialize video stream manager
    manager = VideoStreamManager(args.video_path, model, tokenizer, target_fps=args.fps)

    # Start question input thread
    question_thread = threading.Thread(target=manager.ask_question)
    question_thread.daemon = True  # Daemon thread will exit when the main program exits
    question_thread.start()

    # Start video processing
    queue = Queue()
    output_queue = Queue()

    # Start video stream processes
    p1 = Process(target=manager.frame_generator, args=(queue,))
    p2 = Process(target=manager.process_frames, args=(queue, output_queue))
    p3 = Process(target=manager.inference, args=(output_queue, model))

    p1.start()
    p2.start()
    p3.start()

    p1.join()
    p2.join()
    p3.join()
