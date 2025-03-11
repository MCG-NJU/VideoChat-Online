from datetime import datetime
import gc
import json
import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import csv
import glob
import os.path as osp
import pickle
import random
from matplotlib.pyplot import sca
import numpy as np
import pandas as pd
import torch
import os
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import wandb
import decord
from decord import cpu

# decord.logging.set_level(logging.ERROR)
import io
from ipdb import set_trace

# from internvl.model.internvl_chat.modeling_internvl_sam_chat_finetune import VideoChatOnline_IT_SAM_IT
try:
    from petrel_client.client import Client

    client = Client("~/petreloss.conf")
except:
    pass
import ast

import io
from decord import VideoReader
import wandb

import time
import func_timeout
from func_timeout import func_set_timeout
from einops import rearrange
import numpy as np
import json
import torch
import torch.distributed as dist
import transformers
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.videochat_online import (
    InternVisionConfig,
    InternVisionModel,
    InternVLChatConfig,
    # VideoChatOnline_IT,
    VideoChatOnline_IT,
)
from internvl.patch import (
    concat_pad_data_collator,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_train_sampler,
)
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    VIDEO_CONTEXT_TOKEN,
)
from internvl.train.dataset import (
    ConcatDataset,
    TCSLoader,
    WeightedConcatDataset,
    build_transform,
    dynamic_preprocess,
    preprocess,
    preprocess_internlm,
    preprocess_mpt,
    preprocess_phi3,
    preprocess_internvl2_5,
)
from internvl.train.trainer_monkey_patch import replace_create_optimizer
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)
import signal
import random
import traceback
import os
from PIL import UnidentifiedImageError
import torch
from typing import Dict


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# 设置超时时间
TIMEOUT = 30  # 超时时间，单位为秒

signal.signal(signal.SIGALRM, timeout_handler)


# Apply necessary patches for the transformers library
replace_llama_rmsnorm_with_fused_rmsnorm()
replace_train_sampler()

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print("petrel_client is not installed. Using PIL to load images.")
    has_tcs_loader = False
has_tcs_loader = True
# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the LLM decoder."},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the vision backbone of the model."},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={"help": "Set to True to freeze the MLP layers of the model."},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={
            "help": "Specify the number of ViT layers to unfreeze. Default is 0."
        },
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={
            "help": "Specify the layer of ViT feature map to use. Default is last layer."
        },
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={
            "help": "Set the LoRA adapter rank for the backbone model. Default is 0."
        },
    )
    use_llm_lora: int = field(
        default=0,
        metadata={"help": "Set the LoRA adapter rank for the LLM. Default is 0."},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={"help": "Set to True to unfreeze the language model's head."},
    )
    use_custom_trainer: bool = field(
        default=False,
        metadata={"help": "Set to True to enable the use of a custom trainer."},
    )
    grad_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use gradient checkpointing."},
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={"help": "Set the drop path rate for the ViT model. Default is 0."},
    )
    ps_version: str = field(
        default="v2",
        metadata={
            "help": "Specify the version of pixel shuffle implementation. Default is `v1`."
            "Please use `v2` to fix the bug of transposed image."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments for specifying data input for training and evaluation.
    """

    max_seq_length: Optional[int] = field(
        default=2048,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    force_image_size: Optional[int] = field(
        default=448,
        metadata={"help": "Set the desired size for the image. Default is 224."},
    )
    down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Set the desired down-sampling ratio for the image. Default is 1.0."
        },
    )
    reverse_memory_sample_ratio: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "Reverse Pyramid memory bank Sampling Ratio. (1 / Sampling Ratio)"
        },
    )
    pad2square: Optional[bool] = field(
        default=False,
        metadata={"help": "Pad the image to a square shape if set to True."},
    )
    conv_style: Optional[str] = field(
        default="internlm2-chat", metadata={"help": "Prompt style for a conversation."}
    )
    meta_path: Optional[List[str]] = field(
        default=None,
        metadata={"help": "The path of the meta file of datasets."},
    )
    use_data_resampling: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use data resampling."},
    )
    dynamic_image_size: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to use dynamic image size."},
    )
    use_thumbnail: Optional[bool] = field(
        default=False,
        metadata={"help": "Set to True to add a thumbnail image."},
    )
    min_dynamic_patch: Optional[int] = field(
        default=1,
        metadata={"help": "The minimum number of dynamic patches. Default is 1."},
    )
    max_dynamic_patch: Optional[int] = field(
        default=12,
        metadata={"help": "The maximum number of dynamic patches. Default is 6."},
    )
    normalize_type: Optional[str] = field(
        default="imagenet",
        metadata={"help": "The normalize type for the image. Default is imagenet."},
    )
    ckpt_path: Optional[str] = field(
        default="",
        metadata={"help": "ckpt path"},
    )
    max_num_frame: Optional[int] = field(
        default=64,
        metadata={"help": "The maximum number of frames. Default is 64."},
    )
    avg_pooling_down_sample_ratio: Optional[float] = field(
        default=0.5,
        metadata={"help": "The maximum number of frames. Default is 64."},
    )
    sampling_method: Optional[str] = field(
        default="fps1",
        metadata={"help": "The sampling_method of video. Default is fps=1."},
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=224,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        min_num_frame=1,  # for video data
        max_num_frame=3,  # for video data
        sampling_method="rand",  # for video data
        repeat_time=1,
        normalize_type="imagenet",
        random_seed=0,
        reverse_memory_sample_ratio=None,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f"[Dataset] num_image_token: {num_image_token}")
        logger.info(f"[Dataset] dynamic_image_size: {dynamic_image_size}")
        logger.info(f"[Dataset] use_thumbnail: {use_thumbnail}")
        logger.info(
            f"[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}"
        )

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        logger.info("Formatting inputs...Skip in lazy mode")
        assert meta["annotation"].endswith(
            "json"
        ), f'annotation must be json, but got {meta["annotation"]}'
        with open(meta["annotation"], "r") as f:
            self.raw_data = json.load(f)
        self.num_examples = len(self.raw_data)
        self.rng = np.random.default_rng(seed=random_seed)
        self.rng.shuffle(self.raw_data)

        gc.collect()
        self.root = meta["data_root"]
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type
        self.reverse_memory_sample_ratio = reverse_memory_sample_ratio or [8, 4, 1]

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = (
                {}
            )  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                if "length" in data_item:
                    token_length = data_item[
                        "length"
                    ]  # Use precomputed length if available
                else:
                    if "QA" in data_item:
                        conversations = self.process_qa(data_item["QA"])
                    elif "query_template" in data_item:
                        conversations = [data_item["query_template"]]
                    else:
                        conversations = data_item["conversations"]
                    conversations = "\n".join([temp["value"] for temp in conversations])
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = (
                            token_length
                            + num_image_token * (max_dynamic_patch + use_thumbnail)
                        )
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)
        gc.collect()

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == "Hermes-2":
            preprocess_function = preprocess_mpt
        elif self.template_name == "internlm2-chat":
            preprocess_function = preprocess_internlm
        elif self.template_name == "phi3-chat":
            preprocess_function = preprocess_phi3
        elif self.template_name == "internvl2_5":
            preprocess_function = preprocess_internvl2_5
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and "s3://" in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert("RGB")

    def get_image_path(self, image_path):
        if image_path.startswith("s3://"):  # for ceph
            image_path = self.root + image_path
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if "<image>" not in data_item["conversations"][0]["value"]:
            data_item["conversations"][0]["value"] = (
                "<image>\n" + data_item["conversations"][0]["value"]
            )

        # Merge the image path
        image_path = self.get_image_path(data_item["image"])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if (
            self.dynamic_image_size
        ):  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:  # Otherwise, use the original image as a single patch
            images = [image]
        assert len(images) > 0
        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert (
                num_patches == 1
            ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=torch.tensor([num_patches], dtype=torch.long),
            is_video=torch.tensor([0], dtype=torch.long),
        )
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item["image"])
        for image_path in data_item["image"]:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if (
                self.dynamic_image_size
            ):  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=self.max_dynamic_patch // num_image,
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)

        assert len(images) > 0
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=torch.tensor([num_patches], dtype=torch.long),
            is_video=torch.tensor([0], dtype=torch.long),
        )
        return ret

    def convert_to_seconds(self, timestamp):
        time_obj = datetime.strptime(timestamp, "%H:%M:%S.%f")
        return (
            time_obj.hour * 3600
            + time_obj.minute * 60
            + time_obj.second
            + time_obj.microsecond / 1e6
        )

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        if "QA" in data_item:
            data_item["conversations"] = self.process_qa(data_item["QA"])
        # Ensure the first conversation contains a video placeholder
        data_item["conversations"][0]["value"] = data_item["conversations"][0][
            "value"
        ].replace("<image>", "<video>")
        if "<video>" not in data_item["conversations"][0]["value"]:
            # data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<image>','')
            # data_item['conversations'][0]['value'] = data_item['conversations'][0]['value'].replace('<video>','')
            data_item["conversations"][0]["value"] = (
                "<video>\n" + data_item["conversations"][0]["value"]
            )

        # Get the video file path
        video_file = data_item["video"]
        video_path = os.path.join(self.root, video_file)

        if len(video_path.split(".")) == 1:
            video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
            for fmt in video_formats:  # Added this line
                if os.path.exists(f"{video_path}{fmt}"):
                    video_path = f"{video_path}{fmt}"
                    break
        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        # Get the video file path
        clip = data_item.get("clip", None)
        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type="video",
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=clip,
        )
        assert len(image_list) > 0
        # Generate special tokens for each video frame
        special_tokens = "\n".join(
            ["Frame{}: <image>".format(i + 1) for i in range(len(image_list))]
        )
        data_item["conversations"][0]["value"] = data_item["conversations"][0][
            "value"
        ].replace("<video>", special_tokens)

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        # assert len(pixel_values) > 1, video_path
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        if self.num_image_token == 64:
            num_image_tokens = [self.num_image_token] * num_patches
        else:
            scale = VideoChatOnline_IT.tokens_arrange(
                num_patches,
                self.reverse_memory_sample_ratio,
                [2**s for s in range(len(self.reverse_memory_sample_ratio))],
            )
            num_image_tokens = [self.num_image_token // s**2 for s in scale]

        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=torch.tensor([num_patches], dtype=torch.long),
            is_video=torch.tensor([1], dtype=torch.long),
        )
        return ret

    def online_video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Get the video file path
        image_files = data_item.get("all_image_files", None)
        if image_files is None:
            video_file = data_item["video"]
            video_path = os.path.join(self.root, video_file)

            if len(video_path.split(".")) == 1:
                video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
                for fmt in video_formats:  # Added this line
                    if os.path.exists(f"{video_path}{fmt}"):
                        video_path = f"{video_path}{fmt}"
                        break
            clip = data_item.get("clip", None)
            image_list, timestamps = self.tcs_loader(
                video_path,
                image_type="video",
                max_num_frames=self.max_num_frame,
                min_num_frames=self.min_num_frame,
                sample=self.sampling_method,
                clip=clip,
                return_timestamps=True,
            )
        else:
            fps = data_item.get("fps", 1)  # Default to 1 fps if not specified
            video_file = data_item["video"]
            video_root = os.path.join(self.root, video_file)

            # Uniformly sample to the max_num_frame length
            if len(image_files) > self.max_num_frame:
                # Use np.linspace to generate evenly spaced indices
                sampled_indices = np.linspace(
                    0, len(image_files) - 1, self.max_num_frame, dtype=int
                )
                image_files = [image_files[i] for i in sampled_indices]
                image_bboxes = [data_item["image_bboxes"][i] for i in sampled_indices]
            else:
                image_bboxes = data_item["image_bboxes"]

            # Load all images
            image_list = [
                self.load_image(os.path.join(video_root, img)) for img in image_files
            ]

            # Generate timestamps
            timestamps = [round(bbox["timestamp"], 1) for bbox in image_bboxes]

            # Get the corresponding bbox
            # Randomly select one image's bbox to replace <bbox> in query_template
            random_index = random.randint(0, len(image_bboxes) - 1)
            selected_bbox = image_bboxes[random_index]
            selected_timestamp = timestamps[random_index]

            # Modify query_template, replace bbox and timestamp
            query_template = data_item["query_template"]
            human_query = query_template.copy()
            human_query["timestamps"] = selected_timestamp
            human_query["value"] = human_query["value"].replace(
                "<bbox>", str(selected_bbox["bbox"])
            )
            # f"Track the location and actions of the \"person\" at position {selected_bbox['bbox']} over time. Provide start and end timestamps for each instance in seconds with bounding box coordinates."

            # Generate GPT output (timestamps and bbox from 0 to t)
            gpt_output = {
                "from": "gpt",
                "value": "\n".join(
                    [
                        f"At {t}s, {bbox['bbox']}"
                        for t, bbox in zip(
                            timestamps[: random_index + 1],
                            image_bboxes[: random_index + 1],
                        )
                    ]
                ),
            }

            # Generate subsequent human and GPT data (for time after t)
            conversations = [human_query, gpt_output]
            for i, (image_file, timestamp) in enumerate(
                zip(image_files[random_index + 1 :], timestamps[random_index + 1 :])
            ):
                # Human query for video section
                human_query_after = {
                    "from": "human",
                    "timestamps": timestamp,
                    "image_file": image_file,
                    "value": "<video>\n",
                }
                # GPT response for video section
                gpt_response_after = {
                    "from": "gpt",
                    "value": f"At {timestamp}s, {image_bboxes[random_index+1+i]['bbox']}",
                }
                conversations.extend([human_query_after, gpt_response_after])
            data_item.update({"conversations": conversations})

        if "QA" in data_item:
            data_item["conversations"] = self.process_qa(data_item["QA"])
        # Ensure the first conversation contains a video placeholder
        for i in range(0, len(data_item["conversations"]), 2):
            data_item["conversations"][i]["value"] = data_item["conversations"][i][
                "value"
            ].replace("<image>", "<video>")
            if "<video>" not in data_item["conversations"][i]["value"]:
                data_item["conversations"][i]["value"] = (
                    "<video>\n" + data_item["conversations"][i]["value"]
                )

        if data_item.get("need_reset_timestamp", False):
            timestamps = [t - timestamps[0] for t in timestamps]

        # assert not data_item["need_reset_timestamp"], timestamps

        assert len(image_list) > 1
        # Generate special tokens for each video frame
        special_tokens = [
            f"Frame{i+1} at {round(timestamps[i], 1)}s: <image>"
            for i in range(len(image_list))
        ]
        start_index = 0
        for i in range(0, len(data_item["conversations"]), 2):
            if image_files is not None:
                image_file = data_item["conversations"][i].get("image_file", None)
                if image_file is not None and image_file not in image_file:
                    break

            data_item["conversations"][i]["timestamps"] = min(
                round(timestamps[-1], 1) + 0.1,
                data_item["conversations"][i]["timestamps"],
            )
            end = data_item["conversations"][i]["timestamps"]
            for end_index in range(start_index, len(timestamps)):
                if timestamps[end_index] > end:
                    break
            # end_index = min(random.randint(end_index, end_index+5), len(timestamps)-1)
            special_tokens_split = "\n".join(special_tokens[start_index:end_index])
            data_item["conversations"][i]["value"] = data_item["conversations"][i][
                "value"
            ].replace("<video>", special_tokens_split)
            start_index = end_index

        image_list = image_list[:end_index]

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)
        if num_patches <= 1:
            raise NotImplementedError

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        if self.num_image_token == 64:
            num_image_tokens = [self.num_image_token] * num_patches
        else:
            scale = VideoChatOnline_IT.tokens_arrange(
                num_patches,
                self.reverse_memory_sample_ratio,
                [2**s for s in range(len(self.reverse_memory_sample_ratio))],
            )
            num_image_tokens = [self.num_image_token // s**2 for s in scale]

        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
            num_patches=torch.tensor([num_patches], dtype=torch.long),
            is_video=torch.tensor([1], dtype=torch.long),
        )
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new("RGB", (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=1,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert (
            num_patches == 1
        ), f"The number of patches should be 1, but got {num_patches}."

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        ret = preprocess_function(
            self.template_name,
            [deepcopy(data_item["conversations"])],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=self.group_by_length,
            ds_name=self.ds_name,
        )

        # Create the final return dictionary
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
            num_patches=torch.tensor([num_patches], dtype=torch.long),
            is_video=torch.tensor([0], dtype=torch.long),
        )
        return ret

    def process_qa(self, qa, msg=""):
        # randomly shuffle qa for conversation
        if len(qa) > 1:
            random.shuffle(qa)

        conversation = list()
        # logger.info(f"origin qa {qa}")
        for _, sentence in enumerate(qa):
            i = sentence.get("i", "")
            q = sentence["q"]
            a = sentence["a"]
            user = i
            if q != "":
                user += " " + q
            else:
                # no question, often in caption dataset
                pass
            assistant = a
            conversation.append(
                {
                    "from": "human",
                    "value": user.strip(),
                }
            )
            conversation.append(
                {
                    "from": "gpt",
                    "value": assistant.strip(),
                }
            )
        conversation[0]["value"] = msg.rstrip() + " " + conversation[0]["value"]
        conversation[0]["value"] = conversation[0]["value"].strip()
        assert conversation[0]["from"] == "human"
        # logger.info(f"conversation {conversation}")
        return conversation

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        i = i % len(self.raw_data)
        while True:
            try:
                #    # signal.alarm(TIMEOUT)  # 开始计时
                data_item = self.raw_data[i]
                if "image" in data_item and len(data_item["image"]) != 0:
                    if type(data_item["image"]) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif (
                    "video" in data_item
                    and data_item["video"] is not None
                    and data_item["video"] != ""
                ):
                    if data_item.get("conversations", None) is not None:
                        if (
                            data_item["conversations"][0].get("timestamps", None)
                            is not None
                        ):
                            ret = self.online_video_get_item(data_item)
                        else:
                            ret = self.video_get_item(data_item)
                    elif data_item.get("query_template", None) is not None:
                        ret = self.online_video_get_item(data_item)
                    else:
                        ret = self.video_get_item(data_item)
                else:
                    # ret = self.pure_text_get_item(data_item)
                    raise NotImplementedError
                # signal.alarm(0)  # 成功处理后取消计时
                break
            # except TimeoutException:
            #    print(f'Timeout occurred while processing item {i} in dataset {self.ds_name}. Switching to another sample.', flush=True)
            #    i = random.randint(0, len(self.raw_data) - 1)

            except Exception as e:
                print(e, self.ds_name, flush=True)
                # if not isinstance(e, UnidentifiedImageError):
                #    traceback.print_exc()

                data_item = self.raw_data[i]
                if "image" in data_item:
                    if type(data_item["image"]) == list:
                        images = [self.root + item for item in data_item["image"]]
                        print(
                            f"Failed to load image: {images}, the dataset is: {self.ds_name}"
                        )
                    else:
                        if data_item["image"].startswith("s3://"):
                            data_path = self.root + data_item["image"]
                        else:
                            data_path = os.path.join(self.root, data_item["image"])
                        print(
                            f"Failed to load image: {data_path}, the dataset is: {self.ds_name}"
                        )
                elif "video" in data_item:
                    data_path = os.path.join(self.root, data_item["video"])
                    print(
                        f"Failed to load video: {data_path}, the dataset is: {self.ds_name}"
                    )
                i = random.randint(0, len(self.raw_data) - 1)
        return ret


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    normalize_type="imagenet",
    max_num_frame=3,  # for video data
    sampling_method="rand",  # for video data
):
    datasets = []
    lengths = []
    ds_collections = dict()
    for meta_path in data_args.meta_path:
        ds_collections.update(json.loads(open(meta_path).read()))
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        repeat_time = ds_collections[ds_name]["repeat_time"]
        if "max_dynamic_patch" in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]["max_dynamic_patch"]
            logger.info(
                f"max_dynamic_patch is set to {max_num} according to the meta file"
            )
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]["data_augment"],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            random_seed=ds_idx,
            max_num_frame=max_num_frame,  # for video data
            sampling_method=sampling_method,  # for video data
            reverse_memory_sample_ratio=data_args.reverse_memory_sample_ratio,  # for memory
        )
        logger.info(f"Add dataset: {ds_name} with length: {len(dataset)}")
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))
    if data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def main():
    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    # launcher = os.environ.get('LAUNCHER', 'pytorch')
    init_dist(launcher="pytorch", backend="nccl")
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f"Loading Tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader("~/petreloss.conf") if has_tcs_loader else None

    if model_args.model_name_or_path is not None:
        logger.info("Loading VideoChatOnline_IT...")
        config = InternVLChatConfig.from_pretrained(model_args.model_name_or_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == "internlm2":
            config.llm_config.attn_implementation = "flash_attention_2"  # for InternLM
            logger.info("Using flash_attention_2 for InternLM")
        else:
            config.llm_config._attn_implementation = "flash_attention_2"  # for LLaMA
            logger.info("Using flash_attention_2 for LLaMA")
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        config.reverse_memory_sample_ratio = data_args.reverse_memory_sample_ratio
        model = VideoChatOnline_IT.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            config=config,
            local_files_only=True,
        )

    model.img_context_token_id = img_context_token_id

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info("Loading pretrained MLP projector...")
        state_dict = torch.load(model_args.mlp_path, map_location="cpu")
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info("Finished")

    patch_size = model.config.vision_config.patch_size
    logger.info(f"model.config.force_image_size: {model.config.force_image_size}")
    logger.info(f"data_args.force_image_size: {data_args.force_image_size}")
    logger.info(
        f"model.config.vision_config.image_size: {model.config.vision_config.image_size}"
    )
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f"Resizing position embedding from "
            f"{model.config.vision_config.image_size} "
            f"to {data_args.force_image_size}..."
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio**2)
        * (data_args.avg_pooling_down_sample_ratio**2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()
    print(model)
    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        max_num_frame=data_args.max_num_frame,  # for video data
        sampling_method=data_args.sampling_method,  # for video data
    )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(
            r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora
        )
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(
            r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora
        )
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)
    else:
        model.mlp1.requires_grad = True

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f"Unfreezing ViT layer: {k}")
            v.requires_grad = True
    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    # Initialize our Trainer
    if model_args.use_custom_trainer:
        replace_create_optimizer()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=concat_pad_data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except:
            metrics["train_samples"] = -1

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
