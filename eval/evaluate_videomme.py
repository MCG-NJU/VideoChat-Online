import argparse
import itertools
import json
import os
import random
import time
from functools import partial

import cv2
import imageio
import numpy as np
from sympy import subsets
import torch
from decord import VideoReader, cpu
from internvl.model.videochat_online import (
    VideoChatOnline_Stream,
    VideoChatOnline_IT,
    InternVLChatConfig,
)
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
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
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import io
from internvl.model.videochat_online import (
    VideoChatOnline_IT,
    VideoChatOnline_Stream,
)
import os
from decord import VideoReader, cpu
import decord

decord.bridge.set_bridge("torch")


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_["pixel_values"] for _ in batches], dim=0)
    qa_list = [_["question"] for _ in batches]
    subtitle = [_["subtitle"] for _ in batches]
    # answers = [_["answer"] for _ in batches]
    num_patches_lists = [_["num_patches_list"] for _ in batches]
    task_types = [_["task_type"] for _ in batches]

    sec = [_["sec"] for _ in batches]
    return (
        subtitle[0],
        pixel_values,
        num_patches_lists[0],
        qa_list[0],
        task_types[0],
        sec[0],
    )


# import pysubs2
import re


def clean_text(text):
    cleaned_text = re.sub(r"[^A-Za-z0-9\s]\[\]", "", text)
    return cleaned_text


def read_vtt_and_concatenate(file_path, tokenizer, max_len=4096):
    subs = pysubs2.load(file_path, encoding="utf-8")

    prev = ""
    subtitles = []
    for caption in subs:
        # Split the caption text into individual lines
        lines = caption.text.split("\n")
        for line in lines:
            # Clean the text and check for repetition
            line = clean_text(line)
            if prev != line and line:
                subtitles.append(line)
                prev = line

    # Join subtitles to check length
    full_text = " ".join(subtitles)
    tokenized_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    # If the tokenized length is within the limit, return the full text
    if len(tokenized_ids) <= max_len:
        return full_text

    # Otherwise, we need to trim the text to fit within the limit
    # We will keep the first half and the last half
    half_len = max_len // 2
    start_text = " ".join(subtitles[:half_len])
    end_text = " ".join(subtitles[-half_len:])

    # Re-tokenize to ensure the total length is within the limit
    start_tokenized_ids = tokenizer(start_text, add_special_tokens=False).input_ids
    end_tokenized_ids = tokenizer(end_text, add_special_tokens=False).input_ids

    # Adjust the lengths to fit within the max_len
    while len(start_tokenized_ids) + len(end_tokenized_ids) > max_len:
        if len(start_tokenized_ids) > len(end_tokenized_ids):
            start_tokenized_ids.pop()
        else:
            end_tokenized_ids.pop(0)

    # Combine the adjusted parts
    adjusted_text = (
        tokenizer.decode(start_tokenized_ids)
        + " ... "
        + tokenizer.decode(end_tokenized_ids)
    )

    return adjusted_text


from torchvision.transforms import PILToTensor


class MME_dataset(Dataset):
    def __init__(
        self,
        data_prefix="VideoMME_0629/processed_1fps",
        subtitle_prefix="/mnt/petrelfs/share_data/likunchng/videomme/subtitle_0629",
        anno_path="videomme/Video-MME_0629.json",
        frame_dict_path="video_mme_1fps.json",
        num_segments=16,
        stride=0,  # if stride >= 1, will return all frames according to FPS (1/stride), else return partial frames
        resolution=224,
        hd_num=6,
        max_subtitle_len=4096,  # max_tokens for subtitle
    ):
        self.data_prefix = data_prefix
        self.subtitle_prefix = subtitle_prefix
        with open(anno_path, "r") as f:
            self.data_list = json.load(f)  # [:150]
        # with open(frame_dict_path, 'r') as f:
        #    self.frame_dict = json.load(f)
        random.seed(42)
        random.shuffle(self.data_list)
        self.hd_num = hd_num
        self.num_segments = num_segments
        self.stride = stride
        self.resolution = resolution
        self.max_subtitle_len = max_subtitle_len

        self.transform = build_transform(is_train=False, input_size=448)

    def __len__(self):
        return len(self.data_list)

    def encode_video(self, video_path):
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        sample_fps = max(round(vr.get_avg_fps() / 4), 1)  # FPS
        frame_idx = [i for i in range(0, len(vr), sample_fps)]
        if len(frame_idx) > args.num_segments:
            frame_idx = uniform_sample(frame_idx, args.num_segments)
        frames = vr.get_batch(frame_idx).numpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        print("num frames:", len(frames))
        sec = [f"{i[1]:.1f}" for i in vr.get_frame_timestamp(frame_idx)]
        return frames, sec, round(len(vr) / vr.get_avg_fps(), 1)

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        # question += "Options:\n"
        answer = data["answer"]
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data["options"]):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]["videoID"]
        video_name = os.path.join(self.data_prefix, "videos", video_name)
        image_list, sec, length = self.encode_video(video_name + ".mp4")
        raw_images = []
        num_patches_list = []
        pixel_values = []
        for image in image_list:
            raw_images.append(image)

            patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)
        duration_category = self.data_list[idx]["duration"]
        qa_list = []
        for qa in self.data_list[idx]["questions"]:
            qa_list.append(self.qa_template(qa))

        subtitle = f"This entire video lasts for {length} seconds.\n"
        try:
            subtitle_path = os.path.join(self.subtitle_prefix, video_name + ".srt")
            if os.path.exists(subtitle_path):
                subtitle = read_vtt_and_concatenate(
                    subtitle_path, model.mistral_tokenizer, self.max_subtitle_len
                )
        except Exception:
            subtitle = ""
            print(f"Error for {subtitle_path}")

        return {
            "question": qa_list,
            "pixel_values": pixel_values,
            "subtitle": subtitle,
            "num_patches_list": num_patches_list,
            "task_type": duration_category,
            "sec": sec,
        }


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(
            size, self._world_size, self._rank
        )

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


import os


def remove_extra_open_parens(s):
    # Pattern to match '(Answer:' (optional) followed by one or more '('
    pattern = r"^(\(Answer:)?\s*\(+"
    match = re.match(pattern, s)
    if match:
        # Reconstruct the string, keeping one '(' and removing the rest
        s = "(" + s[match.end() :]
    return s


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
        "Answer:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "").strip()

    matches = re.search(r"[ABCD]", s)
    assert matches is not None

    return matches[0]


def evaluate_chat_model():

    system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"  # newPrompt2
    
    question_prompt = "\nAnswer:  \n"
    instruct = ""
    # question_prompt = "\nAnswer with the option's letter from the given choices directly and only give the best option. The best answer is: "
    answer_prompt = ""
    return_prompt = "("
    stride = -1
    resolution = 336
    hd_num = 6
    max_subtitle_len = 8192

    data_prefix = "/workspace/data/datasets/Video-MME"
    anno_path = "/workspace/data/datasets/Video-MME/Video-MME_0629.json"
    frame_dict_path = "mvbench/video_mme_1fps.json"
    dataset = MME_dataset(
        data_prefix=data_prefix,
        anno_path=anno_path,
        frame_dict_path=frame_dict_path,
        num_segments=16,
        stride=stride,
        resolution=resolution,
        hd_num=hd_num,
        max_subtitle_len=max_subtitle_len,
    )
    #dataset = torch.utils.data.Subset(dataset, list(range(600, 900)))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        sampler=InferenceSampler(len(dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )
    generation_config = dict(
        num_beams=1,
        max_new_tokens=20,
        min_new_tokens=1,
        do_sample=False,
    )

    correct = 0
    total = 0
    res_list = []
    acc_dict = {}
    outputs = []
    model.system_message = system_prompt
    for idx, example in enumerate(tqdm(dataloader)):
        subtitle, pixel_values, num_patches_list, qa_list, duration_category, sec = (
            example
        )

        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        if duration_category not in acc_dict:
            acc_dict[duration_category] = [0, 0]  # correct, total
        qa_count = len(qa_list)
        acc_dict[duration_category][1] += qa_count
        total += qa_count
        for idx, qa in enumerate(qa_list):
            print(f"----------qa_{idx}---------", flush=True)

            # if subtitle != "":
            #    subtitle = f"This video's subtitles are listed below: {subtitle}"
            question = instruct + qa[0] + question_prompt

            pred = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                verbose=True,
                add_generation_prompt=answer_prompt,
                timestamps=None,
            )
            # pred = remove_extra_open_parens(return_prompt + pred.strip().split("\n")[0])

            outputs.append(
                {
                    "question": qa[0],
                    "pred": pred,
                    "gt": qa[1],
                    "task_type": duration_category,
                }
            )
    torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:

        print(f"Evaluating MVBench ...")
        correct, total, acc_dict = 0, 0, {}
        for item in merged_outputs:
            task_type = item["task_type"]
            pred = item["pred"]
            gt = item["gt"]
            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]  # correct, total
            acc_dict[task_type][1] += 1
            total += 1

            print(extract_characters_regex(pred), extract_characters_regex(gt))
            if extract_characters_regex(pred) == extract_characters_regex(gt):
                acc_dict[task_type][0] += 1
                correct += 1

        final_res = {}
        for k, v in acc_dict.items():
            final_res[k] = v[0] / v[1] * 100
        final_res["Avg"] = correct / total * 100
        final_res["total"] = total
        print(final_res)

        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"VideoMME_f{args.num_segments}_{time_prefix}"
        output_path = os.path.join(args.out_dir, results_file)
        with open(f"{output_path}.json", "w") as f:
            json.dump(merged_outputs, f)
        with open(f"{output_path}_result_final.json", "w") as f:
            json.dump(final_res, f)
        print("Results saved to {}".format(output_path))


def check_ans(pred, gt):
    flag = False
    pred = pred.replace("Answer: ", "")

    pred_list = pred.lower().split(" ")
    pred_option, pred_content = pred_list[0], " ".join(pred_list[1:])
    gt_list = gt.lower().split(" ")
    gt_option, gt_content = gt_list[0], " ".join(gt_list[1:])
    if gt_content[-1] == ".":
        gt_content = gt_content[:-1]

    # print(pred_option, gt_option)
    if pred_option.replace(".", "") in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


from datetime import timedelta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--memory_bank", type=int, default=None)
    parser.add_argument("--datasets", type=str, default="mvbench")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--max-num", type=int, default=6)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--num_segments", type=int, default=16)

    parser.add_argument("--short_memory_bank", type=int, default=64)
    parser.add_argument("--mid_memory_bank", type=int, default=64)
    parser.add_argument("--long_memory_bank", type=int, default=64)

    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)
    assert args.batch_size == 1, "Only batch size 1 is supported"

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
        timeout=timedelta(seconds=3000),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, add_eos_token=False, trust_remote_code=True, use_fast=False
    )

    tokenizer.model_max_length = 8192 * 10000
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    config = InternVLChatConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = (
        VideoChatOnline_Stream.from_pretrained(
            args.checkpoint,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    model.long_bank = args.long_memory_bank
    model.mid_bank = args.mid_memory_bank
    model.short_bank = args.short_memory_bank
    print(model)
    model.img_context_token_id = img_context_token_id
    image_size = model.config.force_image_size or model.config.vision_config.image_size

    use_thumbnail = model.config.use_thumbnail

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    if total_params > 20 or args.dynamic:
        args.num_beams = 1
        print(f"[test] total_params: {total_params}B, use num_beams: {args.num_beams}")
    else:
        print(f"[test] total_params: {total_params}B")
    print(f"[test] image_size: {image_size}")
    print(f"[test] template: {model.config.template}")
    print(f"[test] dynamic_image_size: {args.dynamic}")
    print(f"[test] use_thumbnail: {use_thumbnail}")
    print(f"[test] max_num: {args.max_num}")

    evaluate_chat_model()
