import argparse
import itertools
import json
import os
import pprint
import random
import re
import sys
import time
from functools import partial
import cv2
import imageio
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from internvl.model.videochat_online import (
    VideoChatOnline_Stream,
    VideoChatOnline_IT,
    InternVLChatConfig,
)
from collections import defaultdict
import pandas as pd
import os
import json
import time
import torch
import itertools
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
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


class MME_dataset(Dataset):
    def __init__(
        self,
        data_prefix=None,
        anno_path=None,
        num_segments=16,
    ):
        self.data_prefix = data_prefix
        with open(anno_path, "r") as f:
            self.data_list = json.load(f)

        self.num_segments = num_segments
        self.transform = build_transform(is_train=False, input_size=448)
        random.seed(42)
        random.shuffle(self.data_list)
        # Set random seed for shuffle data and load balancing across GPUs

    def __len__(self):
        return len(self.data_list)

    def encode_video(self, video_path, sample_fps, clip):
        # Load video using decord reader
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        
        # Get timestamps for each frame
        frame_timestamps = vr.get_frame_timestamp(range(len(vr)))
        
        # Extract start and end times from clip
        start_time, end_time = clip
        
        # Find start frame index
        start_frame = next(
            i for i, ts in enumerate(frame_timestamps) if ts[1] >= start_time
        )
        
        # Find end frame index
        try:
            end_frame = (
                next(i for i, ts in enumerate(frame_timestamps) if ts[1] > end_time) - 1
            )
        except StopIteration:
            end_frame = len(frame_timestamps) - 1
        
        # Generate frame indices with sampling
        frame_idx = [
            i
            for i in range(
                start_frame,
                end_frame + 1,
                max(1, int(round(vr.get_avg_fps() / sample_fps, 0))),
            )
        ]
        
        # Extract frames and convert to PIL Images
        frames = vr.get_batch(frame_idx).numpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]
        
        # Get corresponding timestamps
        sec = [frame_timestamps[i][1] for i in frame_idx]
        
        return frames, sec

    def encode_frame(self, video_path, video_fps, sample_fps, clip):
        # Count total frames in directory
        num_frames = len(os.listdir(video_path))
        
        # Calculate frame indices from clip times
        start_time, end_time = clip
        end_frame = min(num_frames - 1, int(end_time * video_fps))
        start_frame = max(0, int(start_time * video_fps))
        
        # Generate sampled frame indices
        frame_idx = [
            i
            for i in range(
                start_frame, end_frame + 1, max(1, int(video_fps) // sample_fps)
            )
        ]
        
        # Load images and calculate timestamps
        img_files = os.listdir(video_path)
        sec = [round(i / video_fps, 2) for i in frame_idx]
        frames = [Image.open(os.path.join(video_path, img_files[i])) for i in frame_idx]
        
        return frames, sec

    def qa_template(self, data):
        timestamp = data["middle_frame_timestamp"]
        question = f"Question at {timestamp}s: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        answer = f"({answer}) {data['options'][ord(answer) - ord('A')][3:]}"
        for idx, c in enumerate(data["options"]):
            cur_choice, cur_text = c[0], c[3:]
            question += f"({cur_choice}) {cur_text}\n"
        question = question.rstrip()
        return question, answer

    def __getitem__(self, idx):
        video_name = self.data_list[idx]["video_id"]
        video_name = os.path.join(self.data_prefix, video_name)
        assert os.path.exists(video_name)
        if os.path.isdir(video_name):
            return self.frame_getitem(idx, video_name)
        else:
            return self.video_getitem(idx, video_name)

    def frame_getitem(self, idx, video_name):
        video_fps = self.data_list[idx]["fps"] 

        image_list, sec = self.encode_frame(
            video_name,
            video_fps=video_fps,
            sample_fps=args.fps,
            clip=self.data_list[idx]["clip"],
        )
        raw_images = []
        num_patches_list = []
        pixel_values = []
        for image in image_list:
            raw_images.append(image)

            patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)

        duration_category = [] 
        qa_list = []
        prev_idx, cur_idx = 0, 0
        for question in self.data_list[idx]["questions"]:
            question_t = question["middle_frame_timestamp"]

            for i, t in enumerate(sec):
                cur_idx = i
                if question_t < t:
                    break

            qa_list.append(
                {
                    "question": self.qa_template(question),
                    "question_t": question_t,
                    "idx": [prev_idx, cur_idx],
                    "task_type": question["answer_type"],
                    "sub_task_type": question["sub_answer_type"],
                }
            )
            duration_category.append(question["answer_type"])
            prev_idx = cur_idx
        qa_list[-1]["idx"][1] = len(sec)

        sec = [f"{i:.1f}" for i in sec]
        subtitle = f"The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds.\n"
        return {
            "question": qa_list,
            "pixel_values": pixel_values,
            "subtitle": subtitle,
            "num_patches_list": num_patches_list,
            "task_type": duration_category,
            "sec": sec,
        }

    def video_getitem(self, idx, video_name):
        image_list, sec = self.encode_video(
                video_name,
                sample_fps=args.fps,
                clip=self.data_list[idx]["clip"],
        )

        raw_images = []
        num_patches_list = []
        pixel_values = []
        for image in image_list:
            raw_images.append(image)

            patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)

        answer_type = []
        qa_list = []

        prev_idx, cur_idx = 0, 0
        for question in self.data_list[idx]["questions"]:
            question_t = question["middle_frame_timestamp"]

            for i, t in enumerate(sec):
                cur_idx = i
                if question_t < t:
                    break

            qa_list.append(
                {
                    "question": self.qa_template(question),
                    "question_t": question_t,
                    "idx": [prev_idx, cur_idx],
                    "task_type": question["answer_type"],
                    "sub_task_type": question["sub_answer_type"],
                }
            )
            answer_type.append(question["answer_type"])
            prev_idx = cur_idx
        qa_list[-1]["idx"][1] = len(sec)

        sec = [f"{i:.1f}" for i in sec]
        subtitle = f"The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds.\n"


        return {
            "question": qa_list,
            "pixel_values": pixel_values,
            "subtitle": subtitle,
            "num_patches_list": num_patches_list,
            "task_type": answer_type,
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

def mcq_acc(answer, pred):
    periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
    commaStrip = re.compile("(\d)(\,)(\d)")
    punct = [";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_", "-", ">", "<", "@", "`", ",", "?", "!"]

    def processPunctuation(inText):
        outText = inText
        for p in punct:
            if (p + " " in inText or " " + p in inText) or (re.search(commaStrip, inText) != None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = periodStrip.sub("", outText, re.UNICODE)
        return outText

    def process(answer):
        option_regex = re.compile(r"^([A-E])\.\s*(.+)$", re.IGNORECASE)
        match = option_regex.match(answer.strip())

        if match:
            # If matched, return the option letter in uppercase
            return match.group(1).upper()
        else:
            # If no match, process the answer as before
            answer = answer.replace("\n", " ")
            answer = answer.replace("\t", " ")
            answer = answer.strip()
            answer = processPunctuation(answer)
            answer = answer.strip("'")
            answer = answer.strip('"')
            answer = answer.strip(")")
            answer = answer.strip("(")
            answer = answer.strip().lower()

            # Try to find any single letter (A-E) in the processed answer
            letter_match = re.search(r"\b([A-E])\b", answer, re.IGNORECASE)
            if letter_match:
                return letter_match.group(1).upper()

            return answer

    pred = process(pred)
    answer = process(answer)

    if pred == answer:
        score = 1
    else:
        score = 0

    return score


def evaluate_chat_model():

    system_prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"  # newPrompt2
    question_prompt = "\nAnswer:"
    answer_prompt = ""

    data_prefix = args.data_root 
    anno_path = args.anno_root
    dataset = MME_dataset(
        data_prefix=data_prefix,
        anno_path=anno_path,
        num_segments=args.num_segments,
    )
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
        max_new_tokens=128,
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
        subtitle, pixel_values, num_patches_list, qa_list, duration_categorys, sec = (
            example
        )

        pixel_values = pixel_values.to(torch.bfloat16).cuda()

        if args.time:
            special_image_tokens = [
                f"Frame{i+1} at {round(float(sec[i]), 1)}s: <image>"
                for i in range(len(pixel_values))
            ]
        else:
            special_image_tokens = [
                "Frame{}: <image>".format(i + 1) for i in range(len(pixel_values))
            ]

        for idx, qa_sample in enumerate(qa_list):
            print(f"----------qa_{idx}---------", flush=True)
            qa, clip = qa_sample["question"], qa_sample["idx"]

            question = qa[0] + question_prompt

            pred, history = model.chat(
                    tokenizer,
                    pixel_values[: clip[1]],
                    question,
                    generation_config,
                    num_patches_list=num_patches_list[: clip[1]],
                    history=None,
                    return_history=True,
                    verbose=True,
                    add_generation_prompt=answer_prompt,
                    timestamps=special_image_tokens[0 : clip[1]],
            )
            #print(qa_sample)
            outputs.append(
                {
                    #"question": qa[0],
                    "score": mcq_acc(qa[1], pred),
                    #"pred": pred,
                    #"gt": qa[1],
                    "task_type": qa_sample["task_type"],
                    "sub_task_type": qa_sample["sub_task_type"],  # 添加 sub_task_type
                }
            )
            #break
        #break
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    serialized_outputs = json.dumps(outputs)
    torch.distributed.all_gather_object(merged_outputs, serialized_outputs, group=torch.distributed.group.WORLD)
    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:

        print(f"Evaluating OVBench ...")
        acc_dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # 按 task_type -> sub_task_type 统计

        for item in merged_outputs:
            task_type = item["task_type"]
            sub_task_type = item["sub_task_type"]
            #pred = item["pred"]
            #gt = item["gt"]

            acc_dict[task_type][sub_task_type][1] += 1  # 总数 +1
            acc_dict[task_type][sub_task_type][0] += item["score"]
            #if mcq_acc(gt, pred) == 1:
            #      # 正确数 +1

        # 计算准确率
        final_res = {}
        sub_task_acc = defaultdict(lambda: [0, 0])  # {sub_task: [total_correct, total_count]}

        total_sample = 0
        for task, sub_tasks in acc_dict.items():
            for sub_task, (correct, total) in sub_tasks.items():
                acc = correct / total * 100
                final_res[(task, sub_task)] = acc
                total_sample += total
                #sub_task_acc[sub_task][0] += acc  # 累加准确率
                #sub_task_acc[sub_task][1] += total  # 计数

        # 计算按 sub_task_type 计算的平均准确率
        #avg_sub_task_acc = sum(acc_sum  for acc_sum, count in sub_task_acc.values()) / len(sub_task_acc)
        final_res["Avg"] = sum(final_res.values()) / len(final_res.values())#avg_sub_task_acc
        final_res["total"] = total_sample #sum(count for acc_sum, count in sub_task_acc.values())
        print(final_res)

        # 保存 JSON 结果
        results_file = f"{args.dataset}"
        output_path = os.path.join(args.out_dir, results_file)

        # 生成 CSV
        # 提取所有 task_type 和 sub_task_type
        task_types = []
        sub_task_types = []
        for task in final_res.keys():
            if isinstance(task, tuple):
                task_types.append(task[0])
                sub_task_types.append(task[1])
        task_types = sorted(set(task_types))
        sub_task_types = sorted(set(sub_task_types))
        #task_types = sorted(set(task[0] for task in final_res.keys() if isinstance(task, tuple)))
        #sub_task_types = sorted(set(subtask[1] for subtask in final_res.keys() if isinstance(subtask, tuple)))

        # 生成 DataFrame 结构
        data = []
        columns = []

        for task in task_types:
            for sub_task in sub_task_types:
                if (task, sub_task) in final_res:
                    data.append(final_res[(task, sub_task)])
                    columns.append((task, sub_task))  # 形成 (task_type, sub_task_type) 结构

        # 创建 DataFrame
        df = pd.DataFrame([data], columns=pd.MultiIndex.from_tuples(columns))
        df.loc["Avg"] = df.mean(axis=0, skipna=True)
        df.to_csv(f"{output_path}_result_final.csv", index=False, float_format="%.2f")

        print("Results saved to {}".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="mvbench")

    parser.add_argument("--dataset", type=str, default="mvbench")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--anno_root", type=str, default="")

    parser.add_argument("--short_memory_bank", type=int, default=64)
    parser.add_argument("--mid_memory_bank", type=int, default=64)
    parser.add_argument("--long_memory_bank", type=int, default=64)

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
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--num_segments", type=int, default=16)
    parser.add_argument("--slide_window", type=int, default=2 * 60)
    parser.add_argument("--time", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.datasets = args.datasets.split(",")
    print("datasets:", args.datasets)
    assert args.batch_size == 1, "Only batch size 1 is supported"

    torch.distributed.init_process_group(
        backend="nccl",
        world_size=int(os.getenv("WORLD_SIZE", "1")),
        rank=int(os.getenv("RANK", "0")),
    )

    torch.cuda.set_device(int(os.getenv("LOCAL_RANK", 0)))

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, add_eos_token=False, trust_remote_code=True, use_fast=False
    )

    tokenizer.model_max_length = 8192 * 4
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    from transformers import AutoTokenizer, AutoConfig

    config = InternVLChatConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
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
