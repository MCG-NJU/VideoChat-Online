import argparse
import itertools
import json
import os
import re
from functools import partial
import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from internvl.model.videochat_online import (
    VideoChatOnline_IT,
    InternVLChatConfig,
)
import decord
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# from internvl.model.internvl_chat import InternVLChatModel_IT
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

decord.bridge.set_bridge("torch")

def uniform_sample(l, n):
    idxs = np.linspace(0, len(l) - 1, n, endpoint=True).astype(np.int32)
    return [l[i] for i in idxs]

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


class OVBenchDataset(Dataset):
    def __init__(
        self,
        data_prefix="",
        anno_path="",
        num_segments=16,
    ):
        self.data_prefix = data_prefix
        with open(anno_path, "r") as f:
            self.data_list = json.load(f)  # [:150]
        self.num_segments = num_segments

        self.transform = build_transform(is_train=False, input_size=448)

    def __len__(self):
        return len(self.data_list)

    def encode_video(self, video_path, fps, max_num_frames, timestamp, sliding_window):

        # Load the video
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)

        # Get the frame index corresponding to the timestamp
        frame_timestamps = vr.get_frame_timestamp(range(len(vr)))
        # Get the frame index corresponding to the timestamp
        try:
            end_frame = (
                next(i for i, ts in enumerate(frame_timestamps) if ts[0] > timestamp)
                - 1
            )
        except StopIteration:
            end_frame = len(frame_timestamps) - 1  # 设置为最后一个帧索引

        # Calculate start_frame by sliding_window duration in seconds

        if "AVA_Actions" in self.data_prefix:
            start_time = max(timestamp - sliding_window, 900)
        else:
            start_time = max(timestamp - sliding_window, 0)
        try:
            start_frame = next(
                i for i, ts in enumerate(frame_timestamps) if ts[1] >= start_time
            )
        except StopIteration:
            start_frame = 0

        # Sample frames from start_frame to end_frame with the specified fps
        frame_idx = list(
            reversed(
                [
                    i
                    for i in range(
                        end_frame,
                        start_frame - 1,
                        -1 * max(1, int(round(vr.get_avg_fps() / fps, 0))),
                    )
                ]
            )
        )

        # Ensure that the frame corresponding to the timestamp is included
        # If there are more frames than needed, sample them uniformly
        if len(frame_idx) > max_num_frames:
            frame_idx = uniform_sample(frame_idx, max_num_frames)

        frames = vr.get_batch(frame_idx).numpy()
        frames = [Image.fromarray(v.astype("uint8")) for v in frames]

        print("num frames:", len(frames))

        sec = [f"{i[1]:.1f}" for i in vr.get_frame_timestamp(frame_idx)]

        return frames, sec

    def encode_frame(
        self,
        video_path,
        video_fps,
        sample_fps,
        max_num_frames,
        timestamp,
        sliding_window,
    ):

        num_frames = len(os.listdir(video_path))

        # Get the frame index corresponding to the timestamp
        end_frame = min(num_frames - 1, int(timestamp * video_fps))

        # Calculate start_frame by sliding_window duration in seconds
        start_frame = max(0, end_frame - int(sliding_window * video_fps))

        # Sample frames from start_frame to end_frame with the specified fps
        frame_idx = list(
            reversed(
                [
                    i
                    for i in range(
                        end_frame,
                        start_frame - 1,
                        -1 * max(1, int(round(video_fps / sample_fps, 0))),
                    )
                ]
            )
        )

        # Ensure that the frame corresponding to the timestamp is included
        # If there are more frames than needed, sample them uniformly
        if len(frame_idx) > max_num_frames:
            frame_idx = uniform_sample(frame_idx, max_num_frames)

        img_files = os.listdir(video_path)
        sec = [f"{round(i / video_fps, 2):.1f}" for i in frame_idx]
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
        if os.path.isdir(video_name):
            return self.frame_getitem(idx)
        else:
            return self.video_getitem(idx)

    def frame_getitem(self, idx):
        video_name = self.data_list[idx]["video_id"]
        time_stamp = self.data_list[idx]["middle_frame_timestamp"]
        video_fps = self.data_list[idx]["fps"]
        video_name = os.path.join(self.data_prefix, video_name)

        image_list, sec = self.encode_frame(
            video_name,
            video_fps=video_fps,
            sample_fps=args.fps,
            max_num_frames=args.num_segments,
            timestamp=time_stamp,
            sliding_window=args.slide_window,
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

        duration_category = self.data_list[idx]["answer_type"]
        qa_list = []

        qa_list.append(self.qa_template(self.data_list[idx]))

        subtitle = f"The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds.\n"

        return {
            "question": qa_list,
            "pixel_values": pixel_values,
            "subtitle": subtitle,
            "num_patches_list": num_patches_list,
            "task_type": duration_category,
            "sec": sec,
        }

    def video_getitem(self, idx):
        video_name = self.data_list[idx]["video_id"]
        time_stamp = self.data_list[idx]["middle_frame_timestamp"]
        video_name = os.path.join(self.data_prefix, video_name)
        video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        if len(video_name.split(".")) == 1:
            for fmt in video_formats:  # Added this line
                if os.path.exists(f"{video_name}{fmt}"):
                    video_name = f"{video_name}{fmt}"
                    break
        try:
            image_list, sec = self.encode_video(
                video_name,
                fps=args.fps,
                max_num_frames=args.num_segments,
                timestamp=time_stamp,
                sliding_window=args.slide_window,
            )
        except Exception as e:
            assert False, (e, video_name)
        raw_images = []
        num_patches_list = []
        pixel_values = []
        for image in image_list:
            raw_images.append(image)

            patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)

        duration_category = self.data_list[idx]["answer_type"]
        qa_list = []

        qa_list.append(self.qa_template(self.data_list[idx]))

        subtitle = f"The video contains {len(sec)} frames sampled at {', '.join(sec)} seconds.\n"

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
    model.system_message = system_prompt
    question_prompt = "\nOnly give the best option."
    answer_prompt = "Best option:("
    return_prompt = "("

    data_prefix = args.data_root  # "/workspace/data/datasets/AVA_Actions/raw/trainval"
    anno_path = args.anno_root  # "/workspace/data/hzp/InternVL2/bench/AVA/AVA.json"
    dataset = OVBenchDataset(
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
        max_new_tokens=16,
        min_new_tokens=1,
        do_sample=False,
    )

    correct = 0
    total = 0
    acc_dict = {}
    outputs = []
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

        if args.time:
            special_image_tokens = "\n".join(
                [
                    f"Frame{i+1} at {round(float(sec[i]), 1)}s: <image>"
                    for i in range(len(pixel_values))
                ]
            )
        else:
            special_image_tokens = "\n".join(
                ["Frame{}: <image>".format(i + 1) for i in range(len(pixel_values))]
            )

        for idx, qa in enumerate(qa_list):
            print(f"----------qa_{idx}---------", flush=True)

            question = (
                special_image_tokens
                + "\n"
                + (subtitle if not args.time else "")
                + qa[0]
                + question_prompt
            )
            pred = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                verbose=True,
            )

            outputs.append(
                {
                    "question": qa[0],
                    "pred": extract_characters_regex(pred),
                    "gt": extract_characters_regex(qa[1][1]),
                    "task_type": duration_category,
                }
            )
            torch.cuda.empty_cache()
            
    torch.distributed.barrier()
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, json.dumps(outputs))

    merged_outputs = [json.loads(_) for _ in merged_outputs]
    merged_outputs = [_ for _ in itertools.chain.from_iterable(merged_outputs)]

    if torch.distributed.get_rank() == 0:

        print(f"Evaluating OVBench ...")
        correct, total, acc_dict = 0, 0, {}
        for item in merged_outputs:
            task_type = item["task_type"]
            pred = item["pred"]
            gt = item["gt"]
            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]  # correct, total
            acc_dict[task_type][1] += 1
            total += 1

            if pred == gt:
                acc_dict[task_type][0] += 1
                correct += 1

        final_res = {}
        for k, v in acc_dict.items():
            final_res[k] = v[0] / v[1] * 100
        final_res["Avg"] = correct / total * 100
        final_res["total"] = total
        print(final_res)

        # time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"{args.dataset}_fps{args.fps}_f{args.num_segments}_window{args.slide_window}"
        output_path = os.path.join(args.out_dir, results_file)
        with open(f"{output_path}.json", "w") as f:
            json.dump(outputs, f)
        with open(f"{args.dataset}_result_final.json", "w") as f:
            json.dump(final_res, f)

        df = pd.DataFrame([final_res])  # 将字典转为 DataFrame
        df = df[sorted(df.columns)]
        df.to_csv(f"{output_path}_result_final.csv", index=False, float_format="%.2f")
        print("Results saved to {}".format(output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="mvbench")

    parser.add_argument("--dataset", type=str, default="mvbench")
    parser.add_argument("--data_root", type=str, default="")
    parser.add_argument("--anno_root", type=str, default="")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
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
    from transformers import AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=False,
    )

    tokenizer.model_max_length = 8192 * 4
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    config = InternVLChatConfig.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )
    model = (
        VideoChatOnline_IT.from_pretrained(
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
