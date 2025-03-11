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
import torch
from decord import VideoReader, cpu
from transformers import AutoTokenizer, AutoModel
from internvl.model.videochat_online import (
    VideoChatOnline_Stream,
    VideoChatOnline_IT,
    InternVLChatConfig,
)
from internvl.train.dataset import build_transform, dynamic_preprocess
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

data_root = "/workspace/data/datasets/videochat2/MVBench/video"
data_list = {
    "Action Sequence": (
        "action_sequence.json",
        f"{data_root}/star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Action Prediction": (
        "action_prediction.json",
        f"{data_root}/star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Action Antonym": (
        "action_antonym.json",
        f"{data_root}/ssv2_video/",
        "video",
        False,
    ),
    "Fine-grained Action": (
        "fine_grained_action.json",
        f"{data_root}/Moments_in_Time_Raw/videos/",
        "video",
        False,
    ),
    "Unexpected Action": (
        "unexpected_action.json",
        f"{data_root}/FunQA_test/test/",
        "video",
        False,
    ),
    "Object Existence": (
        "object_existence.json",
        f"{data_root}/clevrer/video_validation/",
        "video",
        False,
    ),
    "Object Interaction": (
        "object_interaction.json",
        f"{data_root}/star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Object Shuffle": (
        "object_shuffle.json",
        f"{data_root}/perception/videos/",
        "video",
        False,
    ),
    "Moving Direction": (
        "moving_direction.json",
        f"{data_root}/clevrer/video_validation/",
        "video",
        False,
    ),
    "Action Localization": (
        "action_localization.json",
        f"{data_root}/sta/sta_video/",
        "video",
        True,
    ),  # has start & end
    "Scene Transition": (
        "scene_transition.json",
        f"{data_root}/scene_qa/video/",
        "video",
        False,
    ),
    "Action Count": (
        "action_count.json",
        f"{data_root}/perception/videos/",
        "video",
        False,
    ),
    "Moving Count": (
        "moving_count.json",
        f"{data_root}/clevrer/video_validation/",
        "video",
        False,
    ),
    "Moving Attribute": (
        "moving_attribute.json",
        f"{data_root}/clevrer/video_validation/",
        "video",
        False,
    ),
    "State Change": (
        "state_change.json",
        f"{data_root}/perception/videos/",
        "video",
        False,
    ),
    "Fine-grained Pose": (
        "fine_grained_pose.json",
        f"{data_root}/nturgbd/",
        "video",
        False,
    ),
    "Character Order": (
        "character_order.json",
        f"{data_root}/perception/videos/",
        "video",
        False,
    ),
    "Egocentric Navigation": (
        "egocentric_navigation.json",
        f"{data_root}/vlnqa/",
        "video",
        False,
    ),
    "Episodic Reasoning": (
        "episodic_reasoning.json",
        f"{data_root}/tvqa/frames_fps3_hq/",
        "frame",
        True,
    ),  # has start & end, read frame
    "Counterfactual Inference": (
        "counterfactual_inference.json",
        f"{data_root}/clevrer/video_validation/",
        "video",
        False,
    ),
}

import os

data_dir = "/workspace/data/datasets/videochat2/MVBench/json"
# from petrel_client.client import Client
from decord import VideoReader, cpu

# client = Client('~/petreloss.conf', enable_mc=False)


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_["pixel_values"] for _ in batches], dim=0)
    questions = [_["question"] for _ in batches]
    answers = [_["answer"] for _ in batches]
    num_patches_lists = [_["num_patches_list"] for _ in batches]
    task_types = [_["task_type"] for _ in batches]
    return pixel_values, questions, answers, num_patches_lists, task_types


class MVBenchDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_dir,
        data_list,
        prompt,
        question_prompt,
        num_segments=16,
        input_size=224,
        dynamic_image_size=False,
        use_thumbnail=False,
        max_num=6,
    ):
        self.data_list = []
        for k, v in data_list.items():
            with open(os.path.join(data_dir, v[0]), "r") as f:
                json_data = json.load(f)
            for data in json_data:
                self.data_list.append(
                    {
                        "task_type": k,
                        "prefix": v[1],
                        "data_type": v[2],
                        "bound": v[3],
                        "data": data,
                    }
                )
        random.seed(42)
        random.shuffle(self.data_list)
        self.decord_method = {
            "video": self.read_video,
            "gif": self.read_gif,
            "frame": self.read_frame,
        }
        self.prompt = prompt
        self.question_prompt = question_prompt
        self.input_size = input_size
        self.num_segments = num_segments
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.data_list)

    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data["task_type"] not in len_list:
                len_list[data["task_type"]] = 0
            len_list[data["task_type"]] += 1
            if data["task_type"] not in option_list:
                option_list[data["task_type"]] = 0
            option_list[data["task_type"]] += len(data["data"]["candidates"])

        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k] / option_list[k] * 100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct / total * 100:.2f}%"
        return res.rstrip()

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        start_idx = min(start_idx, max_frame)

        end_idx = min(round(end * fps), max_frame)
        assert start_idx <= end_idx, (start, end, start_idx, end_idx, max_frame)
        frame_indices = [
            i for i in range(start_idx, end_idx + 1, max(round(fps / 4), 1))
        ]

        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        MAX_NUM_FRAMES = self.num_segments
        if len(frame_indices) > MAX_NUM_FRAMES:
            frame_indices = uniform_sample(frame_indices, MAX_NUM_FRAMES)
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)

        sec = [f"{round(i / fps, 2):.1f}" for i in frame_indices]
        return images_group, sec

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)

        sec = [f"{round(i / fps, 2):.1f}" for i in frame_indices]

        return images_group, sec

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(
            bound, fps, max_frame, first_idx=1
        )  # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f"{frame_index:05d}.jpg"))
            images_group.append(img)

        sec = [f"{round(i / fps, 2):.1f}" for i in frame_indices]
        return images_group, sec

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        #question += "Options:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]["data_type"]]
        bound = None
        if self.data_list[idx]["bound"]:
            bound = (
                self.data_list[idx]["data"]["start"],
                self.data_list[idx]["data"]["end"],
            )
        video_path = os.path.join(
            self.data_list[idx]["prefix"], self.data_list[idx]["data"]["video"]
        )
        image_list, sec = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]["data"])
        question = question + self.question_prompt

        raw_images = []
        num_patches_list = []
        pixel_values = []
        for image in image_list:
            raw_images.append(image)
            if self.dynamic_image_size:
                patches = dynamic_preprocess(
                    image,
                    image_size=self.input_size,
                    use_thumbnail=self.use_thumbnail,
                    max_num=self.max_num,
                )
            else:
                patches = [image]
            num_patches_list.append(len(patches))
            pixel_values.extend([self.transform(patch) for patch in patches])

        pixel_values = torch.stack(pixel_values)

        return {
            "question": question,
            "pixel_values": pixel_values,
            "answer": answer,
            "num_patches_list": num_patches_list,
            "task_type": self.data_list[idx]["task_type"],
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


def check_ans(pred, gt):
    flag = False
    pred = pred.replace("Answer: ", "")

    pred_list = pred.lower().split(" ")
    pred_option, pred_content = pred_list[0], " ".join(pred_list[1:])
    gt_list = gt.lower().split(" ")
    gt_option, gt_content = gt_list[0], " ".join(gt_list[1:])
    if gt_content[-1] == ".":
        gt_content = gt_content[:-1]

    if pred_option.replace(".", "") in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag


def evaluate_chat_model():
    random.seed(args.seed)
    prompt = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
    question_prompt = "\nAnswer:  \n"
    # question_prompt = "\nAnswer with the option's letter from the given choices directly and only give the best option. The best answer is: "
    answer_prompt = ""
    model.system_message = prompt

    vid_dataset = MVBenchDataset(
        data_dir,
        data_list,
        prompt=prompt,
        question_prompt=question_prompt,
        num_segments=args.num_segments,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset=vid_dataset,
        sampler=InferenceSampler(len(vid_dataset)),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
    )

    outputs = []
    for _, (pixel_values, questions, answers, num_patches_lists, task_types) in tqdm(
        enumerate(dataloader)
    ):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=20,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )
        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            num_patches_list=num_patches_lists[0],
            question=questions[0],
            generation_config=generation_config,
            verbose=True,
            add_generation_prompt=answer_prompt,
            timestamps=None,
        )

        outputs.append(
            {
                "question": questions[0],
                "pred": pred,
                "gt": answers[0],
                "task_type": task_types[0],
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

            if check_ans(pred, gt):
                acc_dict[task_type][0] += 1
                correct += 1

        final_res = {}
        for k, v in acc_dict.items():
            final_res[k] = v[0] / v[1] * 100
        final_res["Avg"] = correct / total * 100
        print(final_res)

        time_prefix = time.strftime("%y%m%d%H%M%S", time.localtime())
        results_file = f"MVBench_{time_prefix}"
        output_path = os.path.join(args.out_dir, results_file)
        with open(f"{output_path}.json", "w") as f:
            json.dump(outputs, f)
        with open(f"{output_path}_f{args.num_segments}_result_final.json", "w") as f:
            json.dump(final_res, f)
        print("Results saved to {}".format(output_path))


def load_model_and_tokenizer(args):
    from transformers import AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    tokenizer.model_max_length = 8192 * 4

    config = InternVLChatConfig.from_pretrained(
        "OpenGVLab/InternVL2-4B", trust_remote_code=True
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
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--datasets", type=str, default="mvbench")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
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

    model, tokenizer = load_model_and_tokenizer(args)
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
