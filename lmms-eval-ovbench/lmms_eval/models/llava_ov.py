import math
import re
import copy
import json
import logging
import warnings
from datetime import timedelta
from typing import List, Optional, Union, Tuple
import PIL

import numpy as np
import torch
import transformers
from tqdm import tqdm
from packaging import version
from decord import VideoReader, cpu

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from transformers import AutoConfig

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.load_video import read_video_pyav

import io
from petrel_client.client import Client
client = Client('~/petreloss.conf')

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True

# Import LLaVA modules
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import (
        get_model_name_from_path,
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from llava.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN,
        IGNORE_INDEX,
    )
    from llava.conversation import conv_templates, SeparatorStyle
except ImportError as e:
    eval_logger.debug(f"LLaVA is not installed. Please install LLaVA to use this model.\nError: {e}")


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("llava_ov")
class LLaVa_OV(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "xxx",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name: Optional[str] = None,
        attn_implementation: Optional[str] = best_fit_attn_implementation,
        device_map: Optional[str] = "cuda:0",
        conv_template: Optional[str] = "xxx",
        use_cache: Optional[bool] = True,
        truncate_context: Optional[bool] = False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config: Optional[str] = None,  # ends in json
        max_frames_num: Optional[int] = 32,
        # mm_spatial_pool_stride: Optional[int] = 2,
        # mm_spatial_pool_mode: Optional[str] = "bilinear",
        token_strategy: Optional[str] = "single",  # could be "single" or "multiple", "multiple" denotes adding multiple <image> tokens for each frame
        video_decode_backend: str = "decord", # not use
        time_msg: Optional[str] = None,
        mm_projector_type: Optional[str] = None,
        mm_local_num_frames: Optional[int] = None,
        vision_encode_type: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        assert torch.cuda.device_count() > 0, torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        llava_model_args = {
            "multimodal": True,
        }
        if customized_config is not None:
            llava_model_args["customized_config"] = customized_config
        if attn_implementation is not None:
            llava_model_args["attn_implementation"] = attn_implementation
        if "use_flash_attention_2" in kwargs:
            llava_model_args["use_flash_attention_2"] = kwargs["use_flash_attention_2"]
        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)

        self.pretrained = pretrained
        self.token_strategy = token_strategy
        self.max_frames_num = max_frames_num

        # self.mm_spatial_pool_stride = mm_spatial_pool_stride
        # self.mm_spatial_pool_mode = mm_spatial_pool_mode
        # self.video_decode_backend = video_decode_backend
        self.time_msg = time_msg

        overwrite_config = {}
        # overwrite_config["mm_spatial_pool_stride"] = self.mm_spatial_pool_stride
        # overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode

        if mm_local_num_frames is not None:
            overwrite_config["mm_local_num_frames"] = mm_local_num_frames

        if mm_projector_type is not None:
            overwrite_config["mm_projector_type"] = mm_projector_type

        if vision_encode_type is not None:
            overwrite_config["vision_encode_type"] = vision_encode_type

        cfg_pretrained = AutoConfig.from_pretrained(self.pretrained)

        if cfg_pretrained.architectures[0] == "LlavaLlamaForCausalLM":  # Ugly code, only used in  vicuna that needs ROPE
            raise NotImplementedError
            # if "224" in cfg_pretrained.mm_vision_tower:
            #     least_token_number = self.max_frames_num * (16 // self.mm_spatial_pool_stride) ** 2 + 1000
            # else:
            #     least_token_number = self.max_frames_num * (24 // self.mm_spatial_pool_stride) ** 2 + 1000

            # scaling_factor = math.ceil(least_token_number / 4096)
            # if scaling_factor >= 2:
            #     overwrite_config["rope_scaling"] = {"factor": float(scaling_factor), "type": "linear"}
            #     overwrite_config["max_sequence_length"] = 4096 * scaling_factor
            #     overwrite_config["tokenizer_model_max_length"] = 4096 * scaling_factor

        llava_model_args["overwrite_config"] = overwrite_config
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        except TypeError:
            # for older versions of LLaVA that don't have multimodal argument
            llava_model_args.pop("multimodal", None)
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)

        self._config = self._model.config
        self.model.eval()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
  

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def load_video(self, video_path, max_frames_num, media_dict):
        from llava.video_utils import VIDEO_READER_FUNCS
        if type(video_path) != str:
            assert len(video_path) == 1, video_path
            video_path = video_path[0]
        if 'start' in media_dict:
            clip = [media_dict['start'], media_dict['end']]
        else:
            clip = None
        # print("-------------------------------------------------------------------")
        # print(media_dict['video_read_type'], clip, video_path, max_frames_num)    
        if 'fps' in media_dict:
            frames, frame_indices, fps, duration = VIDEO_READER_FUNCS[media_dict['video_read_type']](video_path=video_path, num_frames=max_frames_num, sample='middle', fix_start=None, min_num_frames=1, max_num_frames=-1, client=client, clip=clip, local_num_frames=-1, fps=media_dict['fps'])
        else:
            frames, frame_indices, fps, duration = VIDEO_READER_FUNCS[media_dict['video_read_type']](video_path=video_path, num_frames=max_frames_num, sample='middle', fix_start=None, min_num_frames=1, max_num_frames=-1, client=client, clip=clip, local_num_frames=-1)

        sec = [str(round(f / fps, 1)) for f in frame_indices]

        if self.time_msg is not None and sec is not None:
            if self.time_msg == 'short':
                msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled from it. "
            else:
                # " " should be added in the start and end
                msg = f"\nThe video lasts for {duration:.2f} seconds, and {len(sec)} frames are uniformly sampled at {', '.join(sec)} seconds. "
        else:
            msg = ""

        return frames, msg

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata # NOTE 多个任务如mvbench的时候会有bug，建议不用
        # metadata["task_type"] = 'video' # NOTE 写死先
        # print('===========================================================================')
        # for request in requests:
        #     print(request)
        #     print('---------------------------------------------------------')
        # print('===========================================================================')
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)

            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            question_input = []

            for visual, context in zip(batched_visuals, batched_contexts):
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                    self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                    eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

                # if type(visual[0]) == PIL.Image.Image and "task_type" not in metadata and "sample_frames" not in metadata:  # For image task
                if type(visual[0]) == PIL.Image.Image: # and "task_type" not in metadata and "sample_frames" not in metadata:  # For image task
                    raise NotImplementedError(f"I don't want image task now: {visual}, {task}, {metadata}")
                    image_tensor = process_images(visual, self._image_processor, self._config)
                    if type(image_tensor) is list:
                        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                    else:
                        image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                    task_type = "image"
                    placeholder_count = len(visual) if isinstance(visual, list) else 1

                # elif "task_type" in metadata and metadata["task_type"] == "video" and type(visual[0]) != str:
                # # elif "task_type" in metadata and metadata["task_type"] == "video" and "sample_frames" in metadata:
                #     assert type(visual) == list, "sample_frames must be specified for video task"
                #     # sample_indices = np.linspace(0, len(visual) - 1, metadata["sample_frames"], dtype=int)
                #     sample_indices = np.linspace(0, len(visual) - 1, self.max_frames_num, dtype=int) # NOTE 
                #     frames = [visual[i] for i in sample_indices]
                #     assert len(frames) == self.max_frames_num # metadata["sample_frames"]

                #     image_tensor = []
                #     frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                #     image_tensor.append(frames)
                    
                #     # image_tensor = process_images(visual, self._image_processor, self._config)
                #     # if type(image_tensor) is list:
                #     #     image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                #     # else:
                #     #     image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)

                #     task_type = "video"
                #     placeholder_count = 1

                elif type(visual[0]) == str:  # For video task
                    if len(visual) > 1:
                        assert len(visual) == 2, visual
                        media_dict = visual[1]
                    else:
                        media_dict = {'video_read_type': 'decord'}
                    image_tensor = []
                    try:
                        frames, time_msg = self.load_video(visual[0], self.max_frames_num, media_dict)
                        frames = self._image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].half().cuda()
                        image_tensor.append(frames)
                    except Exception as e:
                        eval_logger.error(f"Error {e} in loading video")
                        raise e
                        image_tensor = None

                    task_type = "video"
                    placeholder_count = len(frames) if self.token_strategy == "multiple" else 1

                else:
                    raise NotImplementedError(f"I don't want it: {task}")
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    4. For video tasks, we could add a <image> token or multiple <image> tokens for each frame in the context. This depends on the training strategy and should balance in test to decide which is better
                    """
                    # if task_type == "image": # indeed in multi-image case, not the video in frames.
                    #     image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    # elif task_type == "video":
                    # image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count if self.token_strategy == "multiple" else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * placeholder_count
                    image_tokens = " ".join(image_tokens)
                    if task_type == 'video' and time_msg != "":
                        question = image_tokens + "\n" + time_msg.rstrip() + " " + context
                    else:
                        question = image_tokens + "\n" + context
                else:
                    assert task_type != 'video', context
                    question = context

                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()

                if utils.is_json(question):  # conversational question input
                    question = json.loads(question)
                    for idx, item in enumerate(question):
                        role = conv.roles[idx % 2]
                        message = item["value"]
                        conv.append_message(role, message)

                    assert len(conv.messages) % 2 == 1
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)
                else:  # only simple string for question
                    conv.append_message(conv.roles[0], question)
                    conv.append_message(conv.roles[1], None)
                    prompt_question = conv.get_prompt()
                    question_input.append(prompt_question)

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            # message_stage = [
            #     {
            #         "from": "human",
            #         "value": question,
            #     },
            #     {
            #         "from": "gpt",
            #         "value": None,
            #     }
            # ]
            # qwen_input_ids = self.preprocess_qwen(message_stage, self.tokenizer, has_image=True).to(self.device)
            # qwen_result_list = qwen_input_ids.detach().cpu().numpy().tolist()
            # qwen_result_list = [i if i != -200 else 100 for i in qwen_result_list[0]]
            # qwen_input_text = self.tokenizer.decode(qwen_result_list)

            # original_result_list = input_ids.detach().cpu().numpy().tolist()
            # original_result_list = [i if i != -200 else 100 for i in original_result_list[0]]
            # original_input_text = self.tokenizer.decode(original_result_list)

            # print(f"Qwen input text: {qwen_input_text}")
            # print(f"Original input text: {original_input_text}")

            # assert qwen_input_ids == input_ids

            if task_type == "image":
                gen_kwargs["image_sizes"] = [batched_visuals[0][idx].size for idx in range(len(batched_visuals[0]))]
            elif task_type == "video":
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
                gen_kwargs["modalities"] = ["video"]
                gen_kwargs["stopping_criteria"] = [stopping_criteria]
                # self._config.mm_spatial_pool_stride = self.mm_spatial_pool_stride
                # self._config.mm_spatial_pool_mode = self.mm_spatial_pool_mode

            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            if "image_aspect_ratio" in gen_kwargs.keys():
                gen_kwargs.pop("image_aspect_ratio")
            try:
                with torch.inference_mode():
                    cont = self.model.generate(input_ids, attention_mask=attention_masks, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)
                    # cont = self.model.generate(qwen_input_ids, pad_token_id=pad_token_ids, images=image_tensor, use_cache=self.use_cache, **gen_kwargs)

                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
            except Exception as e:
                raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
