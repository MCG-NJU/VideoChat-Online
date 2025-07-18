import torch

from tqdm import tqdm
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from accelerate import Accelerator, DistributedType
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from transformers import AutoModel, AutoTokenizer
from petrel_client.client import Client
client = Client('~/petreloss.conf')

import warnings

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger


@register_model("minicpm_v")
class MiniCPM_V(lmms):
    """
    MiniCPM_V Model
    """

    def __init__(
        self,
        pretrained: str = "openbmb/MiniCPM-V",
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = torch.bfloat16,
        batch_size: Optional[Union[int, str]] = 1,
        trust_remote_code: Optional[bool] = True,
        max_frames_num: Optional[int] = 16,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        # assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self.max_frames_num = max_frames_num
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = device
        self._model = AutoModel.from_pretrained(pretrained, trust_remote_code=trust_remote_code, torch_dtype=dtype, device_map=self._device).to(dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._config = self._model.config
        self.model.eval()
        self.model.tie_weights()
        self.batch_size_per_gpu = int(batch_size)
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
        else:
            self.model.to(self._device)
            self._rank = 0
            self._word_size = 1

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
        return self.tokenizer.decode(tokens)

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "We have not implemented this function for MiniCPM_V yet"

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

        self.time_msg = 'short'
        
        sec = [str(round(f / fps, 1)) for f in frame_indices]

        if self.time_msg is not None and sec is not None:
            if self.time_msg == 'short':
                msg = f"\nThe video segment contains {len(sec)} frames uniformly sampled from the past {(float(sec[-1])-float(sec[0])):.0f} seconds up to the present moment. "
            else:
                # " " should be added in the start and end
                msg = f"\nAnalyze the content of the {len(sec)} frames video segment uniformly sampled from the past {(float(sec[-1])-float(sec[0])):.0f} seconds up to the present moment. "
        else:
            msg = ""
        # print(frames.shape)
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
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            assert self.batch_size_per_gpu == 1, "Do not support batch_size_per_gpu > 1 for now"
            # assert len(visuals) == 1, "MiniCPM_V interface does not support bn_image > 1 for now"
            context = contexts[0]
            if "<image>" in context:
                # minicpm does not expect the <image> tag
                context = context.replace("<image>", "")
                
            
            if len(visuals) > 0:
                if len(visuals) > 1:
                    assert len(visuals) == 2, visuals
                    visual = visuals[0]
                    media_dict = visuals[1]
                else:
                    visual = visuals
                    media_dict = {'video_read_type': 'decord'}
                frames, time_msg = self.load_video(visual, self.max_frames_num, media_dict)
                context = time_msg + context
                # message.append({"role": "user", "content": [{"type": "video", "video": visual, "max_pixels": self.max_pixels}, {"type": "text", "text": context}]})
                # print(message)
            else:
                # message.append({"role": "user", "content": [{"type": "text", "text": context}]})
                raise "error"

            # messages.append(message)
                
            
            msgs = [{"role": "user", "content": context}]


            params = {
                'sampling': False,
                'top_p': 0.8,
                'top_k': 100,
                'temperature': 0.7,
                'repetition_penalty': 1.05,
                "max_new_tokens": 2048
            }

            # gen_kwargs["image_sizes"] = [visuals[idx].size for idx in range(len(visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            # try:
            # ominicpm does not give much information on how they do eval so I just use the chat format.
            response = self.model.chat(
                image=frames,
                msgs=msgs,
                context=None,
                tokenizer=self.tokenizer,
                sampling=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                )
            # print("response", response)
            # except Exception as e:
            #     eval_logger.error(f"Error {e} in generating")
            #     cont = ""
            res.append(response)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), response)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res
