import io
import os
import time
import json
from typing import List, Tuple
from tqdm import tqdm
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.api.instance import Instance
from accelerate import Accelerator, DistributedType

from loguru import logger as eval_logger

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from PIL import Image

NUM_SECONDS_TO_SLEEP = 15
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

from petrel_client.client import Client
client = Client('~/petreloss.conf')

from lmms_eval.models.gemini_tools import *


@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-1.5-flash",
        modality: str = "image",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = "/path_to_your/lmms-eval-ovbench/cache",  # We will cache the Gemini API response in this path and use it for future requests
        max_frames_num: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        set_proxy()
        self.max_frames_num = max_frames_num
        self.model_version = model_version
        self.timeout = timeout
        self.model = genai.GenerativeModel(model_version)
        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        self.response_persistent_folder = response_persistent_folder
        if not os.path.exists(self.response_persistent_folder):
            os.makedirs(self.response_persistent_folder)
        self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")

        if os.path.exists(self.response_persistent_file):
            with open(self.response_persistent_file, "r") as f:
                self.response_cache = json.load(f)
            self.cache_mode = "resume"
        else:
            self.response_cache = {}
            self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        self.modality = modality

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = genai.upload_file(path=video_path)
        return uploaded_obj

    def convert_video(self, images):
        for idx, img in enumerate(images):
            if self.modality == "video" and isinstance(img, str):
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
        return images


    def load_video(self, video_path, max_frames_num, media_dict):
        unset_proxy()
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
        self.time_msg = 'short'
        if self.time_msg is not None and sec is not None:
            if self.time_msg == 'short':
                msg = f"\nThe video segment contains {len(sec)} frames uniformly sampled from the past {(float(sec[-1])-float(sec[0])):.0f} seconds up to the present moment. "
            else:
                # " " should be added in the start and end
                msg = f"\nAnalyze the content of the {len(sec)} frames video segment uniformly sampled from the past {(float(sec[-1])-float(sec[0])):.0f} seconds up to the present moment. "
        else:
            msg = ""
        
        # print("time msg:", msg)
        set_proxy()
        return frames, msg


    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue
                    

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            
            set_proxy()
            for attempt in range(5):
                # try:
                clean_api_chache()
                # visuals = self.convert_video(visuals)
                if len(visuals) > 1:
                    assert len(visuals) == 2, visuals
                    visual = visuals[0]
                    media_dict = visuals[1]
                else:
                    visual = visuals
                    media_dict = {'video_read_type': 'decord'}
                    
                video, time_msg = self.load_video(visual, self.max_frames_num, media_dict)
                contexts = time_msg + contexts
                
                message = [contexts] + video
                
                # message = [contexts]
                
                # print("len of message:", len(message))
                set_proxy()
                content = self.model.generate_content(
                    message,
                )
                content = content.text
                
                break
                # except Exception as e:
                #     eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                #     if isinstance(e, ValueError):
                #         try:
                #             eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                #             content = ""
                #             break
                #         except Exception:
                #             pass
                #     if attempt < 4 :  # If we have retries left, sleep and then continue to next attempt
                #         time.sleep(NUM_SECONDS_TO_SLEEP)
                #     else:  # If this was the last attempt, log and return empty
                #         eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                #         content = ""
            res.append(content)
            pbar.update(1)
            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)
        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"



# Gemini_API()