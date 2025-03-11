# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import bisect
import chunk
import copy
from functools import partial
import warnings
from typing import Any, List, Optional, Tuple, Union

from sympy import N, im
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.videochat_online.modeling_videochat_online_finetune import (
    VideoChatOnline_IT,
)
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
import torch.nn.functional as F
from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.get_logger(__name__)


class VideoChatOnline_Stream(VideoChatOnline_IT):
    def __init__(
        self, config: InternVLChatConfig, vision_model=None, language_model=None
    ):
        super().__init__(config, vision_model, language_model)

    def extract_feature_bank(self, pixel_values, num_patches, is_video, start_id=0):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]
        cls_tokens = vit_embeds.mean(1, keepdim=True)  # vit_embeds[:, 0:1, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        #############downsample##################
        # print("pixel_values", pixel_values.shape, is_video, num_patches)
        ret_vit_embeds = []
        for i, (flag, num_frame) in enumerate(zip(is_video, num_patches)):
            start_idx = sum(num_patches[:i])
            end_idx = sum(num_patches[: i + 1])
            partial_vit_embeds = vit_embeds[start_idx:end_idx]
            partial_cls_tokens = cls_tokens[start_idx:end_idx]
            if len(partial_vit_embeds) == 0:  # pure text
                continue
            if flag == 1:
                partial_vit_embeds, scale, indices = self.extract_feature_stream(
                    partial_vit_embeds, partial_cls_tokens, start_id=start_id
                )
                ret_vit_embeds.append(partial_vit_embeds)
            else:
                partial_vit_embeds, scale = self.extract_feature_image(
                    partial_vit_embeds
                )
                ret_vit_embeds.append(partial_vit_embeds)

        return torch.concat(ret_vit_embeds, dim=0), scale, indices

    def extract_feature(self, pixel_values, num_patches, is_video):
        self.memorybank = None
        stride = 1000  # avoid oom
        for i in range(0, len(pixel_values), stride):
            ret_vit_embeds, scale, indices = self.extract_feature_bank(
                pixel_values[i : i + stride],
                torch.tensor([len(pixel_values[i : i + stride])], dtype=torch.long),
                torch.tensor([1], dtype=torch.long),
                start_id=i,
            )
        self.memorybank = None

        return ret_vit_embeds, scale, indices

    def extract_feature_image(self, vit_embeds):

        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(
            vit_embeds.shape[0], -1, vit_embeds.shape[-1]
        ).flatten(0, 1)

        ret_vit_embeds = self.mlp1(vit_embeds)
        ret_vit_embeds = ret_vit_embeds.unsqueeze(0)
        return ret_vit_embeds.squeeze(0), None

    def extract_feature_stream(self, vit_embeds, cls_tokens, start_id):
        #############downsample##################
        if self.memorybank is None:
            self.memorybank = HierarchicalMemoryBank(
                [self.long_bank, self.mid_bank, self.short_bank], [256, 64, 16]
            )
        scale_ratio = VideoChatOnline_IT.tokens_arrange(
            vit_embeds.shape[0],
            intervals=self.reverse_memory_sample_ratio,
            downsample_ratio=[
                2**s for s in range(len(self.reverse_memory_sample_ratio))
            ],
        )
        ret_vit_embeds = []

        for i, vit_embed in enumerate(vit_embeds):
            ratio = scale_ratio[i]

            vit_embed = vit_embed.unsqueeze(0)
            vit_embed = self.pixel_shuffle(
                vit_embed, scale_factor=self.downsample_ratio
            )

            _, h, w, _ = vit_embed.shape
            vit_embed = vit_embed.permute(0, 3, 1, 2)
            if ratio > 1:
                pooled_vit_embed = F.adaptive_avg_pool2d(
                    vit_embed, (h // ratio, w // ratio)
                )

                self.memorybank.update_memory(pooled_vit_embed, i + start_id, None)
            else:
                self.memorybank.update_memory(vit_embed, i + start_id, None)
        ret_vit_embeds, indices = self.memorybank.output_by_time_order()
        scale = [embed.shape[1] for embed in ret_vit_embeds]
        ret_vit_embeds = torch.concat(ret_vit_embeds, dim=1)
        ret_vit_embeds = self.mlp1(ret_vit_embeds)
        return ret_vit_embeds.squeeze(0), scale, indices

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                "Warning: `image_counts` is deprecated. Please use `num_patches_list` instead."
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        input_ids = model_inputs["input_ids"].to(self.device)  # .cuda()
        attention_mask = model_inputs["attention_mask"].to(self.device)  # .cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        add_generation_prompt="",
        timestamps=None,
    ):
        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history

        num_patches = (torch.tensor([pixel_values.shape[0]], dtype=torch.long),)
        vit_embeds, num_image_tokens_list, frame_idxs = self.extract_feature(
            pixel_values,
            num_patches,
            is_video=torch.tensor([1], dtype=torch.long),
        )
        if timestamps is None:
            special_image_tokens = [
                "Frame{}: <image>".format(frame_idxs[i] + 1)
                for i in range(len(frame_idxs))
            ]
        else:
            special_image_tokens = [timestamps[idx] for idx in frame_idxs]
        question = "\n".join(special_image_tokens) + "\n" + question

        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")
        query += add_generation_prompt
        for num_image_tokens in num_image_tokens_list:

            image_tokens = (
                IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_tokens + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)  # .cuda()
        attention_mask = model_inputs["attention_mask"].to(self.device)  # .cuda()
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_features=vit_embeds,
            num_patches=num_patches_list,
            is_video=torch.tensor([1], dtype=torch.long),
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(
                f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>"
            )
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        num_patches: Optional[torch.LongTensor] = None,
        is_video: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None and len(pixel_values) > 0:
            if visual_features is not None:
                vit_embeds = visual_features
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

class HierarchicalMemoryBank:
    def __init__(self, capacities, reduced_sizes):
        # capacities: list of capacities for each group
        self.groups = [
            {"tokens": [], "capacity": cap, "reduced_size": size}
            for cap, size in zip(capacities, reduced_sizes)
        ]

    def _meanpool(self, tokens):
        return tokens.mean(dim=(2, 3))

    def _find_most_similar_frame(self, group):
        cls_tokens = torch.cat([self._meanpool(g["tokens"]) for g in group], dim=0)
        similarities = F.cosine_similarity(cls_tokens[1:], cls_tokens[:-1], dim=1)
        return torch.argmax(similarities).item()

    def _reduce_tokens(self, tokens, target_size):
        
        H = W = int(target_size**0.5)
        if tokens.size()[-2:] == (H, W):
            return tokens
        return F.interpolate(tokens, size=(H, W), mode="bilinear", align_corners=False)

    def _update_group(self, group, new_tokens, index, cls_token, next_group=None):
        if len(group["tokens"]) >= group["capacity"]:
            if group["capacity"] > 0:
                similar_frame_idx = self._find_most_similar_frame(group["tokens"])
                reduced_tokens = (
                    self._reduce_tokens(
                        group["tokens"][similar_frame_idx]["tokens"],
                        target_size=next_group["reduced_size"],
                    )
                    if next_group
                    else None
                )
                if next_group:
                    self._update_group(
                        next_group,
                        reduced_tokens,
                        group["tokens"][similar_frame_idx]["index"],
                        group["tokens"][similar_frame_idx]["cls_token"],
                    )
                group["tokens"].pop(similar_frame_idx)
        group["tokens"].append(
            {
                "tokens": self._reduce_tokens(
                    new_tokens,
                    target_size=group["reduced_size"],
                ),
                "index": index,
                "cls_token": cls_token,
            }
        )

    def update_memory(self, new_tokens, index, cls_token):
        for i, group in enumerate(self.groups):
            if new_tokens.shape[2] * new_tokens.shape[3] == group["reduced_size"]:
                next_group = self.groups[i + 1] if i + 1 < len(self.groups) else None
                self._update_group(
                    group, new_tokens, index, self._meanpool(new_tokens), next_group
                )
                break
        else:
            raise NotImplementedError(
                f"Unsupported token size: {new_tokens.shape[2] * new_tokens.shape[3]}"
            )

    def output_by_time_order(self):
        all_frames = [frame for group in self.groups for frame in group["tokens"]]
        all_frames.sort(key=lambda x: x["index"])
        tokens_list = [
            frame["tokens"].flatten(2, 3).permute(0, 2, 1) for frame in all_frames
        ]
        indices_list = [frame["index"] for frame in all_frames]
        return tokens_list, indices_list

    def clear(self):
        for group in self.groups:
            group["tokens"].clear()
        self.cls_token_cache.clear()