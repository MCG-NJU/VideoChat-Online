import importlib
import os
import hf_transfer
from loguru import logger
import sys
import hf_transfer

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "minicpm_v": "MiniCPM_V",
    "internvl2": "InternVL2",
    "longva": "LongVA",
    "qwen2_vl": "Qwen2_VL",
    "llava_ov": "LLaVa_OV",
    
    # "batch_gpt4": "BatchGPT4",
    # "claude": "Claude",
    # "from_log": "FromLog",
    # "fuyu": "Fuyu",
    # "gemini_api": "GeminiAPI",
    # "gpt4v": "GPT4V",
    # "idefics2": "Idefics2",
    # "instructblip": "InstructBLIP",
    # "internvl": "InternVLChat",
    # "llama_vid": "LLaMA_Vid",
    # "llava": "Llava",
    # "llava_hf": "LlavaHf",
    # "llava_onevision": "Llava_OneVision",
    # "llava_sglang": "LlavaSglang",
    # "llava_vid": "LlavaVid",
    # "mantis": "Mantis",
    # "mplug_owl_video": "mplug_Owl",
    # "phi3v": "Phi3v",
    # "qwen_vl": "Qwen_VL", 
    # "qwen_vl_api": "Qwen_VL_API",
    # "reka": "Reka",
    # "srt_api": "SRT_API",
    # "tinyllava": "TinyLlava",
    # "videoChatGPT": "VideoChatGPT",
    # "video_llava": "VideoLLaVA",
    # "vila": "VILA",
    # "xcomposer2_4KHD": "XComposer2_4KHD",
    # "xcomposer2d5": "XComposer2D5",
    # "videochat_next": "VideoChat_NeXT",
    "videochat_flash_online_dynamic": "VideoChat_flash_online_dynamic",
    "videochat_next_online": "VideoChat_NeXT_online",
    "videochat_next_online_memory": "VideoChat_NeXT_online_memory",
    "qwen2_5_vl": "Qwen2_5_VL",
    "internvl2_video": "InternVL2_video",
    # "videochat_next_old": "VideoChat_NeXT_old"
}

for model_name, model_class in AVAILABLE_MODELS.items():
    try:
        exec(f"from .{model_name} import {model_class}")
    except Exception as e:
        logger.debug(f"Failed to import {model_class} from {model_name}: {e}")

if os.environ.get("LMMS_EVAL_PLUGINS", None):
    # Allow specifying other packages to import models from
    for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
        m = importlib.import_module(f"{plugin}.models")
        for model_name, model_class in getattr(m, "AVAILABLE_MODELS").items():
            try:
                exec(f"from {plugin}.models.{model_name} import {model_class}")
            except ImportError as e:
                logger.debug(f"Failed to import {model_class} from {model_name}: {e}")
