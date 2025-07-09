import random
import os
import io
import av
import cv2
import decord
import imageio
from decord import VideoReader
import torch
import numpy as np
import math
import gc
from torchvision.transforms.functional import pil_to_tensor
import PIL.Image
import google.generativeai as genai
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
from moviepy.editor import*

def set_proxy(proxy_address = "http://closeai-proxy.pjlab.org.cn:23128"):
    os.environ["http_proxy"] = proxy_address
    os.environ["https_proxy"] = proxy_address
    os.environ["HTTP_PROXY"] = proxy_address
    os.environ["HTTP_PROXYS"] = proxy_address

def unset_proxy():
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTP_PROXYS", None)


def clean_api_chache():
    for file in genai.list_files():
        genai.delete_file(file.name)


def pts_to_secs(pts: int, time_base: float, start_pts: int) -> float:
    """
    Converts a present time with the given time base and start_pts offset to seconds.

    Returns:
        time_in_seconds (float): The corresponding time in seconds.

    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/utils.py#L54-L64
    """
    if pts == math.inf:
        return math.inf

    return int(pts - start_pts) * time_base


def get_pyav_video_duration(video_reader):
    video_stream = video_reader.streams.video[0]
    video_duration = pts_to_secs(
        video_stream.duration,
        video_stream.time_base,
        video_stream.start_time
    )
    return float(video_duration)




def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8
    ):

    # if video_path.endswith('.avi'):
    #     return read_frames_av(video_path=video_path, num_frames=num_frames, sample=sample,
    #                 fix_start=fix_start, min_num_frames=min_num_frames, max_num_frames=max_num_frames, 
    #                 client=client, clip=clip, local_num_frames=local_num_frames)
    if 's3://' in video_path:
        unset_proxy()
        video_bytes = client.get(video_path)
        if video_bytes is None or len(video_bytes) == 0:
            raise ValueError(f"Can't read byte from {video_path}!")
        video_byteio = io.BytesIO(video_bytes)
        video_reader = VideoReader(video_byteio, num_threads=1)
        set_proxy()
        video_cache = '/path_to_your/lmms-eval-ovbench/cache/full_tmp.mp4'
        with open(video_cache, 'wb') as file:
            file.write(video_byteio.getvalue())
        if clip:
            clip_cache = '/path_to_your/lmms-eval-ovbench/cache/clip_tmp.mp4'
            video_cut=CompositeVideoClip([VideoFileClip(video_cache).subclip(clip[0],clip[1])])
            video_cut.write_videofile(clip_cache)
            video_cache = clip_cache
        
        # print("cache:", video_cache)    
        video_upload = genai.upload_file(path = video_cache)
        
    else:
        set_proxy()
        video_byteio = None
        video_reader = VideoReader(video_path, num_threads=1)
        video_cache = video_path
        if clip:
            clip_cache = '/path_to_your/lmms-eval-ovbench/cache/clip_tmp.mp4'
            video_cut=CompositeVideoClip([VideoFileClip(video_path).subclip(clip[0],clip[1])])
            video_cut.write_videofile(clip_cache)
            video_cache = clip_cache
        video_upload = genai.upload_file(path = video_cache)
        
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    

    if clip:
        start, end = clip
        start = max(0, start)
        end = min(duration - 0.1, end) # 防止end超过视频末尾
        duration = end - start
        vlen = int(duration * fps) 
        start_index = int(start * fps)

    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, min_num_frames=min_num_frames, max_num_frames=max_num_frames, local_num_frames=local_num_frames
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]

    # print(fps, frame_indices)
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), torch.uint8
    # https://github.com/dmlc/decord/issues/208
    video_reader.seek(0)

    if video_byteio != None:
        video_byteio.close()
    # frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return [video_upload], frame_indices, float(fps), duration



def get_frame_indices(num_frames, vlen, sample='middle', fix_start=None, input_fps=1, min_num_frames=1, max_num_frames=-1, local_num_frames=8):

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen


    if sample == 'dynamic_fps1':

        duration = float(vlen) / input_fps
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments

        if max_num_frames > 0:
            num_frames = min(num_frames, max_num_frames)
        sample = "middle" # NOTE

        # logger.info(f"? is OK (img), duation={duration} frames={num_frames}!!!!")

    num_frames = max(min_num_frames, num_frames)

    # print(f"\033[0;31m vlen={vlen}, input_fps={input_fps} num_frames={num_frames} \033[0m")
        
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError(f"Not support sample type: {sample}")
    return frame_indices


def read_frames_img(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=1,
        max_num_frames=-1, client=None, clip=None, local_num_frames=8, fps=1
    ):
    img_list=[]
    if "s3://" in video_path:
        unset_proxy()
        for path in sorted(client.list(video_path)):
            img_list.append(path)
        set_proxy()
    else:
        for path in sorted(os.listdir(video_path)):
            img_list.append(path)

    if clip is not None:
        start = float(clip[0])
        end = float(clip[1])
        start = max(0, start)
        end = min((len(img_list)-1) / fps, end) # 防止end超过视频末尾 
        vlen = (end - start) * fps
    else:
        vlen = len(img_list)-1
    
    duration = vlen / fps

    if min_num_frames > vlen:
        if sample == 'dynamic_fps1':
            min_num_frames = (vlen // local_num_frames) * local_num_frames
        else:
            min_num_frames = vlen

    if sample == 'dynamic_fps1':
        num_segments = int(duration // local_num_frames)
        if num_segments == 0:
            num_frames = local_num_frames
        else:
            num_frames = local_num_frames * num_segments
        num_frames = min(num_frames, max_num_frames) 
        num_frames = max(min_num_frames, num_frames)

    if clip is not None:
        def _get_index_by_time(start_sec, end_sec, num_segments=8, fps=1., max_frame=9999):
            start_idx = max(0, round(start_sec * fps))
            end_idx = min(round(end_sec * fps), max_frame)
            if start_idx>=max_frame: 
                start_idx = 0
            seg_size = float(end_idx - start_idx) / (num_segments - 1)
            offsets = np.array([start_idx + int(np.round(seg_size * idx)) for idx in range(num_segments)])
            return offsets
        frame_indices = _get_index_by_time(float(clip[0]), float(clip[1]), num_segments=num_frames, fps=fps, max_frame=len(img_list)-1)
    else:
        frame_indices = get_frame_indices(
            num_frames, vlen, sample=sample, fix_start=fix_start,
            min_num_frames=min_num_frames,
            max_num_frames=max_num_frames, local_num_frames=local_num_frames
        )
    frames = []
    for idx in frame_indices:
        frame_fname = os.path.join(video_path, img_list[idx])
        if "s3://" in video_path:
            unset_proxy()
            img_bytes = client.get(frame_fname)
            img_byteio = io.BytesIO(img_bytes)
            img_cache = PIL.Image.open(img_byteio)
            # set_proxy()
            # cache_path="/path_to_your/lmms-eval-ovbench/cache/cache_image.jpg"
            # img_cache.save(cache_path, format='JPEG')
            # img = genai.upload_file(path=cache_path)
            img = img_cache.resize((400, 400))
        else:
            # set_proxy()
            # img = genai.upload_file(path=frame_fname)
            img_cache = PIL.Image.open(frame_fname)
            img = img_cache.resize((400, 400))
            
        frames.append(img)
        set_proxy()
    # frames = np.array(imgs, dtype=np.uint8)
    return frames, frame_indices, fps, duration # NOTE img直接当1fps处理



VIDEO_READER_FUNCS = {
    'decord': read_frames_decord,
    'img': read_frames_img,
    'frame': read_frames_img,
}