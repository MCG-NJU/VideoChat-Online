
# <img src="assets/videochat_logo.png" style="vertical-align: -10px;" :height="50px" width="50px">VideoChat-Online
<div style="text-align: center;">
    <h2 style="display: inline-block; vertical-align: middle; margin-left: 10px; margin-bottom: 0;">
        <a href="https://videochat-online.github.io">[CVPR2025] Online Video Understanding: OVBench and VideoChat-Online</a>
    </h2>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2501.00584"><img src="https://img.shields.io/badge/arXiv-2501.00584-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT"><img src="https://img.shields.io/badge/🤗%20Dataset-VideoChatOnline--IT-ffca28" alt="Dataset"></a>
  <a href="https://huggingface.co/MCG-NJU/VideoChatOnline-4B"><img src="https://img.shields.io/badge/🤗%20Model-VideoChatOnline-4dc0b0" alt="Model"></a>
  <a href="https://videochat-online.github.io/"><img src="https://img.shields.io/badge/🏆%20Leaderboard-Ranking-8b5cf6" alt="Leaderboard"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>



# 📝 **Highlights**

🚀 **Introducing OVBench**
 OVBench is a benchmark tailored for **real-time video understanding**:

- **Memory, Perception, and Prediction of Temporal Contexts**: Questions are framed to reference the present state of entities, requiring models to memorize/perceive/predict past/present/future temporal contexts over time.
- **Dynamic Spatio-temporal Interaction**: The benchmark demands precise real-time interactions with video content, where actions, objects, and events must be understood in the context of their spatial and temporal relationships.
- **Contextual Awareness at Specific Moments**: Real-time questions are contextual, changing based on the specific timestamp they are asked, requiring a deep understanding of how temporal context evolves.

---

🏗️  **Pyramid Memory Bank**

To tackle the challenges of infinite video streams, we propose a **multi-layered Pyramid Memory Bank** that balances **spatial and temporal information**:  

1. **Spatial Anchors**: The lower layers retain high-resolution features to preserve fine-grained spatial cues, capturing keyframes as "spatial anchors" with a lower sampling rate.
2. **Progressive Abstraction**: As the layers progress, spatial resolution decays  while the temporal sampling rate grows proportionally, forming an abstract representation of fine-grained long-short-term patterns.
3. **Dynamic Eviction**: A dynamic eviction mechanism detects temporal redundancy via similarity, combined with pooling for spatial compression, improving storage efficiency.

---

🎯 **Offline-to-Online Learning Paradigm** 

A novel training strategy designed for online video streams:  

- **Interleaved Dialogue Tuning**: Combines offline video data with online instruction tuning in a dialogue format.  
- **Progressive Learning**: Bridges offline and online video understanding, enhancing real-time adaptability.  

<div style="text-align: center;">
    <div style="display: inline-block; margin-right: 10px;">
        <img src="assets/spatial_perception.png" alt="image-20250311180653255" style="zoom: 15%;" />
    </div>
    <div style="display: inline-block;">
        <img src="assets/past_memory.png" alt="image-20250311184752494" style="zoom: 15%;" />
    </div>
        <div style="display: inline-block;">
        <img src="assets/comparison.png" alt="image-202503111847524942" style="zoom: 15%;" />
    </div>
</div>


## To-Do

- [x] Model checkpoint Upload
- [ ] A more interactive demo

---

# 🏆 OVBench Leaderboard 

See our leaderboard [here](https://videochat-online.github.io/#leaderboard)


## Evaluate your model


Evaluation of Existing Models on OVBench Using [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

#### Preparatory Steps

- Environment Setup: Ensure that all dependencies required by [lmms_eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) are properly installed.

- Please perform a global search for the field `/path_to_your` in the `./lmms-eval-ovbench` directory and replace it with the corresponding file path on your local system.

#### Predefined Model Evaluation

- Execute the script `lmms-eval-ovbench/scripts/eval_models/eval_internvl2-8B.sh` to initiate the benchmark evaluation.

#### Custom Model Evaluation

- Given that the video data used in this benchmark consists of both image sequences and video clips, it is necessary to utilize the `lmms-eval-ovbench/llava/video_utils.py` to read video data correctly.

- You may refer to the implementation of the `load_video` function in `lmms-eval-ovbench/lmms_eval/models/internvl2.py` as a guideline. Integrate this function into your custom model as needed to enable compatibility with the lmms_eval evaluation framework.




## Submit the results

Email xinhaoli00@outlook.com with your result.json or open an issue in this repo.


---


# 🎥 Demo

To launch the demo, use the following script:


https://github.com/user-attachments/assets/4cdc3e57-3dae-4656-8366-a10cc6648884


```bash
bash gradio_demo.sh
```
# 🛠️ Installation
To install the necessary dependencies, use the following commands:

```bash
conda create -n your_env python=3.9
pip install -r requirements.txt
```

---

# 📦 Offline Data Preparation

The anno_data file provides the paths for different types of datasets:

```json
"coin_sl_train": {
    "annotation": "Path to the annotations json file.",
    "data_root": "your data path",
},
...
```
We support the data reading formats `LLaVA` and `VideoChat2-IT` for specific data JSON formats.

---

# 🔄 [Online SFT Data](https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT) Download

For the construction format of online data, please refer to [VideoChatOnline-IT](https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT)

# 📈 Evaluations Results of  VideoChatOnline-4B on Long Video Benchmarks
| Benchmark          | Result                                                 |
|--------------------|--------------------------------------------------------|
| **OVBench**        | 54.9                                                   |
| **VideoMME**       | Short: 65.8<br>Medium: 50.2<br>Long: 47.1<br>Avg: 54.4  |
| **MVBench**        | 65.2                                                   |
| **EgoSchema** | 54.7                                                   |
| **MLVU**           | 60.8                                                   |
| **LongVideoBench** | 54.1                                                   |

---

# 🚀 Training
To run the training, execute the following bash commands for different stages:
```bash
#Offline SFT:
bash shell/online_4b/videochat_online_4b_stage1_ft.sh
```
```bash
#Online & Offline Joint SFT:
bash shell/online_4b/videochat_online_4b_stage2_ft.sh
```

---

# 📊 Evaluation on [OVBench](https://huggingface.co/datasets/MCG-NJU/OVBench)

```bash
#Sliding Window Setting:
bash shell/eval/online_bench_sliding_window.sh
#Streaming Setting:
bash shell/eval/online_bench_stream.sh
```


