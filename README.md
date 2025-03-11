
# <img src="assets/videochat_logo.png" style="vertical-align: -10px;" :height="50px" width="50px">VideoChat-Online
<div style="text-align: center;">
    <h2 style="display: inline-block; vertical-align: middle; margin-left: 10px; margin-bottom: 0;">
        <a href="https://videochat-online.github.io">[CVPR2025] Online Video Understanding: OVBench and Memory-Augmented Method</a>
    </h2>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2501.00584"><img src="https://img.shields.io/badge/arXiv-2501.00584-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT"><img src="https://img.shields.io/badge/🤗%20Dataset-VideoChatOnline--IT-ffca28" alt="Dataset"></a>
  <a href="https://huggingface.co/datasets/MCG-NJU/VideoChatOnline"><img src="https://img.shields.io/badge/🤗%20Model-VideoChatOnline-4dc0b0" alt="Model"></a>
  <a href="https://videochat-online.github.io/"><img src="https://img.shields.io/badge/🏆%20Leaderboard-Ranking-8b5cf6" alt="Leaderboard"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>
<p align="center">
    <img src="assets/comparison.png" width="128000px" style="margin: 20px 0">
</p>

## To-Do List

- [ ] Model checkpoint Upload
- [ ] More comprehensive demo

# 🛠️ Installation
To install the necessary dependencies, use the following commands:

```bash
conda create -n your_env python=3.9
pip install -r requirements.txt
```
# 📦 Offline Data Preparation
The anno_data file provides the paths for different types of datasets:

```json
"coin_sl_train": {
    "annotation": "Path to the annotations json file.",
    "data_root": "your data path",
},
...
```
For specific data json format, we support the data reading formats of `LLaVA` and `VideoChat2-IT`.

# 🔄 [Online SFT Data](https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT) Download

For the construction format of online data, please refer to [VideoChatOnline-IT](https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT)

# 📈 Evaluations Results
| Benchmark          | Result                                                 |
|--------------------|--------------------------------------------------------|
| **OVBench**        | 54.9                                                   |
| **VideoMME**       | Short: 65.8<br>Medium: 50.2<br>Long: 47.1<br>Avg: 54.4  |
| **MVBench**        | 65.2                                                   |
| **EgoSchema** | 54.7                                                   |
| **MLVU**           | 60.8                                                   |
| **LongVideoBench** | 53.6                                                   |


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
# 🎥 Demo
To launch the demo, use the following script:

```bash
bash gradio_demo.sh
```

# 📊 Evaluation on [OVBench](https://huggingface.co/datasets/MCG-NJU/OVBench)

<div style="text-align: center;">
    <div style="display: inline-block; margin-right: 10px;">
        <img src="assets/spatial_perception.png" alt="image-20250311180653255" style="zoom: 28%;" />
    </div>
    <div style="display: inline-block;">
        <img src="assets/past_memory.png" alt="image-20250311184752494" style="zoom: 33%;" />
    </div>
</div>

```bash
#Sliding Window Setting:
bash shell/eval/online_bench_sliding_window.sh
#Streaming Setting:
bash shell/eval/online_bench_stream.sh
```

