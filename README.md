<p align="center">
    <img src="assets/videochat_logo.png" width="320px" style="margin: 20px 0">
</p>

<div align="center">
  
# 🎥 VideoChat-Online 
### CVPR2025 | Online Video Understanding: A Comprehensive Benchmark and Memory-Augmented Method

<p align="center">
    <img src="assets/comparison.png" width="128000px" style="margin: 20px 0">
</p>

  
[![arXiv](https://img.shields.io/badge/arXiv-2501.00584-b31b1b.svg)](https://arxiv.org/abs/2501.00584)

[![Dataset](https://img.shields.io/badge/🤗%20Dataset-VideoChatOnline--IT-ffca28)](https://huggingface.co/datasets/MCG-NJU/VideoChatOnline-IT)
[![Model](https://img.shields.io/badge/🤗%20Model-VideoChatOnline-4dc0b0)](https://huggingface.co/datasets/MCG-NJU/VideoChatOnline)
[![Leaderboard](https://img.shields.io/badge/🏆%20Leaderboard-Ranking-8b5cf6)](https://videochat-online.github.io/) 
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

</div>

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


annotation: 
data_root: Root path of the data.

# 🔄 Online SFT Data Download

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

# 📊 Evaluation on OVBench
```bash
#Sliding Window Setting:
bash shell/eval/online_bench_sliding_window.sh
#Streaming Setting:
bash shell/eval/online_bench_stream.sh
```

