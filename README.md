# CORE：AVE、TRM 与 LLM

项目保留三个纯商品 ID 的会话推荐模型，统一使用 `train.py` 和 `trainer.py` 训练。输入是点击商品序列，目标是下一次点击的商品；模型输出全商品排序分数。三个模型的说明分别见 [AVE](README_AVE.md)、[TRM](README_TRM.md)、[LLM](README_LLM.md)，完整操作说明见 [训练指南](README_TRAINING.md)。

```text
CORE/
  core_ave.py                 # 平均池化模型
  core_trm.py                 # Transformer 学习历史位置权重
  core_llm.py                 # 连续商品向量输入 Qwen 的适配模型
  train.py                   # 统一命令行入口与模型恢复
  trainer.py                 # 共享训练器 CORETrainer
  train_ave.py                # AVE 的 IDE 直接运行入口
  train_trm.py                # TRM 的 IDE 直接运行入口
  train_llm.py                # LLM 的 IDE 直接运行入口
  configs/                   # common.yaml 与 core_*.yaml
  dataset/                   # 已处理的 train / valid / test 数据
  models/hf/                 # LLM 模型缓存
  results/core/              # 新运行结果
  results/phase1/            # 保留的历史实验记录
  presentation/              # 三模型比较 PPT 与构建源文件
  tests/                     # 模型、训练与恢复检查
  docs/research_notes.md      # LLM 方案的历史文献笔记
  requirements.txt           # AVE / TRM 基础依赖
  requirements_llm.txt       # 基础依赖加 LLM 依赖
```

已有项目环境可以直接使用 `.venv`。迁移环境时，AVE/TRM 安装 `requirements.txt`，LLM 安装 `requirements_llm.txt`；后者包含前者。安装 PyTorch 时应按实际设备选择 CUDA 或 CPU 版本。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements_llm.txt
```

先分别运行一轮小数据检查。LLM 默认使用真实 Qwen；第一次运行可能下载模型，已有缓存可以加 `--local-files-only`。

```powershell
.\.venv\Scripts\python.exe train.py --model ave --smoke
.\.venv\Scripts\python.exe train.py --model trm --smoke
.\.venv\Scripts\python.exe train.py --model llm --variant pretrained_lora --smoke
```

正式训练默认使用 Tmall、seed 2020、最多 30 轮、micro-batch 16、累积 16、学习率 0.001，按验证 MRR@20 选择检查点。以下命令分别执行：

```powershell
.\.venv\Scripts\python.exe train.py --model ave
.\.venv\Scripts\python.exe train.py --model trm
.\.venv\Scripts\python.exe train.py --model llm --variant pretrained_lora
```

在 PyCharm 等 IDE 中选择项目 `.venv` 解释器，打开 `train_ave.py`、`train_trm.py` 或 `train_llm.py` 点击 Run，也会启动对应模型的全量训练。三个入口默认 `smoke=False`、`test=False`；修改为 `smoke=True` 可以先检查管线。全量运行默认仅评估验证集，配置固定后用 `--test` 评估测试集。

终端打印 `CORE_RESULT_DIR=...` 和成功结束时的 `CORE_COMPLETE=...`。每次运行独立保存配置、数据哈希、逐轮指标、最佳检查点和推荐示例；历史记录保持原路径。三模型比较 PPT 见 [core_comparison.pptx](presentation/core_comparison.pptx)。2026-10-05 的匹配基线与 2026-10-02 的 LLM 记录属于单 seed 验证比较，LLM 运行中断，三组均未评估全量测试集；详细数值与边界保留在 [LLM 实验记录](README_LLM.md)。

项目基于 CORE 的官方 PyTorch 实现：[CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space](https://arxiv.org/abs/2204.11067)，SIGIR 2022 short。原始 CORE 的核心是让会话表示保持为历史商品向量的加权和；LLM 投影和融合属于扩展变体。数据格式遵循 RecBole，原始处理数据下载入口为 [Google Drive](https://drive.google.com/drive/folders/1dlJ3PzcT5SCN8-Mocr_AIQPGk9DVgTWB?usp=sharing)。方案背景可参考 [历史文献笔记](docs/research_notes.md)，其中的建议不代表本项目已经实现或验证。

实现基于 [RecBole](https://github.com/RUCAIBox/RecBole) 和 [RecBole-GNN](https://github.com/RUCAIBox/RecBole-GNN)。使用代码或处理数据时请引用：

```bibtex
@inproceedings{hou2022core,
  author = {Yupeng Hou and Binbin Hu and Zhiqiang Zhang and Wayne Xin Zhao},
  title = {CORE: Simple and Effective Session-based Recommendation within Consistent Representation Space},
  booktitle = {{SIGIR}},
  year = {2022}
}

@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}
```
