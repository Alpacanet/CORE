# CORE：AVE、TRM 与 LLM

项目保留三个纯商品 ID 的会话推荐模型，统一使用 `train.py` 和 `trainer.py` 训练。输入是点击商品序列，目标是下一次点击的商品；模型输出全商品排序分数。三个模型的说明分别见 [AVE](README_AVE.md)、[TRM](README_TRM.md)、[LLM](README_LLM.md)，完整操作说明见 [训练指南](README_TRAINING.md)。

## 🎯 当前研究主线：review 与 explore

当前研究问题是：LLM 是否改善 explore、却损害 review，还是两组都没有收益？2026-10-09 已完成三组已有检查点的全量验证分组比较：LLM 的 explore 略有改善，review 下降主导整体损失；这是单 seed 验证信号，尚不证明稳定或显著收益。下一步按组分析门控与两路表示，随后做随机主干和多 seed 对照。完整表格与原始结果见 [分组复评记录](README_LLM.md)。

| 分组 | 依据真实下一商品的定义 |
| --- | --- |
| `review` | 目标出现在模型实际可见、排除 padding 的会话历史中 |
| `explore` | 目标没有出现在上述可见历史中 |

模型可见序列长度上限为 50，分组以截断后的输入为准；它不由预测 Top-1 决定，也不表示商品在完整未截断会话、训练集或全局商品表中从未出现。两组共用完整商品候选表和同一次排序，仅排除 padding，不按组过滤候选或屏蔽历史商品。定义和复评操作见 [训练指南](README_TRAINING.md)。

```text
CORE/
  core_ave.py                 # 平均池化模型
  core_trm.py                 # Transformer 学习历史位置权重
  core_llm.py                 # 连续商品向量输入 Qwen 的适配模型
  train.py                   # 统一命令行入口与模型恢复
  trainer.py                 # 共享训练器 CORETrainer
  evaluator.py               # overall / review / explore 指标累计
  evaluate.py                # 恢复已有检查点，仅复评，不训练
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

新训练默认在每轮验证中单次遍历统计 `overall`、`review`、`explore` 的样本数、比例及 Recall/MRR@10/@20，并保存到 `epochs.json`、`metrics.json` 与 `review_explore.json`。空组指标为 `null`；未舍入的整体指标应由非空两组按样本比例加权复原。`predictions.json` 增加真实目标的 `target_group`。2026-10-09 的独立复评已补充三组旧检查点的全量验证分组结果，原始训练记录保持原样；测试集仍未评估。

例如，仅复评已有 AVE 最佳检查点的验证集：

```powershell
.\.venv\Scripts\python.exe evaluate.py --run-directory results/phase1/20261005-115750-tmall-ave-9cf3dc --split valid --device cuda --eval-batch-size 32
```

复评会新建 `results/core/evaluation-.../`，保存 `review_explore.json` 和 `metrics.json`，不训练、不覆盖原运行。默认 `--split valid`；方案固定后显式指定 `--split test` 才评估测试集。加 `--limit 128` 只检查验证集前缀子集，结果范围标为 `prefix_subset`，不能作为全量分组结论；训练 smoke 的范围标为 `smoke_subset`。

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
