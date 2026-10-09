# CORE 统一训练指南

AVE、TRM、LLM 共用 `train.py` 和 `trainer.py` 中的 `CORETrainer`。三模型默认采用相同的数据文件、seed、micro-batch、梯度累积、优化器和评价流程。模型配置是 `configs/core_ave.yaml`、`configs/core_trm.yaml`、`configs/core_llm.yaml`，公共设置在 `configs/common.yaml`。模型原理分别见 [AVE](README_AVE.md)、[TRM](README_TRM.md)、[LLM](README_LLM.md)。

## 环境与数据

选择项目 `.venv` 解释器。迁移环境时，AVE/TRM 安装基础依赖，LLM 安装含基础依赖的扩展清单：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements_llm.txt
```

基础清单包含 PyTorch、NumPy、RecBole、tqdm；LLM 额外包含固定版本的 Transformers、PEFT、Accelerate。PyTorch 的 CUDA/CPU 版本应与实际设备匹配。

数据读取 `dataset/<数据集>/<数据集>.train.inter`、`.valid.inter`、`.test.inter`，不重新划分。字段为 `session_id:token`、`item_id_list:token_seq`、`item_id:token`。RecBole 建立内部商品 ID 映射，0 为 padding；历史输入是 `int64 [B, S]`，下一商品标签是 `int64 [B]`。模型采用全商品交叉熵，评价仅屏蔽 padding，不屏蔽历史商品，因为目标允许重复点击。

LLM 默认使用 `Qwen/Qwen3-0.6B-Base`，固定 revision 为 `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`，缓存位于 `models/hf/`。首次运行会下载约 1.2 GB 权重；已有缓存可加 `--local-files-only`。本实现通过 `inputs_embeds` 输入商品连续向量，不下载 tokenizer。

## 小数据检查与 IDE 入口

先分别执行以下命令：

```powershell
.\.venv\Scripts\python.exe train.py --model ave --smoke
.\.venv\Scripts\python.exe train.py --model trm --smoke
.\.venv\Scripts\python.exe train.py --model llm --variant pretrained_lora --smoke
```

`--smoke` 使用一轮、训练 256 条、验证和测试各 128 条，同时保留完整商品候选表。它检查训练、验证、测试、保存和推荐输出是否连通，短跑分数不能当作正式效果结论。离线工程调试可以使用 `--model llm --variant tiny_random --smoke`；该变体是小型随机 Qwen，不是正式预训练对照。

在 PyCharm 中打开 `train_ave.py`、`train_trm.py` 或 `train_llm.py`，选择项目解释器，点击 Run。三个文件已各自固定模型，默认 `smoke=False`、`test=False`，会启动全量训练；若只想检查管线，先把对应文件的 `smoke` 改为 `True`。修改入口的 `run_experiment(...)` 参数即可调整训练配置。

## 正式训练与 LLM 对照

默认 Tmall、seed 2020、最多 30 epochs、micro-batch 16 × 累积 16 = 有效 batch 256，学习率 0.001，梯度裁剪最大范数 5。序列上限 50，评价 batch 32，按验证 MRR@20 保存最佳检查点。`stopping_step=5` 沿用 RecBole 实现：`cur_step > stopping_step` 才停止，即连续 6 次低于最佳验证指标时触发；持平会重置计数。

分别运行以下命令，每次运行占用一组训练资源：

```powershell
.\.venv\Scripts\python.exe train.py --model ave --epochs 30
.\.venv\Scripts\python.exe train.py --model trm --epochs 30
.\.venv\Scripts\python.exe train.py --model llm --variant pretrained_lora --epochs 30
```

全量默认仅评估验证集，不评估测试集。配置固定后在对应命令加 `--test`，才进行最终测试报告。测试集不用于选择超参数。

LLM 默认 LoRA rank 8 / alpha 16 / dropout 0.05，预训练本体冻结，训练商品表、输入输出投影、LayerNorm、门控与 LoRA。冻结本体仍保留到输入投影的梯度路径。商品 embedding 从头训练，不加载旧 CORE 检查点。输入和输出目前使用单层 Linear 投影。

需要判断预训练和 LoRA 的贡献时，可运行：

```powershell
.\.venv\Scripts\python.exe train.py --model llm --variant pretrained_frozen --epochs 30
.\.venv\Scripts\python.exe train.py --model llm --variant random_lora --epochs 30
```

`pretrained_frozen` 仅训练适配模块等；`random_lora` 使用相同 Qwen 架构、投影、门控、LoRA 设置，但主干随机且冻结，控制冻结预训练权重的贡献。它不等于完整从头训练大 Transformer。`random_full` 全参数 Adam 的显存成本更高，入口拒绝在小于 12 GB 的 GPU 上直接运行；12 GB 也不保证足够。

可选 `--fusion llm_only` 去掉门控 CORE 分支；`--core-branch trm` 则与 TRM 分支融合。正式比较保持相同 micro-batch、累积、seed 与调参预算。累积按实际样本数加权，处理最后不足窗口的 batch；dropout 的随机抽样使它不保证与单个大 batch 数值完全相同。

## 输出与指标

每次运行创建新的 `results/core/日期-数据集-模型或变体-编号/`，打印 `CORE_RESULT_DIR=...`；正常结束还打印 `CORE_COMPLETE=...`。历史 `results/phase1/` 目录保持原样。

- `metrics.json`：运行状态、最佳验证指标、测试指标（如有）；`complete` 为正常结束，`failed` 记录失败原因，主动中止为 `interrupted`。
- `epochs.json`：每轮平均训练 loss、样本数、优化器更新次数、耗时、训练峰值显存。
- `predictions.json`：最佳检查点对 5 条样本的真实历史 ID、目标 ID、Top-20 推荐 ID 与分数。全量不评估测试时来自验证集，查看 `prediction_split`；分数不是概率。
- `config.json` / `manifest.json`：完整配置、版本、参数量、输入文件与商品映射哈希。
- `best.pth`：最佳验证检查点；LLM 冻结权重不重复保存，重建需要相同缓存、模型版本、seed 与商品映射。
- `item_tokens.json`：内部 ID 到原始商品 token 的映射，推荐示例已经反映射。
- `smoke_indices.json`：小数据检查抽取的确定性样本位置，仅 smoke 运行包含。

MRR@20 是选择模型的主指标，Recall@20 衡量目标进入前 20 的比例，MRR@20 同时考虑排名。0.20 表示 20%，不是 0.20%；MRR 不是点击概率，`1 / MRR` 也不是平均排名。训练 loss 下降不保证排序指标提高。

2026-10-05 的 AVE/TRM 与 2026-10-02 的 LLM 原始结果及结论见 [LLM 实验记录](README_LLM.md)。这些记录未因重构重新训练或修改指标。旧 CORE 日志使用不同 batch、设备和预算，只能作历史参考。判断 LLM 是否值得继续，需要稳定优于匹配 CORE 基线、优于随机冻结主干，并结合时间/显存成本；多 seed 可以采用 2020、2021、2022 配对复核。

## 恢复模型与实现检查

统一入口提供 `load_experiment`，可加载新的运行目录，也可传入保留的历史目录：

```python
from train import load_experiment

model, config, splits = load_experiment(
    "results/core/你的运行目录",
    device="cpu",
)
# 历史结果同样通过此接口加载：
# model, config, splits = load_experiment("results/phase1/原运行目录")
```

`core_ave.py`、`core_trm.py`、`core_llm.py` 与模型类名保持稳定，便于恢复旧检查点。加载时核对原始数据文件和商品映射，返回完整数据划分，不是 smoke 子集。默认设备来自保存配置，可显式传 `device="cpu"`。LLM 需要完整模型缓存和 LLM 依赖；BF16 在 CPU/GPU 上可能有数值差异，接近分数的商品排序不保证逐位一致。

早期 tiny LLM 检查点可能缺少严格校验需要的元数据。例如 `results/phase1/20261002-124359-tmall-tiny_random-1bf528` 缺少 `temperature`、`sess_dropout`、`item_dropout` 签名字段，当前加载器会拒绝它。这是已有的旧格式限制；重构保留原检查点，也没有绕过校验。

模型恢复用于预测，不恢复优化器与中断训练进度；重新运行会创建新目录并从头训练。只加载本项目可信的 `.pth` 检查点，它使用 Python pickle 格式。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

单元测试不下载模型，使用本地 tiny Qwen 检查 causal mask、padding、冻结梯度、LoRA 隔离、紧凑检查点恢复和商品映射不匹配拒绝。真实 Tmall smoke 检查 RecBole 与 GPU 管线；二者都不替代科学效果评估。
