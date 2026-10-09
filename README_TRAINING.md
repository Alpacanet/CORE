# CORE 统一训练指南

AVE、TRM、LLM 共用 `train.py` 和 `trainer.py` 中的 `CORETrainer`。三模型默认采用相同的数据文件、seed、micro-batch、梯度累积、优化器和评价流程。模型配置是 `configs/core_ave.yaml`、`configs/core_trm.yaml`、`configs/core_llm.yaml`，公共设置在 `configs/common.yaml`。模型原理分别见 [AVE](README_AVE.md)、[TRM](README_TRM.md)、[LLM](README_LLM.md)。

当前研究首先通过 review/explore 分组回答：LLM 是否在 explore 上获益、在 review 上退步，还是两组均没有改善？新训练会自动保存分组评价；已有可信检查点可以用 [evaluate.ipynb](evaluate.ipynb) 单独复评，无需重新训练。

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

## 🔍 Review/explore 分组评价

分组依据是每条样本的真实下一商品 `item_id` 是否出现在模型实际可见的非 padding `item_id_list` 中：

| 分组 | 条件 |
| --- | --- |
| `review` | 可见历史中至少有一个非零 ID 等于真实目标 ID |
| `explore` | 可见历史中没有与真实目标 ID 相等的非零 ID |

序列长度上限为 50，定义只使用经过数据处理和截断后传入模型的历史。若目标仅出现在已截掉的历史中，该样本仍属于 `explore`。分组不使用预测 Top-1，也不表示目标在完整未截断会话、训练集或全局商品表中从未出现；它描述的是本条样本的可见历史与真实目标的关系。

模型仍为完整商品表产生一次排序，只排除 padding ID 0。两组不分别缩小候选集，不屏蔽历史商品。`evaluator.py` 按同一批全商品 Top-K 排名累计 `overall`、`review`、`explore` 的样本数 `count`、样本比例 `fraction`、Recall/MRR；默认 K 为 10 和 20。空组的 Recall/MRR 为 `null`，不按 0 分处理。非空两组指标按各自样本比例加权应等于未舍入的整体指标，空组不参与加权；RecBole 的常规指标摘要会另行舍入。

先检查分组比例，再用相同验证样本与排名协议比较 AVE/TRM/LLM 的组内指标和 `LLM - 基线` 差值。整体下降可能包含某组收益与另一组损失的抵消，必须用实际分组结果判断。之后才按组分析门控、CORE/LLM 分支表示和排序；随机冻结主干、多 seed 配对及等预算调参继续用于检验归因和稳定性。

## 输出与指标

每次运行创建新的 `results/core/日期-数据集-模型或变体-编号/`，打印 `CORE_RESULT_DIR=...`；正常结束还打印 `CORE_COMPLETE=...`。历史 `results/phase1/` 目录保持原样。

- `metrics.json`：运行状态、最佳验证指标、测试指标（如有），以及 `best_valid_review_explore` / `test_review_explore`；`complete` 为正常结束，`failed` 记录失败原因，主动中止为 `interrupted`。
- `epochs.json`：每轮平均训练 loss、样本数、优化器更新次数、耗时、训练峰值显存，以及 `valid_result` / `valid_review_explore`。整体与分组验证在一次遍历中完成。
- `review_explore.json`：分组定义、候选集和评价范围等元数据，以及最佳验证检查点的 `valid` / `test` 分组结果与 `best_epoch`；未执行测试时不填入测试指标。
- `predictions.json`：最佳检查点对 5 条样本的真实历史 ID、目标 ID、真实目标分组 `target_group`、Top-20 推荐 ID 与分数。全量不评估测试时来自验证集，查看 `prediction_split`；分数不是概率。
- `config.json` / `manifest.json`：完整配置、版本、参数量、输入文件与商品映射哈希。
- `best.pth`：最佳验证检查点；LLM 冻结权重不重复保存，重建需要相同缓存、模型版本、seed 与商品映射。
- `item_tokens.json`：内部 ID 到原始商品 token 的映射，推荐示例已经反映射。
- `smoke_indices.json`：小数据检查抽取的确定性样本位置，仅 smoke 运行包含。

MRR@20 是选择模型的主指标，Recall@20 衡量目标进入前 20 的比例，MRR@20 同时考虑排名。0.20 表示 20%，不是 0.20%；MRR 不是点击概率，`1 / MRR` 也不是平均排名。训练 loss 下降不保证排序指标提高。

2026-10-05 的 AVE/TRM 与 2026-10-02 的 LLM 原始结果及结论见 [LLM 实验记录](README_LLM.md)。这些记录未因重构重新训练或修改指标，原始运行尚未保存 review/explore 分组结果；新统计应通过独立复评补充。旧 CORE 日志使用不同 batch、设备和预算，只能作历史参考。判断 LLM 是否值得继续，先核对两组相对匹配 CORE 的增减，再检验随机冻结主干和多 seed 稳定性，并结合时间/显存成本；多 seed 可以采用 2020、2021、2022 配对复核。

## 🔄 仅复评已有检查点

[evaluate.ipynb](evaluate.ipynb) 恢复配置运行目录中的可信最佳检查点，不进行训练。它在 Notebook 内定义 `evaluate_experiment`，保留原有恢复、分组评价与报告保存方法，复用 `evaluator.py`、`trainer.py` 和 `train.py`。

当前项目 `.venv` 尚未安装 Jupyter 相关包。需要使用 Notebook 时安装可选依赖；这份清单包含基础依赖、`ipykernel>=6` 与 `jupyterlab>=4`：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_notebook.txt
```

如果恢复 LLM，还需要 `requirements_llm.txt` 中的依赖和固定版本的模型缓存；Notebook 依赖清单不包含 LLM 的扩展依赖。

1. 在 PyCharm、VS Code 或 Jupyter 中打开 `evaluate.ipynb`。
2. 选择项目 `.venv` 的 Python 内核，即 `D:\DataSciencePractise\CORE\.venv\Scripts\python.exe` 对应的环境。
3. 在参数单元格选择一个检查点目录、split、设备、batch 和样本范围。
4. 从上到下运行导入、参数、函数定义、执行和展示单元格。

Notebook 的 `RUN_DIRECTORIES` 字典已映射三组已有最佳检查点。参数默认选 AVE，也可改为 TRM 或 LLM；每次执行只复评当前选中的一个模型：

```python
RUN_DIRECTORY = RUN_DIRECTORIES["ave"]
# RUN_DIRECTORY = RUN_DIRECTORIES["trm"]
# RUN_DIRECTORY = RUN_DIRECTORIES["llm"]
SPLIT = "valid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_BATCH_SIZE = 32
LIMIT = None
```

也可把 `RUN_DIRECTORY` 设为新的 `results/core/` 训练目录。原模型的依赖、数据哈希和商品映射校验仍适用。`DEVICE` 在 CUDA 可用时默认 `"cuda"`，否则为 `"cpu"`，也可显式设为 CPU；实际速度和精度差异需记录。评价 batch 默认显式设为 32。

函数定义单元格运行后，执行单元格调用：

```python
result = evaluate_experiment(
    RUN_DIRECTORY,
    split=SPLIT,
    device=DEVICE,
    eval_batch_size=EVAL_BATCH_SIZE,
    limit=LIMIT,
)
```

随后显示 `overall`、`review`、`explore` 的样本数、比例及 Recall/MRR@10/@20 DataFrame。可选的历史比较单元格只读取已保存的 2026-10-09 三组结果，不自动额外复评其他模型。

`LIMIT=None` 表示所选 split 的全量复评；只想检查流程时设 `LIMIT=128`，结果范围标为 `prefix_subset`，只能描述前缀子集，不能视为全量效果报告。新训练的 smoke 范围仍为 `smoke_subset`。默认 `SPLIT="valid"` 不评估测试集；方案固定后显式改为 `SPLIT="test"`，才执行最终测试。

每次复评创建独立的 `results/core/evaluation-.../`，保存 `review_explore.json` 和 `metrics.json`，记录来源运行目录、检查点 SHA256、实际检查点轮次 `best_epoch`、split 和样本范围等元数据。原始训练目录、指标与检查点保持原样。终端打印 `CORE_EVALUATION_DIR`，成功结束打印 `CORE_EVALUATION_COMPLETE`。

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

2026-10-09 的接入检查包含 37 项单元与集成测试，以及 AVE/TRM/tiny LLM 三组工程 smoke。检查同时修正了 RecBole 1.2.1 评价中的旧 NumPy 别名兼容问题和 CUDA 可用时忽略 `--device cpu` 的设备选择问题；兼容处理限于评价调用，不修改已安装的依赖或全局 NumPy。模型结构、训练损失和历史检查点不变。
