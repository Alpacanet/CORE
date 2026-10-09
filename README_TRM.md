# CORE-TRM

`core_trm.py` 中的 `COREtrm` 用小型因果 Transformer 为历史位置学习权重，再对共享商品向量加权求和。配置见 `configs/core_trm.yaml`，统一训练参数见 `configs/common.yaml`。

## 数据格式与模型

原始数据字段和 RecBole 内部 ID 映射与 [AVE](README_AVE.md) 相同；0 表示 padding。对于 batch 大小 B、序列宽度 S、商品数 V、向量维度 D：

```text
历史商品 ID：int64 [B, S]
  → 共享商品向量 x：[B, S, D]
  → x 加位置向量、LayerNorm、dropout
  → 因果 Transformer 隐状态：[B, S, D]
  → Linear(D, 1) 与有效位置 softmax：alpha [B, S, 1]
  → sum(alpha * x) 并归一化：会话向量 [B, D]
  → 全商品点积 / temperature：分数 [B, V]
```

默认 D=100、temperature=0.07。`TransNet` 使用 causal mask 屏蔽未来位置，并对 padding 位置设置极低权重 logits，再沿序列维执行 softmax。Transformer 输出用于计算 `alpha`；最终会话表示是历史商品向量 `x` 的加权和，从而保持与候选商品相同的表示空间。

`COREtrm` 继承 `COREave` 的 `calculate_loss()` 与 `full_sort_predict()`。这些方法调用 `self.forward()` 时会使用 TRM 的实现，因此仍采用目标 `int64 [B]` 和预测 `[B, V]` 的全商品交叉熵。训练使用 session/item dropout，评估关闭 dropout、排除 padding ID 0，并允许推荐历史商品。

## 运行

```powershell
.\.venv\Scripts\python.exe train.py --model trm --smoke
.\.venv\Scripts\python.exe train.py --model trm
```

在 IDE 中打开 `train_trm.py` 点击 Run，会按默认设置启动全量训练：最多 30 轮、micro-batch 16、累积 16、学习率 0.001、seed 2020、`test=False`。需要小数据检查时将 `smoke` 改为 `True`。更多选项和模型恢复见 [训练指南](README_TRAINING.md)。

## 已保存实验

2026-10-05 的 Tmall 匹配基线 [20261005-120616-tmall-trm-32ec4d](results/phase1/20261005-120616-tmall-trm-32ec4d) 正常早停，完成 8 轮；最佳 epoch 为 1，验证 Recall@20=0.3457、MRR@20=0.1936，未评估全量测试集。该记录的 [metrics.json](results/phase1/20261005-120616-tmall-trm-32ec4d/metrics.json) 与 [epochs.json](results/phase1/20261005-120616-tmall-trm-32ec4d/epochs.json) 保留原样。跨模型比较和限制见 [LLM 实验记录](README_LLM.md)。
