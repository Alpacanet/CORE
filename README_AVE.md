# CORE-AVE

`core_ave.py` 中的 `COREave` 将历史商品向量等权平均，得到会话向量，再用共享商品表为全部候选商品打分。配置见 `configs/core_ave.yaml`，统一训练参数见 `configs/common.yaml`。

## 数据格式与模型

原始数据包含 `session_id:token`、`item_id_list:token_seq`、`item_id:token`。RecBole 把原始商品 token 映射为内部整数 ID；0 表示 padding。对于 batch 大小 B、序列宽度 S、商品数 V、向量维度 D，模型中的数据变化为：

```text
历史商品 ID：int64 [B, S]
  → 共享商品表查询：[B, S, D]
  → session dropout
  → 有效位置等权平均并归一化：[B, D]
  → 与归一化候选商品表 [V, D] 点积 / temperature
  → 全商品分数：[B, V]
```

默认 D=100、temperature=0.07。`ave_net()` 根据非零 ID 创建 `[B, S, 1]` 权重，有效历史长度为 n 时，每个有效位置权重为 1/n，padding 权重为 0。会话表示始终来自历史商品向量的加权和，因此保持 CORE 的表示空间一致性。

`calculate_loss()` 接收目标商品 `int64 [B]`，使用全商品交叉熵得到标量 loss。训练时候选商品表也使用 item dropout；`full_sort_predict()` 评估时关闭 dropout，返回 `[B, V]` 分数。训练 CE 包含 padding 类，评价时排除 ID 0；允许推荐历史中出现过的商品。

## 运行

```powershell
.\.venv\Scripts\python.exe train.py --model ave --smoke
.\.venv\Scripts\python.exe train.py --model ave
```

在 IDE 中打开 `train_ave.py` 点击 Run，会按默认设置启动全量训练：最多 30 轮、micro-batch 16、累积 16、学习率 0.001、seed 2020、`test=False`。需要小数据检查时将入口中的 `smoke` 改为 `True`。更多选项和模型恢复见 [训练指南](README_TRAINING.md)。

## 已保存实验

2026-10-05 的 Tmall 匹配基线 [20261005-115750-tmall-ave-9cf3dc](results/phase1/20261005-115750-tmall-ave-9cf3dc) 正常早停，完成 8 轮；最佳 epoch 为 1，验证 Recall@20=0.3471、MRR@20=0.1936，未评估全量测试集。该记录的 [metrics.json](results/phase1/20261005-115750-tmall-ave-9cf3dc/metrics.json) 与 [epochs.json](results/phase1/20261005-115750-tmall-ave-9cf3dc/epochs.json) 保留原样。跨模型比较和限制见 [LLM 实验记录](README_LLM.md)。
