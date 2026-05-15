# 微调Checkpoint评估脚本

评估使用不同 `reason_steps` 训练的微调 checkpoint 的检索性能。

## Checkpoint 配置

| 名称 | Checkpoint路径 | 训练reason_steps |
|------|---------------|-----------------|
| MT0 | `checkpointMT0/.../jina_v4_step_600.pth` | 0 |
| MT1 | `checkpointMT1/.../jina_v4_step_100.pth` | 1 |
| MT3 | `checkpointMT3/.../jina_v4_step_100.pth` | 3 |

## 关键特性

1.  **与现有代码保持一致**: 使用 `mbeir_embedder.py` 和 `mbeir_retriever.py`
2. **独立输出目录**: 每个checkpoint的结果保存在 `runs/eval_finetuned/{MT0,MT1,MT3}/`

## 使用方法

```bash
cd /data/LR1/src/models/jina_v4/jina_v4/configs/large/eval/inbatch

# 运行完整评估
./eval_finetuned_ckpts.sh
```

## 输出结构

```
/data/LR1/runs/eval_finetuned/
├── MT0/
│   ├── embed/JinaV4/Large/Instruct/InBatch/
│   │   ├── test/           # Query embeddings
│   │   └── cand_pool 
│   ├── indices/...
│   └── retrieval_results/...
├── MT1/...
└── MT3/...
```

## 结果汇总

结果保存在: `/data/LR1/retrieval_results_summary/finetuned_ckpts/`
