# 验证损失检查设置文档

## 概述
已成功配置训练代码以支持**验证集损失的多级别检查**：
1. **步骤级验证（Step-level）**：每隔 N 步运行一次验证
2. **周期级验证（Epoch-level）**：每隔 M 个 epoch 运行一次验证

---

## 配置参数

### 文件：`qwen3/configs/2b/train/inbatch/inbatch.yaml`

#### 评估器配置（Evaluator）
```yaml
evaluator:
  enable_eval: true                    # 启用验证
  enable_retrieval_eval: false         # 禁用检索评估（仅使用对比损失）
  eval_freq: 1                         # 每 1 个 epoch 进行一次周期级验证
  eval_steps: 100                      # 每 100 个步骤进行一次步骤级验证（0=禁用）
  print_freq: 10                       # 每 10 个步骤打印一次训练进度
```

#### 训练器配置（Trainer）
```yaml
trainer_config:
  gradient_accumulation_steps: 64      # 梯度累积步数
  init_lr: 1e-4                        # 初始学习率
  special_token_lr: 1e-3               # 特殊 token 学习率
  num_train_epochs: 1                  # 训练 epoch 数
  print_freq: 5                        # 打印频率
  weight_decay: 0.05                   # 权重衰减
  save_steps: 50                       # 每 50 步保存一次检查点
  save_total_limit: 5                  # 保存检查点数量限制
  eval_steps: 100                      # 可选：覆盖 evaluator.eval_steps
```

---

## 验证流程详解

### 1. 步骤级验证（Step-level Validation）
**位置**：[engine.py](engine.py#L373-L392) 中的 `train_one_epoch()` 函数

**触发条件**：
```python
if eval_steps > 0 and global_step % eval_steps == 0:
```

**执行流程**：
1. 每更新一次参数后检查当前 `global_step`
2. 如果 `global_step % eval_steps == 0`，则运行验证
3. 验证集通过 `eval_engine()` 评估
4. 结果输出到控制台和 SwanLab

**输出示例**：
```
============================================================
Running validation at step 100...
============================================================
Validation Results (Step 100):
  loss: 2.156234
  inbatch_accuracy: 0.782456
============================================================
```

### 2. 周期级验证（Epoch-level Validation）
**位置**：[train.py](train.py#L258-L280) 中的 `train()` 函数

**触发条件**：
```python
if val_loader is None or epoch % eval_freq != 0:
    # 不验证
else:
    # 运行验证
    val_stats = eval_engine(...)
```

**执行流程**：
1. 每个 epoch 完成后检查 `epoch % eval_freq`
2. 如果条件满足，运行完整的验证集评估
3. 结果输出到控制台和 SwanLab

---

## 验证损失计算

### 对比损失（InfoNCE Loss）
**计算位置**：[engine.py](engine.py#L176-L234) 中的 `compute_contrastive_loss()`

**损失计算过程**：
1. **归一化**：Query 和 Positive Candidate 嵌入向量
2. **相似度矩阵**：计算 $\text{sim}[i,j] = \frac{\text{query}[i] \cdot \text{pos\_cand}[j]}{\tau}$
   - $\tau$ = 温度参数 (默认 0.07)
3. **In-batch 负样本**：其他正样本作为负样本
4. **交叉熵损失**：$\text{loss} = -\log\frac{\exp(\text{sim}_{i,i})}{\sum_j \exp(\text{sim}_{i,j})}$
5. **准确率**：In-batch 准确率（正样本排名第一的比例）

### 验证集配置
```yaml
data_config:
  val_query_data_path: /data/M-BEIR/query/union_val/mbeir_union_val.jsonl
  val_cand_pool_path: /data/M-BEIR/cand_pool/global/mbeir_union_val_cand_pool.jsonl
  
dataloader_config:
  valid_batch_size: 64                 # 验证集批大小
  num_workers: 2                       # 数据加载线程数
```

---

## 输出与日志

### 控制台输出
每次验证后输出格式化结果：
```
============================================================
Validation Results (Step 100):
  loss: 2.156234
  inbatch_accuracy: 0.782456
============================================================
```

### SwanLab 云端看板记录
自动记录以下指标：
- `val/loss`: 验证集损失
- `val/inbatch_accuracy`: 验证集 in-batch 准确率
- `train/loss`: 训练损失
- `train/lr`: 学习率
- `train/inbatch_accuracy`: 训练 in-batch 准确率
- `train/grad_norm`: 梯度范数

**项目**：`qwen3vl-training`  
**模式**：`cloud`（需要 SwanLab 账户）

---

## 模型评估模式

### 评估时的模型状态
[engine.py](engine.py#L618) 中 `eval_engine()` 函数：
```python
@torch.no_grad()
def eval_engine(model_without_ddp, model, data_loader, gpu_id, config):
    model.eval()  # 设置评估模式
    # - Batch Normalization 使用全局统计
    # - Dropout 禁用
    # - 无梯度计算
```

### Thought Token 配置
- **训练时**：Query 使用 4 个思维 token (`num_thought_tokens: 4`)
- **验证时**：Query 同样使用 4 个思维 token（配置一致）
- **正样本**：无论训练还是验证都不使用思维 token

---

## 配置建议

### 快速测试
```yaml
evaluator:
  eval_steps: 50           # 每 50 步验证一次（快速反馈）

trainer_config:
  gradient_accumulation_steps: 1
  num_train_epochs: 1
```

### 生产训练
```yaml
evaluator:
  eval_steps: 500          # 每 500 步验证一次（减少计算开销）
  eval_freq: 1             # 每 epoch 验证一次

trainer_config:
  gradient_accumulation_steps: 64
  num_train_epochs: 10
```

### 仅周期级验证（禁用步骤级）
```yaml
evaluator:
  eval_steps: 0            # 禁用步骤级验证
  eval_freq: 1             # 仅保留周期级验证
```

---

## 代码修改总结

### 修改的文件
1. **qwen3/configs/2b/train/inbatch/inbatch.yaml**
   - 添加 `evaluator.eval_steps: 100`
   - 添加 `trainer_config.eval_steps: 100`

2. **train.py**
   - 修改 `train()` 函数签名，添加 `eval_steps` 参数传递
   - 添加 SwanLab 日志记录验证指标

3. **engine.py**
   - 修改 `train_one_epoch()` 函数签名，支持 `val_loader`, `eval_steps`, `model_without_ddp` 参数
   - 在梯度更新后添加步骤级验证逻辑（第 373-392 行）
   - 验证结果输出到控制台和 SwanLab

---

## 验证执行时间线

### 总结表
| 参数 | 值 | 含义 |
|------|-----|------|
| 步骤级验证 | 100 步 | 每进行 100 次梯度更新后运行一次验证 |
| 周期级验证 | 1 epoch | 每个 epoch 完成后运行一次验证 |
| 梯度累积 | 64 步 | 64 个批次的梯度累积后更新参数 |
| 训练 epoch | 1 | 总共训练 1 个 epoch |

### 预期验证次数（单个 epoch）
假设训练集有 N 个样本，批大小为 64，梯度累积步数为 64：
- **批数**：$\lceil N / 64 \rceil$
- **梯度更新次数**：$\lceil (N / 64) / 64 \rceil = \lceil N / 4096 \rceil$
- **步骤级验证次数**：$\lfloor \lceil N / 4096 \rceil / 100 \rfloor$

例如，若 $N = 400,000$：
- 梯度更新次数 ≈ 98
- 步骤级验证次数 ≈ 0（少于 100 次更新）
- 周期级验证次数 = 1

---

## 故障排查

### 验证损失为 NaN
**原因**：
- 嵌入向量包含 NaN
- 温度参数过小导致数值不稳定

**解决**：
1. 检查 `max_token_length` 和 `max_visual_pixels` 设置
2. 增加温度参数 `temperature: 0.1`（默认 0.07）

### 验证速度过慢
**原因**：
- `eval_steps` 设置过小
- 验证集过大

**解决**：
- 增加 `eval_steps` 值（如改为 500）
- 减小 `valid_batch_size`

### SwanLab 日志未出现
**原因**：
- SwanLab 未配置或未启用
- 网络连接问题

**检查**：
```yaml
swanlab_config:
  enabled: true            # 确保为 true
  mode: cloud              # 连接到云端
```

---

## 相关文件位置
- 配置文件：[qwen3/configs/2b/train/inbatch/inbatch.yaml](qwen3/configs/2b/train/inbatch/inbatch.yaml)
- 训练脚本：[train.py](train.py)
- 引擎代码：[engine.py](engine.py)
- 数据配置：[qwen3/configs/2b/train/inbatch/inbatch.yaml#L1-L17](qwen3/configs/2b/train/inbatch/inbatch.yaml#L1-L17)
