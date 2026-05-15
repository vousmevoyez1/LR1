# Qwen3a 自回归编码分析报告

## 执行摘要

**结论**：qwen3a模型在**训练和评估时都使用了自回归递归编码**，实现方式一致。

---

## 关键发现

### 1. 自回归递归编码的实现位置

**文件**：[qwen3_thought_wrapper.py:677-811](qwen3_thought_wrapper.py#L677-L811)

**方法**：`_forward_recursive_embedding_reasoning()`

**核心逻辑**：
```python
# 第742行：自回归循环
for step_idx in range(1, num_reasoning_steps + 1):
    # 1. 添加一个新token位置
    append_ids = torch.full((batch_size, 1), fill_value=append_token_id, ...)
    running_input_ids = torch.cat([running_input_ids, append_ids], dim=1)
    
    # 2. 用历史embedding替换新token位置的embedding
    step_inputs_embeds = embed_layer(running_input_ids)
    history_tensor = torch.stack(reasoning_history, dim=1)
    step_inputs_embeds[:, -history_tensor.size(1):, :] = history_tensor
    
    # 3. 前向传播生成新的embedding
    step_outputs = self.model(inputs_embeds=step_inputs_embeds, ...)
    
    # 4. 提取新embedding并加入历史
    e_prev = self._get_single_vector_embedding(step_outputs.last_hidden_state, ...)
    reasoning_history.append(e_prev)
```

**特点**：
- 每个推理步骤都进行一次完整的前向传播
- 每步生成的embedding作为下一步的输入
- 序列长度逐步增长：base_len → base_len+1 → base_len+2 → ... → base_len+K

---

### 2. 训练代码分析

**文件**：[engine.py:472-479](engine.py#L472-L479)

**Query编码**（使用thought tokens）：
```python
query_output = encode_batch_for_training(
    model=model,
    txt_batched=txt_batched,
    image_batched=image_batched,
    txt_mask=txt_mask,
    image_mask=image_mask,
    indices=query_indices,
    task_label=task_label,
    device=gpu_id,
    num_thought_tokens=num_thought_tokens,  # ← 关键参数
    use_final_token=enable_final_token,
    max_token_length=max_token_length,
    max_visual_pixels=max_visual_pixels,
    return_reasoning_steps=True,
)
```

**Candidate编码**（不使用thought tokens）：
```python
pos_cand_embeds = encode_batch_for_training(
    model=model,
    ...
    num_thought_tokens=cand_num_thought_tokens,  # = 0（非对称）或 num_thought_tokens（对称）
    use_final_token=cand_use_final_token,
    ...
)
```

---

### 3. 评估代码分析

**文件**：[engine.py:947-964](engine.py#L947-L964)

**Query编码**（使用thought tokens）：
```python
query_output = encode_batch_for_training(
    model=model,
    txt_batched=txt_batched,
    image_batched=image_batched,
    txt_mask=txt_mask,
    image_mask=image_mask,
    indices=query_indices,
    task_label=task_label,
    device=gpu_id,
    num_thought_tokens=num_thought_tokens,  # ← 关键参数
    use_final_token=enable_final_token,
    reason_steps=reason_steps,  # ← 注意：这个参数被忽略！
    use_cache_for_reasoning=use_cache_for_reasoning,
    max_token_length=max_token_length,
    max_visual_pixels=max_visual_pixels,
    return_reasoning_steps=True,
)
```

**Candidate编码**（不使用thought tokens）：
```python
pos_cand_embeds = encode_batch_for_training(
    model=model,
    ...
    num_thought_tokens=cand_num_thought_tokens,  # = 0（非对称）或 num_thought_tokens（对称）
    use_final_token=cand_use_final_token,
    reason_steps=0,
    ...
)
```

---

### 4. 参数传递链路

**完整调用链**：
```
engine.py: encode_batch_for_training(num_thought_tokens=K, reason_steps=R)
    ↓
engine.py:115-120: model(num_thought_tokens=K, ...)  # reason_steps被丢弃！
    ↓
qwen3_thought_wrapper.py:1390-1410: forward(num_thought_tokens=K, ...)
    ↓
qwen3_thought_wrapper.py:1406-1410: _forward_embeddings(num_thought_tokens=K, ...)
    ↓
qwen3_thought_wrapper.py:1198-1204: _forward_recursive_embedding_reasoning(num_reasoning_steps=K, ...)
    ↓
qwen3_thought_wrapper.py:742: for step_idx in range(1, num_reasoning_steps + 1):
```

**关键转换**（第1202行）：
```python
num_reasoning_steps=max(0, int(num_thought_tokens))
```

**重要发现**：
- ✅ `num_thought_tokens` 参数被正确传递并转换为 `num_reasoning_steps`
- ❌ `reason_steps` 参数在 `encode_batch_for_training` 中被接收但**从未使用**
- ✅ 实际的递归步数完全由 `num_thought_tokens` 决定

---

### 5. 训练日志验证

**用户提供的debug日志**：
```
[DEBUG] recursive-start: num_reasoning_steps=5, batch_size=16, base_seq_len=1234, e0_norm_mean=0.123456
[DEBUG] recursive-step-input: step=1/5, seq_len=1235, valid_len_sample0=1235, history_len=1, tail_replaced=1
[DEBUG] recursive-step-output: step=1/5, ek_norm_mean=0.234567, stored_steps=2
[DEBUG] recursive-step-input: step=2/5, seq_len=1236, valid_len_sample0=1236, history_len=2, tail_replaced=2
...
```

**日志来源**：[qwen3_thought_wrapper.py:726-805](qwen3_thought_wrapper.py#L726-L805)

**验证结果**：
- ✅ 日志显示 `num_reasoning_steps=5`（与配置中的 `num_thought_tokens=5` 一致）
- ✅ 序列长度逐步增长（1234 → 1235 → 1236 → ...）
- ✅ 每步都进行前向传播并生成新embedding
- ✅ 确认使用了自回归递归编码

---

## 训练与评估一致性验证

### 对比表

| 维度 | 训练代码 | 评估代码 | 一致性 |
|------|---------|---------|--------|
| **Query编码函数** | `encode_batch_for_training` | `encode_batch_for_training` | ✅ 相同 |
| **num_thought_tokens** | 从配置读取 | 从配置读取 | ✅ 相同 |
| **use_final_token** | `enable_final_token` | `enable_final_token` | ✅ 相同 |
| **调用链路** | model() → forward() → _forward_embeddings() → _forward_recursive_embedding_reasoning() | model() → forward() → _forward_embeddings() → _forward_recursive_embedding_reasoning() | ✅ 相同 |
| **递归步数** | `num_reasoning_steps = num_thought_tokens` | `num_reasoning_steps = num_thought_tokens` | ✅ 相同 |
| **自回归逻辑** | 742行for循环 | 742行for循环 | ✅ 相同 |

### 结论

**训练和评估使用完全相同的自回归递归编码实现**，不存在不一致问题。

---

## 配置参数说明

### 有效参数

1. **`num_thought_tokens`**（关键参数）
   - 作用：决定自回归递归的步数
   - 位置：配置文件中的 `model.num_thought_tokens`
   - 传递：engine.py → model.forward() → _forward_embeddings() → _forward_recursive_embedding_reasoning()
   - 效果：`num_thought_tokens=5` → 进行5步自回归递归

2. **`use_final_token`**
   - 作用：是否使用 `<final>` token进行pooling
   - 位置：配置文件中的 `model.enable_final_token`
   - 效果：影响最终embedding的提取方式

3. **`use_recursive_embedding_reasoning`**
   - 作用：是否启用自回归递归编码（默认True）
   - 位置：模型属性 `model.use_recursive_embedding_reasoning`
   - 效果：False时回退到legacy并行编码模式

### 无效参数

1. **`reason_steps`**（被忽略）
   - 位置：配置文件中的 `model.reason_steps`
   - 问题：在 `encode_batch_for_training` 中被接收但从未传递给模型
   - 建议：可以从代码中删除此参数，避免混淆

---

## 代码改进建议

### 1. 清理无效参数

**问题**：`reason_steps` 参数造成混淆

**建议**：
```python
# engine.py:23-40
def encode_batch_for_training(
    model,
    txt_batched,
    image_batched: torch.Tensor,
    txt_mask: torch.Tensor,
    image_mask: torch.Tensor,
    indices: List[int],
    task_label: str,
    device: torch.device,
    num_thought_tokens: int = 0,
    use_final_token: bool = False,
    # reason_steps: int = 0,  # ← 删除此行
    # use_cache_for_reasoning: bool = False,  # ← 删除此行（也未使用）
    max_token_length=15000,
    max_visual_pixels=501760,
    return_reasoning_steps: bool = False,
) -> Union[torch.Tensor, Any]:
```

### 2. 更新文档字符串

**当前**（第56行）：
```python
# - reason_steps / use_cache_for_reasoning are kept for compatibility but not used.
```

**建议**：
```python
# - Recursive reasoning steps are controlled by num_thought_tokens parameter.
# - Each thought token triggers one autoregressive reasoning step.
```

### 3. 统一参数命名

**建议**：在配置文件和代码注释中明确说明：
- `num_thought_tokens` = 自回归递归的步数
- 不需要单独的 `reason_steps` 参数

---

## 性能特征

### 计算复杂度

**单个样本的前向传播次数**：
- Base encoding: 1次
- Recursive steps: `num_thought_tokens` 次
- **总计**：`1 + num_thought_tokens` 次

**示例**（`num_thought_tokens=5`）：
- Query编码：6次前向传播（1次base + 5次递归）
- Candidate编码：1次前向传播（无递归）

### 内存占用

**序列长度增长**：
- Step 0: `base_len`
- Step 1: `base_len + 1`
- Step 2: `base_len + 2`
- ...
- Step K: `base_len + K`

**峰值内存**：最后一步的序列长度最长，占用内存最多

---

## 总结

1. ✅ **qwen3a训练和评估都使用自回归递归编码**
2. ✅ **实现方式完全一致**，调用相同的函数和逻辑
3. ✅ **递归步数由 `num_thought_tokens` 参数控制**
4. ⚠️ **`reason_steps` 参数被忽略**，建议从代码中删除
5. ✅ **训练日志验证了自回归递归的正确执行**

**不存在训练评估不一致的问题**。
