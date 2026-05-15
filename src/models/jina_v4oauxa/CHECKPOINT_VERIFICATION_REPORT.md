# Stage-2 训练权重保存/加载验证报告

## 执行日期


## 验证目标
确认Stage-2训练时hard_gate、LoRA、embedding映射层和thought token embeddings的权重是否正确：
1. 加载
2. 训练
3. 保存

---

## 1. 权重冻结逻辑 ✓

**函数**: `freeze_all_except_hard_gate()` (train.py:131-209)

**逻辑**:
```python
1. 冻结所有参数: param.requires_grad = False
2. 解冻hard_gate: gate_module.parameters() -> requires_grad = True
3. 解冻LoRA: 'lora_' in name -> requires_grad = True  
4. 解冻embedding映射: multi_vector_projector -> requires_grad = True
5. 解冻thought embeddings: embed_tokens.weight (通过grad hook)
```

**验证结果**: ✓ 正确
- hard_gate的requires_grad会被正确设置为True
- 打印输出会显示各部分的参数数量

---

## 2. 权重保存逻辑 ✓

**函数**: `save_checkpoint()` (train.py:374-420)

**关键代码** (第393-395行):
```python
adapter_state_dict = {}
for name, param in model_to_save.named_parameters():
    if param.requires_grad:  # 只保存trainable参数
        adapter_state_dict[name] = param.data.cpu()
```

**保存的参数**:
- LoRA参数: `base_model.model.*.lora_A/lora_B`
- Hard gate参数: `base_model.model.step_predictor_gate.*`
- Embedding映射: `base_model.model.multi_vector_projector.*`
- Thought embeddings: `base_model.model.model.language_model.embed_tokens.weight`

**验证结果**: ✓ 正确
- 所有requires_grad=True的参数都会被保存
- 使用`param.data.cpu()`确保保存的是当前训练的权重

---

## 3. 权重加载逻辑 ✓

**函数**: `load_lora_checkpoint()` (train.py:442-506)

**关键代码** (第471-473行):
```python
for name, param in adapter_state_dict.items():
    if name in model_state_dict:
        model_state_dict[name].copy_(param)
```

**验证结果**: ✓ 正确
- checkpoint中的所有参数都会被加载
- 不会过滤或跳过任何参数
- 使用`.copy_()`确保权重被正确覆盖

---

## 4. 执行顺序验证 ✓

### 首次训练 (training_stage=2, resume_training=false)

```
1. 第671行: 加载预训练模型 JinaEmbeddingsV4Model.from_pretrained()
2. 第707行: setup_thought_tokens(5)
3. 第724行: setup_hard_gate() → 创建gate（随机初始化）
4. 第728行: freeze_all_except_hard_gate() → 设置requires_grad
5. 第753行: 创建optimizer（只包含requires_grad=True的参数）
6. 训练...
7. 第609行: save_checkpoint() → 保存所有trainable参数
```

**验证结果**: ✓ 正确

### Resume训练 (training_stage=2, resume_training=true)

```
1. 第671行: 加载预训练模型
2. 第707行: setup_thought_tokens(5)
3. 第724行: setup_hard_gate() → 创建gate结构（随机初始化）
4. 第728行: freeze_all_except_hard_gate() → 设置requires_grad
5. 第753行: 创建optimizer
6. 第770行: load_lora_checkpoint() → 加载checkpoint，覆盖gate权重 ✓
7. 第771行: optimizer.load_state_dict() → 恢复optimizer状态
8. 训练...
```

**验证结果**: ✓ 正确
- setup_hard_gate()在load_checkpoint()之前调用，创建gate结构
- load_checkpoint()会加载训练好的权重，覆盖随机初始化的权重
- setup_hard_gate()有保护机制：如果gate已存在且不强制重新初始化，会直接返回

---

## 5. setup_hard_gate()幂等性验证 ✓

**函数**: `setup_step_predictor_gate()` (modeling_jina_embeddings_v4.py:444-479)

**关键代码** (第464-465行):
```python
if self.step_predictor_gate is not None and not force_reinit:
    return  # 如果gate已存在，直接返回，不重新初始化
```

**验证结果**: ✓ 正确
- 默认情况下force_reinit=False
- 如果gate已存在，不会重新初始化
- 这确保了多次调用setup_hard_gate()是安全的

---

## 6. Optimizer参数验证 ✓

**代码** (train.py:752-758):
```python
trainable_params = [p for p in model.parameters() if p.requires_grad]

optimizer = torch.optim.AdamW(
    params=trainable_params,
    lr=trainer_config.init_lr,
    weight_decay=trainer_config.weight_decay,
)
```

**验证结果**: ✓ 正确
- optimizer只包含requires_grad=True的参数
- 包括hard_gate、LoRA、embedding映射层和thought embeddings

---

## 7. 梯度反向传播验证 ✓

**Stage-2 Loss计算** (engine.py:501-579):
```python
def compute_hard_gate_stage2_loss(...):
    # 1. Gate选择thought embedding
    query_embeds, step_logits, selected_steps = base_model.select_best_thought_embedding(
        thought_tokens,
        query_context_embeddings=query_context_embeds,
    )
    
    # 2. 计算InfoNCE loss
    sim_matrix = torch.matmul(query_norm, pos_norm.t()) * scale
    loss = F.cross_entropy(sim_matrix, targets)
    
    return loss, inbatch_accuracy, stats
```

**验证结果**: ✓ 正确
- loss依赖于gate的输出（selected_steps）
- 梯度会通过gate反向传播
- gate参数会被更新

---

## 8. 潜在问题检查

### 问题1: 配置文件中batch size是否恢复？ ⚠️

**检查**: inbatch.yaml第19-20行
```yaml
train_batch_size: 32  # 从32降到16以减少显存占用
valid_batch_size: 32  # 从32降到16
```

**状态**: ⚠️ 注释说要降到16，但实际值还是32
**建议**: 确认是否需要修改为16

### 问题2: debug_aux_loss是否关闭？ ✓

**检查**: inbatch.yaml第71行
```yaml
debug_aux_loss: false  # 关闭梯度调试以节省显存
```

**状态**: ✓ 已正确关闭

---

## 总结

### ✓ 所有核心逻辑都正确

1. **权重冻结**: ✓ hard_gate的requires_grad正确设置为True
2. **权重保存**: ✓ 所有trainable参数都会被保存
3. **权重加载**: ✓ checkpoint中的所有参数都会被加载
4. **执行顺序**: ✓ setup_hard_gate() -> load_checkpoint()顺序正确
5. **Optimizer**: ✓ 包含所有trainable参数
6. **梯度传播**: ✓ loss依赖gate输出，梯度会反向传播

### ⚠️ 需要确认的配置

1. **batch size**: 配置文件注释说要改为16，但实际值还是32
   - 建议：确认是否需要修改

### 📝 建议

如果训练过程中发现gate权重没有更新，可以添加调试代码：

```python
# 在train.py的train()函数中，每个epoch结束后添加：
if utils.is_main_process() and training_stage == 2:
    gate_params = [p for n, p in model_without_ddp.named_parameters() 
                   if 'step_predictor_gate' in n or 'hard_gate' in n]
    for i, p in enumerate(gate_params):
        print(f"Gate param {i}: norm={p.data.norm().item():.6f}, "
              f"grad_norm={p.grad.norm().item() if p.grad is not None else 0:.6f}")
```

这样可以监控gate参数的更新情况。
