# 特殊Token Embedding 保存和加载检查报告

## 1. 特殊Token Embedding的结构

### 1.1 包装层 (ReasoningTokenEmbeddingWrapper)
**文件**: [`qwen3_thought_wrapper.py`](qwen3_thought_wrapper.py#L68-L144)

```
ReasoningTokenEmbeddingWrapper
├── base_embedding (冻结，不可训练)
│   └── weight: [vocab_size, hidden_dim]
└── special_embedding (可训练)
    └── weight: [num_special_tokens, hidden_dim]  ← 这是我们要保存的部分
```

**关键点**:
- `special_embedding` 是一个小的 `nn.Embedding` 层
- 只有 `special_tokens` 数量的行（通常 K+1 = 5 行）
- 其他整个vocabulary不参与训练，只有这5行梯度会被计算

### 1.2 初始化方法
**文件**: [`qwen3_thought_wrapper.py#L312-L325`](qwen3_thought_wrapper.py#L312-L325)

```python
def _random_init_reasoning_tokens(self, token_ids: List[int], semantic_token: str = "."):
    """
    使用语义 Token (如句号) 进行初始化，并加入微小噪声打破对称性。
    """
    # 1. 获取语义token的预训练embedding
    init_token_id = encode(semantic_token)  # "." -> token_id
    base_embed = embed_weight[init_token_id].clone()
    
    # 2. 为每个思维token加上微小噪声
    for token_id in token_ids:
        noise = randn() * 1e-5
        embed_weight[token_id].copy_(base_embed + noise)
```

**优势**:
- ✅ 基于语义初始化（以句号为基准）
- ✅ 加入微小噪声避免梯度爆炸
- ✅ 相比纯随机初始化更稳定

---

## 2. 特殊Token Embedding的保存

### 2.1 保存机制
**文件**: [`train.py#L73-E88`](train.py#L73-L88)

```python
def _collect_trainable_state_dict(model_to_save):
    """Collect trainable params only (LoRA + thought/final token embeddings)."""
    trainable = {}
    for name, param in model_to_save.named_parameters():
        if param.requires_grad:  # ← 只保存requires_grad=True的参数
            trainable[name] = param.detach().cpu()
    return trainable
```

**可训练参数包括**:
1. LoRA参数 (`peft_model.layers[*].lora_*`)
2. **special_embedding参数** (`model.get_input_embeddings().special_embedding.weight`)
3. **其他新增的可训练层**

### 2.2 Checkpoint保存结构
**文件**: [`train.py#L106-L113`](train.py#L106-L113)

```python
save_obj = {
    "trainable_state_dict": _collect_trainable_state_dict(model_to_save),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "config": config,
    "epoch": epoch,
    "global_step": global_step,
    "scaler": scaler.state_dict() if scaler is not None else None,
}
```

**特殊token embedding在checkpoint中的存储位置**:
```
trainable_state_dict
├── "model.model.get_input_embeddings().special_embedding.weight"  ← 形状 [5, 4096]
├── "model.model.language_model.peft_model.layers[*].lora_*"      ← LoRA参数
└── ... (其他可训练参数)
```

---

## 3. 特殊Token Embedding的加载

### 3.1 加载函数
**文件**: [`train.py#L133-L155`](train.py#L133-L155)

```python
def load_checkpoint_trainable(model, checkpoint_path: str):
    """Load trainable parameters from our checkpoint format."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["trainable_state_dict"]
    model_state = model.state_dict()

    loaded = 0
    missing = 0
    for name, tensor in state.items():
        if name in model_state:
            model_state[name].copy_(tensor)  # ← 直接拷贝（包括特殊token embedding）
            loaded += 1
        else:
            missing += 1

    return model, checkpoint
```

### 3.2 Resume Training流程
**文件**: [`train.py#L427-L437`](train.py#L427-L437)

```python
if ckpt_config.resume_training:
    checkpoint_path = os.path.join(config.uniir_dir, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
    model, checkpoint = load_checkpoint_trainable(model, checkpoint_path)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
```

---

## 4. 推理/测试时的Embedding加载

### 4.1 模型初始化
**文件**: [`qwen3_thought_wrapper.py#L169-L191`](qwen3_thought_wrapper.py#L169-L191)

```python
def __init__(self, model_name_or_path, ...):
    # 1. 从预训练路径加载模型 (包含所有特殊token的embedding)
    self.model = Qwen3VLForEmbedding.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        ...
    )
```

### 4.2 推理时的Setup
**文件**: [`qwen3_thought_wrapper.py#L194-L283`](qwen3_thought_wrapper.py#L194-L283)

```python
def setup_thought_tokens(
    self,
    num_thought_tokens: int,
    semantic_init_token: str = ".",
    skip_init: bool = False,  # ← 关键参数！
    enable_final_token: bool = True,
):
    # 1. 添加特殊token到tokenizer
    self.processor.tokenizer.add_special_tokens({'additional_special_tokens': all_reasoning_tokens})
    
    # 2. 扩展embedding层
    self.model.resize_token_embeddings(new_vocab_size)
    
    # 3. 包装embedding层为ReasoningTokenEmbeddingWrapper
    wrapped_embed = ReasoningTokenEmbeddingWrapper(base_embed, all_ids)
    self.model.set_input_embeddings(wrapped_embed)
```

### 4.3 ⚠️ 关键问题：推理时Embedding的正确加载

**当 `skip_init=False` 时**:
```python
if not skip_init:
    # 调用 _random_init_reasoning_tokens 进行初始化
    # 这会覆盖掉 resume_training 时加载的参数！
```

**解决方案**:
推理/测试时应该：
1. 加载预训练模型
2. 加载已训练的checkpoint
3. 设置 `skip_init=True` 以**避免重新初始化**

---

## 5. 当前代码检查

### 5.1 ✅ 训练时保存
- [x] 特殊token embedding被正确标记为 `requires_grad=True`
- [x] 通过 `_collect_trainable_state_dict()` 自动收集
- [x] 存储在checkpoint的 `trainable_state_dict` 中

### 5.2 ✅ Resume Training时加载
- [x] 通过 `load_checkpoint_trainable()` 恢复参数
- [x] 通过 `.copy_()` 方法正确更新权重

### 5.3 ⚠️ 推理/测试时加载 - 需要修复
- [ ] 需要在推理代码中添加checkpoint加载逻辑
- [ ] 需要设置 `skip_init=True` 以避免覆盖

---

## 6. 推荐的推理代码模板

```python
def load_trained_model(checkpoint_path: str, config_path: str):
    """
    Load trained model with learned special token embeddings.
    
    Args:
        checkpoint_path: Path to checkpoint with trainable_state_dict
        config_path: Path to training config
    
    Returns:
        Model with loaded embeddings
    """
    # 1. 加载配置
    config = OmegaConf.load(config_path)
    model_config = config.model
    
    # 2. 初始化模型（不初始化特殊token embedding）
    model = Qwen3VLThoughtWrapper(
        model_name_or_path=model_config.original_model_name,
        max_length=model_config.mbeir_max_text_length,
        torch_dtype=torch.bfloat16,
    )
    
    # 3. 设置特殊token（注意：skip_init=True！）
    model.setup_thought_tokens(
        num_thought_tokens=model_config.num_thought_tokens,
        semantic_init_token=model_config.semantic_init_token,
        skip_init=True,  # ← 关键：跳过初始化，使用预训练权重
        enable_final_token=model_config.enable_final_token,
    )
    
    # 4. 加载已训练的参数
    from models.qwen3.train import load_checkpoint_trainable
    model, checkpoint = load_checkpoint_trainable(model, checkpoint_path)
    
    # 5. 设置为评估模式
    model.eval()
    
    return model
```

---

## 7. 参数检查清单

### 配置参数一致性检查
```yaml
# 训练配置
trainer_config:
  num_train_epochs: 1
  gradient_accumulation_steps: 64

model:
  num_thought_tokens: 4              # ← 训练时
  enable_final_token: true
  semantic_init_token: "."
  skip_thought_init: false           # ← 训练时应为 false

# 推理配置（应该相同）
model:
  mbeir_num_thought_tokens: 4        # ← 推理时
  enable_final_token: true
  semantic_init_token: "."
  skip_thought_init: true            # ← 推理时应为 true
```

---

## 8. 总结

| 阶段 | 状态 | 说明 |
|------|------|------|
| 初始化 | ✅ | 特殊token以语义方式初始化（句号基底+微小噪声） |
| 训练时保存 | ✅ | 通过 `_collect_trainable_state_dict()` 自动收集 |
| Resume Training | ✅ | 通过 `load_checkpoint_trainable()` 正确恢复 |
| 推理时加载 | ⚠️ | **需要在推理脚本中显式加载checkpoint** |

**关键要点**:
- 推理时必须设置 `skip_init=True` 以避免重新初始化
- 推理时必须显式调用 `load_checkpoint_trainable()` 加载已训练的参数
- 配置中的 `skip_thought_init` 参数应该反映当前模式
