from __future__ import annotations

import math
import warnings
from typing import Any, Optional, Union, List

import torch
import torch.nn as nn

from peft.tuners.lora import LoraLayer

### 本类实现**“多任务、多适配器”的 Linear 封装：同一底层线性层上，维护多组（adapter × task）的 A/B 低秩分解权重，并在前向中按任务选择路由**。
class MultiAdapterLinear(nn.Module, LoraLayer):
    """
    Custom LoRA module supporting multiple adapters for a linear layer.
    
    This module extends the standard LoRA implementation to support multiple task-specific
    adapters that can be dynamically selected during the forward pass. The task_label
    parameter passed to the forward function determines which LoRA adapter(s) to use:
    - If task_label is a string, all examples in the batch use the same adapter
    - If task_label is a list of strings, each example can use a different adapter
    
    This enables efficient multi-task inference where all task-specific LoRA adapters
    are loaded in memory simultaneously and dynamically selected per example, eliminating
    the need to switch adapter states between tasks and allowing optimal throughput
    for mixed-task batches.
    
    Derived from peft.tuners.lora.Linear.
    """

    ### 	
    # •	base_layer：被替换/包裹的原始 nn.Linear（或 shape 相同的线性算子）。
	# •	adapter_name：初始激活的 LoRA 适配器名（如 "default" 或 "retrieval"）。
	# •	task_names：任务枚举（如 ["retrieval", "text-matching", "code"]）；后续会为每个 task构建一套 A/B。
	# •	r/lora_alpha/lora_dropout：LoRA 超参；
	# •	fan_in_fan_out/is_target_conv_1d_layer：与 PEFT LoRA 兼容的形状语义（此实现针对 Linear，不对 Conv 做特殊处理）；
	# •	init_lora_weights：True（Kaiming/零初始化）、"gaussian"（正态）、False（跳过初始化）；
	# •	use_rslora：RS-LoRA 的缩放；
	# •	use_dora：DoRA（Decoupled Rank Allocation）占位参数，这里未真正使用；
	# •	lora_bias：是否为 B 分支添加 bias。
	# •	LoraLayer.__init__：会注册一系列字典属性（lora_A/B, scaling, lora_dropout, r, lora_alpha, use_dora, 等），以及缓存 in_features/out_features、merged/disable_adapters 状态等。
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        task_names: List[str],
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        self.fan_in_fan_out = fan_in_fan_out
        self.task_names = task_names
        self._active_adapter = adapter_name
        ### 通过 update_layer 真正创建：
        # •	指定 adapter_name 下的 ModuleDict of A/B（按 task_names 展开）；
        # •	初始化权重与缩放；
        # •	将模块移动到与 base 权重一致的设备/精度；
        # •	调用 set_adapter 激活适配器。
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer


    def forward(self, x: torch.Tensor, task_label: Union[str, List[str]], *args: Any, **kwargs: Any) -> torch.Tensor:
        ### 调用 LoraLayer 的参数检查（主要验证形状/设备/类型等前置条件）。
        self._check_forward_args(x, *args, **kwargs)

        ### 禁用适配器（纯基座推理）：如训练或评估时关闭 LoRA 分支；如果之前曾 merge，需先 unmerge()（此处未实现，见后文）。
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        ### 已经 merge 的场景（LoRA 权重已并入 base）：直接跑底层线性。
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        ### 正常 LoRA 路径：先做一次原始线性得到 result（形如 [B, *, out_features]），记下 dtype 以便后续还原精度。
        else:
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            ### 遍历当前激活的适配器集合（来自 LoraLayer 的 active_adapters 属性，通常是 list）
            ### 设计点：允许多适配器同时激活并叠加（逐个累加到 result）。这支持“多域同时注入”或混合干预，但也可能导致线性叠加的耦合。一般情况下建议同时只激活一个适配器，除非你确实需要叠加。
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue
                ### 单一任务（整批同一 task）
                if isinstance(task_label, str):
                    lora_A = self.lora_A[active_adapter][task_label]
                    # print("---")
                    # print(lora_A.weight[0][:10])
                    # print("---")
                    lora_B = self.lora_B[active_adapter][task_label]
                    dropout = self.lora_dropout[active_adapter]
                    scaling = self.scaling[active_adapter]
                    x = self._cast_input_dtype(x, lora_A.weight.dtype)
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                ### 逐样本任务路由（混任务批次）：
                else:
                    unique_tasks = list(set(task_label))
                    lora_output = torch.zeros_like(result)
                    
                    for task in unique_tasks:
                        task_indices = [i for i, t in enumerate(task_label) if t == task]
                        task_x = x[task_indices]
                        
                        lora_A = self.lora_A[active_adapter][task]
                        lora_B = self.lora_B[active_adapter][task]
                        dropout = self.lora_dropout[active_adapter]
                        scaling = self.scaling[active_adapter]
                        
                        task_x = self._cast_input_dtype(task_x, lora_A.weight.dtype)
                        task_lora_value = lora_B(lora_A(dropout(task_x))) * scaling
                        
                        for i, idx in enumerate(task_indices):
                            lora_output[idx] = task_lora_value[i]
                    
                    result = result + lora_output

            result = result.to(torch_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        ### 为该 adapter_name 存储超参；配置 Dropout 或 Identity。
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        ### 核心设计：对每个 task_name 构建一套 A: [in_features→r] 与 B: [r→out_features]。
        # •	从而实现**(adapter_name, task_name)** 双维度的路由选择；
        # •	lora_bias 决定 B 端是否含 bias（一般设为 False 保持纯低秩增量）。
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.ModuleDict({
            task_name: nn.Linear(self.in_features, r, bias=False)
            for task_name in self.task_names
        })
        self.lora_B[adapter_name] = nn.ModuleDict({
            task_name: nn.Linear(r, self.out_features, bias=lora_bias)
            for task_name in self.task_names
        })
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        ### 初始化权重（见下节）；
        # •	将新注册的 A/B 层搬到与 base 层一致的设备与 dtype；
        # •	将 use_dora 标记为 False（未实现 DoRA）；
        # •	调 set_adapter，保证激活集包含新 adapter（PEFT 会维护 active_adapters）。
        self.reset_lora_parameters(adapter_name, init_lora_weights)
        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.use_dora[adapter_name] = False
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if init_lora_weights is True:
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            for task_name in self.task_names:
                nn.init.kaiming_uniform_(self.lora_A[adapter_name][task_name].weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            for task_name in self.task_names:
                nn.init.normal_(self.lora_A[adapter_name][task_name].weight, std=1 / self.r[adapter_name])
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        for task_name in self.task_names:
            nn.init.zeros_(self.lora_B[adapter_name][task_name].weight)
        if self.lora_bias[adapter_name]:
            for task_name in self.task_names:
                nn.init.zeros_(self.lora_B[adapter_name][task_name].bias)
    
    ### 未实现
    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights
        """
        raise NotImplementedError("Merge operation is not supported")

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        raise NotImplementedError("Unmerge operation is not supported")
