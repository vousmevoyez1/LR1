# LR1
## jina_v4
/data/LR1/src/models/jina_v4：
原始将后续hiddenstate拼接到序列尾部再送回模型的方法。


/data/LR1/src/models/jina_v4o：
可学习的thought token方法。
只有对各个thought token相应的embedding进行监督。无后续说的强渐进监督。


/data/LR1/src/models/jina_v4oaux：
可学习的thought token方法。
对各个thought token相应的embedding进行监督+后续说的强渐进监督。


/data/LR1/src/models/jina_v4oauxa：
可学习的thought token方法。
对各个thought token相应的embedding进行监督+后续说的强渐进监督。
+selector模块的gate参数不冻结，允许训练过程中更新。


/data/LR1/src/models/jina_v4od：
可忽略的版本，未使用。
用于 
一次编码 + 一次评估，完成三种指标：
1) Max-Sim
2) Oracle Best-Step
3) Last-Step

## qwen
注意：qwen环境与jina有差异。


/data/LR1/src/models/qwen3：
可学习的thought token方法。
对各个thought token相应的embedding进行监督+后续说的强渐进监督。


/data/LR1/src/models/qwen3a：
原始将后续hiddenstate拼接到序列尾部再送回模型的方法。