# T2C-CLIP 完整设计文档

## 1. 项目定位

T2C-CLIP 面向 Image-to-Image 行人重识别。检索输入始终是 query 图像和
gallery 图像，不使用自然语言 caption，不做 Text-to-Image 检索，也不使用
CUHK-PEDES 等 Text-ReID 数据集。

项目品牌和 Python 包名继续使用 T2C-CLIP / `t2c_clip`。模型基座固定为：

```text
google/siglip2-so400m-patch14-384
```

旧 OpenAI CLIP backend、CLI 参数和 checkpoint 不再兼容。

## 2. 最终检索特征

模型保留图像流和 prompt 文本流：

```text
f_v_raw = SigLIP2_ImageEncoder(image)
f_v     = FeatureHead(f_v_raw)              # Identity or BNNeck
f_t     = normalize(SigLIP2_TextEncoder(prompt))
f       = normalize(f_v + beta * f_t)
```

推理只允许身份无关 prompt：

```text
training prompt  = global + camera + identity
inference prompt = global + camera
```

`identity prompt` 只能参与训练对齐和训练身份 anchor 构造，禁止进入 query / gallery
特征。`image_only` 模式返回归一化 `FeatureHead(f_v_raw)`；`fused` 模式返回上述融合
特征。Stage-1 和 Stage-2 alignment、triplet 仍使用归一化或未归一化的视觉 raw feature，
具体见损失章节。

## 3. 模块边界

```text
path + person/camera metadata
  -> Rust batch decoder/augmenter (default) or torchvision reference backend
  -> normalized contiguous BCHW float32 tensor
  -> pinned host memory + non-blocking CUDA transfer
  -> fixed SigLIP 2 vision_model + positional interpolation
  -> f_v_raw
  -> feature head (Identity or BNNeck)
  -> f_v

PromptBank
  -> checkpoint-native fixed-length token sequence
  -> bidirectional text encoder + text_model.head
  -> f_t

f_v + beta * f_t
  -> normalize
  -> retrieval

f_v_raw / f_v
  -> alignment / triplet / ID losses
```

主要实现文件：

- `siglip2_backbone.py`：固定 BCHW 视觉/SIE adapter、文本 prompt 注入和格式校验。
- `prompts.py`：global、camera、training identity prompt bank。
- `model.py`：`T2CSiglip2Model` 的 Stage-1、Stage-2 和推理路径。
- `losses.py`：监督式 SigLIP、全身份 anchor SigLIP、batch-hard triplet。
- `training.py`：两阶段 loss 组合。
- `jobs/siglip2_reid.py`：数据、模型、冻结、优化器、precision、缓存和验证。
- `precision.py`：precision 解析、autocast、GradScaler 和 optimizer step。
- `datasets.py` / `transforms.py`：Python reference dataset、共享变换配置、Rust batch collator。
- `native.py` / `rust/src/`：原生 ABI、自带 JPEG/PNG 数据管线、指标聚合和稀疏 rerank。
- `evaluation.py`：Torch 分块矩阵计算、Python reference 和 Rust backend 调度。
- `loops.py` / `scripts/train.py`：epoch、checkpoint、resume、W&B 回调。

不存在旧 `clip_backbone.py`、`jobs/clip_reid.py` 或兼容 alias。

## 4. SigLIP 2 视觉边界

### 4.1 输入尺寸

默认 ReID 输入为：

```text
height = 392
width  = 196
patch  = 14
spatial shape = 28 x 14
patch count = 392
```

该尺寸保持 2:1 行人比例，并确保两个维度都能被 patch14 整除。启动阶段必须校验：

1. 高宽均为正数。
2. 默认 `392x196` 高宽都能被 patch14 整除；固定 `SiglipModel` 也允许官方
   `384x384` valid-stride Conv2d 的 floor 语义。
3. patch 数不超过模型位置 embedding 的实际行数。
4. 位置 embedding 数为完全平方数，保证非方形 ReID 网格可从预训练方形网格插值。
5. 默认固定 checkpoint 的 processor 预训练尺寸、模型 `image_size`、视觉 embedding
   和 patch budget 一致。

`google/siglip2-so400m-patch14-384` 虽然是 SigLIP 2 权重，但 Hugging Face Hub
将其固定分辨率格式注册为 `model_type=siglip`。因此必须使用 `AutoModel` 按 checkpoint
配置加载；强制使用 `Siglip2Model.from_pretrained` 会错误地产生默认 256 patch budget，
且权重结构不受支持。训练 builder 对模型 ID 使用硬 allowlist，任何其他固定或 NaFlex
checkpoint 都在下载前失败。

### 4.2 图像增强

训练增强保留：

- horizontal flip；
- color jitter；
- resize 后 pad + random crop；
- normalization 后 random erasing。

验证只做 bilinear resize、tensor 转换和 checkpoint processor mean/std normalization。
不做 center crop，避免高瘦行人图像丢失头部和脚部。

默认 `data_backend=rust` 以 batch 为边界读取 JPEG/PNG 并输出拥有底层内存的 NumPy
`BCHW float32`，Python 通过 `torch.from_numpy` 共享该 allocation。每个 collate 从
PyTorch worker RNG 获取一个 batch seed，再用 ChaCha 派生逐图 seed；相同版本 Rust
backend 在固定全局 seed 下可重放，但不要求与 torchvision 的逐像素结果或随机序列
一致。`data_backend=python` 保留为显式 reference。PK identity sampler 继续使用 Python，
不改变 micro-batch 的 P-K 结构。

### 4.3 Transformers 固定视觉输入

目标 checkpoint 直接接收 BCHW tensor。adapter 调用官方：

```text
vision_model(pixel_values=images, interpolate_pos_encoding=True)
```

其 stride-14 convolution 完成 patch embedding，`392x196` 得到 `28x14 = 392`
tokens；位置 embedding 从预训练的 `27x27 = 729` 网格插值到非方形网格。训练 job
不接受 patchified/NaFlex checkpoint。

### 4.4 视觉前向与 SIE

无 SIE 时调用官方 `vision_model`，读取 `pooler_output`。

有 SIE 时在 patch/token embedding 后加入 camera embedding，再继续官方 encoder、
post-layernorm 和 attention pooling head。固定 BCHW 格式不使用 patch attention mask。

零值 SIE embedding 的结果必须与官方前向 bitwise 相同。SIE embedding 跟随当前 stage
的视觉塔冻结状态。

## 5. SigLIP 2 文本边界

### 5.1 Prompt 参数空间

三类 prompt 参数都位于 `text_model.embeddings.token_embedding` 的 hidden dimension：

```text
P_global              [context_length, text_hidden]
P_cam[c]              [context_length, text_hidden]
P_id[y] (training)    [context_length, text_hidden]
```

它们不是最终 feature dimension，也不经过随机投影层。

### 5.2 模板与 checkpoint-native padding

模板片段使用同 checkpoint tokenizer，并以 `add_special_tokens=False` 编码：

```text
prefix = "a photo of a"
suffix = "person ."
```

最终序列长度固定为 `max_position_embeddings`（目标模型为 64）。目标固定 checkpoint
使用 Gemma tokenizer 的右填充布局：

```text
prefix + learnable slots + suffix + EOS + PAD ... PAD
```

官方 processor 的文本输入只声明 `input_ids`，因此 adapter 不构造 padding attention
mask，也不额外插入 BOS。learnable prompt 只替换 slots 的 token embedding。

目标固定 checkpoint 的 `SiglipConfig` 包含遗留默认 special-token ID，与 checkpoint 的
Gemma tokenizer 不一致。adapter 以 tokenizer ID 为准，并验证这些 ID 位于模型词表内。

### 5.3 文本前向

SigLIP 2 文本塔不是 CLIP causal transformer。正确路径为：

```text
token embedding with injected slots
  + position embedding
  -> optional bidirectional padding mask
  -> text_model.encoder
  -> final_layer_norm
  -> final sequence position
  -> text_model.head
  -> L2 normalize
```

目标 checkpoint 的 final sequence position 是 PAD，这是 Transformers 官方
`SiglipTextModel` 的预训练 pooling 语义。禁止使用 causal mask、最大 token ID pooling
或不存在的额外 `text_projection`。

## 6. 两阶段训练

### 6.1 Stage-1：监督式 SigLIP prompt 对齐

Stage-1 默认冻结图像塔，只训练 prompt bank；文本塔默认冻结。对 PK micro-batch：

```text
S_ij = exp(logit_scale) * cosine(f_v[i], f_t[j]) + logit_bias
Y_ij = +1, person_id[i] == person_id[j]
       -1, otherwise

L_alignment = -mean_i sum_j log_sigmoid(Y_ij * S_ij)
```

所有同身份 image/text 组合都为正样本，不局限于对角线。完整 pairwise 矩阵同时覆盖
两个模态方向的关系，不再计算旧式双向 softmax CE。

Stage-1 feature cache 只在图像塔冻结时合法。缓存提取使用 eval transform 和当前
autocast policy，之后每个 epoch 只做 prompt/text 前向。

### 6.2 Stage-2：ReID + 全身份 SigLIP anchor

Stage-2 默认解冻视觉塔，保留冻结文本塔。每个训练身份生成 camera-agnostic anchor：

```text
anchor[y] = TextEncoder(P_global + P_id[y])
```

每张图像只把其标签对应 anchor 视为正，其余所有训练身份 anchor 为负：

```text
S_iy = exp(logit_scale) * cosine(f_v[i], anchor[y]) + logit_bias
Y_iy = +1, y == person_id[i]
       -1, otherwise
```

loss 采用 SigLIP 原生 reduction：按 anchor 维求和，再按 image row 求均值。其数值会随
训练身份数变化，这是有意行为，`alignment_weight` 需要通过真实实验调优。

Stage-2 总损失：

```text
L_total = L_id
        + L_triplet
        + alignment_weight * L_alignment
        + tfc_weight * L_TFC
```

- `L_id`：BNNeck/linear classifier cross entropy。
- `L_triplet`：BN 前视觉 feature 的 batch-hard triplet。
- `L_alignment`：全身份文本 anchor SigLIP loss。
- `L_TFC`：融合检索 feature 的中心约束。
- `label_smoothing` 只作用于 `L_id`。

`logit_scale` 和 `logit_bias` 读取预训练值后永久冻结，不进入 optimizer。归一化、logit、
bias 和 `logsigmoid` 在 FP32 中计算。

## 7. 冻结、缓存与优化器

默认冻结策略：

```text
Stage-1 vision: frozen
Stage-2 vision: trainable
text tower: frozen
Stage-2 prompt bank: trainable
logit_scale / logit_bias: frozen
```

文本身份 anchor：

- prompt bank 和 text tower 都冻结：全 Stage-2 只编码一次。
- 任一可训练：每个 Stage-2 epoch 重新编码并 detach。

camera retrieval text：

- prompt bank 和 text tower 都冻结：每个 camera 只编码一次。
- 否则在线编码。

AdamW 参数分组：

- `vision_model.*` 使用 `image_encoder_lr`。
- prompt、classifier、TFC、BNNeck、可训练文本塔使用 `lr`。
- 一维参数、bias、prompt 和 SIE 不做 weight decay。
- 其他矩阵参数使用 `1e-4` weight decay。

## 8. 24GB 单卡 precision 与梯度累积

默认：

```text
micro batch = 8
gradient accumulation = 4
effective optimizer batch = 32
eval batch = 16
gradient checkpointing = on
precision = auto
```

`batch-size` 是真实 pairwise / triplet mining 范围。累积 4 次不会把 SigLIP matrix 或
hard mining 扩展到 32。

precision 解析：

```text
auto + CUDA BF16 support -> bf16
auto + other CUDA        -> fp16 + GradScaler
auto + CPU               -> fp32
```

显式请求不支持的低精度必须启动失败，禁止静默回退。

梯度累积以窗口为单位：

1. 每个窗口开始执行一次 `zero_grad(set_to_none=True)`。
2. 每个 micro-batch loss 除以该窗口的实际长度后 backward。
3. 每个窗口只执行一次 optimizer/scaler step。
4. epoch 尾部不足 4 个 micro-batch 时按实际长度归一化。
5. FP16 overflow 跳过的窗口不计为 W&B optimizer update step。

Stage-1 cache、anchor/camera cache、训练和验证使用同一 precision controller。落到 CPU
进行 ReID 距离计算的最终 feature 强制转换为 FP32。

## 9. 推理与评估

每个 query/gallery 样本执行同一路径：

1. 解析 camera ID。
2. 编码 SigLIP 2 图像 feature。
3. `fused` 模式组合 global + camera prompt 并编码文本。
4. 应用共享 feature head。
5. 输出归一化 feature。
6. 计算 cosine retrieval。

标准协议排除同身份同 camera gallery。主指标始终是无 rerank mAP/CMC；rerank 只能作为
额外报告，不能覆盖主指标。

默认评估 backend 为 Rust：Torch 保留 L2 normalization 与矩阵乘法，按 query chunk
生成完整 gallery 分数；Rust 对每个 query 执行确定性排序、同身份同 camera 过滤和
AP/CMC 聚合。同分时使用 gallery 原始索引升序作为 tie-break。该路径不常驻完整
`Q x G` score matrix；显式 `evaluation_backend=python` 仅用于差分和诊断。

k-reciprocal rerank 的主结果地位不变。Rust production 路径分块计算精确全局距离，
只保留所需 top-k，并以 CSR affinity 与 CSC posting list 执行 reciprocal expansion、
query expansion 和 Jaccard minima。列归一化转置语义、`round(k1/2)` banker-rounding
和 lambda 混合必须与 Python 稠密 reference 一致。reciprocal CSR 确定后，Torch 以第二次
分块距离 pass 提取每条 affinity edge 的精确距离，避免 Rust 标量 dot reduction 改变
排序。最终 rerank 距离在 `1e-6` 容差内视为并列，并按 gallery 原始索引排序；Python
reference 与 Rust 使用同一规则。该实现将常驻内存从稠密 `O(N^2)`
降为 `O(chunk*N + sparse affinity)`，但精确近邻计算仍为 `O(N^2 D)`。

## 10. Checkpoint 契约

新 checkpoint schema version 为 2，并保存：

- `backbone_family = siglip2`；
- Hugging Face model ID；
- feature dimension；
- image size、patch size、patch count、最大 patch budget、vision input format；
- text padding side、BOS/padding-mask layout；
- resolved precision；
- model 和 optimizer state；
- FP16 GradScaler auxiliary state；
- epoch、stage、best mAP 和验证指标。

resume 必须先验证 metadata，再加载 model state。旧 checkpoint 缺少 SigLIP 2 schema，
或 model/image/feature/precision 字段不一致时明确失败。旧 OpenAI CLIP state dict 不做转换。

## 11. W&B 契约

Stage-1 训练指标：

```text
loss
alignment_loss
lr
```

Stage-2 训练指标：

```text
loss
alignment_loss
reid_loss
triplet_loss
tfc_loss
lr
```

`stage*_train_step` 统计成功的 optimizer update window。step 指标为窗口内 micro-batch
均值；epoch 指标为所有 micro-batch 的均值。metadata 必须记录 requested/resolved
precision、micro/effective/eval batch、累积步数、gradient checkpointing、patch 信息、
`data_backend`、`evaluation_backend`、worker/pin/prefetch/persistent 配置和 evaluation
chunk size。运行后端字段只作为实验 provenance，不进入 schema 2 checkpoint 兼容性
校验，因此现有 schema 2 Stage-2 checkpoint 仍可恢复。

## 12. 验收标准

实现必须满足：

1. 默认模型 ID 为 `google/siglip2-so400m-patch14-384`。
2. 默认固定格式以 BCHW 接收 `392x196` 并产生 392 tokens；H-W-C patch utility
   的低层顺序测试同样生成 392 patches。
3. tiny 固定 SigLIP 2 无 SIE、零 SIE 与官方 vision forward 一致。
4. learnable slots 使用真实 token embedding 时，文本 adapter 与官方固定右填充 forward
   一致。
5. 图像和文本最终 feature dimension 相同且归一化。
6. Stage-1 同身份多正样本和 Stage-2 全身份 anchor 使用原生 sigmoid loss。
7. `logit_scale/logit_bias` 冻结且不在 optimizer。
8. `label_smoothing` 不影响 alignment loss。
9. accumulation 尾窗口梯度不被低估，W&B step 按 optimizer window 计数。
10. FP16 scaler 和 SigLIP metadata 可 checkpoint/resume。
11. 推理路径永远不访问 identity prompt。
12. `uv run cargo test --manifest-path rust/Cargo.toml --locked` 和
    `uv run python -m unittest discover -s tests` 全部通过。
13. Rust/Python 主评估差分在相同 feature 上达到 `1e-12` 聚合一致性；小规模稀疏
    rerank 与稠密 reference 距离/指标误差不超过 `1e-6`。
14. 原生 eval resize 的 shape/dtype/contiguous 契约一致，fixture 最大误差不超过一个
    8-bit 量化步长对应的归一化值；固定 seed 的 Rust 增强可重放。
15. `python -m scripts.train --help` 只显示 SigLIP 2 与 Rust backend 公共参数，不显示旧
    `--clip-*` 参数。

## 13. 明确不做

- 不加入旧 CLIP backend 或兼容 alias。
- 不迁移旧 T2C-CLIP checkpoint。
- 不引入 DDP、FSDP、DeepSpeed。
- 不将默认固定 backbone 改为 NaFlex。
- 不改成 Text-to-Image ReID。
- 不让测试身份 prompt 进入推理。
- 不把 rerank 指标当作主结果。
- 不加入 GPU JPEG decode、HNSW/ANN 或近似 rerank。
- 不支持缺失原生扩展时的自动 Python fallback；Python backend 只能显式选择。
