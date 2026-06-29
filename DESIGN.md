# T2C-CLIP 完整设计文档

## 1. 项目定位

T2C-CLIP 面向 Image-to-Image 行人重识别任务。检索输入始终是 query 图像和 gallery 图像，不使用自然语言 caption，不做 Text-to-Image 检索，不使用 CUHK-PEDES 等 Text-ReID 数据集。

模型基座是 CLIP ViT 双流结构。图像流提供主视觉特征，文本流通过 learnable prompt 提供可学习的语义/摄像头条件补充。最终检索特征为：

```text
f_v = normalize(CLIP_ImageEncoder(image))
f_t = normalize(CLIP_TextEncoder(prompt))
f   = normalize(f_v + beta * f_t)
```

其中 `beta` 控制文本流对最终检索特征的影响。主实验默认使用无 rerank 的 cosine 检索协议，并报告 mAP、Rank-1、Rank-5、Rank-10。

## 2. 明确不做的内容

- 不做 Text-to-Image 检索。
- 不使用自然语言 caption 监督。
- 不使用 CUHK-PEDES 等 Text-ReID 数据。
- 不在测试时使用 identity prompt。
- 不使用测试时优化、rerank、memory bank 检索、图传播或邻域搜索作为主结果。
- 不返回 mock 指标，不吞掉训练/验证错误，不通过 silent fallback 制造“能跑”的假象。

## 3. 当前问题与修复目标

当前工程曾经使用随机线性层 `PromptTextEncoder` 近似文本分支：

```text
prompt embeddings -> mean pooling -> Linear -> feature
```

这不等价于真实 CLIP text encoder。它会导致 `clip_loss` 下降不代表图像特征进入真实 CLIP text space，也会让训练期和推理期文本分支分布不一致。因此完整修复目标是方案 B：

1. 使用真实 CLIP text encoder。
2. 将 learnable prompt 注入 CLIP token embedding 空间。
3. 补齐 Stage-1 prompt alignment。
4. Stage-2 使用真实图文融合特征训练 ReID。
5. 推理只使用 `global + camera` prompt，不使用训练 ID prompt。

## 4. 总体架构

```text
Image
  -> CLIP Image Encoder
  -> f_v

Prompt Bank
  -> global prompt
  -> camera prompt
  -> identity prompt, training only
  -> CLIP Text Encoder
  -> f_t

f_v + beta * f_t
  -> normalized retrieval feature f
  -> identity classifier
  -> triplet loss
  -> TFC loss
  -> evaluation retrieval
```

模块边界：

- `clip_backbone.py`: Transformers CLIP image/text encoder adapter。
- `prompts.py`: learnable prompt bank，不直接依赖数据集文件结构。
- `model.py`: T2C-CLIP 前向路径，区分 Stage-1、Stage-2、Inference。
- `training.py`: loss 组合与训练阶段输入输出结构。
- `jobs/clip_reid.py`: Market-1501/MSMT17 数据加载、模型构建、训练回调、验证回调。
- `loops.py`: epoch/batch 进度、checkpoint、验证调度、MLflow 回调。

## 5. 数据与评估协议

### 5.1 数据集

支持两个 Image-ReID 数据集：

- MSMT17：主数据集。
- Market-1501：辅助验证数据集。

MSMT17 需要：

```text
list_train.txt
list_query.txt
list_gallery.txt
train/
test/
```

Market-1501 需要：

```text
bounding_box_train/
query/
bounding_box_test/
```

### 5.2 身份与摄像头映射

训练集身份 ID 映射只用于训练：

```text
raw_train_pid -> train_person_id
```

query/gallery 使用原始身份 ID 做评估匹配，不能复用训练 ID 映射。camera ID 可跨 train/query/gallery 构建统一映射，因为标准 ReID 协议下测试 camera ID 可见。

### 5.3 评估

评估使用标准 Image-to-Image ReID 协议：

```text
similarity = normalize(query_features) @ normalize(gallery_features).T
```

对每个 query，过滤同身份同摄像头 gallery 样本，然后计算 AP 和 CMC。主指标：

- mAP
- Rank-1
- Rank-5
- Rank-10

不做 rerank。

## 6. Prompt 设计

### 6.1 Prompt 类型

Prompt bank 包含三类 learnable prompt：

```text
P_global: shared global context prompt
P_cam[c]: camera-specific prompt
P_id[y]: training identity prompt
```

三类 prompt 都位于 CLIP text token embedding 维度中，而不是 CLIP projection 维度中。

### 6.2 Prompt 组合

训练阶段：

```text
P_train = P_global + P_cam[c_i] + P_id[y_i]
```

推理/验证阶段：

```text
P_eval = P_global + P_cam[c_i]
```

测试阶段禁止使用 `P_id`，因为测试身份未见过。训练 ID prompt 只用于训练期对齐和正则化文本分支。

### 6.3 CLIP Text Encoder 注入方式

需要使用真实 CLIP text encoder：

```text
token embedding + learnable prompt embedding
  -> CLIP text transformer
  -> pooled/eos hidden state
  -> text_projection
  -> f_t
```

learnable prompt 应该替换或插入到 prompt token 位置，不应通过随机线性层直接投影成 CLIP feature。输出的 `f_t` 必须与 `CLIP_ImageEncoder` 输出的 `f_v` 位于同一 projection space。

## 7. 模型前向路径

模型应显式区分三条路径。

### 7.1 Stage-1 前向

```text
f_v = normalize(CLIP_ImageEncoder(image))
f_t_id = normalize(CLIP_TextEncoder(P_global + P_cam[c] + P_id[y]))
```

输出：

```text
{
  "visual": f_v,
  "text": f_t_id
}
```

Stage-1 不计算 ReID classifier、triplet、TFC。

### 7.2 Stage-2 前向

Stage-2 computes two text features:

```text
f_v = normalize(CLIP_ImageEncoder(image))
f_t_train = normalize(CLIP_TextEncoder(P_global + P_cam[c] + P_id[y]))
f_t_eval  = normalize(CLIP_TextEncoder(P_global + P_cam[c]))
f_reid    = normalize(f_v + beta * f_t_eval)
```

输出：

```text
{
  "visual": f_v,
  "text": f_t_train,
  "retrieval": f_reid
}
```

`L_id`, `L_triplet`, and `L_TFC` act on `f_reid`, so the training target
matches the validation and inference feature. `f_t_train` is used only for
the auxiliary CLIP alignment loss, which prevents identity prompts from
becoming a training-time shortcut that disappears during validation.

### 7.3 推理/验证前向

```text
f_v = normalize(CLIP_ImageEncoder(image))
f_t_eval = normalize(CLIP_TextEncoder(P_global + P_cam[c]))
f_eval = normalize(f_v + beta * f_t_eval)
```

输出 `f_eval` 用于 query/gallery 检索。

## 8. 两阶段训练

### 8.1 Stage-1: Prompt Alignment

Stage-1 目标是让训练身份 prompt 通过真实 CLIP text encoder 后，与对应图像特征进入同一语义空间。

推荐配置：

```text
--stage1-epochs 20
--freeze-image-encoder-stage1
```

Stage-1 默认冻结 CLIP image encoder，只训练：

- prompt bank
- 必要的 text-side prompt 参数

Stage-1 loss：

```text
L_stage1 = L_clip_dual(f_v, f_t_id)
```

`L_clip_dual` 使用 batch 内双向 image-text contrastive loss。

### 8.2 Stage-2: ReID Training

Stage-2 目标是在真实 CLIP 图文融合空间训练行人检索特征。

Stage-2 loss：

```text
L_total =
  L_id
  + L_triplet
  + clip_weight * L_clip_dual
  + tfc_weight * L_TFC
```

推荐默认：

```text
--clip-weight 0.1
--tfc-weight 1.0
--triplet-margin 0.3
--beta 0.1
```

`clip_weight` 不应硬编码为 1。过大的 CLIP 对齐权重会压制 ReID 目标，尤其在 prompt 尚未稳定时会损害 mAP。

### 8.3 冻结策略

推荐阶段化策略：

```text
Stage-1:
  freeze image encoder
  train prompt/text-side parameters

Stage-2:
  train prompt bank
  train classifier
  fine-tune CLIP image encoder (default; matches CLIP-ReID Stage-2 and is the
  only path for ReID gradients to reach f_v)
  --freeze-image-encoder-stage2 opts back into prompt-only mode (caps mAP near
  the frozen CLIP image-only floor)
```

是否 fine-tune image encoder 应作为显式参数控制，不能 silently fallback。

## 9. Loss 设计

### 9.1 Image-Text Contrastive Loss

输入：

```text
visual: [B, D]
text: [B, D]
```

计算：

```text
logits = visual @ text.T * logit_scale
loss_i2t = CE(logits, labels)
loss_t2i = CE(logits.T, labels)
L_clip_dual = (loss_i2t + loss_t2i) / 2
```

### 9.2 Identity Classification Loss

使用推理一致的检索特征：

```text
logits = classifier(f_reid)
L_id = CE(logits, train_person_id)
```

### 9.3 Batch-Hard Triplet Loss

使用推理一致的检索特征：

```text
L_triplet = batch_hard_triplet_loss(f_reid, train_person_id)
```

训练 batch 必须 identity-balanced，至少包含有效 positive 和 negative pair。

### 9.4 TFC Loss

TFC 维护训练身份的 EMA center：

```text
center_y <- momentum * center_y + (1 - momentum) * mean(f_reid | y)
```

损失：

```text
L_TFC = mean(1 - cosine(f_reid_i, stopgrad(center_yi)))
```

TFC 只用于训练，不参与推理和验证。

## 10. 训练入口设计

`scripts/train.py` 应支持：

```bash
python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --clip-model-name openai/clip-vit-base-patch16 \
  --stage1-epochs 20 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints \
  --batch-size 384 \
  --num-workers 4 \
  --lr 0.0001 \
  --device cuda \
  --beta 0.1 \
  --clip-weight 0.1 \
  --tfc-weight 1.0 \
  --freeze-image-encoder-stage1 \
  --enable-mlflow \
  --tracking-db mlflow/t2c_clip.db \
  --artifact-root mlruns \
  --experiment-name T2C-CLIP \
  --run-name msmt17-stage1-stage2
```

参数语义：

- `--stage1-epochs`: Stage-1 prompt alignment 轮数。
- `--epochs`: Stage-2 轮数。
- `--validation-interval`: Stage-2 验证间隔。
- `--clip-weight`: Stage-2 中 CLIP 对齐损失权重。
- `--tfc-weight`: Stage-2 中 TFC 损失权重。
- `--beta`: 图文特征融合系数。
- `--freeze-image-encoder-stage1`: Stage-1 冻结图像编码器。

## 11. 进度条与日志

### 11.1 终端输出

每个 epoch 应有 batch 级进度条：

```text
stage1 epoch 3/20: loss=... clip_loss=... lr=...
stage2 epoch 5/120: loss=... reid_loss=... triplet_loss=... clip_loss=... tfc_loss=... lr=...
```

验证完成后用 `tqdm.write` 输出：

```text
stage2 epoch=5 mAP=... rank1=... rank5=... rank10=... best_mAP=... best=True
```

### 11.2 MLflow 指标

Stage-1 batch step：

```text
stage1_train_step_loss
stage1_train_step_clip_loss
stage1_train_step_lr
```

Stage-1 epoch：

```text
stage1_train_loss
stage1_train_clip_loss
stage1_lr
```

Stage-2 batch step：

```text
stage2_train_step_loss
stage2_train_step_reid_loss
stage2_train_step_triplet_loss
stage2_train_step_clip_loss
stage2_train_step_tfc_loss
stage2_train_step_lr
```

Stage-2 epoch：

```text
stage2_train_loss
stage2_train_reid_loss
stage2_train_triplet_loss
stage2_train_clip_loss
stage2_train_tfc_loss
stage2_lr
```

Validation：

```text
mAP
rank_1
rank_5
rank_10
best_mAP
is_best
```

MLflow params/tags 应记录：

```text
dataset
clip_model_name
stage1_epochs
stage2_epochs
batch_size
lr
beta
clip_weight
tfc_weight
retrieval_mode=fused
```

## 12. Checkpoint 设计

保存文件：

```text
last.pth
best.pth
```

`last.pth` 每个 Stage-2 epoch 后保存。`best.pth` 只在验证 mAP 创新高时保存。

checkpoint payload 应包含：

```text
stage
epoch
best_map
metrics
model_state
optimizer_state
config
```

如果 Stage-1 单独保存初始化结果，可额外保存：

```text
stage1_last.pth
```

但主训练恢复路径应优先围绕 `last.pth`。

## 13. Sanity Check

完整训练前必须先跑短程 sanity check：

`--retrieval-mode image_only` must skip the text branch and evaluate
`normalize(f_v)` directly. This result is the CLIP image-only baseline sanity
check and must be above the random floor before long Stage-2 training.

1. image-only CLIP baseline mAP 高于随机水平。
2. Stage-1 `clip_loss` 明显低于 `ln(batch_size)`。
3. Stage-2 `reid_loss` 明显低于 `ln(num_train_ids)`。
4. Stage-2 前 5 到 10 个 epoch 的 mAP 不应长期贴近随机。
5. query/gallery 验证路径不使用 identity prompt。
6. `f_v`、`f_t`、`f` 都是二维 tensor，且最后一维相同。
7. 检索特征输出前必须 L2 normalize。

如果 sanity check 失败，应暴露错误或输出明确诊断，不添加静默 fallback。

## 14. Ablation 设计

主消融：

| Variant | Image Stream | Text Stream | Stage-1 | Camera Prompt | TFC |
|---|---:|---:|---:|---:|---:|
| Image-only baseline | yes | no | no | no | no |
| Fused without Stage-1 | yes | yes | no | yes | no |
| Stage-1 + fused | yes | yes | yes | yes | no |
| Full T2C-CLIP | yes | yes | yes | yes | yes |

Prompt 消融：

| Variant | Global | Camera | Train ID | Eval ID |
|---|---:|---:|---:|---:|
| Global only | yes | no | no | no |
| Global + camera | yes | yes | no | no |
| Stage-1 ID alignment | yes | yes | yes | no |

Beta 消融：

| Variant | Feature |
|---|---|
| beta = 0 | `f = f_v` |
| fixed beta | `f = normalize(f_v + beta * f_t)` |
| learnable beta | `f = normalize(f_v + beta_learned * f_t)` |

TFC 消融：

| Variant | TFC Target |
|---|---|
| Without TFC | none |
| Main TFC | fused feature `f` |
| Visual TFC | visual feature `f_v` |

主结果必须使用无 rerank 指标。

## 15. 实现验收标准

代码修复完成后应满足：

1. 不再使用随机线性 `PromptTextEncoder` 作为主文本路径。
2. 真实 CLIP text encoder 参与 Stage-1、Stage-2 和 inference。
3. Stage-1 只训练图文 prompt alignment，不计算 ReID/TFC。
4. Stage-2 计算 ReID、triplet、weighted CLIP alignment、weighted TFC。
5. 验证路径不使用 identity prompt。
6. MLflow 同时记录 stage-aware batch step、epoch 平均和验证指标。
7. `best.pth` 和 `last.pth` 语义稳定。
8. 测试覆盖 text encoder path、Stage-1、Stage-2、inference prompt、MLflow 指标、checkpoint。

## 16. 推荐实施顺序

1. 新增真实 CLIP text encoder adapter。
2. 重写 prompt bank，使 prompt 参数处于 CLIP token embedding 维度。
3. 改 `T2CClipModel`，增加 `forward_stage1`、`forward_stage2`、`encode_retrieval`。
4. 改 loss 输入结构，加入 `clip_weight`。
5. 改真实 training job，支持 Stage-1 + Stage-2。
6. 改训练入口参数和 MLflow stage-aware logging。
7. 补 image-only baseline 评估命令，作为 sanity check。
8. 更新 README 训练流程。
9. 跑短程 sanity check，再跑完整训练。

## 17. 预期训练流程

启动 MLflow：

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow/t2c_clip.db \
  --default-artifact-root mlruns \
  --host 127.0.0.1 \
  --port 6006
```

训练：

```bash
python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --clip-model-name openai/clip-vit-base-patch16 \
  --stage1-epochs 20 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints \
  --batch-size 384 \
  --num-workers 4 \
  --lr 0.0001 \
  --device cuda \
  --beta 0.1 \
  --clip-weight 0.1 \
  --tfc-weight 1.0 \
  --freeze-image-encoder-stage1 \
  --enable-mlflow \
  --tracking-db mlflow/t2c_clip.db \
  --artifact-root mlruns \
  --experiment-name T2C-CLIP \
  --run-name msmt17-full-t2c
```

短程调试建议先跑：

```bash
python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --clip-model-name openai/clip-vit-base-patch16 \
  --stage1-epochs 2 \
  --epochs 5 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints/debug \
  --batch-size 128 \
  --num-workers 4 \
  --lr 0.0001 \
  --device cuda \
  --beta 0.1 \
  --clip-weight 0.1 \
  --tfc-weight 1.0 \
  --freeze-image-encoder-stage1 \
  --enable-mlflow \
  --tracking-db mlflow/t2c_clip.db \
  --artifact-root mlruns \
  --experiment-name T2C-CLIP \
  --run-name sanity-msmt17
```

## 18. 风险与约束

- 真实 CLIP text encoder prompt 注入比随机线性 text branch 更复杂，但这是修复 mAP 随机化问题的必要条件。
- Stage-1 会增加训练时间，但能降低 Stage-2 初期随机对齐对 ReID 目标的干扰。
- MSMT17 验证耗时明显长于训练 epoch，需要在验证开始/结束时输出明确日志。
- batch size 增大后需要监控吞吐、显存峰值和验证阶段显存占用。
- 所有失败必须显式暴露，不通过 mock 指标或静默降级隐藏。
