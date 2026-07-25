# T2C-CLIP

Train2Central-CLIP foundation for Image-to-Image person ReID experiments.
The project name and Python package remain `T2C-CLIP` / `t2c_clip`, while the
foundation vision-language model is **SigLIP 2 So400m**.

The implementation uses a two-stage training pipeline:

- `google/siglip2-so400m-patch14-384` image and text towers.
- Market-1501 and MSMT17 person/camera parsing.
- Global, camera, and training-identity learnable prompts in the SigLIP 2 text
  token embedding space.
- Stage-1 supervised SigLIP alignment between image features and identity-aware
  prompt features.
- Stage-2 ReID identity, batch-hard triplet, SigLIP all-identity alignment, and
  TFC losses.
- Fused or image-only no-rerank cosine retrieval.
- Stage-aware Weights & Biases tracking and resumable checkpoints.

The architecture contract is maintained in [DESIGN.md](DESIGN.md). Research
hypotheses and experiment tables are separate in
[docs/research-blueprint.md](docs/research-blueprint.md).

## Retrieval Contract

The image and text towers produce features in the same SigLIP 2 output space:

```text
f_v_raw = SigLIP2_ImageEncoder(image)
f_v     = FeatureHead(f_v_raw)              # Identity or BNNeck
f_t     = normalize(SigLIP2_TextEncoder(prompt))
f       = normalize(f_v + beta * f_t)
```

Training identity prompts are never used for query/gallery retrieval:

```text
training prompt  = global + camera + identity
inference prompt = global + camera
```

`--retrieval-mode image_only` returns normalized `FeatureHead(f_v_raw)` without
the text branch. `--retrieval-mode fused` uses the inference prompt above.

## Environment

The project uses `uv`:

```bash
uv sync
uv run python -m unittest discover -s tests
```

Core requirements are declared in `pyproject.toml`:

- Python 3.14+
- PyTorch 2.13+
- torchvision 0.28+
- Transformers 5.14.1+
- Weights & Biases 0.28+

The first real run downloads the selected Hugging Face checkpoint, which is
several GB for So400m.

## Data

Supported datasets:

- `market1501`
- `msmt17`

Market-1501 expects the standard directories:

```text
Market-1501-v15.09.15/
  bounding_box_train/
  query/
  bounding_box_test/
```

MSMT17 expects the standard manifests and `train` / `test` image trees. The
training split combines `list_train.txt` and `list_val.txt`.

## Training

A 24GB single-GPU starting point is:

```bash
uv run python -m scripts.train \
  --job-builder t2c_clip.jobs.siglip2_reid:build_training_job \
  --dataset msmt17 \
  --data-root path/to/MSMT17_V1 \
  --siglip2-model-name google/siglip2-so400m-patch14-384 \
  --stage1-epochs 20 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints/siglip2-so400m \
  --batch-size 8 \
  --gradient-accumulation-steps 4 \
  --eval-batch-size 16 \
  --precision auto \
  --gradient-checkpointing \
  --num-instances 2 \
  --num-workers 4 \
  --lr 1e-4 \
  --image-encoder-lr 5e-6 \
  --beta 0.1 \
  --alignment-weight 1.0 \
  --tfc-weight 1.0 \
  --freeze-image-encoder-stage1 \
  --no-freeze-image-encoder-stage2 \
  --freeze-text-encoder
```

The defaults already select the model, `392x196` input, batch `8`, accumulation
`4`, eval batch `16`, automatic precision, and gradient checkpointing.

### Stage 1

Stage-1 trains identity-aware prompts with the supervised SigLIP objective.
Every image/text pair sharing a person ID is positive; all other pairs in the
PK micro-batch are negative. The image tower is frozen by default.

`--stage1-feature-cache` is enabled by default. It extracts the frozen
train-split image features once through the eval transform and reuses them for
all Stage-1 epochs. The flag is rejected if the Stage-1 image tower is
trainable.

### Stage 2

Stage-2 defaults to an unfrozen vision tower and frozen text tower. Each image
is aligned against the camera-agnostic text anchors of **all** training
identities. The total loss is:

```text
L_total = L_id + L_triplet
        + alignment_weight * L_siglip
        + tfc_weight * L_TFC
```

`label_smoothing` applies only to `L_id`. The SigLIP loss keeps its native
binary targets and native row-mean / column-sum reduction.

The pretrained `logit_scale` and `logit_bias` are frozen constants:

```text
logits = exp(logit_scale) * cosine(image, text) + logit_bias
L_siglip = -mean_rows(sum_columns(logsigmoid(sign_target * logits)))
```

### Image Input

The default whole-person input is `392x196`, a 2:1 aspect ratio. With patch14,
this is `28x14 = 392` patches. Images remain BCHW tensors through the dataset
and augmentation pipeline. Hugging Face publishes the selected fixed-resolution
SigLIP 2 checkpoint with `model_type=siglip`; the adapter therefore calls its
official BCHW vision path with positional interpolation enabled. The model's
stride-14 convolution produces the 392 tokens.

The default dimensions are exactly patch-aligned. The fixed Transformers model
also retains its official valid-stride behavior for sizes such as `384x384`;
patch counts use floor division and must not exceed the checkpoint's positional
budget. The checkpoint model type, processor pretraining size, patch settings,
and positional embeddings are validated at startup. The training builder
rejects every other model ID, including NaFlex variants.

### Memory Semantics

`--batch-size` is the actual SigLIP pairwise and triplet-mining micro-batch.
Gradient accumulation does not enlarge either mining scope:

```text
micro-batch: 8
accumulation steps: 4
effective optimizer batch: 32
alignment/triplet scope: 8
```

`--precision auto` resolves as follows:

- CUDA with BF16 support: `bf16`
- other CUDA: `fp16` with `torch.amp.GradScaler`
- CPU: `fp32`

Explicit unsupported low precision fails at startup instead of silently
falling back. FP16 scaler state is saved and restored with the checkpoint.

## Main CLI Options

Backbone and input:

- `--siglip2-model-name google/siglip2-so400m-patch14-384` (the only accepted value)
- `--siglip2-checkpoint PATH` (optional strict additional state dict)
- `--image-height 392`
- `--image-width 196`
- `--sie-coe 0.0`
- `--context-length 4`

Batching and memory:

- `--batch-size 8`
- `--eval-batch-size 16`
- `--num-instances 2`
- `--gradient-accumulation-steps 4`
- `--precision auto|fp32|bf16|fp16`
- `--gradient-checkpointing / --no-gradient-checkpointing`

Optimization:

- `--lr 1e-4`
- `--image-encoder-lr 5e-6`
- `--alignment-weight 1.0`
- `--tfc-weight 1.0`
- `--triplet-margin 0.3`
- `--triplet-metric euclidean|cosine`
- `--label-smoothing 0.0`
- `--stage2-lr-scheduler none|cosine`
- `--stage2-warmup-epochs 0`
- `--beta 0.1`
- `--beta-warmup-epochs 0`

Freezing and retrieval:

- `--freeze-image-encoder-stage1 / --no-freeze-image-encoder-stage1`
- `--freeze-image-encoder-stage2 / --no-freeze-image-encoder-stage2`
- `--freeze-text-encoder / --no-freeze-text-encoder`
- `--freeze-prompt-bank-stage2 / --no-freeze-prompt-bank-stage2`
- `--reid-head linear|bnneck`
- `--retrieval-mode fused|image_only`
- `--report-rerank`

## Checkpoints And Resume

Stage-1 writes `stage1_last.pth`. Stage-2 writes `last.pth` and `best.pth`.
New checkpoints include:

- schema version and `backbone_family=siglip2`
- Hugging Face model ID
- feature dimension
- image size, patch size, patch count, maximum patch budget, and vision input format
- tokenizer padding/pooling layout
- resolved precision
- model, optimizer, and FP16 scaler state

Resume a Stage-2 run with the same architecture and precision:

```bash
uv run python -m scripts.train \
  --job-builder t2c_clip.jobs.siglip2_reid:build_training_job \
  --dataset msmt17 \
  --data-root path/to/MSMT17_V1 \
  --stage1-epochs 20 \
  --epochs 120 \
  --resume checkpoints/siglip2-so400m/last.pth
```

This migration intentionally does not support OpenAI CLIP weights or old
T2C-CLIP training checkpoints. The removed options `--clip-model-name`,
`--clip-checkpoint`, and `--clip-weight` are argparse errors. Incompatible
resume metadata fails before model weights are loaded.

## Weights & Biases

Enable online tracking:

```bash
uv run wandb login
uv run python -m scripts.train \
  --job-builder t2c_clip.jobs.siglip2_reid:build_training_job \
  --dataset msmt17 \
  --data-root path/to/MSMT17_V1 \
  --enable-wandb \
  --wandb-project T2C-CLIP \
  --run-name msmt17-siglip2-so400m
```

Training metrics include:

- Stage-1: `loss`, `alignment_loss`, `lr`
- Stage-2: `loss`, `alignment_loss`, `reid_loss`, `triplet_loss`, `tfc_loss`, `lr`
- Validation: `mAP`, `best_mAP`, `rank_1`, `rank_5`, `rank_10`

`stage1_train_step` and `stage2_train_step` count successful optimizer update
windows, not micro-batches. Window metrics are means across their constituent
micro-batches; epoch metrics average all micro-batches.

## Feature Evaluation CLI

Evaluate pre-extracted query/gallery features from `.npz`:

```bash
uv run python -m t2c_clip.cli.evaluate path/to/features.npz \
  --output metrics.json \
  --ranks 1 5 10
```

The evaluator applies the standard Image-to-Image ReID protocol and excludes
same-identity, same-camera gallery samples. Primary metrics are no-rerank;
`--report-rerank` adds separate rerank metrics during training validation.

## Verification

Run the offline suite and inspect the CLI:

```bash
uv run python -m unittest discover -s tests
uv run python -m scripts.train --help
```

The suite uses tiny randomly initialized fixed Transformers SigLIP models,
focused fakes, and a low-level H-W-C patch-order conformance test. It validates
official text/vision forward equivalence, SIE, native sigmoid losses,
accumulation, mixed precision, checkpoint compatibility, two-stage training,
prompt isolation, retrieval, and W&B behavior without downloading So400m.
