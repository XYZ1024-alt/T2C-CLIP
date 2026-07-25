# T2C-ReID

Train2Central ReID is a research codebase for Image-to-Image person ReID.
The public project name is `T2C-ReID` and the Python package is `t2c_reid`;
the foundation vision-language model is **SigLIP 2 So400m**.

The implementation uses a two-stage training pipeline:

- `google/siglip2-so400m-patch14-384` image and text towers.
- Market-1501 and MSMT17 person/camera parsing.
- Global, camera, and training-identity learnable prompts in the SigLIP 2 text
  token embedding space.
- Stage-1 supervised SigLIP alignment between image features and identity-aware
  prompt features.
- Stage-2 ReID identity, batch-hard triplet, SigLIP all-identity alignment, and
  camera-aware cross-modal TFC losses with per-identity/per-camera prototypes.
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

The project uses `uv` and builds a mandatory Rust extension with `maturin`:

```bash
uv sync
uv run python -c "from t2c_reid.native import NATIVE_VERSION; print(NATIVE_VERSION)"
uv run python -m unittest discover -s tests
```

Install stable Rust `1.85+` before `uv sync`. Windows x86_64 requires the
MSVC Rust target and Visual Studio Build Tools; Linux x86_64 requires a C
linker. The extension is built as the private CPython module
`t2c_reid._native`. There is no automatic Python fallback when that module is
missing or has an incompatible ABI.

Core requirements are declared in `pyproject.toml`:

- Python 3.14+
- Rust 1.85+ and maturin 1.14+
- NumPy 2.4+
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

With MSMT17 placed at `data/MSMT17_V1`, start the complete default recipe with:

```bash
uv run train
```

The default run uses Stage-1 for 20 epochs, Stage-2 for 120 epochs, validates
every 5 Stage-2 epochs, and writes to `checkpoints/msmt17-siglip2-tfc`.
Only specify values that differ from the baseline recipe. For example:

```bash
uv run train \
  --data-root D:/datasets/MSMT17_V1 \
  --tfc-weight 0.5 \
  --alignment-weight 0.25 \
  --run-name msmt17-tfc-weight-sweep
```

The defaults select MSMT17 at `data/MSMT17_V1`, the fixed model, `392x196`
input, Stage-1 `20`, Stage-2 `120`, batch `8`, accumulation `4`, eval batch
`16`, automatic precision, and gradient checkpointing. `--job-builder` remains
available for test fixtures or external jobs but is not required for normal
T2C-ReID training.

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

Camera-aware TFC maintains FP32 visual/text EMA centers for every observed
`(person_id, camera_id)`. Camera-local centers are aggregated with equal camera
weight into global identity centers, then fused with the current Stage-2 beta:

```text
P_local[y,c] = normalize(V_local[y,c] + beta * T_local[y,c])
P_global[y]  = normalize(V_global[y]  + beta * T_global[y])

L_TFC = weighted_mean(L_local, L_global, L_cross_modal, L_cross_camera)
      + transfer_reg_weight * KL(camera_prior || camera_transfer)
```

`L_cross_camera` is multi-positive InfoNCE over initialized prototypes: the
same identity in other cameras is positive and every different identity is
negative. A learned directed `C x C` row-stochastic transfer matrix weights the
available positive cameras. The text center teacher exactly encodes
`global + camera + identity` under `no_grad`; identity prompts remain absent
from query/gallery retrieval.

Long-tail identities use higher EMA momentum between `--tfc-momentum` and
`--tfc-tail-momentum`. Effective-number class weights are applied to every
sample-level TFC component. Set `--tfc-weight 0` to skip the teacher text
forward, all center updates, and all TFC losses.

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

Run and data defaults:

- `uv run train`
- `--dataset msmt17`
- `--data-root data/MSMT17_V1`
- `--stage1-epochs 20`
- `--epochs 120` (Stage-2 epochs)
- `--validation-interval 5`
- `--checkpoint-dir checkpoints/msmt17-siglip2-tfc`
- `--job-builder t2c_reid.jobs.siglip2_reid:build_training_job`

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
- `--num-workers 4`
- `--data-backend rust|python` (default `rust`; Python is a reference backend)
- `--prefetch-factor 2`
- `--pin-memory / --no-pin-memory` (defaults on for CUDA)
- `--persistent-workers / --no-persistent-workers` (defaults on with workers)
- `--rust-data-threads 1`
- `--evaluation-backend rust|python` (default `rust`)
- `--evaluation-chunk-size 256`
- `--precision auto|fp32|bf16|fp16`
- `--gradient-checkpointing / --no-gradient-checkpointing`

Optimization:

- `--lr 1e-4`
- `--image-encoder-lr 5e-6`
- `--alignment-weight 1.0`
- `--tfc-weight 1.0`
- `--tfc-momentum 0.5` (highest-frequency identity)
- `--tfc-tail-momentum 0.9` (lowest-frequency identity)
- `--tfc-class-balance-beta 0.9999`
- `--tfc-local-weight 1.0`
- `--tfc-global-weight 1.0`
- `--tfc-cross-modal-weight 0.5`
- `--tfc-cross-camera-weight 0.1`
- `--tfc-contrast-temperature 0.07`
- `--tfc-transfer-reg-weight 0.01`
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
New checkpoints use schema version 3 and include:

- `backbone_family=siglip2`, dataset, Hugging Face model ID, and feature dimension
- training identity/camera counts and a deterministic pid-camera count fingerprint
- Camera-aware TFC version and every momentum, weight, temperature, class-balance,
  beta schedule, and Stage-2 epoch-offset setting
- visual/text local and global prototypes, initialized masks, statistics, and camera transfer logits
- image size, patch size, patch count, maximum patch budget, and vision input format
- tokenizer padding/pooling layout
- resolved precision
- model, optimizer, and FP16 scaler state

Resume a Stage-2 run with the same architecture and precision:

```bash
uv run train \
  --resume checkpoints/msmt17-siglip2-tfc/last.pth
```

This migration intentionally rejects schema 2 Stage-2 checkpoints, OpenAI CLIP
weights, and old T2C-CLIP training checkpoints. The removed options
`--clip-model-name`, `--clip-checkpoint`, and `--clip-weight` are argparse
errors. Incompatible resume metadata fails before model weights are loaded.

## Weights & Biases

Enable online tracking:

```bash
uv run wandb login
uv run train \
  --enable-wandb \
  --wandb-project T2C-ReID \
  --run-name msmt17-siglip2-camera-tfc
```

Training metrics include:

- Stage-1: `loss`, `alignment_loss`, `lr`
- Stage-2: `loss`, `alignment_loss`, `reid_loss`, `triplet_loss`, `tfc_loss`,
  `tfc_local_loss`, `tfc_global_loss`, `tfc_cross_modal_loss`,
  `tfc_cross_camera_loss`, `tfc_transfer_reg_loss`,
  `tfc_cross_camera_coverage`, `lr`
- Validation: `mAP`, `best_mAP`, `rank_1`, `rank_5`, `rank_10`

`stage1_train_step` and `stage2_train_step` count successful optimizer update
windows, not micro-batches. Window metrics are means across their constituent
micro-batches; epoch metrics average all micro-batches.

## Feature Evaluation CLI

Evaluate pre-extracted query/gallery features from `.npz`:

```bash
uv run python -m t2c_reid.cli.evaluate path/to/features.npz \
  --output metrics.json \
  --ranks 1 5 10
```

The evaluator applies the standard Image-to-Image ReID protocol and excludes
same-identity, same-camera gallery samples. Rust evaluates exact cosine scores
in query chunks and aggregates deterministic mAP/CMC without retaining the
complete `Q x G` score matrix. Primary metrics remain no-rerank.

Add exact sparse k-reciprocal metrics without replacing the primary result:

```bash
uv run python -m t2c_reid.cli.evaluate path/to/features.npz \
  --report-rerank \
  --rerank-k1 20 \
  --rerank-k2 6 \
  --rerank-lambda 0.3
```

Sparse rerank keeps exact all-sample neighbor search and therefore still has
`O(N^2 D)` compute. After reciprocal edges are known, a second chunked Torch
pass extracts their exact affinity distances; resident affinity and Jaccard
structures remain sparse. Rerank distances within `1e-6` are treated as ties
and ordered by gallery index in both backends.

## Native Data Pipeline

The default training loader sends path/ID records to a batch collator. Rust
reads JPEG/PNG images, converts to RGB, applies flip, randomized ColorJitter,
bilinear resize, padded crop, SigLIP normalization, and normalized-space random
erasing, then transfers an owned contiguous `BCHW float32` allocation to NumPy
and `torch.from_numpy` without copying the element buffer.

A fixed `--seed` is repeatable within the same Rust pipeline version. The Rust
augmentation parameters and operation order match the torchvision pipeline,
but stochastic pixel values and random-number sequences are not bitwise
compatible with the Python backend. Eval resize regression fixtures differ by
at most one 8-bit quantization step after normalization.

## Performance Benchmark

Run the self-contained synthetic benchmark:

```bash
uv run python -m t2c_reid.cli.benchmark_native \
  --mode all \
  --runs 5 \
  --warmup-runs 1 \
  --output output/native-benchmark.json
```

Use real training images by adding `--dataset market1501|msmt17 --data-root
PATH`. The benchmark defaults to two Rust threads per data worker because that
was the first configuration to clear the synthetic throughput gate; production
training retains the conservative `--rust-data-threads 1` default and should be
tuned against available CPU cores. The JSON reports median/p95 duration, throughput, sampled RSS, backend
speedup, metric parity, and the acceptance gates: data `1.5x`, primary
evaluation `3x`, rerank `2x`, and rerank RSS ratio `<=0.4`. Benchmark gates are
not automated assertions because timing is hardware-sensitive. The recorded synthetic
Windows run and its real-dataset follow-up command are in
[docs/native-performance.md](docs/native-performance.md).

## Verification

Run the offline suite and inspect the CLI:

```bash
uv run cargo test --manifest-path rust/Cargo.toml --locked
uv run python -m unittest discover -s tests
uv run python -m compileall -q t2c_reid scripts tests
uv run train --help
uv run python -m t2c_reid.cli.evaluate --help
uv run python -m t2c_reid.cli.benchmark_native --help
```

The suite uses tiny randomly initialized fixed Transformers SigLIP models,
focused fakes, and a low-level H-W-C patch-order conformance test. It validates
official text/vision forward equivalence, SIE, native sigmoid losses,
accumulation, mixed precision, checkpoint compatibility, two-stage training,
prompt isolation, retrieval, and W&B behavior without downloading So400m.
