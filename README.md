# T2C-CLIP

Train2Central-CLIP foundation for Image-to-Image person ReID experiments.

The implementation follows `DESIGN.md` (the consolidated design doc) and
`docs/2026-06-27-t2c-clip-design.md` (research blueprint). The current
codebase trains a real CLIP dual-stream model with a two-stage pipeline:

- Market-1501 and MSMT17 person/camera parsing.
- Global, camera, and training-ID learnable prompt composition that lives
  in the CLIP text **token embedding space** (not the projection space).
- Real CLIP text encoder injection through `TransformersCLIPTextEncoder`.
- Stage-1 prompt alignment: `f_v` vs training-ID prompt text feature
  `f_t_id` via bidirectional image-text contrastive loss.
- Stage-2 ReID training: classifier, triplet, and TFC losses act on
  `f_eval = normalize(f_v + beta * f_t_eval)`, where `f_t_eval` uses only
  global + camera prompts. The identity-aware text feature is used only by
  the auxiliary CLIP alignment loss.
- Inference: `f_eval = normalize(f_v + beta * f_t_eval)` with only
  global + camera prompts. `--retrieval-mode image_only` evaluates normalized
  image features without the text branch.
- Training-time Feature Centralization (TFC) EMA centers act on the
  fused retrieval feature during Stage-2 only.
- tqdm-based training loop scheduling with configurable mAP validation
  intervals and stage-aware checkpoint naming.
- `best.pth`, `last.pth` (Stage-2) plus `stage1_last.pth` (Stage-1).
- MLflow tracking with stage-aware batch-step / epoch / validation metrics.
- No-rerank cosine ReID evaluation.
- Transformers CLIP-backed Market-1501/MSMT17 training job builder.
- `.npz` no-rerank evaluation CLI.

## Environment

Use the WSL conda environment named `reid`:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m unittest discover -s tests -v
```

## Evaluate Extracted Features

The evaluator expects a `.npz` file with these arrays:

- `query_features`: float matrix `[num_query, dim]`
- `gallery_features`: float matrix `[num_gallery, dim]`
- `query_ids`: integer vector `[num_query]`
- `gallery_ids`: integer vector `[num_gallery]`
- `query_cams`: integer vector `[num_query]`
- `gallery_cams`: integer vector `[num_gallery]`

Run:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m t2c_clip.cli.evaluate features.npz --output metrics.json --ranks 1 5 10
```

The evaluator performs standard Image-to-Image ReID scoring without rerank. Same-identity same-camera gallery samples are excluded for each query.
Training keeps this no-rerank mAP as the primary metric. Pass `--report-rerank`
to log additional `rerank_mAP` and `rerank_rank_1` values for comparison only;
do not mix those values into the main no-rerank result table.

## Training Loop Checkpoints

`run_training_loop` provides reusable epoch scheduling around project-specific training and validation callables:

- `TrainingLoopConfig(validation_interval=5)` validates mAP every 5 epochs by default.
- Set `validation_interval` to another positive integer to validate at a different cadence.
- The loop wraps epochs with `tqdm`.
- Validation epochs write `mAP`, Rank-1, best mAP, and best-status to the progress output.
- `last.pth` is saved after every epoch.
- `best.pth` is saved only when the latest validation mAP is better than the previous best.

The loop calls real `torch.save` and lets training, validation, tqdm, and checkpoint errors surface directly.

## Train Entrypoint

`scripts/train.py` wires a project-specific training job into `run_training_loop`.
The recommended real training builder is `t2c_clip.jobs.clip_reid:build_training_job`,
which produces a `TwoStageTrainingJob` when `--stage1-epochs > 0` and a single
Stage-2 `TrainingJob` otherwise.

Stage-1 prompt alignment runs first (image encoder frozen by default), then
Stage-2 ReID training runs. CLIP-ReID's standard Stage-2 recipe fine-tunes
the vision encoder so the ReID signal can actually reach the retrieval
feature; this project therefore defaults to **unfrozen at Stage-2**. Pass
`--freeze-image-encoder-stage2` to opt back into the prompt-tuning-only
mode (which on person ReID tops out near the frozen CLIP image-only floor).
Identity prompts present in Stage-1 and Stage-2 training are **never** used
at inference; `encode_retrieval` only ever composes `global + camera`
prompts.

Market-1501 expects the standard directories under `--data-root`:

- `bounding_box_train/`
- `query/`
- `bounding_box_test/`

MSMT17 expects the standard manifests and image folders under `--data-root`:

- `list_train.txt`
- `list_query.txt`
- `list_gallery.txt`
- `train/`
- `test/`

Run Stage-1 + Stage-2 training:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python scripts/train.py \
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
  --image-encoder-lr 0.00005 \
  --beta-warmup-epochs 5 \
  --sanity-gate-epochs 10 \
  --freeze-image-encoder-stage1
```

The MSMT17 command above fine-tunes the CLIP vision encoder during Stage-2
(see `--image-encoder-lr`). The smaller backbone learning rate
(`5e-5` by default) keeps the pretrained features from being catastrophically
forgotten while still letting the ReID losses tune them. The blended
text branch ramp goes from `0` at epoch 1 (pure image feature) to the
configured `--beta` at `--beta-warmup-epochs + 1`, so the random
camera-conditioned text feature does not pull the retrieval feature below
the image-only floor at startup. `--sanity-gate-epochs` fails the run early
if mAP stays pinned to the random floor (sign the ReID signal isn't acting
on `f_eval`).

Useful training arguments:

- `--dataset market1501|msmt17`
- `--data-root PATH`
- `--clip-model-name openai/clip-vit-base-patch16`
- `--clip-checkpoint /path/to/clip_state.pth` (optional local CLIP/ReID
  checkpoint; missing, unexpected, or incomplete keys fail at startup)
- `--stage1-epochs N` (Stage-1 prompt alignment epochs; `0` skips Stage-1)
- `--epochs N` (Stage-2 ReID training epochs)
- `--validation-interval N` (Stage-2 mAP validation cadence)
- `--batch-size 384`
- `--num-workers 4`
- `--lr 0.0001` (learning rate for the prompt bank, classifier, and unfrozen
  text encoder)
- `--image-encoder-lr 0.00005` (learning rate for the CLIP vision backbone
  and visual projection; smaller than `--lr` to avoid catastrophic
  forgetting of the pretrained visual features)
- `--device cuda`
- `--beta 0.1`
- `--beta-warmup-epochs 0` (linear ramp of the fused retrieval beta from
  `0` at epoch 1 to `--beta` at `warmup-epochs + 1`)
- `--context-length 4`
- `--tfc-momentum 0.5`
- `--triplet-margin 0.3`
- `--tfc-weight 1.0`
- `--clip-weight 0.1`
- `--label-smoothing 0.0` (Stage-2 identity classification label smoothing;
  strong ReID recipes commonly use `0.1`)
- `--reid-head linear|bnneck` (`bnneck` classifies BN-normalized retrieval
  features while triplet/TFC/eval keep the T2C retrieval feature)
- `--retrieval-mode fused|image_only` (`fused` validates global + camera prompt fusion; `image_only` validates normalized CLIP image features)
- `--report-rerank` (logs extra rerank metrics while keeping primary `mAP`
  no-rerank)
- `--freeze-image-encoder-stage1 / --no-freeze-image-encoder-stage1` (default frozen)
- `--freeze-image-encoder-stage2 / --no-freeze-image-encoder-stage2` (default **unfrozen**; CLIP-ReID Stage-2 fine-tunes the vision encoder. Pass `--freeze-image-encoder-stage2` to fall back to prompt-tuning-only mode, which on person ReID tops out near the frozen CLIP image-only mAP floor.)
- `--freeze-text-encoder / --no-freeze-text-encoder` (default frozen — CoOp/CLIP-ReID-style prompt tuning)
- `--freeze-prompt-bank-stage2 / --no-freeze-prompt-bank-stage2` (default
  trainable; pass `--freeze-prompt-bank-stage2` for a CLIP-ReID-faithful
  Stage-2 run that treats Stage-1 prompts as fixed text descriptors)
- `--sanity-gate-epochs 0` (when > 0, fails the training run early at the
  first Stage-2 validation at-or-past epoch offset if best mAP is below
  `sanity-gate-factor × first-validation mAP`. Catches regressions where
  training never escapes the random-init floor.)
- `--sanity-gate-factor 1.5`

Training uses identity-balanced batches for batch-hard triplet loss. Each
train batch samples two images per identity, so `--batch-size 64` means
`32 identities x 2 images`. If the training split cannot provide enough
identities with positive pairs for the configured batch size, startup
fails with an explicit error.

Add `--enable-mlflow` to initialize the SQLite-backed MLflow store before training:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python scripts/train.py \
  --job-builder t2c_clip.jobs.clip_reid:build_training_job \
  --dataset msmt17 \
  --data-root MSMT17_V1 \
  --epochs 120 \
  --validation-interval 5 \
  --checkpoint-dir checkpoints \
  --enable-mlflow \
  --tracking-db mlflow/t2c_clip.db \
  --artifact-root mlruns \
  --experiment-name T2C-CLIP \
  --run-name msmt17-stage2
```

When MLflow is enabled, training logs stage-aware metrics:

- Stage-1 batch step: `stage1_train_step_loss`, `stage1_train_step_clip_loss`, `stage1_train_step_lr`
- Stage-1 epoch average: `stage1_train_loss`, `stage1_train_clip_loss`, `stage1_lr`
- Stage-2 batch step: `stage2_train_step_loss`, `stage2_train_step_reid_loss`,
  `stage2_train_step_triplet_loss`, `stage2_train_step_clip_loss`,
  `stage2_train_step_tfc_loss`, `stage2_train_step_lr`
- Stage-2 epoch average: `stage2_train_loss`, `stage2_train_reid_loss`,
  `stage2_train_triplet_loss`, `stage2_train_clip_loss`, `stage2_train_tfc_loss`, `stage2_lr`
- Validation (Stage-2 only): `mAP`, `best_mAP`, `rank_1`, `rank_5`, `rank_10`, `is_best`

Run params recorded with `--enable-mlflow`:

- `dataset`, `clip_model_name`, `stage1_epochs`, `stage2_epochs`,
  `validation_interval`, `batch_size`, `lr`, `beta`, `clip_weight`,
  `tfc_weight`, `freeze_image_encoder_stage1`, `freeze_image_encoder_stage2`,
  `freeze_text_encoder`, `retrieval_mode`.

## MLflow SQLite Tracking

Initialize the local SQLite-backed MLflow store:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python -m t2c_clip.cli.mlflow --tracking-db mlflow/t2c_clip.db --artifact-root mlruns --experiment-name T2C-CLIP
```

Start the MLflow UI on port 6006:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid mlflow ui --backend-store-uri sqlite:///mlflow/t2c_clip.db --default-artifact-root mlruns --host 127.0.0.1 --port 6006
```

The SQLite database and artifact directory are local runtime outputs and are ignored by git.

## CLIP Encoder Boundary

`T2CClipModel` injects real CLIP image and text encoders. The recommended
`clip_reid` builder:

- wraps a real Transformers CLIP image encoder through
  `TransformersCLIPImageEncoder`,
- wraps a real CLIP text encoder through `TransformersCLIPTextEncoder`, which
  injects learnable prompt parameters into the CLIP text **token embedding**
  space and runs `text_model.encoder` + `text_projection` so `f_t` lives in the
  shared CLIP projection space with `f_v`,
- never invents mock metrics, returns fake training success, silently
  switches devices, or hides missing dependency and dataset errors.

## Sanity Check

Before running 120-epoch Stage-2 training, run a short sanity check on
MSMT17 to validate the Stage-1 + Stage-2 pipeline:

```bash
wsl --cd /mnt/d/Code/T2C-CLIP /home/xyz10/miniconda3/bin/conda run -n reid python scripts/train.py \
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

Sanity gate:

1. image-only CLIP baseline mAP > random.
2. Stage-1 `clip_loss` clearly below `ln(batch_size)`.
3. Stage-2 `reid_loss` clearly below `ln(num_train_ids)`.
4. Stage-2 mAP after 5/10 epochs must not stay pinned to the random floor.
5. **Inference path never uses identity prompts** — verified by unit tests
   (`test_features_prompts`, `test_evaluation_model`,
   `test_two_stage_training`) and by the `encode_retrieval` implementation
   in `t2c_clip/model.py`.
