"""Train2Central-CLIP core package."""

from t2c_clip.data import ReIDSample
from t2c_clip.evaluation import ReIDMetrics, evaluate_reid
from t2c_clip.features import fuse_features, l2_normalize
from t2c_clip.losses import batch_hard_triplet_loss, bidirectional_contrastive_loss
from t2c_clip.loops import (
    DEFAULT_VALIDATION_INTERVAL,
    EpochResult,
    TrainingLoopConfig,
    TrainingLoopResult,
    run_training_loop,
    should_validate_epoch,
)
from t2c_clip.model import T2CClipModel
from t2c_clip.wandb import WandbConfig, WandbTracker, start_wandb_run
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.tfc import TFCCenterBank
from t2c_clip.training import (
    Stage1LossConfig,
    Stage2LossBreakdown,
    Stage2LossConfig,
    Stage2LossInputs,
    TrainingBatch,
    stage1_alignment_loss,
    stage2_loss_breakdown,
)

__all__ = [
    "DEFAULT_VALIDATION_INTERVAL",
    "EpochResult",
    "PromptBank",
    "PromptConfig",
    "ReIDMetrics",
    "ReIDSample",
    "Stage1LossConfig",
    "Stage2LossBreakdown",
    "Stage2LossConfig",
    "Stage2LossInputs",
    "T2CClipModel",
    "TFCCenterBank",
    "TrainingBatch",
    "TrainingLoopConfig",
    "TrainingLoopResult",
    "WandbConfig",
    "WandbTracker",
    "batch_hard_triplet_loss",
    "bidirectional_contrastive_loss",
    "evaluate_reid",
    "fuse_features",
    "l2_normalize",
    "run_training_loop",
    "should_validate_epoch",
    "stage1_alignment_loss",
    "stage2_loss_breakdown",
    "start_wandb_run",
]
