"""Train2Central ReID core package."""

from t2c_reid.data import ReIDSample
from t2c_reid.evaluation import ReIDMetrics, evaluate_reid
from t2c_reid.features import fuse_features, l2_normalize
from t2c_reid.losses import batch_hard_triplet_loss, supervised_siglip_loss
from t2c_reid.loops import (
    DEFAULT_VALIDATION_INTERVAL,
    EpochResult,
    TrainingLoopConfig,
    TrainingLoopResult,
    run_training_loop,
    should_validate_epoch,
)
from t2c_reid.model import T2CReIDModel
from t2c_reid.wandb import WandbConfig, WandbTracker, start_wandb_run
from t2c_reid.prompts import PromptBank, PromptConfig
from t2c_reid.tfc import TFCCenterBank
from t2c_reid.training import (
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
    "T2CReIDModel",
    "TFCCenterBank",
    "TrainingBatch",
    "TrainingLoopConfig",
    "TrainingLoopResult",
    "WandbConfig",
    "WandbTracker",
    "batch_hard_triplet_loss",
    "supervised_siglip_loss",
    "evaluate_reid",
    "fuse_features",
    "l2_normalize",
    "run_training_loop",
    "should_validate_epoch",
    "stage1_alignment_loss",
    "stage2_loss_breakdown",
    "start_wandb_run",
]
