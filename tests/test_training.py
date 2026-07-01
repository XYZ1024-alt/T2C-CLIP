import unittest

import torch

from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.tfc import TFCCenterBank
from t2c_clip.training import (
    Stage1LossConfig,
    Stage2LossConfig,
    Stage2LossInputs,
    TrainingBatch,
    stage1_alignment_loss,
    stage2_loss_breakdown,
)


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class PromptMeanEncoder(torch.nn.Module):
    def forward(self, prompts):
        return prompts.mean(dim=1)


class TrainingLossTest(unittest.TestCase):
    def test_stage1_alignment_uses_training_identity_prompts(self):
        model = _training_model(beta=0.0)
        batch = TrainingBatch(
            images=torch.eye(2),
            camera_ids=torch.tensor([0, 1]),
            person_ids=torch.tensor([0, 1]),
        )

        breakdown = stage1_alignment_loss(model, batch, Stage1LossConfig(logit_scale=10.0))

        self.assertLess(float(breakdown.total.detach()), 0.01)
        self.assertTrue(torch.allclose(breakdown.total, breakdown.clip_dual))

    def test_stage2_loss_breakdown_combines_clip_reid_and_tfc(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(10.0 * torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 0, 1]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)
        inputs = Stage2LossInputs(
            classifier=classifier,
            tfc_bank=tfc_bank,
            config=Stage2LossConfig(tfc_weight=0.5, clip_weight=1.0),
        )

        breakdown = stage2_loss_breakdown(model, batch, inputs)

        expected = (
            breakdown.identity + breakdown.triplet
            + breakdown.clip_weight * breakdown.clip_dual
            + breakdown.tfc_weight * breakdown.tfc
        )
        self.assertTrue(torch.allclose(breakdown.total, expected))
        self.assertEqual(breakdown.clip_weight, 1.0)
        self.assertEqual(breakdown.tfc_weight, 0.5)

    def test_stage2_loss_uses_configured_label_smoothing(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)
        config = Stage2LossConfig(label_smoothing=0.1)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, config=config),
        )

        self.assertGreater(float(breakdown.identity.detach()), 0.0)

    def test_stage2_loss_scales_identity_logits(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)

        unscaled = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank),
        )
        scaled = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(
                classifier=classifier,
                tfc_bank=tfc_bank,
                config=Stage2LossConfig(id_logit_scale=10.0),
            ),
        )

        self.assertLess(float(scaled.identity.detach()), float(unscaled.identity.detach()))

    def test_stage2_loss_classifies_feature_head_output(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(10.0 * torch.eye(2))
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            head.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([1, 1, 0, 0]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, feature_head=head),
        )

        self.assertLess(float(breakdown.identity.detach()), 0.01)


def _training_model(beta: float) -> T2CClipModel:
    bank = PromptBank(PromptConfig(num_cameras=2, num_train_ids=2, context_length=1, embedding_dim=2))
    with torch.no_grad():
        bank.global_prompt.zero_()
        bank.camera_prompts.zero_()
        bank.identity_prompts[0] = torch.tensor([[1.0, 0.0]])
        bank.identity_prompts[1] = torch.tensor([[0.0, 1.0]])
    return T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), bank, beta=beta)
