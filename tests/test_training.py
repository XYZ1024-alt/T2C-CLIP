import unittest

import torch

from t2c_clip.losses import supervised_bidirectional_contrastive_loss
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

    def test_stage1_alignment_treats_same_identity_as_positive(self):
        # PK-sampled batches repeat identities; the stage-1 alignment loss must
        # score same-person rows as positives, not arange negatives.
        model = _training_model(beta=0.0)
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]),
            camera_ids=torch.tensor([0, 1, 0]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        config = Stage1LossConfig(logit_scale=5.0)
        outputs = model.forward_stage1(batch.images, batch.camera_ids, batch.person_ids)
        expected = supervised_bidirectional_contrastive_loss(
            outputs["visual"], outputs["text"], batch.person_ids, logit_scale=5.0
        )

        breakdown = stage1_alignment_loss(model, batch, config)

        self.assertTrue(torch.allclose(breakdown.clip_dual, expected))

    def test_stage2_clip_alignment_treats_same_identity_as_positive(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 1, 0]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"], batch.person_ids)
        outputs = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)
        expected = supervised_bidirectional_contrastive_loss(
            outputs["visual"], outputs["text"], batch.person_ids, logit_scale=Stage2LossConfig().logit_scale
        )

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank),
        )

        self.assertTrue(torch.allclose(breakdown.clip_dual, expected))

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

    def test_stage2_loss_updates_tfc_centers_through_feature_head(self):
        # Stage-2 must initialize/update TFC centers from the SAME forward's
        # feature-head output before scoring — no separate no-grad forward and
        # no RuntimeError on a fresh center bank.
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]),
            camera_ids=torch.tensor([0, 0, 1]),
            person_ids=torch.tensor([0, 0, 1]),
        )
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, feature_head=head),
        )

        self.assertTrue(bool(tfc_bank.initialized.all()))
        # Centers live in feature-head space: the head zeroes dimension 1.
        self.assertTrue(torch.allclose(tfc_bank.centers[:, 1], torch.zeros(2)))
        self.assertGreaterEqual(float(breakdown.tfc.detach()), 0.0)

    def test_stage2_triplet_and_tfc_use_feature_head_output(self):
        model = _training_model(beta=0.0)
        classifier = torch.nn.Linear(2, 2, bias=False)
        head = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            classifier.weight.copy_(torch.eye(2))
            head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        batch = TrainingBatch(
            images=torch.tensor([[1.0, 0.5], [1.0, -0.5], [-1.0, 0.5], [-1.0, -0.5]]),
            camera_ids=torch.tensor([0, 0, 1, 1]),
            person_ids=torch.tensor([0, 0, 1, 1]),
        )
        raw_features = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)["retrieval"]
        headed_features = head(raw_features)
        tfc_bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        tfc_bank.update(headed_features, batch.person_ids)

        breakdown = stage2_loss_breakdown(
            model,
            batch,
            Stage2LossInputs(classifier=classifier, tfc_bank=tfc_bank, feature_head=head),
        )

        self.assertLess(float(breakdown.triplet.detach()), 0.01)
        self.assertLess(float(breakdown.tfc.detach()), 0.01)


def _training_model(beta: float) -> T2CClipModel:
    bank = PromptBank(PromptConfig(num_cameras=2, num_train_ids=2, context_length=1, embedding_dim=2))
    with torch.no_grad():
        bank.global_prompt.zero_()
        bank.camera_prompts.zero_()
        bank.identity_prompts[0] = torch.tensor([[1.0, 0.0]])
        bank.identity_prompts[1] = torch.tensor([[0.0, 1.0]])
    return T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), bank, beta=beta)
