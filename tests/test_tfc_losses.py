import math
import unittest

import torch
import torch.nn.functional as F

from t2c_reid.losses import (
    batch_hard_triplet_loss,
    siglip_identity_anchor_loss,
    supervised_siglip_loss,
)
from t2c_reid.tfc import TFCCenterBank


class TFCLossTest(unittest.TestCase):
    def test_tfc_updates_identity_centers_with_ema(self):
        bank = TFCCenterBank(num_train_ids=2, feature_dim=2, momentum=0.5)
        bank.update(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), torch.tensor([0, 1]))
        bank.update(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))

        expected = torch.tensor([2 ** -0.5, 2 ** -0.5])
        self.assertTrue(torch.allclose(bank.centers[0], expected, atol=1e-6))

    def test_tfc_loss_uses_existing_centers(self):
        bank = TFCCenterBank(num_train_ids=1, feature_dim=2, momentum=0.5)
        bank.update(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        loss = bank.loss(torch.tensor([[0.0, 1.0]]), torch.tensor([0]))

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))


class SiglipAlignmentLossTest(unittest.TestCase):
    def test_native_reduction_matches_hand_computation(self):
        images = torch.eye(2)
        texts = torch.eye(2)
        person_ids = torch.tensor([0, 1])

        loss = supervised_siglip_loss(
            images,
            texts,
            person_ids,
            logit_scale=1.0,
            logit_bias=0.0,
        )

        # Each row has one positive logit 1 and one negative logit 0.
        expected = F.softplus(torch.tensor(-1.0)) + math.log(2.0)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_scale_and_bias_are_applied_before_logsigmoid(self):
        images = torch.tensor([[1.0, 0.0]])
        texts = torch.tensor([[1.0, 0.0]])

        loss = supervised_siglip_loss(
            images,
            texts,
            torch.tensor([0]),
            logit_scale=3.0,
            logit_bias=-1.0,
        )

        self.assertTrue(torch.allclose(loss, F.softplus(torch.tensor(-2.0))))

    def test_same_identity_pairs_are_all_positive(self):
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        loss = supervised_siglip_loss(
            images,
            texts,
            torch.tensor([0, 0]),
            logit_scale=1.0,
            logit_bias=0.0,
        )
        loss.backward()

        # Moving image 0 toward text 1 lowers the all-positive objective.
        self.assertLess(float(images.grad[0] @ texts[1]), 0.0)

    def test_pairwise_loss_runs_in_fp32_under_low_precision_inputs(self):
        loss = supervised_siglip_loss(
            torch.eye(2, dtype=torch.bfloat16),
            torch.eye(2, dtype=torch.bfloat16),
            torch.tensor([0, 1]),
            logit_scale=2.0,
            logit_bias=-0.5,
        )
        self.assertEqual(loss.dtype, torch.float32)

    def test_pairwise_loss_disables_active_autocast(self):
        images = torch.eye(2)
        texts = torch.eye(2)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss = supervised_siglip_loss(
                images,
                texts,
                torch.tensor([0, 1]),
                logit_scale=2.0,
                logit_bias=-0.5,
            )

        reference = supervised_siglip_loss(
            images,
            texts,
            torch.tensor([0, 1]),
            logit_scale=2.0,
            logit_bias=-0.5,
        )
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.equal(loss, reference))

    def test_supervised_loss_validates_person_ids(self):
        with self.assertRaises(ValueError):
            supervised_siglip_loss(torch.eye(2), torch.eye(2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            supervised_siglip_loss(
                torch.eye(2), torch.eye(2), torch.tensor([0.5, 1.5])
            )


class SiglipIdentityAnchorLossTest(unittest.TestCase):
    def test_stage2_uses_one_positive_and_all_other_anchors_as_negatives(self):
        anchors = torch.eye(2)
        visual = torch.tensor([[2.0, 0.0], [0.0, 3.0]])

        loss = siglip_identity_anchor_loss(
            visual,
            anchors,
            torch.tensor([0, 1]),
            logit_scale=1.0,
            logit_bias=0.0,
        )

        expected = F.softplus(torch.tensor(-1.0)) + math.log(2.0)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_correct_anchor_assignment_scores_lower_than_wrong(self):
        anchors = torch.eye(2)
        visual = torch.eye(2)
        correct = siglip_identity_anchor_loss(
            visual, anchors, torch.tensor([0, 1]), logit_scale=5.0
        )
        wrong = siglip_identity_anchor_loss(
            visual, anchors, torch.tensor([1, 0]), logit_scale=5.0
        )
        self.assertLess(float(correct), float(wrong))

    def test_extra_negative_anchor_participates_in_native_sum(self):
        visual = torch.tensor([[1.0, 0.0]])
        ids = torch.tensor([0])
        two = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        three = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])

        without_extra = siglip_identity_anchor_loss(
            visual, two, ids, logit_scale=5.0
        )
        with_extra = siglip_identity_anchor_loss(
            visual, three, ids, logit_scale=5.0
        )

        self.assertGreater(float(with_extra), float(without_extra))

    def test_stage2_alignment_validates_shapes_and_ids(self):
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.ones(2, 2, 2), torch.eye(2), torch.tensor([0, 1])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(torch.eye(2), torch.ones(2), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.ones(2, 3), torch.ones(4, 2), torch.tensor([0, 1])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(
                torch.eye(2), torch.eye(2), torch.tensor([0.5, 1.5])
            )
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(torch.eye(2), torch.eye(2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            siglip_identity_anchor_loss(torch.eye(2), torch.eye(2), torch.tensor([0, 2]))


class TripletLossTest(unittest.TestCase):
    def test_batch_hard_triplet_penalizes_close_negative(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertGreater(float(loss), 0.0)

    def test_euclidean_default_is_hand_computed(self):
        features = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertTrue(torch.allclose(loss, torch.tensor(0.15), atol=1e-6))

    def test_cosine_and_euclidean_differ_for_scaled_direction(self):
        features = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])
        euclidean = batch_hard_triplet_loss(features, labels, metric="euclidean")
        cosine = batch_hard_triplet_loss(features, labels, metric="cosine")
        self.assertFalse(torch.allclose(euclidean, cosine))

    def test_unknown_triplet_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            batch_hard_triplet_loss(
                torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]),
                torch.tensor([0, 0, 1]),
                metric="manhattan",
            )


if __name__ == "__main__":
    unittest.main()
