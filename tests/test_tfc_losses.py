import math
import unittest

import torch

from t2c_clip.losses import (
    batch_hard_triplet_loss,
    bidirectional_contrastive_loss,
    image_to_text_cross_entropy,
    supervised_bidirectional_contrastive_loss,
)
from t2c_clip.tfc import TFCCenterBank


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

    def test_bidirectional_contrastive_loss_is_low_for_matching_pairs(self):
        image = torch.eye(2)
        text = torch.eye(2)
        loss = bidirectional_contrastive_loss(image, text, logit_scale=10.0)
        self.assertLess(float(loss), 0.01)

    def test_batch_hard_triplet_loss_penalizes_close_negative(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        labels = torch.tensor([0, 0, 1])
        loss = batch_hard_triplet_loss(features, labels, margin=0.3)
        self.assertGreater(float(loss), 0.0)

    def test_batch_hard_triplet_loss_defaults_to_euclidean_hand_computed(self):
        # d(a0,a1)=5, d(a0,a2)=10, d(a1,a2)=5 in raw euclidean space.
        # anchor0: relu(5 - 10 + 0.3) = 0; anchor1: relu(5 - 5 + 0.3) = 0.3;
        # anchor2 has no positive -> skipped. mean = 0.15.
        features = torch.tensor([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0]])
        labels = torch.tensor([0, 0, 1])

        loss = batch_hard_triplet_loss(features, labels, margin=0.3)

        self.assertTrue(torch.allclose(loss, torch.tensor(0.15), atol=1e-6))

    def test_batch_hard_triplet_loss_euclidean_ignores_feature_norm_direction(self):
        # Euclidean must NOT l2-normalize: [3,4] and [6,8] share a direction
        # (cosine distance 0) but are euclidean distance 5 apart.
        features = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1])

        euclidean = batch_hard_triplet_loss(features, labels, margin=0.3, metric="euclidean")
        cosine = batch_hard_triplet_loss(features, labels, margin=0.3, metric="cosine")

        self.assertGreater(float(euclidean), 0.0)
        self.assertFalse(torch.allclose(euclidean, cosine))

    def test_batch_hard_triplet_loss_cosine_preserves_previous_formula(self):
        # Old formula: distances = 1 - l2n(f) @ l2n(f).T. For this fixture both
        # valid anchors contribute 1 - (1 - 2**-0.5) + 0.3 = 0.3 + 2**-0.5.
        features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        labels = torch.tensor([0, 0, 1])

        loss = batch_hard_triplet_loss(features, labels, margin=0.3, metric="cosine")

        expected = torch.tensor(0.3 + 2 ** -0.5)
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_batch_hard_triplet_loss_rejects_unknown_metric(self):
        features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]])
        labels = torch.tensor([0, 0, 1])

        with self.assertRaises(ValueError):
            batch_hard_triplet_loss(features, labels, margin=0.3, metric="manhattan")


    def test_supervised_contrastive_matches_arange_loss_for_unique_ids(self):
        torch.manual_seed(0)
        image = torch.randn(4, 3)
        text = torch.randn(4, 3)
        ids = torch.tensor([3, 1, 0, 2])

        supervised = supervised_bidirectional_contrastive_loss(image, text, ids, logit_scale=7.0)
        plain = bidirectional_contrastive_loss(image, text, logit_scale=7.0)

        self.assertTrue(torch.allclose(supervised, plain, atol=1e-6))

    def test_supervised_contrastive_pulls_same_identity_text_toward_image(self):
        # PK batches always contain same-person pairs. The plain arange loss
        # pushes image 0 away from the same person's other text row; the
        # supervised loss must treat it as a positive instead.
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
        texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        ids = torch.tensor([0, 0])

        loss = supervised_bidirectional_contrastive_loss(images, texts, ids, logit_scale=1.0)
        loss.backward()

        # Moving image 0 toward the same-person text 1 must decrease the loss.
        self.assertLess(float(images.grad[0] @ texts[1]), 0.0)

    def test_supervised_contrastive_validates_person_ids(self):
        with self.assertRaises(ValueError):
            supervised_bidirectional_contrastive_loss(torch.eye(2), torch.eye(2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            supervised_bidirectional_contrastive_loss(torch.eye(2), torch.eye(2), torch.tensor([0.5, 1.5]))


class ImageToTextCrossEntropyTest(unittest.TestCase):
    def test_i2t_hand_computed_with_orthogonal_anchors(self):
        # Each row's logits are [1, 0] (or [0, 1]) at logit_scale=1, so the
        # cross entropy is log(1 + e^-1) for both samples.
        anchors = torch.eye(2)
        visual = torch.tensor([[2.0, 0.0], [0.0, 3.0]])  # non-unit: must be L2-normalized inside
        person_ids = torch.tensor([0, 1])

        loss = image_to_text_cross_entropy(visual, anchors, person_ids, logit_scale=1.0)

        expected = torch.tensor(math.log(1.0 + math.exp(-1.0)))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))

    def test_i2t_correct_target_scores_lower_than_wrong_target(self):
        anchors = torch.eye(2)
        visual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

        correct = image_to_text_cross_entropy(visual, anchors, torch.tensor([0, 1]), logit_scale=5.0)
        wrong = image_to_text_cross_entropy(visual, anchors, torch.tensor([1, 0]), logit_scale=5.0)

        self.assertLess(float(correct), float(wrong))

    def test_i2t_classifies_against_all_anchor_rows_not_just_batch(self):
        # Anchor 2 is close to the sample; including it must raise the loss
        # even though no batch sample carries person id 2.
        visual = torch.tensor([[1.0, 0.0]])
        person_ids = torch.tensor([0])
        two_anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        three_anchors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])

        without_extra = image_to_text_cross_entropy(visual, two_anchors, person_ids, logit_scale=5.0)
        with_extra = image_to_text_cross_entropy(visual, three_anchors, person_ids, logit_scale=5.0)

        self.assertGreater(float(with_extra), float(without_extra))

    def test_i2t_label_smoothing_raises_separable_loss(self):
        anchors = torch.eye(2)
        visual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        person_ids = torch.tensor([0, 1])

        plain = image_to_text_cross_entropy(visual, anchors, person_ids, logit_scale=20.0)
        smoothed = image_to_text_cross_entropy(
            visual, anchors, person_ids, logit_scale=20.0, label_smoothing=0.2
        )

        self.assertLess(float(plain), 0.01)
        self.assertGreater(float(smoothed), float(plain))

    def test_i2t_validates_shapes_and_person_ids(self):
        with self.assertRaises(ValueError):
            image_to_text_cross_entropy(torch.ones(2, 2, 2), torch.eye(2), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            image_to_text_cross_entropy(torch.eye(2), torch.ones(2), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            image_to_text_cross_entropy(torch.ones(2, 3), torch.ones(4, 2), torch.tensor([0, 1]))
        with self.assertRaises(ValueError):
            image_to_text_cross_entropy(torch.eye(2), torch.eye(2), torch.tensor([0.5, 1.5]))
        with self.assertRaises(ValueError):
            image_to_text_cross_entropy(torch.eye(2), torch.eye(2), torch.tensor([0]))
        with self.assertRaises(ValueError):
            image_to_text_cross_entropy(torch.eye(2), torch.eye(2), torch.tensor([0, 2]))
