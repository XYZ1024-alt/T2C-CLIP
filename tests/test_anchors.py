import unittest

import torch

from t2c_reid.anchors import IdentityAnchorProvider
from t2c_reid.model import T2CReIDModel
from t2c_reid.prompts import PromptBank, PromptConfig


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class CountingPromptMeanEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, prompts):
        self.call_count += 1
        return prompts.mean(dim=1)


class IdentityAnchorProviderTest(unittest.TestCase):
    def test_anchor_matrix_has_one_detached_row_per_identity(self):
        model, encoder = _anchor_model(num_train_ids=5)

        provider = IdentityAnchorProvider(model, num_train_ids=5, frozen=True)
        anchors = provider.anchors()

        self.assertEqual(tuple(anchors.shape), (5, 2))
        self.assertFalse(anchors.requires_grad)
        self.assertIsNone(anchors.grad_fn)

    def test_anchors_are_not_computed_at_construction(self):
        model, encoder = _anchor_model(num_train_ids=5)

        IdentityAnchorProvider(model, num_train_ids=5, frozen=True)

        self.assertEqual(encoder.call_count, 0)

    def test_anchors_are_encoded_in_chunks(self):
        model, encoder = _anchor_model(num_train_ids=5)

        chunked = IdentityAnchorProvider(model, num_train_ids=5, frozen=True, chunk_size=2).anchors()
        chunk_calls = encoder.call_count
        single = IdentityAnchorProvider(model, num_train_ids=5, frozen=True, chunk_size=16).anchors()

        self.assertEqual(chunk_calls, 3)
        self.assertEqual(encoder.call_count - chunk_calls, 1)
        self.assertTrue(torch.equal(chunked, single))

    def test_frozen_provider_computes_once_and_ignores_epoch_boundaries(self):
        model, encoder = _anchor_model(num_train_ids=5)
        provider = IdentityAnchorProvider(model, num_train_ids=5, frozen=True, chunk_size=2)

        first = provider.anchors()
        provider.start_epoch()
        with torch.no_grad():
            model.prompt_bank.identity_prompts.fill_(3.0)
        second = provider.anchors()

        self.assertEqual(encoder.call_count, 3)
        self.assertTrue(torch.equal(first, second))

    def test_unfrozen_provider_recomputes_at_epoch_start(self):
        model, encoder = _anchor_model(num_train_ids=5)
        provider = IdentityAnchorProvider(model, num_train_ids=5, frozen=False, chunk_size=2)

        first = provider.anchors()
        provider.anchors()  # same epoch: cached
        self.assertEqual(encoder.call_count, 3)

        provider.start_epoch()
        with torch.no_grad():
            model.prompt_bank.identity_prompts.fill_(3.0)
        second = provider.anchors()

        self.assertEqual(encoder.call_count, 6)
        self.assertFalse(torch.equal(first, second))

    def test_anchors_exclude_camera_prompts(self):
        model, _ = _anchor_model(num_train_ids=3)
        with torch.no_grad():
            model.prompt_bank.camera_prompts.fill_(9.0)
        polluted = IdentityAnchorProvider(model, num_train_ids=3, frozen=True).anchors()
        with torch.no_grad():
            model.prompt_bank.camera_prompts.zero_()
        clean = IdentityAnchorProvider(model, num_train_ids=3, frozen=True).anchors()

        self.assertTrue(torch.equal(polluted, clean))

    def test_anchors_are_computed_without_grad_even_for_trainable_prompts(self):
        model, _ = _anchor_model(num_train_ids=2)
        self.assertTrue(model.prompt_bank.identity_prompts.requires_grad)

        anchors = IdentityAnchorProvider(model, num_train_ids=2, frozen=False).anchors()

        self.assertFalse(anchors.requires_grad)
        self.assertIsNone(anchors.grad_fn)

    def test_provider_validates_positive_sizes(self):
        model, _ = _anchor_model(num_train_ids=2)

        with self.assertRaises(ValueError):
            IdentityAnchorProvider(model, num_train_ids=0, frozen=True)
        with self.assertRaises(ValueError):
            IdentityAnchorProvider(model, num_train_ids=2, frozen=True, chunk_size=0)


def _anchor_model(num_train_ids: int) -> tuple[T2CReIDModel, CountingPromptMeanEncoder]:
    bank = PromptBank(
        PromptConfig(num_cameras=2, num_train_ids=num_train_ids, context_length=1, embedding_dim=2)
    )
    with torch.no_grad():
        bank.global_prompt.fill_(0.5)
        bank.camera_prompts.zero_()
        for person_id in range(num_train_ids):
            bank.identity_prompts[person_id] = torch.tensor([[1.0 + person_id, 2.0 - person_id]])
    encoder = CountingPromptMeanEncoder()
    return T2CReIDModel(IdentityEncoder(), encoder, bank, beta=0.0), encoder


if __name__ == "__main__":
    unittest.main()
