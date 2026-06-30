import unittest

import torch

from t2c_clip.evaluation import RerankConfig, evaluate_reid, evaluate_reid_with_rerank
from t2c_clip.model import T2CClipModel
from t2c_clip.prompts import PromptBank, PromptConfig
from t2c_clip.retrieval import IMAGE_ONLY_RETRIEVAL


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class PromptMeanEncoder(torch.nn.Module):
    def forward(self, prompts):
        return prompts.mean(dim=1)


class EvaluationModelTest(unittest.TestCase):
    def test_evaluate_reid_excludes_same_camera_matches(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = evaluate_reid(
            query,
            gallery,
            query_ids=[1],
            gallery_ids=[1, 2],
            query_cams=[1],
            gallery_cams=[1, 2],
            ranks=(1,),
        )
        self.assertEqual(metrics.map, 0.0)
        self.assertEqual(metrics.cmc[1], 0.0)

    def test_evaluate_reid_reports_rank1_and_map(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        metrics = evaluate_reid(
            query,
            gallery,
            query_ids=[1],
            gallery_ids=[2, 1],
            query_cams=[1],
            gallery_cams=[2, 2],
            ranks=(1,),
        )
        self.assertEqual(metrics.map, 0.5)
        self.assertEqual(metrics.cmc[1], 0.0)

    def test_rerank_metric_is_not_the_primary_map(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
        metrics = evaluate_reid(
            query,
            gallery,
            query_ids=[1],
            gallery_ids=[1, 2],
            query_cams=[1],
            gallery_cams=[2, 2],
            ranks=(1,),
        )

        self.assertEqual(metrics.map, 1.0)
        self.assertEqual(metrics.extras, {})

    def test_evaluate_reid_with_rerank_reports_metrics(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
        metrics = evaluate_reid_with_rerank(
            query,
            gallery,
            query_ids=[1],
            gallery_ids=[1, 2],
            query_cams=[1],
            gallery_cams=[2, 2],
            ranks=(1,),
            config=RerankConfig(k1=1, k2=1),
        )

        self.assertGreaterEqual(metrics.map, 0.0)
        self.assertLessEqual(metrics.map, 1.0)
        self.assertIn(1, metrics.cmc)

    def test_evaluate_reid_with_rerank_requires_keyword_metadata(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[0.9, 0.1], [0.0, 1.0]])

        with self.assertRaises(TypeError):
            evaluate_reid_with_rerank(query, gallery, [1], [1, 2], [1], [2, 2])

    def test_evaluate_reid_requires_keyword_metadata(self):
        query = torch.tensor([[1.0, 0.0]])
        gallery = torch.tensor([[0.9, 0.1], [0.0, 1.0]])

        with self.assertRaises(TypeError):
            evaluate_reid(query, gallery, [1], [1, 2], [1], [2, 2])

    def test_evaluate_reid_rejects_empty_query_or_gallery(self):
        with self.assertRaisesRegex(ValueError, "query_features must contain at least one row"):
            evaluate_reid(
                torch.empty(0, 2),
                torch.empty(1, 2),
                query_ids=[],
                gallery_ids=[1],
                query_cams=[],
                gallery_cams=[1],
            )

        with self.assertRaisesRegex(ValueError, "gallery_features must contain at least one row"):
            evaluate_reid(
                torch.empty(1, 2),
                torch.empty(0, 2),
                query_ids=[1],
                gallery_ids=[],
                query_cams=[1],
                gallery_cams=[],
            )

    def test_model_inference_uses_global_and_camera_prompts(self):
        prompt_bank = PromptBank(PromptConfig(num_cameras=1, num_train_ids=1, context_length=1, embedding_dim=2))
        with torch.no_grad():
            prompt_bank.global_prompt.zero_()
            prompt_bank.camera_prompts[0] = torch.tensor([[0.0, 1.0]])
            prompt_bank.identity_prompts[0] = torch.tensor([[10.0, 0.0]])
        model = T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)

        output = model.encode_retrieval(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
        self.assertTrue(torch.allclose(output, expected))

    def test_encode_retrieval_image_only_returns_visual_feature(self):
        prompt_bank = PromptBank(PromptConfig(num_cameras=1, num_train_ids=1, context_length=1, embedding_dim=2))
        with torch.no_grad():
            prompt_bank.global_prompt.zero_()
            prompt_bank.camera_prompts[0] = torch.tensor([[0.0, 1.0]])
            prompt_bank.identity_prompts[0] = torch.tensor([[10.0, 0.0]])
        model = T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)

        output = model.encode_retrieval(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([0]),
            retrieval_mode=IMAGE_ONLY_RETRIEVAL,
        )

        expected = torch.tensor([[1.0, 0.0]])
        self.assertTrue(torch.allclose(output, expected))

    def test_stage2_reid_feature_matches_inference_feature_without_identity_prompt(self):
        prompt_bank = PromptBank(PromptConfig(num_cameras=1, num_train_ids=1, context_length=1, embedding_dim=2))
        with torch.no_grad():
            prompt_bank.global_prompt.zero_()
            prompt_bank.camera_prompts[0] = torch.tensor([[0.0, 1.0]])
            prompt_bank.identity_prompts[0] = torch.tensor([[10.0, 0.0]])
        model = T2CClipModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)
        images = torch.tensor([[1.0, 0.0]])
        camera_ids = torch.tensor([0])
        person_ids = torch.tensor([0])

        outputs = model.forward_stage2(images, camera_ids, person_ids)
        inference = model.encode_retrieval(images, camera_ids)

        self.assertTrue(torch.allclose(outputs["retrieval"], inference))
        self.assertFalse(torch.allclose(outputs["text"], inference))
