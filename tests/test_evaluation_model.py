import unittest

import numpy as np
import torch

from t2c_reid.evaluation import RerankConfig, evaluate_reid, evaluate_reid_with_rerank
from t2c_reid.model import T2CReIDModel
from t2c_reid.prompts import PromptBank, PromptConfig
from t2c_reid.retrieval import IMAGE_ONLY_RETRIEVAL


class IdentityEncoder(torch.nn.Module):
    def forward(self, inputs, camera_ids=None):
        return inputs


class PromptMeanEncoder(torch.nn.Module):
    def forward(self, prompts):
        return prompts.mean(dim=1)


class EvaluationModelTest(unittest.TestCase):
    def test_rerank_lambda_term_uses_transposed_normalized_distance(self):
        # Zhong et al. build the original-distance term from the TRANSPOSED
        # column-normalized matrix (np.transpose(dist / np.max(dist, axis=0))).
        # With lambda=1 the rerank distance must equal exactly that slice.
        from t2c_reid.evaluation import _pairwise_distance, _rerank_distance
        from t2c_reid.features import l2_normalize

        torch.manual_seed(0)
        query = torch.randn(3, 4)
        gallery = torch.randn(5, 4)
        config = RerankConfig(k1=3, k2=2, lambda_value=1.0)

        features = l2_normalize(torch.cat([query, gallery], dim=0))
        original = _pairwise_distance(features)
        original = original / original.max(dim=0, keepdim=True).values.clamp_min(1e-12)
        expected = original.T.contiguous()[: query.shape[0], query.shape[0]:]

        result = _rerank_distance(query, gallery, config)

        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

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

    def test_rust_backend_matches_python_across_chunks_and_ties(self):
        query = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        gallery = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        )
        metadata = {
            "query_ids": [1, 2, 3],
            "gallery_ids": [9, 1, 2, 3],
            "query_cams": [0, 0, 0],
            "gallery_cams": [1, 1, 1, 1],
        }

        python_metrics = evaluate_reid(
            query, gallery, **metadata, ranks=(1, 2, 4), backend="python"
        )
        rust_metrics = evaluate_reid(
            query,
            gallery,
            **metadata,
            ranks=(1, 2, 4),
            backend="rust",
            query_chunk_size=2,
        )

        self.assertEqual(rust_metrics, python_metrics)

    def test_rust_backend_accepts_non_contiguous_features(self):
        torch.manual_seed(2)
        query = torch.randn(4, 6)[:, ::2]
        gallery = torch.randn(7, 6)[:, ::2]
        metadata = {
            "query_ids": [0, 1, 2, 3],
            "gallery_ids": [0, 1, 2, 3, 8, 9, 10],
            "query_cams": [0, 0, 0, 0],
            "gallery_cams": [1, 1, 1, 1, 2, 2, 2],
        }

        python_metrics = evaluate_reid(query, gallery, **metadata, backend="python")
        rust_metrics = evaluate_reid(
            query, gallery, **metadata, backend="rust", query_chunk_size=3
        )

        self.assertEqual(rust_metrics, python_metrics)

    def test_rust_sparse_rerank_matches_dense_reference(self):
        torch.manual_seed(3)
        query = torch.randn(4, 8)
        gallery = torch.randn(7, 8)
        metadata = {
            "query_ids": [0, 1, 2, 3],
            "gallery_ids": [0, 1, 2, 3, 0, 1, 9],
            "query_cams": [0, 0, 0, 0],
            "gallery_cams": [1, 1, 1, 1, 2, 2, 2],
        }
        for config in (
            RerankConfig(k1=1, k2=1, lambda_value=0.0),
            RerankConfig(k1=1, k2=3, lambda_value=0.0),
            RerankConfig(k1=3, k2=2, lambda_value=0.3),
            RerankConfig(k1=5, k2=3, lambda_value=1.0),
        ):
            with self.subTest(config=config):
                dense = evaluate_reid_with_rerank(
                    query, gallery, **metadata, ranks=(1, 5), config=config, backend="python"
                )
                sparse = evaluate_reid_with_rerank(
                    query,
                    gallery,
                    **metadata,
                    ranks=(1, 5),
                    config=config,
                    backend="rust",
                    query_chunk_size=3,
                )
                self.assertAlmostEqual(sparse.map, dense.map, places=12)
                self.assertEqual(sparse.cmc, dense.cmc)

    def test_rerank_ties_use_deterministic_combined_index_order(self):
        query = torch.ones(2, 3)
        gallery = torch.ones(4, 3)
        metadata = {
            "query_ids": [0, 1],
            "gallery_ids": [9, 0, 1, 8],
            "query_cams": [0, 0],
            "gallery_cams": [1, 1, 1, 1],
        }
        config = RerankConfig(k1=1, k2=1, lambda_value=0.3)

        dense = evaluate_reid_with_rerank(
            query, gallery, **metadata, ranks=(1, 4), config=config, backend="python"
        )
        sparse = evaluate_reid_with_rerank(
            query, gallery, **metadata, ranks=(1, 4), config=config, backend="rust"
        )

        self.assertEqual(sparse, dense)

    def test_sparse_rerank_regression_for_k1_one_with_query_expansion(self):
        torch.manual_seed(2)
        query = torch.randn(5, 11)
        gallery = torch.randn(9, 11)
        metadata = {
            "query_ids": [0, 1, 2, 3, 4],
            "gallery_ids": [0, 1, 2, 3, 4, 0, 1, 8, 9],
            "query_cams": [0] * 5,
            "gallery_cams": [1, 1, 1, 1, 1, 2, 2, 2, 2],
        }
        config = RerankConfig(k1=1, k2=3, lambda_value=0.0)

        dense = evaluate_reid_with_rerank(
            query, gallery, **metadata, ranks=(1, 3, 7), config=config, backend="python"
        )
        sparse = evaluate_reid_with_rerank(
            query,
            gallery,
            **metadata,
            ranks=(1, 3, 7),
            config=config,
            backend="rust",
            query_chunk_size=4,
        )

        self.assertEqual(sparse, dense)

    def test_native_primary_signed_zero_uses_gallery_index_tie_break(self):
        from t2c_reid.native import native_extension

        average_precision, cmc, count = native_extension.evaluate_scores(
            np.array([[-0.0, 0.0]], dtype=np.float32),
            np.array([1], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([1, 1], dtype=np.int64),
            [1],
        )

        self.assertEqual((average_precision, cmc, count), (1.0, [1], 1))

    def test_native_topk_uses_index_tie_break_and_reports_row_max(self):
        from t2c_reid.native import native_extension

        distances = np.array([[0.5, 0.1, 0.1, 0.8]], dtype=np.float32)
        indices, values, maxima = native_extension.select_topk_distances(distances, 3)

        self.assertEqual(indices.tolist(), [[1, 2, 0]])
        self.assertTrue(np.allclose(values, [[0.1, 0.1, 0.5]]))
        self.assertTrue(np.allclose(maxima, [0.8]))

    def test_evaluation_rejects_invalid_backend_chunk_and_rank(self):
        query = torch.ones(1, 2)
        gallery = torch.ones(1, 2)
        metadata = {
            "query_ids": [1],
            "gallery_ids": [1],
            "query_cams": [0],
            "gallery_cams": [1],
        }

        with self.assertRaisesRegex(ValueError, "unsupported evaluation backend"):
            evaluate_reid(query, gallery, **metadata, backend="unknown")
        with self.assertRaisesRegex(ValueError, "query_chunk_size"):
            evaluate_reid(query, gallery, **metadata, query_chunk_size=0)
        with self.assertRaisesRegex(ValueError, "ranks"):
            evaluate_reid(query, gallery, **metadata, ranks=(0,))

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
        model = T2CReIDModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)

        output = model.encode_retrieval(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5]])
        self.assertTrue(torch.allclose(output, expected))

    def test_encode_retrieval_image_only_returns_visual_feature(self):
        prompt_bank = PromptBank(PromptConfig(num_cameras=1, num_train_ids=1, context_length=1, embedding_dim=2))
        with torch.no_grad():
            prompt_bank.global_prompt.zero_()
            prompt_bank.camera_prompts[0] = torch.tensor([[0.0, 1.0]])
            prompt_bank.identity_prompts[0] = torch.tensor([[10.0, 0.0]])
        model = T2CReIDModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)

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
        model = T2CReIDModel(IdentityEncoder(), PromptMeanEncoder(), prompt_bank, beta=1.0)
        images = torch.tensor([[1.0, 0.0]])
        camera_ids = torch.tensor([0])
        person_ids = torch.tensor([0])

        outputs = model.forward_stage2(images, camera_ids, person_ids)
        inference = model.encode_retrieval(images, camera_ids)

        self.assertTrue(torch.allclose(outputs["retrieval"], inference))
        self.assertNotIn("text", outputs)


class CameraRecordingImageEncoder(torch.nn.Module):
    """Records the camera_ids keyword the model threads into the image encoder."""

    def __init__(self):
        super().__init__()
        self.received_camera_ids: list = []

    def forward(self, images, camera_ids=None):
        self.received_camera_ids.append(camera_ids)
        return images


class CameraIdThreadingTest(unittest.TestCase):
    def _model(self) -> tuple[T2CReIDModel, CameraRecordingImageEncoder]:
        prompt_bank = PromptBank(PromptConfig(num_cameras=2, num_train_ids=2, context_length=1, embedding_dim=2))
        encoder = CameraRecordingImageEncoder()
        return T2CReIDModel(encoder, PromptMeanEncoder(), prompt_bank, beta=0.5), encoder

    def test_encode_retrieval_passes_camera_ids_to_image_encoder(self):
        model, encoder = self._model()
        camera_ids = torch.tensor([0, 1])

        model.encode_retrieval(torch.eye(2), camera_ids)

        self.assertIs(encoder.received_camera_ids[-1], camera_ids)

    def test_forward_stage1_passes_camera_ids_to_image_encoder(self):
        model, encoder = self._model()
        camera_ids = torch.tensor([1, 0])

        model.forward_stage1(torch.eye(2), camera_ids, torch.tensor([0, 1]))

        self.assertIs(encoder.received_camera_ids[-1], camera_ids)

    def test_forward_stage2_passes_camera_ids_to_image_encoder(self):
        model, encoder = self._model()
        camera_ids = torch.tensor([1, 1])

        model.forward_stage2(torch.eye(2), camera_ids, torch.tensor([0, 1]))

        self.assertIs(encoder.received_camera_ids[-1], camera_ids)

    def test_encode_visual_without_camera_ids_calls_encoder_with_images_only(self):
        model, encoder = self._model()

        model.encode_visual(torch.eye(2))

        self.assertEqual(encoder.received_camera_ids, [None])
