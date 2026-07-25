import unittest

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import Siglip2Config, Siglip2Model, SiglipConfig, SiglipModel

from t2c_clip.siglip2_backbone import (
    TransformersSiglip2ImageEncoder,
    TransformersSiglip2TextEncoder,
    patchify_siglip2_images,
    siglip2_feature_dim,
    siglip2_max_num_patches,
    siglip2_patch_size,
    siglip2_text_hidden_dim,
    siglip2_uses_patchified_inputs,
    validate_siglip2_image_size,
)
from t2c_clip.transforms import Siglip2ImageTransform, Siglip2TrainImageTransform
from tests._siglip2_fakes import FakeSiglip2, FakeSiglip2ImageProcessor


class Siglip2TransformTest(unittest.TestCase):
    def test_eval_transform_preserves_full_person_and_uses_default_size(self):
        transform = Siglip2ImageTransform(FakeSiglip2ImageProcessor())
        image = Image.new("RGB", (64, 128), "green")
        image.paste(Image.new("RGB", (64, 32), "red"), (0, 0))
        image.paste(Image.new("RGB", (64, 32), "blue"), (0, 96))

        output = transform(image)

        self.assertEqual(tuple(output.shape), (3, 392, 196))
        # [0.5, 0.5] normalization maps a saturated channel to +1 and zero to -1.
        self.assertGreater(float(output[0, 5, :].mean()), 0.8)
        self.assertLess(float(output[2, 5, :].mean()), -0.8)
        self.assertGreater(float(output[2, -6, :].mean()), 0.8)

    def test_transform_uses_processor_normalization(self):
        processor = FakeSiglip2ImageProcessor()
        processor.image_mean = [0.25, 0.25, 0.25]
        processor.image_std = [0.5, 0.5, 0.5]
        transform = Siglip2ImageTransform(processor, image_size=(14, 14))

        output = transform(Image.new("RGB", (8, 8), (128, 128, 128)))

        expected = torch.full_like(output, (128.0 / 255.0 - 0.25) / 0.5)
        self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    def test_transform_requires_processor_stats(self):
        with self.assertRaises(ValueError):
            Siglip2ImageTransform(object())

    def test_train_transform_keeps_shape_and_random_erases(self):
        transform = Siglip2TrainImageTransform(
            FakeSiglip2ImageProcessor(),
            image_size=(28, 14),
            color_jitter=(0.0, 0.0, 0.0, 0.0),
            crop_padding=0,
            erase_prob=1.0,
            erase_scale=(0.25, 0.25),
            erase_ratio=(1.0, 1.0),
        )

        output = transform(Image.new("RGB", (14, 28), "white"))

        self.assertEqual(tuple(output.shape), (3, 28, 14))
        self.assertTrue(bool(torch.any(output == 0.0)))


class Siglip2PatchifyTest(unittest.TestCase):
    def test_patchify_uses_h_w_c_order_within_each_patch(self):
        image = torch.tensor(
            [[
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [[11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
            ]]
        )

        result = patchify_siglip2_images(image, patch_size=2, max_num_patches=4)

        expected = torch.tensor(
            [[[1.0, 11.0, 2.0, 12.0, 5.0, 15.0, 6.0, 16.0],
              [3.0, 13.0, 4.0, 14.0, 7.0, 17.0, 8.0, 18.0]]]
        )
        self.assertTrue(torch.equal(result.pixel_values, expected))
        self.assertTrue(torch.equal(result.pixel_attention_mask, torch.ones(1, 2, dtype=torch.long)))
        self.assertTrue(torch.equal(result.spatial_shapes, torch.tensor([[1, 2]])))

    def test_default_reid_shape_produces_392_patch14_tokens(self):
        result = patchify_siglip2_images(
            torch.zeros(1, 3, 392, 196),
            patch_size=14,
            max_num_patches=729,
        )

        self.assertEqual(tuple(result.pixel_values.shape), (1, 392, 588))
        self.assertTrue(torch.equal(result.spatial_shapes, torch.tensor([[28, 14]])))

    def test_patchify_rejects_non_divisible_size(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            patchify_siglip2_images(
                torch.zeros(1, 3, 15, 14), patch_size=14, max_num_patches=16
            )

    def test_patchify_rejects_patch_budget_overflow(self):
        with self.assertRaisesRegex(ValueError, "exceeding"):
            patchify_siglip2_images(
                torch.zeros(1, 3, 28, 28), patch_size=14, max_num_patches=3
            )

    def test_model_image_size_validation(self):
        model = FakeSiglip2(num_patches=400)
        self.assertEqual(validate_siglip2_image_size((392, 196), model), 392)
        with self.assertRaisesRegex(ValueError, "divisible"):
            validate_siglip2_image_size((391, 196), model)

    def test_non_square_position_budget_is_rejected(self):
        model = FakeSiglip2(num_patches=399)
        with self.assertRaisesRegex(ValueError, "perfect square"):
            siglip2_max_num_patches(model)


class Siglip2ImageEncoderTest(unittest.TestCase):
    def test_no_sie_matches_official_vision_forward(self):
        torch.manual_seed(0)
        model = _tiny_siglip2().eval()
        encoder = TransformersSiglip2ImageEncoder(model).eval()
        images = torch.randn(2, 3, 4, 4)
        inputs = patchify_siglip2_images(images, patch_size=2, max_num_patches=16)

        with torch.no_grad():
            actual = encoder(images)
            official = model.vision_model(
                pixel_values=inputs.pixel_values,
                pixel_attention_mask=inputs.pixel_attention_mask,
                spatial_shapes=inputs.spatial_shapes,
            ).pooler_output

        self.assertTrue(torch.equal(actual, official))

    def test_zero_sie_matches_official_vision_forward(self):
        torch.manual_seed(0)
        model = _tiny_siglip2().eval()
        encoder = TransformersSiglip2ImageEncoder(
            model, num_cameras=2, sie_coe=1.0
        ).eval()
        encoder.sie_embedding.weight.data.zero_()
        images = torch.randn(2, 3, 4, 4)
        cameras = torch.tensor([0, 1])
        inputs = patchify_siglip2_images(images, patch_size=2, max_num_patches=16)

        with torch.no_grad():
            actual = encoder(images, cameras)
            official = model.vision_model(
                pixel_values=inputs.pixel_values,
                pixel_attention_mask=inputs.pixel_attention_mask,
                spatial_shapes=inputs.spatial_shapes,
            ).pooler_output

        self.assertTrue(torch.equal(actual, official))

    def test_nonzero_sie_distinguishes_cameras(self):
        torch.manual_seed(0)
        encoder = TransformersSiglip2ImageEncoder(
            _tiny_siglip2().eval(), num_cameras=2, sie_coe=1.0
        ).eval()
        image = torch.randn(1, 3, 4, 4).repeat(2, 1, 1, 1)

        with torch.no_grad():
            output = encoder(image, torch.tensor([0, 1]))

        self.assertFalse(torch.allclose(output[0], output[1]))

    def test_sie_requires_valid_camera_ids(self):
        encoder = TransformersSiglip2ImageEncoder(
            _tiny_siglip2(), num_cameras=2, sie_coe=1.0
        )
        with self.assertRaisesRegex(ValueError, "camera_ids"):
            encoder(torch.randn(1, 3, 4, 4))
        with self.assertRaisesRegex(ValueError, "camera_ids"):
            encoder(torch.randn(1, 3, 4, 4), torch.tensor([2]))


class FixedSiglip2CheckpointAdapterTest(unittest.TestCase):
    def test_fixed_checkpoint_accepts_canonical_non_divisible_pretrain_size(self):
        model = _tiny_fixed_siglip2(image_size=384, patch_size=14)

        self.assertEqual(siglip2_max_num_patches(model), 729)
        self.assertEqual(validate_siglip2_image_size((384, 384), model), 729)
        self.assertEqual(validate_siglip2_image_size((392, 196), model), 392)

    def test_fixed_vision_adapter_matches_official_non_square_forward(self):
        torch.manual_seed(0)
        model = _tiny_fixed_siglip2().eval()
        encoder = TransformersSiglip2ImageEncoder(model).eval()
        images = torch.randn(2, 3, 8, 4)

        with torch.no_grad():
            actual = encoder(images)
            official = model.vision_model(
                pixel_values=images,
                interpolate_pos_encoding=True,
            ).pooler_output

        self.assertFalse(siglip2_uses_patchified_inputs(model))
        self.assertTrue(torch.equal(actual, official))

    def test_fixed_vision_zero_sie_matches_official_forward(self):
        torch.manual_seed(0)
        model = _tiny_fixed_siglip2().eval()
        encoder = TransformersSiglip2ImageEncoder(
            model, num_cameras=2, sie_coe=1.0
        ).eval()
        encoder.sie_embedding.weight.data.zero_()
        images = torch.randn(2, 3, 8, 4)

        with torch.no_grad():
            actual = encoder(images, torch.tensor([0, 1]))
            official = model.vision_model(
                pixel_values=images,
                interpolate_pos_encoding=True,
            ).pooler_output

        self.assertTrue(torch.equal(actual, official))

    def test_fixed_text_adapter_matches_official_right_padded_forward(self):
        torch.manual_seed(0)
        model = _tiny_fixed_siglip2().eval()
        slots = (10, 11)
        adapter = TransformersSiglip2TextEncoder(
            model,
            context_length=2,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            prefix_token_ids=(5, 6),
            suffix_token_ids=(7,),
            left_padding=False,
            include_bos_token=False,
            mask_padding=False,
        ).eval()
        input_ids = torch.tensor([[5, 6, *slots, 7, 1, 0, 0, 0, 0, 0, 0]])

        with torch.no_grad():
            prompt_embeddings = model.text_model.embeddings.token_embedding(
                torch.tensor([slots])
            )
            actual = adapter(prompt_embeddings)
            official = model.text_model(input_ids=input_ids).pooler_output
            official = F.normalize(official, p=2, dim=-1, eps=1e-12)

        self.assertTrue(torch.equal(actual, official))


class Siglip2TextEncoderTest(unittest.TestCase):
    def test_prompt_adapter_matches_official_left_padded_text_forward(self):
        torch.manual_seed(0)
        model = _tiny_siglip2().eval()
        prefix = (5, 6)
        suffix = (7,)
        slots = (10, 11)
        adapter = TransformersSiglip2TextEncoder(
            model,
            context_length=2,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            prefix_token_ids=prefix,
            suffix_token_ids=suffix,
        ).eval()
        padding = model.config.text_config.max_position_embeddings - 7
        input_ids = torch.tensor([[*[0] * padding, 2, *prefix, *slots, *suffix, 1]])
        attention_mask = torch.tensor([[*[0] * padding, *[1] * 7]])

        with torch.no_grad():
            prompt_embeddings = model.text_model.embeddings.token_embedding(
                torch.tensor([slots])
            )
            actual = adapter(prompt_embeddings)
            official = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).pooler_output
            official = F.normalize(official, p=2, dim=-1, eps=1e-12)

        self.assertTrue(torch.equal(actual, official))

    def test_prompt_sequence_is_left_padded_and_eos_is_last(self):
        model = FakeSiglip2(hidden_size=8, max_position_embeddings=12)
        adapter = TransformersSiglip2TextEncoder(
            model,
            context_length=2,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
            prefix_token_ids=(5, 6),
            suffix_token_ids=(7,),
        )
        prompts = torch.randn(1, 2, 8)

        adapter(prompts)

        embeddings = model.text_model.embeddings
        expected_ids = torch.tensor([[0, 0, 0, 0, 0, 2, 5, 6, 0, 0, 7, 1]])
        expected_tokens = embeddings.token_embedding(expected_ids).clone()
        expected_tokens[:, 8:10] = prompts
        positions = embeddings.position_embedding(torch.arange(12).unsqueeze(0))
        received = model.text_model.encoder.last_inputs_embeds
        self.assertTrue(torch.allclose(received, expected_tokens + positions))

    def test_text_features_are_normalized_and_in_shared_dimension(self):
        model = FakeSiglip2(hidden_size=8, feature_dim=4)
        adapter = TransformersSiglip2TextEncoder(
            model, context_length=2, bos_token_id=2, eos_token_id=1, pad_token_id=0
        )

        output = adapter(torch.randn(3, 2, 8))

        self.assertEqual(tuple(output.shape), (3, 4))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(3), atol=1e-6))

    def test_text_adapter_validates_prompt_shape_and_length(self):
        model = FakeSiglip2(hidden_size=8, max_position_embeddings=8)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            TransformersSiglip2TextEncoder(
                model,
                context_length=7,
                bos_token_id=2,
                eos_token_id=1,
                pad_token_id=0,
            )
        adapter = TransformersSiglip2TextEncoder(
            model,
            context_length=2,
            bos_token_id=2,
            eos_token_id=1,
            pad_token_id=0,
        )
        with self.assertRaises(ValueError):
            adapter(torch.randn(2, 8))
        with self.assertRaisesRegex(ValueError, "hidden dim"):
            adapter(torch.randn(2, 2, 7))


class Siglip2ConfigBoundaryTest(unittest.TestCase):
    def test_feature_and_hidden_dimensions(self):
        model = FakeSiglip2(hidden_size=8, feature_dim=4)
        self.assertEqual(siglip2_text_hidden_dim(model), 8)
        self.assertEqual(siglip2_feature_dim(model), 4)
        self.assertEqual(siglip2_patch_size(model), 14)

    def test_mismatched_text_and_vision_features_fail(self):
        model = FakeSiglip2(hidden_size=8, feature_dim=4)
        model.config.text_config.projection_size = 3
        with self.assertRaisesRegex(ValueError, "must match"):
            siglip2_feature_dim(model)


def _tiny_fixed_siglip2(
    image_size: int = 8,
    patch_size: int = 2,
) -> SiglipModel:
    config = SiglipConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 12,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "projection_size": 8,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": image_size,
            "patch_size": patch_size,
        },
    )
    return SiglipModel(config)


def _tiny_siglip2() -> Siglip2Model:
    config = Siglip2Config(
        text_config={
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "max_position_embeddings": 12,
            "pad_token_id": 0,
            "bos_token_id": 2,
            "eos_token_id": 1,
            "projection_size": 8,
        },
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 2,
            "num_patches": 16,
        },
    )
    return Siglip2Model(config)


if __name__ == "__main__":
    unittest.main()
