import unittest

from argparse import Namespace

from PIL import Image
import torch
import torch.nn.functional as F

from t2c_clip.clip_backbone import (
    PromptTextEncoder,
    TransformersCLIPImageEncoder,
    TransformersCLIPTextEncoder,
    clip_projection_dim,
    clip_text_hidden_dim,
)
from t2c_clip.transforms import CLIPImageTransform, CLIPTrainImageTransform
from tests._clip_fakes import FakeCLIP, FakeImageProcessor


class CLIPTrainingComponentsTest(unittest.TestCase):
    def test_clip_image_transform_resizes_whole_image_without_center_crop(self):
        transform = CLIPImageTransform(FakeImageProcessor())
        image = Image.new("RGB", (64, 128), "green")
        image.paste(Image.new("RGB", (64, 32), "red"), (0, 0))      # head quarter
        image.paste(Image.new("RGB", (64, 32), "blue"), (0, 96))    # feet quarter

        output = transform(image)

        self.assertEqual(tuple(output.shape), (3, 256, 128))
        # Identity normalization: channels stay in [0, 1]. A center crop would
        # discard the red head and blue feet stripes entirely.
        self.assertGreater(float(output[0, 4, :].mean()), 0.8)
        self.assertLess(float(output[2, 4, :].mean()), 0.2)
        self.assertGreater(float(output[2, -5, :].mean()), 0.8)
        self.assertLess(float(output[0, -5, :].mean()), 0.2)

    def test_clip_image_transform_normalizes_with_processor_mean_std(self):
        processor = FakeImageProcessor()
        processor.image_mean = [0.5, 0.5, 0.5]
        processor.image_std = [0.5, 0.5, 0.5]
        transform = CLIPImageTransform(processor)

        output = transform(Image.new("RGB", (64, 128), (128, 128, 128)))

        expected = torch.full_like(output, (128.0 / 255.0 - 0.5) / 0.5)
        self.assertTrue(torch.allclose(output, expected, atol=1e-4))

    def test_clip_image_transform_requires_processor_normalization_stats(self):
        class StatlessProcessor:
            pass

        with self.assertRaises(ValueError):
            CLIPImageTransform(StatlessProcessor())

    def test_clip_train_image_transform_erases_after_normalize_and_keeps_shape(self):
        transform = CLIPTrainImageTransform(
            FakeImageProcessor(),
            color_jitter=(0.0, 0.0, 0.0, 0.0),
            erase_prob=1.0,
            erase_scale=(0.25, 0.25),
            erase_ratio=(1.0, 1.0),
        )

        output = transform(Image.new("RGB", (64, 128), (255, 255, 255)))

        self.assertEqual(tuple(output.shape), (3, 256, 128))
        self.assertTrue(bool(torch.any(output == 0.0)))
        self.assertTrue(bool(torch.any(output == 1.0)))

    def test_clip_train_image_transform_pads_and_random_crops_to_target_size(self):
        transform = CLIPTrainImageTransform(
            FakeImageProcessor(),
            flip_prob=0.0,
            color_jitter=(0.0, 0.0, 0.0, 0.0),
            erase_prob=0.0,
            crop_padding=10,
        )
        image = Image.new("RGB", (64, 128), (255, 255, 255))

        torch.manual_seed(0)
        output = transform(image)

        self.assertEqual(tuple(output.shape), (3, 256, 128))
        # The pad-then-crop canvas is zero-filled: a random crop over the
        # padded image carries some black border into the white image.
        self.assertTrue(bool(torch.any(output == 0.0)))

    def test_clip_train_image_transform_zero_crop_padding_matches_eval_resize(self):
        transform = CLIPTrainImageTransform(
            FakeImageProcessor(),
            flip_prob=0.0,
            color_jitter=(0.0, 0.0, 0.0, 0.0),
            erase_prob=0.0,
            crop_padding=0,
        )
        baseline = CLIPImageTransform(FakeImageProcessor())
        image = Image.new("RGB", (64, 128), "green")
        image.paste(Image.new("RGB", (64, 32), "red"), (0, 0))

        self.assertTrue(torch.equal(transform(image), baseline(image)))

    def test_text_encoder_rejects_sequence_longer_than_position_embeddings(self):
        # tiny config has max_position_embeddings=32; SOT + slots + EOS must fit.
        clip = _tiny_clip_model()

        with self.assertRaisesRegex(ValueError, "position"):
            TransformersCLIPTextEncoder(
                clip,
                context_length=31,
                sot_token_id=997,
                eos_token_id=998,
            )

    def test_transformers_clip_image_encoder_calls_get_image_features(self):
        clip = FakeCLIP(projection_dim=2)
        encoder = TransformersCLIPImageEncoder(clip)

        output = encoder(torch.ones(2, 3, 2, 2))

        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(output.device.type, "cpu")

    def test_transformers_clip_image_encoder_interpolates_position_encoding(self):
        # Non-square 256x128 ReID inputs require CLIP position-embedding
        # interpolation; without it the real CLIPVisionEmbeddings raises.
        clip = FakeCLIP(projection_dim=2)
        encoder = TransformersCLIPImageEncoder(clip)

        encoder(torch.ones(2, 3, 256, 128))

        self.assertTrue(clip.last_interpolate_pos_encoding)

    def test_prompt_text_encoder_projects_mean_prompt(self):
        encoder = PromptTextEncoder(prompt_embedding_dim=3, output_dim=2)
        with torch.no_grad():
            weight = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            encoder.projection.weight.copy_(weight)
            encoder.projection.bias.zero_()

        output = encoder(torch.tensor([[[1.0, 2.0, 9.0], [3.0, 4.0, 9.0]]]))

        self.assertTrue(torch.equal(output, torch.tensor([[2.0, 3.0]])))

    def test_clip_projection_dim_reads_positive_config_value(self):
        self.assertEqual(clip_projection_dim(FakeCLIP(projection_dim=7)), 7)

    def test_clip_text_hidden_dim_reads_token_embedding_width(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=2)
        self.assertEqual(clip_text_hidden_dim(clip), 8)

    def test_text_encoder_requires_3d_prompt_embeddings(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=2)
        encoder = TransformersCLIPTextEncoder(clip, context_length=2, sot_token_id=49406, eos_token_id=49407)

        with self.assertRaises(ValueError):
            encoder(torch.zeros(2, 8))

    def test_text_encoder_rejects_mismatched_context_length(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=2)
        encoder = TransformersCLIPTextEncoder(clip, context_length=3, sot_token_id=49406, eos_token_id=49407)

        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 2, 8))

    def test_text_encoder_returns_normalized_projection_space_features(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=4)
        encoder = TransformersCLIPTextEncoder(clip, context_length=3, sot_token_id=49406, eos_token_id=49407)
        prompt_embeddings = torch.randn(4, 3, 8)

        output = encoder(prompt_embeddings)

        self.assertEqual(output.shape, (4, 4))
        norms = output.norm(dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones(4), atol=1e-5))

    def test_text_encoder_requires_positive_context_length(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=2)
        with self.assertRaises(ValueError):
            TransformersCLIPTextEncoder(clip, context_length=0, sot_token_id=49406, eos_token_id=49407)

    def test_text_encoder_default_template_pools_eos_directly_after_prompt_block(self):
        # Backward compatibility: empty prefix/suffix keep the old
        # [SOT] + slots + [EOS] sequence with EOS at context_length + 1.
        clip = FakeCLIP(hidden_size=8, projection_dim=4)
        encoder = TransformersCLIPTextEncoder(clip, context_length=3, sot_token_id=49406, eos_token_id=49407)

        output = encoder(torch.randn(2, 3, 8))

        self.assertEqual(encoder.prefix_token_ids, ())
        self.assertEqual(encoder.suffix_token_ids, ())
        expected = _expected_fake_eos_feature(clip, eos_token_id=49407, eos_index=4, batch_size=2)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_text_encoder_template_shifts_eos_pooling_after_suffix(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=4)
        encoder = TransformersCLIPTextEncoder(
            clip,
            context_length=2,
            sot_token_id=49406,
            eos_token_id=49407,
            prefix_token_ids=(11, 12, 13),
            suffix_token_ids=(21, 22),
        )

        output = encoder(torch.randn(3, 2, 8))

        # EOS index = 1 (SOT) + 3 (prefix) + 2 (slots) + 2 (suffix) = 8.
        expected = _expected_fake_eos_feature(clip, eos_token_id=49407, eos_index=8, batch_size=3)
        self.assertTrue(torch.allclose(output, expected, atol=1e-6))

    def test_text_encoder_template_builds_sot_prefix_slots_suffix_eos_sequence(self):
        clip = FakeCLIP(hidden_size=8, projection_dim=4)
        encoder = TransformersCLIPTextEncoder(
            clip,
            context_length=2,
            sot_token_id=49406,
            eos_token_id=49407,
            prefix_token_ids=(11, 12),
            suffix_token_ids=(21,),
        )
        prompt_embeddings = torch.randn(1, 2, 8)

        encoder(prompt_embeddings)

        embeddings = clip.text_model.embeddings
        template_ids = torch.tensor([[49406, 11, 12, 0, 0, 21, 49407]])
        expected_tokens = embeddings.token_embedding(template_ids).clone()
        expected_tokens[:, 3:5, :] = prompt_embeddings
        expected_positions = embeddings.position_embedding(torch.arange(7).unsqueeze(0))
        received = clip.text_model.encoder.last_inputs_embeds
        self.assertEqual(tuple(received.shape), (1, 7, 8))
        self.assertTrue(torch.allclose(received, expected_tokens + expected_positions, atol=1e-6))

    def test_text_encoder_with_template_matches_official_clip_text_forward(self):
        # The custom inputs_embeds path must reproduce the official HF
        # CLIPTextModel forward when the slot positions hold the token
        # embeddings of concrete token ids.
        from transformers import CLIPConfig, CLIPModel

        torch.manual_seed(0)
        config = CLIPConfig(
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "max_position_embeddings": 32,
                "vocab_size": 1000,
                "bos_token_id": 997,
                "eos_token_id": 998,
                "pad_token_id": 0,
            },
            vision_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "image_size": 32,
                "patch_size": 16,
            },
            projection_dim=16,
        )
        clip = CLIPModel(config).eval()
        prefix = (10, 11, 12)
        suffix = (20, 21)
        slot_ids = (5, 6, 7, 8)
        encoder = TransformersCLIPTextEncoder(
            clip,
            context_length=len(slot_ids),
            sot_token_id=997,
            eos_token_id=998,
            prefix_token_ids=prefix,
            suffix_token_ids=suffix,
        )

        with torch.no_grad():
            prompt_embeddings = clip.text_model.embeddings.token_embedding(torch.tensor([slot_ids]))
            output = encoder(prompt_embeddings)
            input_ids = torch.tensor([[997, *prefix, *slot_ids, *suffix, 998]])
            pooled = clip.text_model(input_ids=input_ids).pooler_output
            official = F.normalize(clip.text_projection(pooled), p=2.0, dim=-1, eps=1e-12)

        self.assertTrue(torch.equal(output, official))


class SIECameraEmbeddingTest(unittest.TestCase):
    def test_sie_disabled_creates_no_embedding_and_accepts_images_only(self):
        clip = FakeCLIP(projection_dim=2)
        encoder = TransformersCLIPImageEncoder(clip, num_cameras=3, sie_coe=0.0)

        output = encoder(torch.ones(2, 3, 2, 2))

        self.assertIsNone(encoder.sie_embedding)
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(clip.last_interpolate_pos_encoding)

    def test_sie_disabled_when_num_cameras_is_none(self):
        encoder = TransformersCLIPImageEncoder(FakeCLIP(projection_dim=2), sie_coe=0.5)

        self.assertIsNone(encoder.sie_embedding)

    def test_sie_disabled_accepts_unused_camera_ids(self):
        clip = FakeCLIP(projection_dim=2)
        encoder = TransformersCLIPImageEncoder(clip)

        output = encoder(torch.ones(2, 3, 2, 2), camera_ids=torch.tensor([0, 1]))

        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(clip.image_feature_calls, 1)

    def test_sie_manual_forward_with_zero_embedding_matches_get_image_features(self):
        # The manual embeddings -> pre_layrnorm -> encoder -> post_layernorm ->
        # visual_projection path must replicate the official HF forward exactly
        # (bit-for-bit) on a non-square ReID-shaped input when the SIE term is zero.
        torch.manual_seed(0)
        clip = _tiny_clip_model().eval()
        encoder = TransformersCLIPImageEncoder(clip, num_cameras=2, sie_coe=0.5)
        with torch.no_grad():
            encoder.sie_embedding.weight.zero_()
        images = torch.randn(2, 3, 32, 16)

        with torch.no_grad():
            output = encoder(images, camera_ids=torch.tensor([0, 1]))
            # v5 get_image_features returns the vision outputs with pooler_output
            # replaced by the projected image features.
            official = clip.get_image_features(
                pixel_values=images, interpolate_pos_encoding=True
            ).pooler_output

        self.assertTrue(torch.equal(output, official))

    def test_sie_enabled_distinguishes_cameras(self):
        torch.manual_seed(0)
        clip = _tiny_clip_model().eval()
        encoder = TransformersCLIPImageEncoder(clip, num_cameras=3, sie_coe=1.0)
        image = torch.randn(1, 3, 32, 16)
        images = torch.cat([image, image], dim=0)

        with torch.no_grad():
            different = encoder(images, camera_ids=torch.tensor([0, 1]))
            same = encoder(images, camera_ids=torch.tensor([1, 1]))

        self.assertFalse(torch.allclose(different[0], different[1]))
        self.assertTrue(torch.equal(same[0], same[1]))

    def test_sie_coe_scales_the_camera_injection(self):
        # A dropped `sie_coe *` factor (effectively 1.0) must not stay green:
        # the same encoder with the same embedding weights but a different
        # coefficient must produce a different output.
        torch.manual_seed(0)
        clip = _tiny_clip_model().eval()
        encoder = TransformersCLIPImageEncoder(clip, num_cameras=2, sie_coe=0.5)
        images = torch.randn(1, 3, 32, 16)
        camera_ids = torch.tensor([0])

        with torch.no_grad():
            half = encoder(images, camera_ids=camera_ids)
            encoder.sie_coe = 2.0
            double = encoder(images, camera_ids=camera_ids)

        self.assertFalse(torch.allclose(half, double))

    def test_sie_enabled_requires_camera_ids(self):
        encoder = TransformersCLIPImageEncoder(_tiny_clip_model(), num_cameras=2, sie_coe=0.5)

        with self.assertRaisesRegex(ValueError, "camera_ids"):
            encoder(torch.randn(1, 3, 32, 16))

    def test_sie_enabled_rejects_out_of_range_camera_ids(self):
        encoder = TransformersCLIPImageEncoder(_tiny_clip_model(), num_cameras=2, sie_coe=0.5)

        with self.assertRaisesRegex(ValueError, "camera_ids"):
            encoder(torch.randn(1, 3, 32, 16), camera_ids=torch.tensor([2]))

    def test_sie_enabled_rejects_non_long_camera_ids(self):
        encoder = TransformersCLIPImageEncoder(_tiny_clip_model(), num_cameras=2, sie_coe=0.5)

        with self.assertRaisesRegex(ValueError, "camera_ids"):
            encoder(torch.randn(1, 3, 32, 16), camera_ids=torch.tensor([0.0]))

    def test_sie_enabled_requires_vision_hidden_dim(self):
        clip = FakeCLIP(projection_dim=4)
        clip.config = Namespace(projection_dim=4)  # no vision_config

        with self.assertRaisesRegex(ValueError, "vision_config"):
            TransformersCLIPImageEncoder(clip, num_cameras=2, sie_coe=0.5)


def _tiny_clip_model():
    from transformers import CLIPConfig, CLIPModel

    config = CLIPConfig(
        text_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "max_position_embeddings": 32,
            "vocab_size": 1000,
            "bos_token_id": 997,
            "eos_token_id": 998,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "image_size": 32,
            "patch_size": 16,
        },
        projection_dim=16,
    )
    return CLIPModel(config)


def _expected_fake_eos_feature(
    clip: FakeCLIP, eos_token_id: int, eos_index: int, batch_size: int
) -> torch.Tensor:
    """Expected encoder output for FakeCLIP, whose transformer is the identity.

    The pooled EOS hidden state is exactly
    ``token_embedding(EOS) + position_embedding(eos_index)``.
    """
    embeddings = clip.text_model.embeddings
    with torch.no_grad():
        hidden = embeddings.token_embedding(torch.tensor([eos_token_id]))
        hidden = hidden + embeddings.position_embedding(torch.tensor([eos_index]))
        projected = clip.text_projection(hidden)
    return F.normalize(projected, p=2.0, dim=-1, eps=1e-12).expand(batch_size, -1)


if __name__ == "__main__":
    unittest.main()
