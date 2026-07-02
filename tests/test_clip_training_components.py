import unittest

from PIL import Image
import torch

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


if __name__ == "__main__":
    unittest.main()
