# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing."""

from collections.abc import Mapping
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchFeature, Mistral3Config

from vllm.model_executor.models.lightonocr import LightOnOCRProcessingInfo
from vllm.model_executor.models.mistral3 import (
    Mistral3HFEncoderInfo,
    Mistral3MultiModalProcessor,
    Mistral3ProcessingInfo,
)
from vllm.model_executor.models.pixtral import PixtralHFEncoderInfo
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.transformers_utils.processors.pixtral import (
    MistralCommonImageProcessor,
    MistralCommonPixtralProcessor,
)

from ...utils import build_model_context

pytestmark = pytest.mark.skip_global_cleanup

# This repo ships both params.json (Mistral) and config.json (HF). Auto config
# selects PixtralForConditionalGeneration; force HF to exercise Mistral3.
_MODEL_CONFIG_KWARGS = {"config_format": "hf"}
_MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
_LIGHTON_MODEL_ID = "lightonai/LightOnOCR-1B-1025"


class _ProcessorContext:
    def __init__(
        self,
        tokenizer: object,
        hf_config: object | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.hf_config = hf_config or SimpleNamespace(image_token_index=2)
        self.processor_cls: type[object] | None = None
        self.processor_kwargs: dict[str, object] | None = None

    def get_tokenizer(self) -> object:
        return self.tokenizer

    def get_merged_mm_kwargs(
        self, kwargs: Mapping[str, object]
    ) -> Mapping[str, object]:
        return kwargs

    def get_hf_config(self, config_type: type[object]) -> object:
        return self.hf_config

    def get_hf_processor(self, processor_cls: type[object], **kwargs: object) -> object:
        self.processor_cls = processor_cls
        self.processor_kwargs = kwargs
        return SimpleNamespace(processor_cls=processor_cls, patch_size=1)


class _NativeTextTokenizer:
    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.kwargs = kwargs
        return {"input_ids": [[11]], "attention_mask": [[1]]}


class _NativeImageEncoder:
    special_ids = SimpleNamespace(img_break=1, img=2, img_end=3)

    def __call__(self, image_chunk: object) -> SimpleNamespace:
        return SimpleNamespace(image=torch.ones(3, 32, 48))

    def _image_to_num_tokens(self, image: Image.Image) -> tuple[int, int]:
        return 2, 1


def _mistral_tokenizer() -> MistralTokenizer:
    tokenizer = object.__new__(MistralTokenizer)
    tokenizer.transformers_tokenizer = _NativeTextTokenizer()
    tokenizer.instruct = SimpleNamespace(mm_encoder=_NativeImageEncoder())
    return tokenizer


def _native_pixtral_processor() -> MistralCommonPixtralProcessor:
    tokenizer = _mistral_tokenizer()
    return MistralCommonPixtralProcessor(
        tokenizer=tokenizer,
        image_processor=MistralCommonImageProcessor(tokenizer.instruct.mm_encoder),
    )


def test_mistral3_selects_native_processor_for_mistral_tokenizer() -> None:
    ctx = _ProcessorContext(_mistral_tokenizer())
    info = Mistral3ProcessingInfo(ctx)

    processor = info.get_hf_processor()

    assert isinstance(processor, MistralCommonPixtralProcessor)
    assert ctx.processor_cls is None


def test_mistral3_rejects_native_image_token_id_mismatch_at_setup() -> None:
    ctx = _ProcessorContext(
        _mistral_tokenizer(), hf_config=SimpleNamespace(image_token_index=99)
    )

    with pytest.raises(ValueError, match="Mistral3.*image token id"):
        Mistral3ProcessingInfo(ctx).get_hf_processor()


def test_mistral3_keeps_hf_processor_for_hf_tokenizer() -> None:
    ctx = _ProcessorContext(object())
    info = Mistral3ProcessingInfo(ctx)

    info.get_hf_processor(size={"longest_edge": 448})

    assert ctx.processor_kwargs == {"size": {"longest_edge": 448}}


def test_native_pixtral_processor_tokenizes_text_and_images() -> None:
    processor = _native_pixtral_processor()

    output = processor(
        text="plain text",
        images=[Image.new("RGB", (48, 32))],
        return_tensors="pt",
    )

    assert output["input_ids"] == [[11]]
    assert output["attention_mask"] == [[1]]
    assert output["images"][0].shape == (3, 32, 48)
    assert output["images"][0].dtype == torch.float32
    assert processor.tokenizer.kwargs["add_special_tokens"] is False


def test_mistral3_normalizes_native_images_to_pixel_values() -> None:
    native_processor = _native_pixtral_processor()
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )
    processed_data = BatchFeature({"images": [torch.ones(3, 32, 48)]})

    output = multimodal_processor._postprocess_hf_mm_data(
        {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
    )

    assert torch.equal(output["pixel_values"][0], torch.ones(3, 32, 48))
    assert "images" not in output


def test_mistral3_rejects_invalid_native_image_rank() -> None:
    native_processor = _native_pixtral_processor()
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )

    with pytest.raises(ValueError, match="Mistral3.*3-D tensor"):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32))]},
            {},
            BatchFeature({"images": [torch.ones(1, 3, 32, 48)]}),
        )


@pytest.mark.parametrize(
    ("images", "error_match"),
    [
        ([torch.ones(4, 32, 48)], "3 channels"),
        ([torch.ones(3, 32, 48, dtype=torch.int64)], "floating-point dtype"),
        ([torch.ones(3, 0, 48)], "positive spatial dimensions"),
    ],
)
def test_mistral3_rejects_invalid_native_image_contract(
    images: list[torch.Tensor],
    error_match: str,
) -> None:
    native_processor = _native_pixtral_processor()
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )

    with pytest.raises(ValueError, match=error_match):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32))]},
            {},
            BatchFeature({"images": images}),
        )


def test_mistral3_rejects_native_patch_count_mismatch() -> None:
    native_processor = _native_pixtral_processor()
    native_processor.image_processor.get_number_of_image_patches = (
        lambda height, width: (2, 1, 2) if (height, width) == (32, 48) else (1, 1, 1)
    )
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )

    with pytest.raises(ValueError, match="Mistral3 pixel_values.*patches"):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32))]},
            {},
            BatchFeature({"images": [torch.ones(3, 16, 16)]}),
        )


def test_mistral3_rejects_size_for_native_tokenizer() -> None:
    processor = object.__new__(Mistral3MultiModalProcessor)
    processor.info = Mistral3ProcessingInfo(_ProcessorContext(_mistral_tokenizer()))

    with pytest.raises(ValueError, match="Mistral tokenizer mode.*size"):
        processor.validate_mm_processor_kwargs({"size": {"longest_edge": 448}})


def test_mistral3_native_prompt_updates_do_not_replace_full_grid() -> None:
    native_processor = _native_pixtral_processor()
    config = Mistral3Config()
    config.image_token_index = native_processor.image_token_id
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor,
        get_hf_config=lambda: config,
        get_tokenizer=lambda: _mistral_tokenizer(),
        get_vision_encoder_info=lambda kwargs: object(),
    )
    images = SimpleNamespace(
        get_image_size=lambda item_idx: SimpleNamespace(width=48, height=32)
    )
    mm_items = SimpleNamespace(get_items=lambda modality, item_type: images)

    updates = multimodal_processor._get_prompt_updates(mm_items, {}, None)
    resolved = updates[0].resolve(0)

    assert resolved.target == []
    assert resolved.content.full.count(native_processor.image_token_id) == 2


def _processed_pixel_values(
    hf_processor,
    images: list[Image.Image],
    mm_processor_kwargs: dict[str, object],
) -> list[torch.Tensor]:
    """Resize via the HF image processor and un-pad to per-image H×W."""
    image_processor = hf_processor.image_processor
    hf_out = image_processor(images=images, return_tensors="pt", **mm_processor_kwargs)
    pixel_values = hf_out["pixel_values"]
    image_sizes = hf_out["image_sizes"]
    return [p[:, :h, :w] for p, (h, w) in zip(pixel_values, image_sizes)]


def _placeholder_count_from_prompt_updates(
    processor,
    images: list[Image.Image],
    pixel_values: list[torch.Tensor],
    mm_processor_kwargs: dict[str, object],
) -> int:
    hf_inputs = BatchFeature({"pixel_values": pixel_values})
    fields_config = processor._get_mm_fields_config(hf_inputs, mm_processor_kwargs)
    out_mm_kwargs = MultiModalKwargsItems.from_hf_inputs(hf_inputs, fields_config)
    # Prompt updates use raw PIL sizes and must predict the processed grid.
    mm_items = processor.info.parse_mm_data({"image": images})
    updates = processor._get_prompt_updates(
        mm_items, mm_processor_kwargs, out_mm_kwargs
    )
    image_token_id = processor.info.get_hf_config().image_token_index

    total = 0
    for item_idx in range(len(images)):
        details = updates[0].resolve(item_idx).content
        total += details.full.count(image_token_id)
    return total


def _expected_placeholder_tokens_per_image(
    hf_processor,
    pixel_values: torch.Tensor,
) -> int:
    """Count projected tokens from the actual HF-processed H×W."""
    image_h, image_w = pixel_values.shape[-2:]
    patch_size = hf_processor.image_processor.patch_size
    if isinstance(patch_size, dict):
        patch_h = patch_size["height"]
        patch_w = patch_size["width"]
    else:
        patch_h = patch_w = int(patch_size)

    spatial_merge_size = getattr(hf_processor, "spatial_merge_size", 1)
    assert image_h % patch_h == 0
    assert image_w % patch_w == 0

    return (image_h // (patch_h * spatial_merge_size)) * (
        image_w // (patch_w * spatial_merge_size)
    )


@pytest.mark.parametrize("model_id", [_MODEL_ID])
@pytest.mark.parametrize(
    ("mm_processor_kwargs", "image_size", "expected_toks_per_img"),
    [
        ({}, (448, 448), 256),
        ({"size": {"longest_edge": 1008}}, (1540, 1540), 1296),
        ({"size": {"longest_edge": 1288}}, (1536, 1187), 1656),
        ({"size": {"longest_edge": 1008}}, (29, 29), 1),
        ({"size": {"longest_edge": 1000}}, (1540, 1700), 1152),
    ],
)
@pytest.mark.parametrize("num_imgs", [1, 2])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_size_override(
    model_id: str,
    mm_processor_kwargs: dict[str, object],
    image_size: tuple[int, int],
    expected_toks_per_img: int,
    num_imgs: int,
    kwargs_on_init: bool,
):
    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": num_imgs},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    if isinstance(ctx.tokenizer, MistralTokenizer):
        object.__setattr__(ctx, "tokenizer", ctx.tokenizer.transformers_tokenizer)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, tokenizer=ctx.tokenizer
    )
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs
    hf_processor = processor.info.get_hf_processor(**hf_processor_mm_kwargs)

    dummy_image = Image.new("RGB", image_size, color=(127, 127, 127))
    images = [dummy_image] * num_imgs
    merged_mm_kwargs = processor.info.ctx.get_merged_mm_kwargs(hf_processor_mm_kwargs)
    pixel_values = _processed_pixel_values(hf_processor, images, merged_mm_kwargs)

    image_token_count = _placeholder_count_from_prompt_updates(
        processor, images, pixel_values, hf_processor_mm_kwargs
    )
    expected_from_pixel_values = _expected_placeholder_tokens_per_image(
        hf_processor, pixel_values[0]
    )
    assert expected_from_pixel_values == expected_toks_per_img
    assert image_token_count == expected_from_pixel_values * num_imgs


def test_lightonocr_keeps_vision_config_image_size():
    ctx = build_model_context(
        _LIGHTON_MODEL_ID,
        mm_processor_kwargs={"size": {"longest_edge": 1008}},
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    assert isinstance(processor.info, LightOnOCRProcessingInfo)
    encoder_info = processor.info.get_vision_encoder_info()
    assert isinstance(encoder_info, PixtralHFEncoderInfo)
    assert not isinstance(encoder_info, Mistral3HFEncoderInfo)
    assert encoder_info.get_image_size() == (
        processor.info.get_hf_config().vision_config.image_size
    )
