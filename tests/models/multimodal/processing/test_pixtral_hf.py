# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.llava import (
    LlavaForConditionalGeneration,
    LlavaProcessingInfo,
    PixtralHFMultiModalProcessor,
    PixtralHFProcessingInfo,
)
from vllm.transformers_utils.processors.pixtral import MistralCommonPixtralProcessor

from .test_mistral3 import _mistral_tokenizer, _ProcessorContext

pytestmark = pytest.mark.skip_global_cleanup


def test_pixtral_hf_selects_native_processor_for_mistral_tokenizer() -> None:
    info = PixtralHFProcessingInfo(_ProcessorContext(_mistral_tokenizer()))

    processor = info.get_hf_processor()

    assert isinstance(processor, MistralCommonPixtralProcessor)


def test_pixtral_hf_rejects_native_image_token_id_mismatch_at_setup() -> None:
    ctx = _ProcessorContext(
        _mistral_tokenizer(), hf_config=SimpleNamespace(image_token_index=99)
    )

    with pytest.raises(ValueError, match="Llava.*image token id"):
        PixtralHFProcessingInfo(ctx).get_hf_processor()


def test_llava_keeps_hf_processor_for_non_pixtral_vision() -> None:
    ctx = _ProcessorContext(object())
    info = LlavaProcessingInfo(ctx)

    info.get_hf_processor()

    assert ctx.processor_cls is not None
    assert ctx.processor_cls.__name__ == "LlavaProcessor"


def test_pixtral_hf_accepts_size_for_hf_tokenizer() -> None:
    processor = object.__new__(PixtralHFMultiModalProcessor)
    processor.info = PixtralHFProcessingInfo(_ProcessorContext(object()))

    processor.validate_mm_processor_kwargs({"size": {"longest_edge": 448}})


def test_pixtral_hf_normalizes_native_images_to_pixel_values() -> None:
    native_processor = PixtralHFProcessingInfo(
        _ProcessorContext(_mistral_tokenizer())
    ).get_hf_processor()
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )
    processed_data = BatchFeature({"images": [torch.ones(3, 32, 48)]})

    output = multimodal_processor._postprocess_hf_mm_data(
        {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
    )

    assert output["pixel_values"][0].shape == (3, 32, 48)
    assert "images" not in output


def test_pixtral_hf_rejects_invalid_native_image_count() -> None:
    native_processor = PixtralHFProcessingInfo(
        _ProcessorContext(_mistral_tokenizer())
    ).get_hf_processor()
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )

    with pytest.raises(ValueError, match="Llava.*same number of images"):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32)), Image.new("RGB", (48, 32))]},
            {},
            BatchFeature({"images": torch.ones(1, 3, 32, 48)}),
        )


def test_pixtral_hf_rejects_size_for_native_tokenizer() -> None:
    processor = object.__new__(PixtralHFMultiModalProcessor)
    processor.info = PixtralHFProcessingInfo(_ProcessorContext(_mistral_tokenizer()))

    with pytest.raises(ValueError, match="Mistral tokenizer mode.*size"):
        processor.validate_mm_processor_kwargs({"size": {"longest_edge": 448}})


def _pixtral_llava_model_for_input_validation() -> LlavaForConditionalGeneration:
    model = object.__new__(LlavaForConditionalGeneration)
    model.config = SimpleNamespace(
        vision_config=SimpleNamespace(model_type="pixtral", num_channels=3)
    )
    return model


@pytest.mark.parametrize(
    ("pixel_values", "error_match"),
    [
        (torch.ones(1, 3, 2), "4-D tensor"),
        (torch.ones(1, 4, 2, 2), "3 channels"),
        (torch.ones(1, 3, 2, 2, dtype=torch.int64), "floating-point dtype"),
        (torch.ones(1, 3, 0, 2), "positive spatial dimensions"),
    ],
)
def test_pixtral_hf_validates_direct_pixel_values(
    pixel_values: torch.Tensor,
    error_match: str,
) -> None:
    model = _pixtral_llava_model_for_input_validation()

    with pytest.raises(ValueError, match=error_match):
        model._parse_and_validate_image_input(pixel_values=pixel_values)


def test_pixtral_hf_preserves_valid_ragged_direct_pixel_values() -> None:
    model = _pixtral_llava_model_for_input_validation()
    pixel_values = [torch.ones(3, 2, 2), torch.ones(3, 3, 4)]

    image_input = model._parse_and_validate_image_input(pixel_values=pixel_values)

    assert image_input is not None
    assert image_input["pixel_values"] is pixel_values
