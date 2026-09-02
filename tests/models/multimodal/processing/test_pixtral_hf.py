# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchFeature

from vllm.model_executor.models.llava import (
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
