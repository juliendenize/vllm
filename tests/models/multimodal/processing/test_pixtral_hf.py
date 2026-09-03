# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.pixtral import PixtralProcessor

from vllm.config import ModelConfig
from vllm.model_executor.models.llava import (
    LlavaDummyInputsBuilder,
    PixtralHFMultiModalProcessor,
    PixtralHFProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    InputProcessingContext,
    TimingContext,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processors.pixtral import MistralCommonPixtralProcessor
from vllm.utils.mistral import is_mistral_tokenizer

from ...registry import HF_EXAMPLE_MODELS
from .test_mistral3 import (
    _assert_native_dummy_inputs_build_budget,
    _assert_native_dummy_inputs_match_cache_paths,
    _mistral_tokenizer,
    _NativeDummyInfo,
    _ProcessorContext,
)

pytestmark = pytest.mark.skip_global_cleanup

_PIXTRAL_MODEL_ID = "mistral-community/pixtral-12b"


def _build_pixtral_context(
    *,
    tokenizer_mode: str,
    limit_mm_per_prompt: dict[str, int],
    mm_processor_cache_gb: int,
) -> InputProcessingContext:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(_PIXTRAL_MODEL_ID)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )
    model_config = ModelConfig(
        _PIXTRAL_MODEL_ID,
        tokenizer="mistralai/Pixtral-12B-2409",
        tokenizer_mode=tokenizer_mode,
        config_format="hf",
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        dtype="auto",
        seed=0,
        limit_mm_per_prompt=limit_mm_per_prompt,
        mm_processor_cache_gb=mm_processor_cache_gb,
        hf_overrides=model_info.hf_overrides,
    )
    return InputProcessingContext(
        model_config,
        tokenizer=cached_tokenizer_from_config(model_config),
    )


@pytest.mark.parametrize(
    ("tokenizer_mode", "processor_type"),
    [
        ("hf", PixtralProcessor),
        ("mistral", MistralCommonPixtralProcessor),
        ("auto", MistralCommonPixtralProcessor),
    ],
)
def test_pixtral_hf_tokenizer_matrix(
    tokenizer_mode: str,
    processor_type: type[object],
) -> None:
    ctx = _build_pixtral_context(
        tokenizer_mode=tokenizer_mode,
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=0,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, tokenizer=ctx.tokenizer
    )
    hf_processor = processor.info.get_hf_processor()
    assert type(hf_processor) is processor_type
    assert is_mistral_tokenizer(ctx.tokenizer) == (tokenizer_mode != "hf")

    images = [Image.new("RGB", (48, 32)), Image.new("RGB", (64, 32))]
    processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 2},
        mm_options={},
        mm_data={"image": images},
    )
    output = processor.apply(processor_inputs, TimingContext(enabled=False))
    mm_data = output["mm_kwargs"].get_data()

    assert set(output) == {
        "type",
        "prompt_token_ids",
        "mm_kwargs",
        "mm_placeholders",
        "mm_hashes",
    }
    assert set(mm_data) == {"pixel_values"}
    assert [tuple(value.shape) for value in mm_data["pixel_values"]] == [
        (3, 32, 48),
        (3, 32, 64),
    ]
    assert [item.get_num_embeds() for item in output["mm_placeholders"]["image"]] == [
        6,
        8,
    ]
    assert (
        output["prompt_token_ids"].count(ctx.model_config.hf_config.image_token_index)
        == 14
    )


def test_pixtral_hf_keeps_hf_processor_without_tokenizer() -> None:
    ctx = _ProcessorContext(None)
    info = PixtralHFProcessingInfo(ctx)

    info.get_hf_processor()

    assert ctx.processor_cls is PixtralProcessor


def test_pixtral_hf_native_dummy_inputs_render_full_image_grids() -> None:
    info = _NativeDummyInfo()
    builder = LlavaDummyInputsBuilder(info)
    images = [
        Image.new("RGB", (32, 32)),
        Image.new("RGB", (64, 32)),
    ]

    inputs = builder.get_dummy_processor_inputs(
        seq_len=128,
        mm_counts={"image": 2},
        mm_options={},
        mm_data={"image": images},
    )

    assert info.parse_validate is False
    assert info.chat_tokenizer.calls == [images]
    assert inputs.prompt == [2, 2, 3, 2, 2, 2, 2, 3]


def test_pixtral_hf_normalizes_native_images_to_pixel_values() -> None:
    native_processor = PixtralHFProcessingInfo(
        _ProcessorContext(_mistral_tokenizer())
    ).get_hf_processor()
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )
    native_images = [torch.ones(1, 3, 32, 48)]

    class NativeBatchFeature(BatchFeature):
        def __contains__(self, key: object) -> bool:
            if key == "image_sizes":
                raise AssertionError("native output must not access image_sizes")
            return super().__contains__(key)

    processed_data = NativeBatchFeature({"images": native_images})

    output = multimodal_processor._postprocess_hf_mm_data(
        {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
    )

    assert output["pixel_values"] is native_images
    assert "images" not in output


def test_pixtral_hf_hf_crops_pixel_values_to_image_sizes() -> None:
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: SimpleNamespace()
    )
    pixel_values = [torch.arange(20).reshape(1, 4, 5)]
    processed_data = BatchFeature(
        {"pixel_values": pixel_values, "image_sizes": [(2, 3)]}
    )

    output = multimodal_processor._postprocess_hf_mm_data(
        {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
    )

    torch.testing.assert_close(output["pixel_values"][0], pixel_values[0][:, :2, :3])


def test_pixtral_hf_hf_rejects_pixel_values_image_sizes_mismatch() -> None:
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: SimpleNamespace()
    )
    processed_data = BatchFeature(
        {
            "pixel_values": [torch.ones(1, 4, 5), torch.ones(1, 4, 5)],
            "image_sizes": [(2, 3)],
        }
    )

    with pytest.raises(ValueError, match="same number of images"):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
        )


def test_pixtral_hf_hf_requires_image_sizes_for_pixel_values() -> None:
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: SimpleNamespace()
    )
    processed_data = BatchFeature({"pixel_values": [torch.ones(1, 4, 5)]})

    with pytest.raises(KeyError, match="image_sizes"):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
        )


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_pixtral_hf_native_dummy_inputs_match_cache_paths(
    cache_enabled: bool,
) -> None:
    ctx = _build_pixtral_context(
        tokenizer_mode="mistral",
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=4 if cache_enabled else 0,
    )
    _assert_native_dummy_inputs_match_cache_paths(
        ctx=ctx,
        cache_enabled=cache_enabled,
    )


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_pixtral_hf_native_dummy_inputs_build_budget(cache_enabled: bool) -> None:
    ctx = _build_pixtral_context(
        tokenizer_mode="mistral",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=4,
    )

    _assert_native_dummy_inputs_build_budget(
        ctx=ctx,
        cache_enabled=cache_enabled,
    )
