# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image
from transformers.models.pixtral import PixtralProcessor

from vllm.config import ModelConfig
from vllm.model_executor.models.llava import (
    LlavaDummyInputsBuilder,
    PixtralHFMultiModalProcessor,
    PixtralHFProcessingInfo,
)
from vllm.model_executor.models.pixtral import get_mistral_common_pixtral_processor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import (
    InputProcessingContext,
    TimingContext,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processors.pixtral import (
    MistralCommonPixtralHFProcessor,
)
from vllm.utils.mistral import is_mistral_tokenizer

from ...registry import HF_EXAMPLE_MODELS
from .test_mistral3 import (
    _CACHE_CASES,
    _TOKENIZER_PROCESSOR_CASES,
    _assert_hf_crops_pixel_values_to_image_sizes,
    _assert_hf_rejects_pixel_values_image_sizes_mismatch,
    _assert_hf_requires_image_sizes_for_pixel_values,
    _assert_native_dummy_inputs_build_budget,
    _assert_native_dummy_inputs_match_cache_paths,
    _assert_native_dummy_inputs_render_full_image_grids,
    _assert_native_images_normalize_to_pixel_values,
    _assert_native_prompt_updates_do_not_replace_full_grid,
    _assert_tokenizer_processor_case,
    _expected_placeholder_tokens_per_image,
    _mistral_tokenizer,
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
    _TOKENIZER_PROCESSOR_CASES,
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
    _assert_tokenizer_processor_case(
        tokenizer_mode=tokenizer_mode,
        actual_uses_mistral_tokenizer=is_mistral_tokenizer(ctx.tokenizer),
        processor=hf_processor,
        expected_processor_type=processor_type,
    )

    processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 2},
        mm_options={},
    )
    images = processor_inputs.mm_data_items["image"].get_all()
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
    assert len(mm_data["pixel_values"]) == len(images)
    assert all(value.shape[0] == 3 for value in mm_data["pixel_values"])
    if isinstance(hf_processor, MistralCommonPixtralHFProcessor):
        expected_patch_counts = [
            hf_processor.image_processor.get_number_of_image_patches(
                height=image.height,
                width=image.width,
            )[0]
            for image in images
        ]
    else:
        expected_patch_counts = [
            _expected_placeholder_tokens_per_image(hf_processor, value)
            for value in mm_data["pixel_values"]
        ]

    assert [
        item.get_num_embeds() for item in output["mm_placeholders"]["image"]
    ] == expected_patch_counts
    assert output["prompt_token_ids"].count(
        ctx.model_config.hf_config.image_token_index
    ) == sum(expected_patch_counts)


def test_pixtral_hf_keeps_hf_processor_without_tokenizer() -> None:
    ctx = _ProcessorContext(None)
    info = PixtralHFProcessingInfo(ctx)

    info.get_hf_processor()

    assert ctx.processor_cls is PixtralProcessor


def test_pixtral_hf_processor_tokenizes_text_and_images() -> None:
    processor = get_mistral_common_pixtral_processor(_mistral_tokenizer())
    assert processor is not None

    output = processor(
        text="plain text",
        images=[Image.new("RGB", (48, 32))],
        return_tensors="pt",
        add_special_tokens=True,
    )

    assert output["input_ids"] == [[11]]
    assert output["attention_mask"] == [[1]]
    assert output["images"][0].shape == (3, 32, 48)
    assert output["images"][0].dtype == torch.float32
    assert processor.tokenizer.kwargs["add_special_tokens"] is False


def test_pixtral_hf_native_dummy_inputs_render_full_image_grids() -> None:
    _assert_native_dummy_inputs_render_full_image_grids(LlavaDummyInputsBuilder)


def test_pixtral_hf_normalizes_native_images_to_pixel_values() -> None:
    native_processor = PixtralHFProcessingInfo(
        _ProcessorContext(_mistral_tokenizer())
    ).get_hf_processor()
    _assert_native_images_normalize_to_pixel_values(
        processor_cls=PixtralHFMultiModalProcessor,
        native_processor=native_processor,
    )


def test_pixtral_hf_hf_crops_pixel_values_to_image_sizes() -> None:
    _assert_hf_crops_pixel_values_to_image_sizes(PixtralHFMultiModalProcessor)


def test_pixtral_hf_hf_rejects_pixel_values_image_sizes_mismatch() -> None:
    _assert_hf_rejects_pixel_values_image_sizes_mismatch(PixtralHFMultiModalProcessor)


def test_pixtral_hf_hf_requires_image_sizes_for_pixel_values() -> None:
    _assert_hf_requires_image_sizes_for_pixel_values(PixtralHFMultiModalProcessor)


def test_pixtral_hf_native_prompt_updates_skip_hf_state() -> None:
    native_processor = get_mistral_common_pixtral_processor(_mistral_tokenizer())
    assert native_processor is not None
    _assert_native_prompt_updates_do_not_replace_full_grid(
        processor_cls=PixtralHFMultiModalProcessor,
        native_processor=native_processor,
    )


@pytest.mark.parametrize("cache_enabled", _CACHE_CASES)
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


@pytest.mark.parametrize("cache_enabled", _CACHE_CASES)
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
