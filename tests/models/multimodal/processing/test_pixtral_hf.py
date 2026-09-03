# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.pixtral import PixtralProcessor

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.llava import (
    LlavaDummyInputsBuilder,
    LlavaProcessingInfo,
    PixtralHFMultiModalProcessor,
    PixtralHFProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import (
    InputProcessingContext,
    TimingContext,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processors.pixtral import MistralCommonPixtralProcessor
from vllm.utils.mistral import is_mistral_tokenizer

from ...registry import HF_EXAMPLE_MODELS
from .test_mistral3 import (
    _DummyTextTokenizer,
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


def test_pixtral_hf_hf_dummy_inputs_preserve_supplied_data() -> None:
    parsed_mm_data: MultiModalDataDict | None = None
    parsed_validate: bool | None = None
    image = Image.new("RGB", (32, 32))

    def parse_mm_data(
        mm_data: MultiModalDataDict,
        *,
        validate: bool,
    ) -> dict[str, ImageProcessorItems]:
        nonlocal parsed_mm_data, parsed_validate
        parsed_mm_data = mm_data
        parsed_validate = validate
        return {"image": ImageProcessorItems(mm_data["image"])}

    info = SimpleNamespace(
        ctx=SimpleNamespace(tokenizer=_DummyTextTokenizer()),
        get_hf_processor=lambda: SimpleNamespace(image_token="<image>"),
        get_image_size_with_most_features=lambda: (32, 32),
        parse_mm_data=parse_mm_data,
    )
    builder = LlavaDummyInputsBuilder(info)

    inputs = builder.get_dummy_processor_inputs(
        seq_len=128,
        mm_counts={"image": 1},
        mm_options={},
        mm_data={"image": [image]},
    )

    assert parsed_mm_data is not None
    assert parsed_mm_data["image"][0] is image
    assert parsed_validate is False
    assert inputs.prompt == [7]
    assert inputs.mm_data_items["image"].get_all() == [image]


def test_pixtral_hf_normalizes_native_images_to_pixel_values() -> None:
    native_processor = PixtralHFProcessingInfo(
        _ProcessorContext(_mistral_tokenizer())
    ).get_hf_processor()
    multimodal_processor = object.__new__(PixtralHFMultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: native_processor
    )
    native_images = [torch.ones(1, 3, 32, 48)]
    processed_data = BatchFeature({"images": native_images})

    output = multimodal_processor._postprocess_hf_mm_data(
        {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
    )

    assert output["pixel_values"] is native_images
    assert "images" not in output


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_pixtral_hf_native_dummy_inputs_match_cache_paths(
    cache_enabled: bool,
) -> None:
    ctx = _build_pixtral_context(
        tokenizer_mode="mistral",
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=4 if cache_enabled else 0,
    )
    cache = MultiModalProcessorOnlyCache(ctx.model_config) if cache_enabled else None
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config, cache=cache)
    mm_config = ctx.model_config.get_multimodal_config()
    processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 2},
        mm_options=mm_config.limit_per_prompt,
    )
    native_processor = processor.info.get_hf_processor()
    images = processor_inputs.mm_data_items["image"].get_all()
    expected_patch_counts = [
        native_processor.image_processor.get_number_of_image_patches(
            height=image.height,
            width=image.width,
        )[0]
        for image in images
    ]

    output = processor.apply(processor_inputs, TimingContext(enabled=False))

    assert processor_inputs.prompt.count(native_processor.image_token_id) == sum(
        expected_patch_counts
    )
    assert [
        item.get_num_embeds() for item in output["mm_placeholders"]["image"]
    ] == expected_patch_counts

    if cache_enabled:
        assert cache is not None
        cached_output = processor.apply(
            processor_inputs,
            TimingContext(enabled=False),
        )
        uncached_processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
        uncached_output = uncached_processor.apply(
            processor_inputs,
            TimingContext(enabled=False),
        )
        assert cache.make_stats().hits > 0
        assert cached_output["prompt_token_ids"] == uncached_output["prompt_token_ids"]
        assert cached_output["mm_hashes"] == uncached_output["mm_hashes"]
        assert cached_output["mm_placeholders"] == uncached_output["mm_placeholders"]
        cached_data = cached_output["mm_kwargs"].get_data()
        uncached_data = uncached_output["mm_kwargs"].get_data()
        assert cached_data.keys() == uncached_data.keys()
        for key in cached_data:
            assert len(cached_data[key]) == len(uncached_data[key])
            for cached_value, uncached_value in zip(
                cached_data[key], uncached_data[key]
            ):
                assert cached_value.shape == uncached_value.shape
                assert cached_value.dtype == uncached_value.dtype
                torch.testing.assert_close(cached_value, uncached_value)


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_pixtral_hf_native_dummy_inputs_build_budget(cache_enabled: bool) -> None:
    ctx = _build_pixtral_context(
        tokenizer_mode="mistral",
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=4,
    )

    budget = MultiModalBudget(
        VllmConfig(
            model_config=ctx.model_config,
            device_config=DeviceConfig(device="cpu"),
        ),
        MULTIMODAL_REGISTRY,
        enable_cache=cache_enabled,
    )

    assert budget.mm_max_toks_per_item["image"] > 0
