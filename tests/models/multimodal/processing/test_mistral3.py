# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Mistral3's multimodal preprocessing."""

from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from PIL import Image
from transformers import AutoProcessor, BatchFeature, Mistral3Config
from transformers.models.pixtral import PixtralProcessor

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.models.lightonocr import LightOnOCRProcessingInfo
from vllm.model_executor.models.mistral3 import (
    Mistral3DummyInputsBuilder,
    Mistral3HFEncoderInfo,
    Mistral3MultiModalProcessor,
    Mistral3ProcessingInfo,
)
from vllm.model_executor.models.pixtral import (
    PixtralDummyInputsBuilder,
    PixtralHFEncoderInfo,
    get_mistral_common_pixtral_dummy_inputs,
    get_mistral_common_pixtral_processor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import ImageProcessorItems
from vllm.multimodal.processing import (
    InputProcessingContext,
    TimingContext,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.transformers_utils.processors.pixtral import (
    MistralCommonImageProcessor,
    MistralCommonPixtralHFProcessor,
    MistralCommonPixtralProcessor,
)

from ...registry import HF_EXAMPLE_MODELS
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
        self.hf_config = hf_config or SimpleNamespace(
            image_token_index=2,
            vision_config=SimpleNamespace(patch_size=16),
            spatial_merge_size=1,
        )
        self.processor_cls: type[object] | None = None
        self.processor_kwargs: dict[str, object] | None = None

    def get_tokenizer(self) -> object:
        if self.tokenizer is None:
            raise ValueError(
                "You cannot pass text prompts when `skip_tokenizer_init=True`"
            )
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


class _NativeChatTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[Image.Image]] = []
        self.texts: list[str] = []

    def encode_chat_completion(self, request: ChatCompletionRequest) -> SimpleNamespace:
        content = request.messages[0].content
        self.texts.append(content[0].text)
        images = [chunk.image for chunk in content[1:]]
        self.calls.append(images)
        tokens: list[int] = []
        for image in images:
            tokens.extend([2] * (image.width // 16))
            tokens.append(3)
        return SimpleNamespace(tokens=tokens)


class _NativeImageEncoder:
    special_ids = SimpleNamespace(img_break=1, img=2, img_end=3)

    def __init__(
        self,
        image_patch_size: int = 16,
        spatial_merge_size: int = 1,
    ) -> None:
        self.image_config = SimpleNamespace(
            image_patch_size=image_patch_size,
            max_image_size=1024,
            spatial_merge_size=spatial_merge_size,
        )

    def __call__(self, image_chunk: object) -> SimpleNamespace:
        return SimpleNamespace(image=torch.ones(3, 32, 48))

    def _image_to_num_tokens(self, image: Image.Image) -> tuple[int, int]:
        return 2, 1


def _mistral_tokenizer(
    image_patch_size: int = 16,
    spatial_merge_size: int = 1,
) -> MistralTokenizer:
    tokenizer = object.__new__(MistralTokenizer)
    tokenizer.transformers_tokenizer = _NativeTextTokenizer()
    tokenizer.instruct = SimpleNamespace(
        mm_encoder=_NativeImageEncoder(
            image_patch_size=image_patch_size,
            spatial_merge_size=spatial_merge_size,
        )
    )
    return tokenizer


def _native_pixtral_processor() -> MistralCommonPixtralProcessor:
    tokenizer = _mistral_tokenizer()
    return MistralCommonPixtralProcessor(
        tokenizer=tokenizer,
        image_processor=MistralCommonImageProcessor(tokenizer.instruct.mm_encoder),
    )


def _build_mistral3_processing_context(
    tokenizer_mode: str,
) -> InputProcessingContext:
    model_info = HF_EXAMPLE_MODELS.find_hf_info(_MODEL_ID)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(
        on_fail="skip",
        check_max_version=False,
        check_version_reason="vllm",
    )
    model_config = ModelConfig(
        _MODEL_ID,
        tokenizer=_MODEL_ID,
        tokenizer_mode=tokenizer_mode,
        config_format="hf",
        revision=model_info.revision,
        trust_remote_code=model_info.trust_remote_code,
        dtype="auto",
        seed=0,
        limit_mm_per_prompt={"image": 2},
        hf_overrides=model_info.hf_overrides,
    )
    tokenizer = cached_tokenizer_from_config(model_config)
    if tokenizer_mode == "hf":
        tokenizer = AutoProcessor.from_pretrained(
            _MODEL_ID,
            revision=model_info.revision,
            trust_remote_code=model_info.trust_remote_code,
        ).tokenizer

    return InputProcessingContext(model_config, tokenizer=tokenizer)


@pytest.mark.parametrize(
    ("tokenizer_mode", "processor_type"),
    [
        ("hf", PixtralProcessor),
        ("mistral", MistralCommonPixtralHFProcessor),
        ("auto", MistralCommonPixtralHFProcessor),
    ],
)
def test_mistral3_hf_format_tokenizer_matrix(
    tokenizer_mode: str,
    processor_type: type[object],
) -> None:
    ctx = _build_mistral3_processing_context(tokenizer_mode)
    processor = MULTIMODAL_REGISTRY.create_processor(
        ctx.model_config, tokenizer=ctx.tokenizer
    )
    hf_processor = cast(
        PixtralProcessor | MistralCommonPixtralProcessor,
        processor.info.get_hf_processor(),
    )
    assert type(hf_processor) is processor_type
    assert isinstance(ctx.tokenizer, MistralTokenizer) == (tokenizer_mode != "hf")

    if isinstance(hf_processor, MistralCommonPixtralProcessor):
        images = [Image.new("RGB", (48, 32)), Image.new("RGB", (64, 32))]
        processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
            seq_len=ctx.model_config.max_model_len,
            mm_counts={"image": 2},
            mm_options={},
            mm_data={"image": images},
        )
        output = processor.apply(processor_inputs, TimingContext(enabled=False))
        mm_data = output["mm_kwargs"].get_data()
        expected_patch_counts = [
            hf_processor.image_processor.get_number_of_image_patches(
                height=image.height,
                width=image.width,
            )[0]
            for image in images
        ]

        assert set(mm_data) == {"pixel_values"}
        assert [tuple(value.shape) for value in mm_data["pixel_values"]] == [
            (3, 56, 56),
            (3, 56, 84),
        ]
        assert processor_inputs.prompt.count(hf_processor.image_token_id) == sum(
            expected_patch_counts
        )
        assert [
            item.get_num_embeds() for item in output["mm_placeholders"]["image"]
        ] == expected_patch_counts
        return

    images = [Image.new("RGB", (448, 448)), Image.new("RGB", (448, 672))]
    processor_inputs = processor.dummy_inputs.get_dummy_processor_inputs(
        seq_len=ctx.model_config.max_model_len,
        mm_counts={"image": 2},
        mm_options={},
        mm_data={"image": images},
    )
    output = processor.apply(processor_inputs, TimingContext(enabled=False))
    mm_data = output["mm_kwargs"].get_data()
    expected_patch_counts = [
        _expected_placeholder_tokens_per_image(hf_processor, value)
        for value in mm_data["pixel_values"]
    ]

    assert set(output) == {
        "type",
        "prompt_token_ids",
        "mm_kwargs",
        "mm_placeholders",
        "mm_hashes",
    }
    assert output["type"] == "multimodal"
    assert set(mm_data) == {"pixel_values"}
    assert [tuple(value.shape) for value in mm_data["pixel_values"]] == [
        (3, 448, 448),
        (3, 672, 448),
    ]
    assert len(output["mm_hashes"]["image"]) == len(images)
    assert [
        item.get_num_embeds() for item in output["mm_placeholders"]["image"]
    ] == expected_patch_counts
    assert output["prompt_token_ids"].count(
        ctx.model_config.hf_config.image_token_index
    ) == sum(expected_patch_counts)


class _NativeDummyInfo:
    def __init__(self) -> None:
        self.processor = _native_pixtral_processor()
        self.chat_tokenizer = _NativeChatTokenizer()
        self.parse_validate: bool | None = None
        self.tokenizer = SimpleNamespace(mistral=self.chat_tokenizer)

    def get_hf_processor(self) -> MistralCommonPixtralProcessor:
        return self.processor

    def get_tokenizer(self) -> object:
        return self.tokenizer

    def get_image_size_with_most_features(self) -> tuple[int, int]:
        return 64, 32

    def parse_mm_data(
        self,
        mm_data: MultiModalDataDict,
        *,
        validate: bool = True,
    ) -> dict[str, ImageProcessorItems]:
        self.parse_validate = validate
        return {"image": ImageProcessorItems(mm_data["image"])}


def test_mistral3_native_vision_info_uses_image_config() -> None:
    info = Mistral3ProcessingInfo(_ProcessorContext(_mistral_tokenizer()))

    encoder_info = info.get_vision_encoder_info()

    assert encoder_info.get_image_size() == 1024


def test_mistral3_keeps_hf_processor_without_tokenizer() -> None:
    ctx = _ProcessorContext(None)
    info = Mistral3ProcessingInfo(ctx)

    info.get_hf_processor(size={"longest_edge": 448})

    assert ctx.processor_cls is PixtralProcessor
    assert ctx.processor_kwargs == {"size": {"longest_edge": 448}}


def test_mistral3_native_dummy_inputs_render_full_image_grids() -> None:
    info = _NativeDummyInfo()
    builder = Mistral3DummyInputsBuilder(info)
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


def test_native_pixtral_dummy_inputs_preserve_dummy_text() -> None:
    info = _NativeDummyInfo()

    class _DummyTextPixtralInputsBuilder(PixtralDummyInputsBuilder):
        def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
            return "native dummy text"

    builder = _DummyTextPixtralInputsBuilder(info)

    builder.get_dummy_processor_inputs(
        seq_len=128,
        mm_counts={"image": 0},
        mm_options={},
        mm_data={"image": []},
    )

    assert info.chat_tokenizer.texts == ["native dummy text"]


def test_shared_native_dummy_inputs_skip_validation() -> None:
    info = _NativeDummyInfo()
    images = [Image.new("RGB", (32, 32)), Image.new("RGB", (64, 32))]

    inputs = get_mistral_common_pixtral_dummy_inputs(
        tokenizer=info.get_tokenizer(),
        mm_data={"image": images},
        parse_mm_data=info.parse_mm_data,
    )

    assert info.parse_validate is False
    assert info.chat_tokenizer.calls == [images]
    assert inputs.prompt == [2, 2, 3, 2, 2, 2, 2, 3]


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


def test_native_pixtral_processor_accepts_image_kwargs() -> None:
    processor = _native_pixtral_processor()
    image = Image.new("RGB", (48, 32))

    output = processor(
        images=[image],
        size={"longest_edge": 448},
    )
    image_output = processor.image_processor(
        images=[image],
        do_resize=False,
    )

    assert output["images"][0].shape == (3, 32, 48)
    assert image_output["images"][0].shape == (3, 32, 48)


def test_native_pixtral_processor_keeps_native_processor_type() -> None:
    processor = _native_pixtral_processor()

    assert type(processor) is MistralCommonPixtralProcessor
    assert not hasattr(processor, "image_token")
    assert not hasattr(processor, "image_break_token")
    assert not hasattr(processor, "image_end_token")


def test_hf_pixtral_processor_uses_hf_adapter() -> None:
    processor = get_mistral_common_pixtral_processor(_mistral_tokenizer())

    assert type(processor) is MistralCommonPixtralHFProcessor
    assert processor.image_token == "[IMG]"
    assert processor.image_break_token == "[IMG_BREAK]"
    assert processor.image_end_token == "[IMG_END]"


def test_mistral3_normalizes_native_images_to_pixel_values() -> None:
    native_processor = _native_pixtral_processor()
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
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


def test_mistral3_hf_crops_pixel_values_to_image_sizes() -> None:
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
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


def test_mistral3_hf_rejects_pixel_values_image_sizes_mismatch() -> None:
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
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


def test_mistral3_hf_requires_image_sizes_for_pixel_values() -> None:
    multimodal_processor = object.__new__(Mistral3MultiModalProcessor)
    multimodal_processor.info = SimpleNamespace(
        get_hf_processor=lambda **kwargs: SimpleNamespace()
    )
    processed_data = BatchFeature({"pixel_values": [torch.ones(1, 4, 5)]})

    with pytest.raises(KeyError, match="image_sizes"):
        multimodal_processor._postprocess_hf_mm_data(
            {"images": [Image.new("RGB", (48, 32))]}, {}, processed_data
        )


def test_mistral3_native_prompt_updates_do_not_replace_full_grid() -> None:
    native_processor = _native_pixtral_processor()
    config = Mistral3Config()
    config.image_token_index = native_processor.image_token_id
    config.vision_config.patch_size = (
        native_processor.image_processor.mm_encoder.image_config.image_patch_size
    )
    config.spatial_merge_size = (
        native_processor.image_processor.mm_encoder.image_config.spatial_merge_size
    )
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


def _assert_native_dummy_inputs_match_cache_paths(
    ctx: InputProcessingContext,
    cache_enabled: bool,
) -> None:
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
def test_mistral3_native_dummy_inputs_match_cache_paths(
    cache_enabled: bool,
) -> None:
    ctx = build_model_context(
        _MODEL_ID,
        limit_mm_per_prompt={"image": 2},
        mm_processor_cache_gb=4 if cache_enabled else 0,
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )

    _assert_native_dummy_inputs_match_cache_paths(
        ctx=ctx,
        cache_enabled=cache_enabled,
    )


def _assert_native_dummy_inputs_build_budget(
    ctx: InputProcessingContext,
    cache_enabled: bool,
) -> None:
    budget = MultiModalBudget(
        VllmConfig(
            model_config=ctx.model_config,
            device_config=DeviceConfig(device="cpu"),
        ),
        MULTIMODAL_REGISTRY,
        enable_cache=cache_enabled,
    )

    assert budget.mm_max_toks_per_item["image"] > 0


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_mistral3_native_dummy_inputs_build_budget(cache_enabled: bool) -> None:
    ctx = build_model_context(
        _MODEL_ID,
        limit_mm_per_prompt={"image": 1},
        mm_processor_cache_gb=4,
        model_config_kwargs=_MODEL_CONFIG_KWARGS,
    )

    _assert_native_dummy_inputs_build_budget(
        ctx=ctx,
        cache_enabled=cache_enabled,
    )


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
