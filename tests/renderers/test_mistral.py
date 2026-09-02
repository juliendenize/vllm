# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import time
from dataclasses import dataclass
from typing import Any, cast
from unittest.mock import Mock

import pytest
from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy
from transformers.models.pixtral import PixtralProcessor

from vllm.config import DeviceConfig, LoadConfig, ModelConfig, VllmConfig
from vllm.renderers import ChatParams, renderer_from_config
from vllm.renderers.hf import HfRenderer
from vllm.renderers.mistral import MistralRenderer, safe_apply_chat_template
from vllm.renderers.params import TokenizeParams
from vllm.tokenizers.mistral import MistralTokenizer
from vllm.transformers_utils.processors.pixtral import MistralCommonPixtralProcessor
from vllm.utils.mistral import is_mistral_tokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
HF_MISTRAL3_MODEL_NAME = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
pytestmark = pytest.mark.skip_global_cleanup


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    runner_type = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    max_model_len: int = 100
    tokenizer_revision = None
    tokenizer_mode = "mistral"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = True
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


class _MMProcessorPolicy:
    def __init__(self, events: list[str], error: Exception | None = None) -> None:
        self.events = events
        self.error = error
        self.received_kwargs: object | None = None

    def validate_mm_processor_kwargs(self, kwargs: object) -> None:
        self.events.append("policy")
        self.received_kwargs = kwargs
        if self.error is not None:
            raise self.error


def _make_renderer(events: list[str]) -> MistralRenderer:
    mock_model_config = MockModelConfig(skip_tokenizer_init=True)
    mock_tokenizer = Mock(spec=MistralTokenizer)

    def apply_chat_template(*_args: object, **_kwargs: object) -> list[int]:
        events.append("template")
        return [1, 2, 3]

    mock_tokenizer.apply_chat_template = apply_chat_template
    return MistralRenderer(
        MockVllmConfig(mock_model_config, parallel_config=MockParallelConfig()),
        tokenizer=mock_tokenizer,
    )


@pytest.mark.parametrize(
    ("tokenizer_mode", "uses_mistral_tokenizer", "processor_type"),
    [
        ("hf", False, PixtralProcessor),
        ("mistral", True, MistralCommonPixtralProcessor),
        ("auto", True, MistralCommonPixtralProcessor),
    ],
)
def test_hf_mistral3_renderer_tokenizer_matrix(
    tokenizer_mode: str,
    uses_mistral_tokenizer: bool,
    processor_type: type[object],
) -> None:
    model_config = ModelConfig(
        HF_MISTRAL3_MODEL_NAME,
        tokenizer=HF_MISTRAL3_MODEL_NAME,
        tokenizer_mode=tokenizer_mode,
        config_format="hf",
        dtype="auto",
        seed=0,
    )
    renderer = cast(
        HfRenderer | MistralRenderer,
        renderer_from_config(
            VllmConfig(
                model_config=model_config,
                load_config=LoadConfig(load_format="hf"),
                device_config=DeviceConfig(device="cpu"),
            )
        ),
    )

    if tokenizer_mode == "hf":
        assert type(renderer) is HfRenderer
    else:
        assert type(renderer) is MistralRenderer
    assert is_mistral_tokenizer(renderer.tokenizer) == uses_mistral_tokenizer
    assert renderer.mm_processor is not None
    assert isinstance(renderer.mm_processor.info.get_hf_processor(), processor_type)


@pytest.mark.asyncio
async def test_async_mistral_tokenizer_does_not_block_event_loop():
    expected_tokens = [1, 2, 3]

    # Mock the blocking version to sleep
    def mocked_apply_chat_template(*_args, **_kwargs):
        time.sleep(2)
        return expected_tokens

    mock_model_config = MockModelConfig(skip_tokenizer_init=True)
    mock_tokenizer = Mock(spec=MistralTokenizer)
    mock_tokenizer.apply_chat_template = mocked_apply_chat_template
    mock_renderer = MistralRenderer(
        MockVllmConfig(mock_model_config, parallel_config=MockParallelConfig()),
        tokenizer=mock_tokenizer,
    )

    task = mock_renderer.render_messages_async([], ChatParams())

    # Ensure the event loop is not blocked
    blocked_count = 0
    for _i in range(20):  # Check over ~2 seconds
        start = time.perf_counter()
        await asyncio.sleep(0)
        elapsed = time.perf_counter() - start

        # an overly generous elapsed time for slow machines
        if elapsed >= 0.5:
            blocked_count += 1

        await asyncio.sleep(0.1)

    # Ensure task completes
    _, prompt = await task
    assert prompt["prompt_token_ids"] == expected_tokens, (
        "Mocked blocking tokenizer was not called"
    )
    assert blocked_count == 0, "Event loop blocked during tokenization"


def test_renderer_noop_policy_preserves_chat_rendering():
    events: list[str] = []
    renderer = _make_renderer(events)
    renderer.mm_processor = _MMProcessorPolicy(events)

    _, prompts = renderer.render_chat(
        conversations=[[]],
        chat_params=ChatParams(mm_processor_kwargs={"size": 42}),
        tok_params=TokenizeParams(max_total_tokens=100),
    )

    assert prompts[0]["prompt_token_ids"] == [1, 2, 3]
    assert events == ["policy", "template"]


def test_renderer_policy_receives_exact_kwargs_before_sync_chat_rendering():
    events: list[str] = []
    renderer = _make_renderer(events)
    policy = _MMProcessorPolicy(events, ValueError("invalid multimodal kwargs"))
    renderer.mm_processor = policy
    mm_processor_kwargs = {"size": 42}

    with pytest.raises(ValueError, match="invalid multimodal kwargs"):
        renderer.render_chat(
            conversations=[[]],
            chat_params=ChatParams(mm_processor_kwargs=mm_processor_kwargs),
            tok_params=TokenizeParams(max_total_tokens=100),
        )

    assert policy.received_kwargs is mm_processor_kwargs
    assert events == ["policy"]


def test_renderer_policy_runs_before_async_chat_rendering():
    events: list[str] = []
    renderer = _make_renderer(events)
    renderer.mm_processor = _MMProcessorPolicy(
        events, ValueError("invalid multimodal kwargs")
    )

    with pytest.raises(ValueError, match="invalid multimodal kwargs"):
        asyncio.run(
            renderer.render_chat_async(
                conversations=[[]],
                chat_params=ChatParams(mm_processor_kwargs={"size": 42}),
                tok_params=TokenizeParams(max_total_tokens=100),
            )
        )

    assert events == ["policy"]


def test_apply_mistral_chat_template_thinking_chunk():
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "thinking",
                    "closed": True,
                    "thinking": "Only return the answer when you are confident.",
                },
            ],
        },
        {"role": "user", "content": "What is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me think about it."},
                {"type": "thinking", "closed": True, "thinking": "2+2 = 4"},
                {
                    "type": "text",
                    "text": "The answer is 4.",
                },
            ],
        },
        {"role": "user", "content": "Thanks, what is 3+3?"},
    ]
    mistral_tokenizer = MistralTokenizer.from_pretrained(
        "mistralai/Magistral-Small-2509"
    )

    tokens_ids = safe_apply_chat_template(
        mistral_tokenizer, messages, chat_template=None, tools=None
    )

    string_tokens = mistral_tokenizer.mistral.decode(
        tokens_ids, special_token_policy=SpecialTokenPolicy.KEEP
    )

    expected_tokens = (
        r"<s>[SYSTEM_PROMPT]You are a helpful assistant.[THINK]Only return the"
        r" answer when you are confident.[/THINK][/SYSTEM_PROMPT]"
        r"[INST]What is 2+2?[/INST]"
        r"Let me think about it.[THINK]2+2 = 4[/THINK]The answer is 4.</s>"
        r"[INST]Thanks, what is 3+3?[/INST]"
    )

    assert string_tokens == expected_tokens
