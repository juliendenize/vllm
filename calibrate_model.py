import json
from functools import partial

import tqdm
from datasets import concatenate_datasets, load_dataset
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from merge_checkpoints import merge_checkpoints
from save_shards import save_shards_if_rank0
from vllm import LLM, SamplingParams
from pathlib import Path
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

if __name__ == "__main__":
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"
    DATASET_ID_LONG = "openai/mrcr"
    NUM_CALIBRATION_SAMPLES = 100
    NUM_CALIBRATION_SAMPLES_LONG = 100
    MAX_LEN = 300000
    TP = 8
    ORIGIN_CKPT = (
        "/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_weight_quantized"
    )
    SAVE_ACTI = "/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_activation_calibrated_long_3"
    MERGE_CKPT = "/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_nvfp4_long_3"

    tokenizer = MistralTokenizer.from_file(
        "/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_bf16/tekken.json"
    )

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds_long = load_dataset(
        DATASET_ID_LONG, split=f"train[:{NUM_CALIBRATION_SAMPLES_LONG}]"
    )
    ds = concatenate_datasets([ds, ds_long])
    ds = ds.shuffle(seed=42)

    # Create a sampling params object for greedy sampling
    sampling_params = SamplingParams(
        temperature=0.80, top_p=0.95, max_tokens=100, min_tokens=1
    )
    llm = LLM(
        model="/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_weight_quantized",
        tensor_parallel_size=TP,
        enforce_eager=True,
        max_model_len=MAX_LEN,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_num_seqs=1,
        gpu_memory_utilization=0.80,
        max_num_batched_tokens=MAX_LEN // 8,
        attention_backend="FLASH_ATTN_MLA",
    )
    counter = 0
    for item in tqdm.tqdm(ds):
        try:
            prompt = json.loads(item["prompt"])
        except json.JSONDecodeError:
            prompt = [{"role": "user", "content": item["prompt"]}]
        len_prompt = len(
            tokenizer.encode_chat_completion(
                ChatCompletionRequest(messages=prompt)
            ).tokens
        )
        if len_prompt > MAX_LEN - 1:
            print(f"skip {counter}")
            counter += 1
            continue

        print(f"{counter=} {len_prompt=}")
        outputs = llm.chat(prompt, sampling_params)
        output = outputs[0].outputs[0].text.strip()
        print(f"Generated {counter}: {output}\n")
        counter += 1

    save_shards = partial(save_shards_if_rank0, dir=SAVE_ACTI)

    # Save updated model
    llm.apply_model(save_shards)
    merge_checkpoints(
        main_checkpoint_dir=Path(ORIGIN_CKPT),
        shards_checkpoint_dir=Path(SAVE_ACTI),
        output_dir=Path(MERGE_CKPT),
    )
