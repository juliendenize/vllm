# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from save_shards import save_shards_if_rank0

if __name__ == "__main__":
    import json

    import tqdm
    from datasets import load_dataset
    from mistral_common.protocol.instruct.request import ChatCompletionRequest
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    from vllm import LLM, SamplingParams

    tokenizer = MistralTokenizer.from_file(
        "/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_bf16/tekken.json"
    )

    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_ID_LONG = "openai/mrcr"
    DATASET_SPLIT = "train_sft"

    # Select number of samples
    NUM_CALIBRATION_SAMPLES = 100
    # NUM_CALIBRATION_SAMPLES_LONG = 5

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    # ds_long = load_dataset(
    #     DATASET_ID_LONG, split=f"train[:{NUM_CALIBRATION_SAMPLES_LONG}]"
    # )
    # ds = concatenate_datasets([ds, ds_long])
    ds = ds.shuffle(seed=42)

    # Create a sampling params object for greedy sampling
    sampling_params = SamplingParams(
        temperature=0.80, top_p=0.95, max_tokens=1, min_tokens=1
    )
    llm = LLM(
        model="/mnt/vast/shared/julien.denize/MS4_release/MS4_candidate_weight_quantized",
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=1024,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        # max_num_seqs=1,
        gpu_memory_utilization=0.90,
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
        print(f"Prompt {counter}: {len_prompt}")
        if len_prompt > 1023:
            print(f"skip {counter}")
            counter += 1
            continue
        outputs = llm.chat(prompt, sampling_params)
        output = outputs[0].outputs[0].text.strip()
        print(f"Generated {counter}: {output}\n")
        counter += 1

    # Save updated model
    llm.apply_model(save_shards_if_rank0)
