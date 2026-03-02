# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from save_shards import save_shards_if_rank0

if __name__ == "__main__":
    from datasets import load_dataset

    from vllm import EngineArgs, LLMEngine, SamplingParams

    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Select number of samples
    NUM_CALIBRATION_SAMPLES = 10

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    # Create a sampling params object for greedy sampling
    sampling_params = SamplingParams(
        temperature=0.80, top_p=0.95, max_tokens=1, min_tokens=1
    )
    engine_args = EngineArgs(
        model="/mnt/vast/shared/julien.denize/MS4-conversion/mistral-small-4-nvfp-4",
        tensor_parallel_size=8,
        enforce_eager=True,
        max_model_len=262144,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        attention_backend="FLASH_ATTN_MLA",
    )
    llm = LLMEngine.from_engine_args(engine_args)
    counter = 0
    for item in ds:
        llm.add_request(
            request_id=str(counter), prompt=item["prompt"], params=sampling_params
        )
        counter += 1

    # Process requests
    outputs = {}
    step_count = 0
    while llm.has_unfinished_requests():
        step_outputs = llm.step()
        step_count += 1
        for output in step_outputs:
            if output.finished:
                outputs[output.request_id] = output

                generated_text = output.outputs[0].text
                print(f"Generated: {generated_text}\n")

    # Save updated model
    llm.apply_model(save_shards_if_rank0)
