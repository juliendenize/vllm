
from save_shards import save_shards_if_rank0

if __name__ == '__main__':
    from vllm import LLMEngine, SamplingParams, EngineArgs
    import torch
    from datasets import load_dataset


    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Select number of samples
    NUM_CALIBRATION_SAMPLES = 100

    ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    # Create a sampling params object for greedy sampling
    sampling_params = SamplingParams(temperature=0.80, top_p=0.95, max_tokens=100, min_tokens=1)
    engine_args = EngineArgs(
        model="/raid/engine/dsikka/models--mistralai--Mistral-Small-4-119B-2602-NVFP4-PTQ/snapshots/f87462a4e1a5b5104cd89a9e3cc456392b53e0a5",
        attention_backend="TRITON_MLA",
        enforce_eager=True
    )
    llm = LLMEngine.from_engine_args(engine_args)
    counter = 0
    for item in ds:
        llm.add_request(
            request_id=str(counter),
            prompt=item["prompt"],
            params=sampling_params
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
                print(f"Generated: {generated_text}")

    # Save updated model
    llm.apply_model(save_shards_if_rank0)
