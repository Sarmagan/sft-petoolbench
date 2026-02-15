import os
import re
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from datasets import load_dataset
from huggingface_hub import login

# ── Configuration ──────────────────────────────────────────────────────────────
INSTRUCT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FINETUNED_ADAPTER = "SArmagan/petoolbench-2026-02-15_05.18.28"
REVISION="c56c84ad7d688adbc41e9d7170c78c5eadf2eba3"

capability = torch.cuda.get_device_capability()
use_bf16 = capability[0] >= 8

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_quantized_model(model_name: str):
    """Load a model with 4-bit quantization + tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"  # left-pad for batched generation

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant_config, device_map="auto"
    )
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model, tokenizer

def evaluate_model(model, tokenizer, test_path="user_entries_test_r.json", max_new_tokens=256, batch_size=1):
    with open(test_path, "r") as f:
        data = json.load(f)

    # data = data[:100]
    results = []

    for batch_start in tqdm(range(0, len(data), batch_size)):
        batch = data[batch_start:batch_start + batch_size]

        prompts = []
        for entry in batch:
            messages = [
                {"role": "system", "content": entry["instruction_ratings"]},
                {"role": "user", "content": entry["query"]},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        for i, entry in enumerate(batch):
            prompt_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum()
            response = tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True)

            results.append({
                "entry_id": batch_start + i,
                "response": response,
                "api_call_ground_truth": entry["api_call_ground_truth"],
            })

    calculate_scores(results)
    # return results


def calculate_scores(results):
    scores_tool = 0
    scores_parameters = 0
    total = len(results)

    for entry in results:
        if not entry["response"]:
            continue

        pattern = r'\{.*"tool_name":\s*".*?",\s*"parameters":\s*\{.*?\}\}'
        match = re.search(pattern, entry["response"], re.DOTALL)

        if match:
            try:
                response = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"Entry {entry['entry_id']}: invalid JSON")
                continue
        else:
            print(f"Entry {entry['entry_id']}: no valid JSON found")
            continue

        if entry["api_call_ground_truth"]["tool_name"] == response["tool_name"]:
            scores_tool += 1
        if entry["api_call_ground_truth"]["parameters"] == response["parameters"]:
            scores_parameters += 1

    print(f"Tool accuracy: {scores_tool}/{total} ({scores_tool/total:.2%})")
    print(f"Parameters accuracy: {scores_parameters}/{total} ({scores_parameters/total:.2%})")


# ── Usage ──────────────────────────────────────────────
print("Loading fine-tuned model (base + LoRA adapter) …")
ft_model, ft_tokenizer = load_quantized_model(INSTRUCT_MODEL)
ft_model = PeftModel.from_pretrained(ft_model, FINETUNED_ADAPTER, revision=REVISION)
ft_model.eval()

evaluate_model(ft_model, ft_tokenizer, test_path="user_entries_test_r.json")

# Evaluate base instruct model
print("\nLoading base instruct model …")
base_model, base_tokenizer = load_quantized_model(INSTRUCT_MODEL)

print("\n=== Base Instruct Model ===")
evaluate_model(base_model, base_tokenizer, test_path="user_entries_test_r.json")