# evaluate.py
import argparse
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import regex as re
import ast

def format_reward_func(prompt: str, completion: str, example: dict) -> float:
    try:
        completion_full = "<think>" + completion
        match = re.search(r"<think>.*?</think>\n<answer>.*?</answer>", completion_full, re.DOTALL)
        return 1.0 if match else 0.0
    except Exception as e:
        print(f"[format_reward_func] Error: {e}")
        return 0.0

def equation_reward_func(prompt: str, completion: str, example: dict) -> float:
    try:
        completion_full = "<think>" + completion
        match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", completion_full)
        if not match:
            return 0.0
        equation = match.group(1).strip()
        used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
        target_nums = example["nums"]
        if isinstance(target_nums, str):
            target_nums = ast.literal_eval(target_nums)
        if sorted(used_numbers) != sorted(target_nums):
            return 0.0
        if not re.match(r'^[\d+\-*/().\s]+$', equation):
            return 0.0
        result = eval(equation, {"__builtins__": None}, {})
        return 1.0 if abs(float(result) - float(example["target"])) < 1e-5 else 0.0
    except Exception as e:
        print(f"[equation_reward_func] Error: {e}")
        return 0.0

def format_row(row):
    nums = row["nums"]
    if isinstance(nums, str):
        try:
            nums = ast.literal_eval(nums)
        except:
            nums = []
    return f"""<|im_start|>system
You are a helpful assistant. You first think about the reasoning process step by step and then provide the user with an answer.<|im_end|>
<|im_start|>user
Using the numbers {nums}, create an equation that equals {row['target']}. You can use basic arithmetic operations (+, -, *, /) and parentheses, and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>.<|im_end|>
<|im_start|>assistant
Let me solve this step by step.
<think>"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--base-model', type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument('--output-file', type=str, default="evaluation_results.json")
    parser.add_argument('--num-examples', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-new-tokens', type=int, default=150)
    return parser.parse_args()

def main():
    args = parse_args()

    print("📥 Loading dataset with schema workaround...")
    try:
        # First approach: Try loading with direct download
        from huggingface_hub import hf_hub_download
        import pandas as pd
        import os
        
        # Download the parquet file directly
        file_path = hf_hub_download(repo_id="predibase/countdown", filename="test.parquet", repo_type="dataset")
        df = pd.read_parquet(file_path).head(args.num_examples)
        print(f"Successfully loaded dataset from {file_path}")
    except Exception as e:
        print(f"Direct download failed: {e}")
        # Fallback approach: Use CSV representation to avoid schema issues
        print("Falling back to manual download...")
        import requests
        import io
        import pandas as pd
        
        # Manually download a CSV version if available or create your own small test set
        test_data = [
            {"nums": [1, 2, 3, 4], "target": 24},
            {"nums": [2, 3, 5, 7], "target": 35},
            {"nums": [4, 5, 6, 7], "target": 46},
            {"nums": [3, 4, 7, 9], "target": 50},
            {"nums": [1, 5, 8, 9], "target": 40}
        ]
        df = pd.DataFrame(test_data).head(args.num_examples)

    prompts = [format_row(row) for _, row in df.iterrows()]

    print(f"🚀 Loading model: {args.base_model}")
    import os
    import tempfile
    
    # Create temp directory for offloading if needed
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary offload directory: {temp_dir}")
    
    # Memory optimizations
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=compute_dtype,
        device_map="auto",          # Automatically decide what goes where
        offload_folder=temp_dir,    # Specify offload directory
        trust_remote_code=True,
        load_in_8bit=True if torch.cuda.is_available() else False,  # 8-bit quantization if possible
    )
    model.config.pad_token_id = model.config.eos_token_id
    
    print(f"Loading adapter from: {args.model_dir}")
    try:
        # First approach - standard way
        model = PeftModel.from_pretrained(model, args.model_dir, offload_folder=temp_dir)
    except ValueError as e:
        if "offload_dir" in str(e):
            print("Retrying with explicit offload directory setting...")
            # Try with explicit device map
            model = PeftModel.from_pretrained(
                model, 
                args.model_dir,
                device_map={"base_model": "cpu", "modules_to_save": "cuda" if torch.cuda.is_available() else "cpu"},
                offload_folder=temp_dir
            )
        else:
            raise e
            
    model.eval()
    print("Model loaded successfully")

    print("🔍 Evaluating...")
    results = []
    format_scores = []
    equation_scores = []

    for i, (prompt, row) in enumerate(zip(prompts, df.to_dict(orient="records"))):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        completion = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        format_score = format_reward_func(prompt, completion, row)
        equation_score = equation_reward_func(prompt, completion, row)

        results.append({
            "prompt": prompt,
            "completion": completion,
            "format_score": float(format_score),
            "equation_score": float(equation_score)
        })
        format_scores.append(format_score)
        equation_scores.append(equation_score)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} examples")

    metrics = {
        "avg_format_score": sum(format_scores) / len(format_scores),
        "avg_equation_score": sum(equation_scores) / len(equation_scores),
        "combined_score": (sum(format_scores) + sum(equation_scores)) / len(format_scores),
        "num_examples": len(format_scores)
    }

    with open(args.output_file, "w") as f:
        json.dump({"metrics": metrics, "examples": results}, f, indent=2)

    print(f"\n✅ Evaluation complete. Results saved to {args.output_file}")
    print(f"📊 Format Score:   {metrics['avg_format_score']:.4f}")
    print(f"🧮 Equation Score: {metrics['avg_equation_score']:.4f}")
    print(f"🔗 Combined Score: {metrics['combined_score']:.4f}")

if __name__ == "__main__":
    main()
