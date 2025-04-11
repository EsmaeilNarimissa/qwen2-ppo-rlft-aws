import os
import argparse
import json
import torch
import ast
import regex as re
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--base-model', type=str, default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument('--lora-r', type=int, default=16)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--learning-rate', type=float, default=1.41e-5)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--save-steps', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--mini-batch-size', type=int, default=4)
    return parser.parse_args()


def format_reward_func(prompt: str, completion: str, example: dict) -> float:
    try:
        if "<think>" in completion and "<answer>" in completion:
            return 1.0
    except:
        pass
    return 0.0


def equation_reward_func(prompt: str, completion: str, example: dict) -> float:
    try:
        match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", completion)
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
        if abs(float(result) - float(example["target"])) < 1e-5:
            return 1.0
    except:
        return 0.0
    return 0.0


def main():
    # ✅ Memory & perf config
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    step_logs = []
    log_file_path = os.path.join(args.output_data_dir, "ppo_training_log.csv")
    tensorboard_log_dir = os.path.join(args.output_data_dir, "runs", "ppo_rewards")
    os.makedirs(os.path.dirname(tensorboard_log_dir), exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_log_dir)

    # ✅ Load dataset
    dataset = load_dataset("predibase/countdown")
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # ✅ Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Fix warning for decoder-only models

    # ✅ Compute dtype
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    # ✅ Load base model w/ dtype
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map={"": 0},
        trust_remote_code=True,
        torch_dtype=compute_dtype
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()  # ✅ Reduces memory

    # ✅ LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, peft_config)
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

    # ✅ PPOConfig with stability
    ppo_config = PPOConfig(
        model_name=args.base_model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optimize_cuda_cache=True,
        use_score_scaling=True,
        use_score_norm=True,
    )

    def build_prompt(example):
        return f"<think>Let's try solving this with {example['nums']} for {example['target']}.</think>"

    def tokenize_for_model(example):
        prompt = build_prompt(example)
        encoded = tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    tokenized_data = train_dataset.map(tokenize_for_model, batched=False)
    tokenized_data.set_format(type='torch')

    metadata_list = []
    for example in train_dataset:
        metadata_list.append({
            "prompt_text": build_prompt(example),
            "nums": example["nums"],
            "target": example["target"]
        })

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_data
    )

    generation_kwargs = {
        "max_new_tokens": 20,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    try:
        for epoch, batch in enumerate(ppo_trainer.dataloader):
            if epoch >= args.max_steps:
                break

            query_tensors = batch['input_ids']
            query_tensors_list = [q for q in query_tensors]

            response_tensors = ppo_trainer.generate(
                query_tensors_list,
                return_prompt=False,
                **generation_kwargs
            )

            batch['response'] = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

            rewards = []
            format_rewards_batch = []
            equation_rewards_batch = []
            total_rewards_batch = []

            for i in range(len(query_tensors)):
                original_index = epoch * args.batch_size + i
                if original_index >= len(metadata_list):
                    continue
                sample_metadata = metadata_list[original_index]
                prompt = sample_metadata['prompt_text']
                example_dict = {
                    'nums': sample_metadata['nums'],
                    'target': sample_metadata['target']
                }
                completion = batch['response'][i]

                format_reward = format_reward_func(prompt, completion, example_dict)
                equation_reward = equation_reward_func(prompt, completion, example_dict)
                total_reward = format_reward + equation_reward

                format_rewards_batch.append(format_reward)
                equation_rewards_batch.append(equation_reward)
                total_rewards_batch.append(total_reward)

                rewards.append(torch.tensor(total_reward, device=ppo_trainer.accelerator.device))

                step_logs.append({
                    "step": epoch,
                    "sample_index_in_batch": i,
                    "format_reward": format_reward,
                    "equation_reward": equation_reward,
                    "total_reward": total_reward
                })

            writer.add_scalar("Reward/Format_Mean", torch.mean(torch.tensor(format_rewards_batch)).item(), epoch)
            writer.add_scalar("Reward/Equation_Mean", torch.mean(torch.tensor(equation_rewards_batch)).item(), epoch)
            writer.add_scalar("Reward/Total_Mean", torch.mean(torch.tensor(total_rewards_batch)).item(), epoch)

            stats = ppo_trainer.step(query_tensors_list, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards)

            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.max_steps:
                if step_logs:
                    header = not os.path.exists(log_file_path)
                    pd.DataFrame(step_logs).to_csv(log_file_path, mode='a', index=False, header=header)
                    print(f"[Step {epoch+1}] Saved logs.")
                    step_logs = []

            print(f"[Step {epoch+1}/{args.max_steps}] Mean reward: {torch.mean(torch.stack(rewards)).item():.2f}")

    finally:
        if step_logs:
            header = not os.path.exists(log_file_path)
            pd.DataFrame(step_logs).to_csv(log_file_path, mode='a', index=False, header=header)
            print(f"Final logs saved.")

        writer.close()
        print("TensorBoard writer closed.")

    # ✅ Save reward log into model artifacts
    import shutil
    try:
        shutil.copy(log_file_path, os.path.join(args.model_dir, "ppo_training_log.csv"))
        print("✅ Copied reward log to model_dir for artifact preservation.")
    except Exception as e:
        print(f"❌ Failed to copy reward log: {e}")

    # ✅ Save model & tokenizer
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    print(f"✅ Model saved to {args.model_dir}")


if __name__ == "__main__":
    main()
