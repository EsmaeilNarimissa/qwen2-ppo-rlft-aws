# Qwen2.5 Reinforcement Learning Fine-Tuning (RLFT) with PPO on SageMaker

## Overview

This repository presents *Qwen2.5 Reinforcement Learning Fine-Tuning (RLFT) with PPO on SageMaker*, a complete and modular pipeline for aligning the Qwen/Qwen2.5-0.5B-Instruct language model with task-specific reward signals. The model is trained using **Proximal Policy Optimization (PPO)** and **Low-Rank Adaptation (LoRA)**, and fine-tuned to generate mathematically correct equations from a list of given numbers, following a structured format:

`<think>...</think><answer>...</answer>`

The project is built to run entirely on **AWS SageMaker**, taking advantage of scalable infrastructure, optional Spot Instance pricing, and Docker-based environment management.

**Key Features**

-   **Model**: Qwen/Qwen2.5-0.5B-Instruct
-   **Dataset**: `predibase/countdown`, a structured math reasoning dataset
-   **Fine-Tuning Strategy**: Reinforcement Learning via PPO using the `trl` library
-   **Efficiency**: LoRA-based adapter tuning through the `peft` library for minimal GPU overhead
-   **Reward Mechanism**: Custom format and math evaluation functions that provide feedback on both structure and correctness
-   **Platform**: AWS SageMaker with support for PyTorch estimators and managed compute
-   **Environment**: Self-contained Docker image defined via `Dockerfile` for portable deployment and reproducibility


## File Structure

*   `train.py`: The main script for PPO training with LoRA. Parses arguments from SageMaker and runs the training loop.
*   `evaluate.py`: Script to evaluate the fine-tuned model's performance on the test set using the defined reward metrics.
*   `sagemaker_PPO_training_eval.ipynb`: Jupyter Notebook to configure hyperparameters, set up the SageMaker session, define the estimator, and launch the training/evaluation jobs.
*   `requirements.txt`: Lists all necessary Python dependencies.
*   `Dockerfile`: Defines the container environment for running the training script on SageMaker.

## Setup

1.  **Prerequisites:**
    *   AWS Account with appropriate SageMaker permissions.
    *   AWS CLI configured locally (if running the notebook locally to launch SageMaker jobs).
    *   Docker installed (if building the container image locally).
2.  **Dependencies:** The required Python packages are listed in `requirements.txt`. These will be installed within the Docker container or SageMaker environment.

## Usage

The primary entry point for setting up and launching the training process is the `sagemaker_PPO_training_eval.ipynb` notebook. Follow the steps within the notebook to:
1.  Configure S3 bucket paths.
2.  Define hyperparameters (ensure batch size constraints for PPO are met).
3.  Set up the SageMaker Estimator (using either a pre-built SageMaker image or the custom Docker image).
4.  Launch the training job using `estimator.fit()`.
5.  (Optional) Adapt the notebook to run evaluation using `evaluate.py`.

## Script Details

### `train.py`: PPO Fine-tuning

This script orchestrates the fine-tuning process using Proximal Policy Optimization (PPO). Here's a breakdown of the algorithm and process:

1.  **Initialization:**
    *   Loads the base language model (e.g., `Qwen/Qwen2.5-0.5B-Instruct`) and tokenizer.
    *   Applies LoRA (Low-Rank Adaptation) configuration to the model using `peft` for efficient training. This involves adding small, trainable adapter layers instead of retraining all model parameters.
    *   Loads the `predibase/countdown` dataset.
2.  **PPO Setup:**
    *   Initializes the `PPOTrainer` from the `trl` library. This trainer manages the PPO algorithm steps.
    *   Requires specific hyperparameters (`batch_size`, `mini_batch_size`, `gradient_accumulation_steps`) that must satisfy the constraint: `batch_size` must be divisible by (`mini_batch_size` * `gradient_accumulation_steps`).
3.  **Training Loop:**
    *   Iterates through the dataset in batches.
    *   For each batch of prompts (`query_tensor`), the model generates responses (`response_tensor`).
    *   **Reward Calculation:** The generated prompt-response pairs are evaluated using custom reward functions:
        *   `format_reward_func`: Checks if the output contains the required `<think>` and `<answer>` tags.
        *   `equation_reward_func`: Parses the equation from the `<answer>` tag and checks if it evaluates correctly using the numbers from the prompt.
        *   These individual rewards are combined to produce a final `reward` tensor for the batch.
    *   **PPO Step:** The `ppo_trainer.step` function is called with the prompts, responses, and rewards. This function performs the core PPO update, calculating policy losses and value losses and updating the model (specifically the LoRA adapters) to maximize the expected reward.
    *   Logging: Metrics (like mean reward) are logged using TensorBoard and CSV.
4.  **Model Saving:** After training, the final LoRA adapter weights are saved.

### `evaluate.py`: Model Evaluation

This script evaluates the performance of the fine-tuned model (base model + trained LoRA adapter) on the test split of the dataset:

1.  **Model Loading:**
    *   Loads the base language model specified during training.
    *   Loads the saved LoRA adapter weights from the training output path and merges them into the base model.
2.  **Dataset Loading:** Loads the test split of the `predibase/countdown` dataset.
3.  **Inference:**
    *   Iterates through the test dataset.
    *   For each example, it formats the prompt and uses the model's `generate` function to produce a completion (response).
4.  **Scoring:**
    *   Applies the *same* `format_reward_func` and `equation_reward_func` used during training to score each generated response against the prompt and the ground truth example.
5.  **Metrics Calculation:** Calculates overall performance metrics based on the collected scores (e.g., average format reward, average equation reward, overall success rate).
6.  **Output:** Saves the detailed results, including individual scores and generated examples, to a JSON file in the specified output directory.

### `sagemaker_PPO_training_eval.ipynb`: SageMaker Orchestration

This Jupyter Notebook serves as the command center for managing the training and potentially evaluation processes on AWS SageMaker.

1.  **Initialization:**
    *   Imports necessary libraries (`sagemaker`, `boto3`, etc.).
    *   Initializes the SageMaker session and retrieves the default execution role and AWS region.
    *   Defines S3 bucket paths for storing training data, model artifacts, and outputs.
2.  **Hyperparameter Configuration:**
    *   Sets up a dictionary containing all hyperparameters required by the `train.py` script. This includes:
        *   `base-model`: The Hugging Face model identifier.
        *   `dataset`: The dataset identifier.
        *   PPO-specific parameters (`ppo_epochs`, `batch_size`, `mini_batch_size`, `gradient_accumulation_steps`, `learning_rate`, etc.).
    *   **Crucially, this section includes logic or comments emphasizing the need to ensure the PPO batch size constraint (`batch_size` divisible by `mini_batch_size * gradient_accumulation_steps`) is met.**
    *   *Final Training Run Example Hyperparameters:* 
        *   `base-model`: `Qwen/Qwen2.5-0.5B-Instruct`
        *   `lora-r`: 16, `lora-alpha`: 32
        *   `learning-rate`: 1.41e-5
        *   `batch-size`: 16, `mini-batch-size`: 4, `gradient-accumulation-steps`: 2 (Satisfies constraint)
3.  **SageMaker Estimator Setup:**
    *   Defines a SageMaker `PyTorch` estimator. This object configures the training job environment. Key configurations include:
        *   `entry_point`: Set to `train.py`.
        *   `role`: The IAM role SageMaker will assume.
        *   `instance_count`: Number of machines for training (likely 1 for this setup).
        *   `instance_type`: The EC2 instance type to use (e.g., `ml.g4dn.xlarge`, `ml.g5.xlarge`). The choice depends on model size, budget, and required GPU memory.
        *   `image_uri` (Optional): If using the custom `Dockerfile`, this specifies the ECR path to the built image. Alternatively, a framework version (`pytorch_version`, `transformers_version`) can be specified to use a pre-built SageMaker image.
        *   `hyperparameters`: Passes the defined dictionary to the training script.
        *   `use_spot_instances`: Set to `True` to leverage lower-cost Spot Instances for training, significantly reducing cost. A `max_wait` time is typically specified.
        *   `keep_alive_period_in_seconds`: Helps prevent job termination due to inactivity during spot instance interruptions.
    *   *Final Training Run Example Estimator Config:* 
        *   `instance_type`: `ml.g5.2xlarge` 
        *   Used SageMaker pre-built image: `framework_version='2.4.0'`, `py_version='py311'`
        *   `use_spot_instances`: `False` (but recommended to set to `True` with a `max_wait` for cost savings)
        *   Checkpointing enabled to S3.
4.  **Training Job Launch:**
    *   Calls the `estimator.fit()` method, passing the S3 input data paths. This submits the configuration to SageMaker, which provisions the specified instance(s), downloads the data, runs the `train.py` script within the defined environment (container), and uploads the resulting model artifacts to S3.
5.  **Evaluation & Analysis (Post-Training):**
    *   The notebook contains steps to run model evaluation, typically by:
        *   Creating a SageMaker Processing job that executes the `evaluate.py` script. This job takes the trained model artifacts (from S3) as input and runs inference on the test dataset.
    *   It also includes code to:
        *   Download the evaluation results (the JSON file produced by `evaluate.py`) from S3.
        *   Load and analyze these results (e.g., calculate overall metrics, view example outputs).
        *   Potentially visualize the evaluation metrics (e.g., histograms of reward scores).
6.  **Training Visualization (Post-Training):**
    *   Includes code to download the training metrics logs (e.g., TensorBoard logs or CSV files) generated by `train.py` during the SageMaker job.
    *   Uses libraries like `matplotlib` or `pandas` to plot the PPO reward curves (e.g., `env/reward_mean`) over the training steps, helping to visualize learning progress.
7.  **Deployment (Optional):**
    *   Provides example code demonstrating how to deploy the fine-tuned model (base model + LoRA adapter) to a SageMaker real-time inference endpoint using the `estimator.deploy()` method. This makes the model available for live predictions.
