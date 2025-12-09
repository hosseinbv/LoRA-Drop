# LoRA-Drop (Based on LLaMA-Factory)

This repository implements **LoRA-Drop** on top of the **LLaMA-Factory** training framework.  
All training, configuration, and launching mechanisms directly follow the LLaMA-Factory pipeline.

This repo is **specifically built and tested for Qwen1.5-7B and Qwen-family architectures** with minimal adaptation required for other HuggingFace causal language models.

---

## 1. Installation

### 1.1 Create and Activate Environment

```bash
conda create -n lora-drop python=3.10 -y
conda activate lora-drop
pip install -r requirements.txt
pip install accelerate
accelerate config

```
# 2. Dataset Preparation

Prepare your dataset in HuggingFace format (JSON, JSONL, or Arrow).
Then register it in:

```bash
data/dataset_info.json
```
Example:
```bash
{
  "my_dataset": {
    "file_name": "my_dataset.json",
    "format": "alpaca"
  }
}

```
# 3. Model Preparation (HuggingFace Compatibility)
This repository is built for Qwen1.5-7B and similar Qwen-family models.

Step 1: Download the Model
```bash
huggingface-cli download Qwen/Qwen1.5-7B-Chat --local-dir ./models/Qwen1.5-7B-Chat

```
Step 2: Modify the Model Configuration

Open:
```bash
models/Qwen1.5-7B-Chat/config.json

```
change: 
```bash
"architectures": ["Qwen2ForCausalLM"],
"model_type": "qwen2"

```
To:
```bash
"architectures": ["Qwen2_Lora_DropForCausalLM"],
"model_type": "qwen2_lora_drop"
```
Save the file after modification.

# 4. Training with Accelerate
```bash
accelerate launch src/train.py examples/train_full/LoRA-Drop-Qwen1.5-7B-Chat.yaml
```

