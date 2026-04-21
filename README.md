# LLM Mac Fine-Tune

A modular workspace for high-performance Large Language Model (1B-7B) fine-tuning on Apple Silicon (macOS).

## Features
- **Dual-Engine Support:** Choose between **MLX** (native Apple Silicon performance) and **PyTorch/Hugging Face** (ecosystem compatibility).
- **Quantization:** Native 4-bit and 8-bit support via MLX.
- **PEFT/LoRA:** Efficient fine-tuning using Low-Rank Adaptation.
- **Structured Configuration:** Manage models, training, and data through YAML files.

## Prerequisites
- macOS 13.5+
- Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended)

## Quick Start

1. **Setup Environment:**
   ```bash
   cp .env.example .env
   # Edit .env with your HF_TOKEN
   ```

2. **Install Dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Configure Your Experiment:**
   Edit files in `configs/` to specify your model, dataset, and hyperparameters.

4. **Run Training (MLX Example):**
   ```bash
   python training/train.py --engine mlx --config configs/training.yaml
   ```

## Directory Structure
- `configs/`: YAML configuration files.
- `models/`: Model loading abstractions for MLX and HF.
- `training/`: Core training loops and logic.
- `finetuning/`: LoRA and QLoRA configurations.
- `data/`: Data processing and loading scripts.
- `inference/`: Generation and chat utilities.
- `evaluation/`: Benchmarking and metric calculations.
