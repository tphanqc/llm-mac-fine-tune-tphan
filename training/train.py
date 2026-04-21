import argparse
import yaml
import os
from mlx_lm import train as mlx_train
from data.datasets import load_and_preprocess_dataset
import tempfile
import json

def run_mlx_training(config, training_config, data_config):
    """
    Run training using the MLX engine.
    """
    # mlx_lm.train expects data in a directory with train.jsonl, valid.jsonl
    # We'll create a temporary directory for this if not provided
    
    dataset = load_and_preprocess_dataset(data_config["dataset_name"])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        def save_to_jsonl(ds, filename):
            path = os.path.join(tmp_dir, filename)
            with open(path, "w") as f:
                for item in ds:
                    # Basic formatting
                    text = f"Instruction: {item[data_config['prompt_column']]}\nResponse: {item[data_config['completion_column']]}"
                    f.write(json.dumps({"text": text}) + "\n")
            return path

        save_to_jsonl(dataset["train"], "train.jsonl")
        save_to_jsonl(dataset["test"], "valid.jsonl")
        
        print(f"Starting MLX training with model: {config['base_model']}")
        mlx_train(
            model=config["base_model"],
            data=tmp_dir,
            iters=training_config["iters"],
            batch_size=training_config["batch_size"],
            learning_rate=training_config["learning_rate"],
            steps_per_report=training_config["steps_per_report"],
            steps_per_eval=training_config["steps_per_eval"],
            adapter_path=training_config["adapter_path"],
            save_every=training_config["save_every"],
            lora_layers=-1,
            # Pass additional lora params if needed
        )

def main():
    parser = argparse.ArgumentParser(description="LLM Mac Fine-Tune Trainer")
    parser.add_argument("--config", type=str, default="configs/training.yaml", help="Path to training config")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--engine", type=str, choices=["mlx", "hf"], help="Override engine in config")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        training_config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)
        
    engine = args.engine or training_config.get("engine", "mlx")
    
    if engine == "mlx":
        run_mlx_training(model_config, training_config, data_config)
    else:
        print("Hugging Face training engine implementation in progress...")
        # Placeholder for HF Trainer implementation

if __name__ == "__main__":
    main()
