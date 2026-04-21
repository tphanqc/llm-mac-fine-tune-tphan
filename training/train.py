import argparse
import yaml
import os
from mlx_lm.lora import run as mlx_lora_run, build_parser
from data.datasets import load_and_preprocess_dataset
import tempfile
import json

def run_mlx_training(config, training_config, data_config):
    """
    Run training using the MLX engine.
    """
    dataset = load_and_preprocess_dataset(data_config["dataset_name"])
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        def save_to_jsonl(ds, filename):
            path = os.path.join(tmp_dir, filename)
            with open(path, "w") as f:
                for item in ds:
                    text = f"Instruction: {item[data_config['prompt_column']]}\nResponse: {item[data_config['completion_column']]}"
                    f.write(json.dumps({"text": text}) + "\n")
            return path

        save_to_jsonl(dataset["train"], "train.jsonl")
        save_to_jsonl(dataset["test"], "valid.jsonl")
        
        print(f"Starting MLX training with model: {config['base_model']}")
        
        # Use the official parser to get defaults
        parser = build_parser()
        # Create a list of arguments to parse
        cmd_args = [
            "--model", config["base_model"],
            "--train",
            "--data", tmp_dir,
            "--batch-size", str(training_config["batch_size"]),
            "--iters", str(training_config["iters"]),
            "--learning-rate", str(training_config["learning_rate"]),
            "--steps-per-report", str(training_config["steps_per_report"]),
            "--steps-per-eval", str(training_config["steps_per_eval"]),
            "--adapter-path", training_config["adapter_path"],
            "--save-every", str(training_config["save_every"]),
            "--max-seq-length", str(data_config.get("max_seq_length", 2048)),
            "--seed", "42",
            "--num-layers", "16" # Specific to Llama-3.2-1B
        ]
        
        if training_config.get("resume_adapter_file"):
            cmd_args.extend(["--resume-adapter-file", training_config["resume_adapter_file"]])
            
        args = parser.parse_args(cmd_args)
        
        # Manually set all missing attributes that train_model might expect
        defaults = {
            "grad_accumulation_steps": 1,
            "optimizer": "adam",
            "fine_tune_type": "lora",
            "lr_schedule": None,
            "mask_prompt": False,
            "val_batches": 25,
            "test": False,
            "test_batches": 500,
            "grad_checkpoint": False,
            "clear_cache_threshold": 0,
            "report_to": None,
            "project_name": None,
            "optimizer_config": {}
        }
        for k, v in defaults.items():
            if not hasattr(args, k) or getattr(args, k) is None:
                setattr(args, k, v)
            
        # Override lora_parameters in the args namespace
        lora_params = training_config.get("lora_parameters", {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.0,
            "target_modules": ["q_proj", "v_proj"]
        })
        if "scale" not in lora_params:
            lora_params["scale"] = lora_params.get("alpha", 16) / lora_params.get("rank", 8)
            
        args.lora_parameters = lora_params
        
        mlx_lora_run(args)

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

if __name__ == "__main__":
    main()
