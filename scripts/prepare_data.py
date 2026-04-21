import argparse
import yaml
import os
from data.datasets import load_and_preprocess_dataset
import json

def main():
    parser = argparse.ArgumentParser(description="LLM Mac Data Preparator")
    parser.add_argument("--data_config", type=str, default="configs/data.yaml", help="Path to data config")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    with open(args.data_config, "r") as f:
        data_config = yaml.safe_load(f)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading dataset: {data_config['dataset_name']}")
    dataset = load_and_preprocess_dataset(data_config["dataset_name"])
    
    def save_to_jsonl(ds, split_name):
        path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        print(f"Saving {split_name} split to {path}")
        with open(path, "w") as f:
            for item in ds:
                text = f"Instruction: {item[data_config['prompt_column']]}\nResponse: {item[data_config['completion_column']]}"
                f.write(json.dumps({"text": text}) + "\n")

    save_to_jsonl(dataset["train"], "train")
    save_to_jsonl(dataset["test"], "valid")
    
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
