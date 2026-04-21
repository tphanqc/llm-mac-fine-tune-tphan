import argparse
import yaml
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def main():
    parser = argparse.ArgumentParser(description="LLM Mac Fine-Tune Generator")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--model", type=str, help="Path to model (overrides config)")
    parser.add_argument("--adapter", type=str, help="Path to adapter (overrides config)")
    parser.add_argument("--config", type=str, default="configs/inference.yaml", help="Path to inference config")
    parser.add_argument("--model_config", type=str, default="configs/model.yaml", help="Path to model config")
    
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        inference_config = yaml.safe_load(f)
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
        
    model_path = args.model or model_config["base_model"]
    adapter_path = args.adapter or inference_config.get("adapter_path")
    
    print(f"Loading model: {model_path}")
    model, tokenizer = load(model_path, adapter_path=adapter_path)
    
    sampler = make_sampler(
        temp=inference_config.get("temp", 0.7),
        top_p=inference_config.get("top_p", 0.9),
    )
    
    print(f"Generating for prompt: {args.prompt}")
    response = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=inference_config["max_tokens"],
        sampler=sampler,
    )
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
