# MLX-LM handles LoRA/QLoRA via its training loop directly.
# This file provides a configuration helper for MLX LoRA.

def get_mlx_lora_config(rank=8, alpha=16, dropout=0.05, target_modules=None):
    """
    Returns a dictionary of LoRA parameters for MLX.
    """
    return {
        "lora_layers": -1, # -1 means all layers
        "lora_parameters": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": target_modules or ["q_proj", "v_proj"]
        }
    }
