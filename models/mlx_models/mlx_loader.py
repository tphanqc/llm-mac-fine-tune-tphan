from mlx_lm import load
from ..base.registry import registry

@registry.register("mlx")
class MLXLoader:
    @staticmethod
    def load_model(model_path, adapter_path=None, **kwargs):
        """
        Load an MLX model and tokenizer.
        """
        model, tokenizer = load(model_path, adapter_path=adapter_path, **kwargs)
        return model, tokenizer
