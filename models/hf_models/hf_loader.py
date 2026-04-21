import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..base.registry import registry

@registry.register("hf")
class HFLoader:
    @staticmethod
    def load_model(model_path, device="mps", trust_remote_code=True, **kwargs):
        """
        Load a Hugging Face model and tokenizer on MPS.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        
        # Note: bitsandbytes quantization not natively supported on MPS
        # This loader assumes standard float16/float32 loading on MPS
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            **kwargs
        )
        return model, tokenizer
