from peft import LoraConfig, get_peft_model, TaskType

def apply_lora_hf(model, rank=8, alpha=16, dropout=0.05, target_modules=None):
    """
    Apply LoRA to a Hugging Face model using PEFT.
    """
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    return model, peft_config
