from datasets import load_dataset

def load_and_preprocess_dataset(dataset_name, train_split="train", test_size=0.1, chat_template=True, **kwargs):
    """
    Load and preprocess a dataset from Hugging Face.
    """
    dataset = load_dataset(dataset_name, **kwargs)
    
    if "test" not in dataset and test_size > 0:
        dataset = dataset[train_split].train_test_split(test_size=test_size)
    
    return dataset

def format_dataset_for_mlx(dataset, tokenizer, prompt_col="instruction", completion_col="output"):
    """
    Format dataset for MLX training (JSONL-like dictionary format).
    """
    # MLX expects 'text' or 'messages'
    def format_row(example):
        return {"text": f"Instruction: {example[prompt_col]}\nResponse: {example[completion_col]}"}
        
    return dataset.map(format_row)
