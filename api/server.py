from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from mlx_lm import load, generate
import os
import yaml

app = FastAPI(title="LLM Mac Fine-Tune API")

# Global state for model and tokenizer
model = None
tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temp: float = 0.7

@app.on_event("startup")
def load_model_on_startup():
    global model, tokenizer
    config_path = "configs/model.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_path = config["base_model"]
        print(f"Loading model: {model_path}")
        model, tokenizer = load(model_path)
    else:
        print("Config file not found. Model not loaded.")

@app.post("/generate")
def api_generate(request: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    response = generate(
        model,
        tokenizer,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temp=request.temp
    )
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
