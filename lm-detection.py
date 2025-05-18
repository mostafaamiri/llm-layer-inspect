from typing import Union
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import JSONResponse, HTMLResponse
import torch
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel




class GenerationConfig(BaseModel):
    text: str
    k: int
    steps: int = 1

class ChatTemplateConfig(BaseModel):
    user_text: str
    system_text: str

class ModelConfig(BaseModel):
    name: str

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
app = FastAPI()

models = {
    "Llama3.1_8B_Instruct":"/home/mostafa/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/8c22764a7e3675c50d4c7c9a4edb474456022b16/",
    "Aya-expense-8B":"/home/mostafa/.cache/huggingface/hub/models--CohereForAI--aya-expanse-8b/snapshots/e46040a1bebe4f32f4d2f04b0a5b3af2c523d11b",
    "Qwen2.5-7B": "/home/mostafa/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
    "cooking-trans": "/home/mostafa/Matina_AI_assistant/models/COOKING_SFTIII_MERGED",
    "cooking": "/home/mostafa/Matina_AI_assistant/models/MATINA_COOKING_MERGED"
}
model_id = models["Llama3.1_8B_Instruct"]
app.tokenizer = AutoTokenizer.from_pretrained(model_id)
app.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def read_root():
    with open("llm-detection.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content, status_code=status.HTTP_200_OK)

@app.get("/models")
def get_models():
    return JSONResponse({"models": [k for k in models]}, status_code=status.HTTP_200_OK)

@app.post("/change_model")
def change_model(model_config: ModelConfig):
    
    del(app.model)
    del(app.tokenizer)
    torch.cuda.empty_cache()
    model_id = models[model_config.name]
    app.tokenizer = AutoTokenizer.from_pretrained(model_id)
    app.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    return JSONResponse({"result": "model eas changed"}, status_code=status.HTTP_200_OK)

@app.post("/set_chat_template")
def set_chat_template(chat_template_config: ChatTemplateConfig):
    messages = [
        {"role": "system", "content": chat_template_config.system_text},
        {"role": "user", "content": chat_template_config.user_text}
    ]
    return JSONResponse({"text": app.tokenizer.apply_chat_template(messages, tokenize=False)}, status_code=status.HTTP_200_OK)


@app.post("/get_next_token", summary="SUPPLIER")
def get_next_token(gen_conf: GenerationConfig):    
    prompt = gen_conf.text
    app.model.eval()
    app.model.to("cuda")
    response = {
        "next_token":[], 
        "alt": [], 
        "prob":[], 
        "tokens":[],
        "layer_tokens": []}
    for i in range(gen_conf.steps):
        input_ids = app.tokenizer(prompt, return_tensors='pt').to('cuda')
        output = app.model(**input_ids,output_hidden_states=True)
        next_token = app.tokenizer.decode(torch.argmax(output.logits[:,-1,:]))
        prompt += next_token
        layer_tokens = []
        for l in range(len(app.model.model.layers)):
            layer_output = app.model.lm_head(app.model.model.norm(output.hidden_states[l]))
            layer_tokens.append(app.tokenizer.decode(torch.argmax(layer_output[:,-1,:])))
        response["next_token"].append(next_token)
        response["alt"].append([app.tokenizer.decode(k.item()) for k in torch.topk(output.logits[:,-1,:], k=gen_conf.k).indices[0].to("cpu")])
        response["prob"].append([k.item() for k in torch.topk(torch.softmax(output.logits[:,-1,:], dim=-1), k=gen_conf.k).values[0].to("cpu")])
        response["tokens"].append(app.tokenizer(prompt)['input_ids'])
        response["layer_tokens"].append(layer_tokens)
    return JSONResponse(response, status_code=status.HTTP_200_OK)

if __name__ == "__main__":
    
    uvicorn.run("lm-detection:app", host="0.0.0.0", port=5000, log_level="info", reload=True)