from .graphlearning import *
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import re


# === 加载微调模型 ===
def load_fine_tuned(base_model_path, lora_adapter_dir):
    # 1) 加载 tokenizer（如果你把 tokenizer 也放到 adapter 目录里，可改为 from_pretrained(lora_adapter_dir)）
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 2) 先加载“干”的基模型（不含 LoRA 权重）
    base = AutoModelForCausalLM.from_pretrained(base_model_path,device_map='auto',torch_dtype=torch.bfloat16)

    # 3) 把之前只保存的 LoRA Adapter 权重加载进来
    model_lora = PeftModel.from_pretrained(base, lora_adapter_dir, device_map='auto', torch_dtype=torch.bfloat16)

    # 4) 包装上图嵌入注入逻辑
    model = GraphInjectedModel(model_lora)
    # 把 LayerNorm 放到正确的 device 上
    device = next(model_lora.parameters()).device
    model.norm = model.norm.to(device)

    model.eval()
    return tokenizer, model

# === 生成微调模型后的回复 ===
def generate_response_onecase(instruct_data, embed_data, llm_tokenizer ,llm_model, target):
     # Load fine-tuned model
    tokenizer = llm_tokenizer
    model = llm_model
    model.eval()
    
    # Load graph embedding
    if isinstance(embed_data, np.ndarray):
        embed_data = torch.from_numpy(embed_data)
    graph_vec = embed_data.to(torch.float32)
    instruct = format_instruction(target, instruct_data)
    SYSTEM_MESSAGE = "You are a professional security expert."
    prefix = (
            f"<s><|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{instruct}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )

    inputs = tokenizer(prefix, return_tensors='pt', add_special_tokens=False)
    input_ids = inputs['input_ids'].to(model.model.device)
    attention_mask = inputs['attention_mask'].to(model.model.device)

    # Prepare embeddings
    inputs_embeds = model.model.get_input_embeddings()(input_ids)
    graph = graph_vec.unsqueeze(0).to(model.norm.weight.dtype).to(model.model.device)
    graph_normed = model.norm(graph)
    graph_scaled = graph_normed * model.scale
    inputs_embeds = inputs_embeds.clone()
    inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + graph_scaled

    # Generate response
    outputs = model.model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=1000,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = re.sub(r'\(name:[^)]+\)', '', response).replace('  ', ' ')
    return response
