import sys
sys.path.append('..')
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from agents.graphlearning import *
import dgl

SYSTEM_MESSAGE = "You are a professional security expert."


# ======= Testing Fine-Tuned Model =======
# Import PeftModel for loading LoRA weights\ nfrom peft import PeftModel

# Function to load and prepare fine-tuned model
def load_fine_tuned(base_model_path, lora_dir):
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map='auto', torch_dtype=torch.bfloat16
    )
     # Load LoRA adapter weights
    model_lora = PeftModel.from_pretrained(base, lora_dir)
    # Wrap with graph injection
    model = GraphInjectedModel(model_lora)
    # Move LayerNorm to same device as model_lora parameters
    device = next(model_lora.parameters()).device
    model.norm = model.norm.to(device)
    model.eval()
    return tokenizer, model

# test all    
def test_model_all(graph_path, embed_test_data, model_path ,lora_dir_n, label_data_path):
    # Paths must match training setup
    
    with open(label_data_path, 'r') as f:
        true_labels = [int(line.strip()) for line in f]
    assert len(true_labels) == 944, "标签数量与样本数不匹配"
    
    # Load fine-tuned model
    wahin, metadata = dgl.load_graphs(graph_path)
    tokenizer, model = load_fine_tuned(model_path, lora_dir_n)
    model.eval()

    confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    # Load graph embeddings and select a sample
    embeds = np.load(embed_test_data)

    fp_records = []
    fn_records = []

    for i in range(944):
        idx = i  # test the first example
        key_infor ={"Target": f"ID={idx}","LLM task type": "label inference", "Graph task type": "node classification"} 
        instruct_data = generate_user_instruction(f'ID={idx}','',key_infor, wahin[0])
        graph_vec = torch.tensor(embeds[idx], dtype=torch.float32)
        # instruction = dataset.records[idx]['instruction']
        target, _ = get_node_name_from_string(wahin[0], "Hacker", key_infor['Target'])
        instruction = format_instruction(target, instruct_data)
        # Build prefix
        prefix = (
            f"<s><|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}<|im_end|>\n"
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
            max_new_tokens=50,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = re.sub(r'\(name:[^)]+\)', '', response).replace('  ', ' ')
        print("[Test] Instruction:", instruct_data)
        print("[Test] Graph Token:", embeds[idx])
        print("[Test] Response:", response)

        # Determine predicted label
        pred_label = 1 if "hacker group" in response.lower() else 0
        true_label = true_labels[idx]
        
        # Update confusion matrix
        if true_label == 1 and pred_label == 1:
            confusion['TP'] += 1
        elif true_label == 0 and pred_label == 1:
            confusion['FP'] += 1
            fp_records.append({
                'instruction': instruct_data,
                'response': response,
                'true_label': true_label,
                'pred_label': pred_label
            })
        elif true_label == 0 and pred_label == 0:
            confusion['TN'] += 1
        else:
            confusion['FN'] += 1
            fn_records.append({
                'instruction': instruct_data,
                'response': response,
                'true_label': true_label,
                'pred_label': pred_label
            })

    # Calculate and print metrics
    accuracy = (confusion['TP'] + confusion['TN']) / 944
    precision = confusion['TP'] / (confusion['TP'] + confusion['FP']) if (confusion['TP'] + confusion['FP']) > 0 else 0
    recall = confusion['TP'] / (confusion['TP'] + confusion['FN']) if (confusion['TP'] + confusion['FN']) > 0 else 0
    
    print("Confusion Matrix:")
    print(f"True Positives (TP): {confusion['TP']}")
    print(f"False Positives (FP): {confusion['FP']}")
    print(f"True Negatives (TN): {confusion['TN']}")
    print(f"False Negatives (FN): {confusion['FN']}")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # import json
    # with open('./output/fp_records.json', 'w') as f:
    #     json.dump(fp_records, f, indent=2, ensure_ascii=False)
    # with open('./output/fn_records.json', 'w') as f:
    #     json.dump(fn_records, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # Paths
    GRAPH_TEST = '../dataset/hacked/graph_hacker_com.bin'
    EMBED_NPY_TEST = './Hacker_processed_test.npy'
    MODEL_PATH = '../model/Qwen/Qwen3-8B'
    LORA_DIR = '../model/Lora/Qwen3_wahin_lora'
    LABEL_PATH = './labels.txt'
    test_model_all(GRAPH_TEST, EMBED_NPY_TEST, MODEL_PATH, LORA_DIR, LABEL_PATH)