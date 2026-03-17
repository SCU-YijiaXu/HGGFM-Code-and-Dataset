from ast import Str
from dgl.ops import sddmm
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import os
import numpy as np
from dgl.nn.pytorch import GraphConv
import re
from datasets import Dataset
from transformers import TrainerCallback
import json

# 定义MetaPathEncoder
class MetaPathEncoder(nn.Module):
    def __init__(self, g, metapaths, in_dim, out_dim):
        super().__init__()
        self.metapaths = metapaths
        self.g_list = [dgl.metapath_reachable_graph(g, mp) for mp in metapaths]
        self.conv_layers = nn.ModuleList([
            GraphConv(in_dim, out_dim) for _ in self.g_list
        ])

    def forward(self, features_dict):
        outputs_by_type = {ntype: [] for ntype in features_dict}
        for g_sub, conv in zip(self.g_list, self.conv_layers):
            ntype = g_sub.ntypes[0]
            input_feat = features_dict[ntype]
            g_sub = g_sub.to(input_feat.device)  # ✅ 确保图也在 GPU 或 CPU 上
            h = conv(g_sub, input_feat)
            outputs_by_type[ntype].append(h)

        # 对每种类型节点进行平均池化
        final_embeds = {
            ntype: torch.mean(torch.stack(embs), dim=0)
            for ntype, embs in outputs_by_type.items() if embs
        }
        return final_embeds

# 定义DGI模型
class DGI(nn.Module):
    def __init__(self, encoder, ntypes):
        super().__init__()
        self.encoder = encoder
        self.ntypes = ntypes
        self.readout = lambda h_dict: {k: h.mean(dim=0) for k, h in h_dict.items()}
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, features_dict):
        pos = self.encoder(features_dict)
        neg_input = {k: v[torch.randperm(v.size(0))] for k, v in features_dict.items()}
        neg = self.encoder(neg_input)
        summary = self.readout(pos)

        loss = 0
        for ntype in self.ntypes:
            if ntype in pos:
                p, n = pos[ntype], neg[ntype]
                s = summary[ntype]
                pos_score = (p * s).sum(dim=1)
                neg_score = (n * s).sum(dim=1)
                scores = torch.cat([pos_score, neg_score])
                labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
                loss += self.bce(scores, labels)
        return loss, pos

# 定义RGCN模型
class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        self.conv1 = HeteroGraphConv({
            rel: GraphConv(in_feats, hid_feats) for rel in rel_names
        })
        self.conv2 = HeteroGraphConv({
            rel: GraphConv(hid_feats, out_feats) for rel in rel_names
        })

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h

# 获取id和名称的字典
def get_id2name_dict(g, column):
    node_dict = {}
    for node_id in g.nodes(column):
        name = fiter_char(g, column, node_id)
        if name:
            node_dict[str(int(node_id))] = name
    return node_dict

# 过滤字符
def fiter_char(g, ntype, node_id):
    if 'char_feat' not in g.nodes[ntype].data:
        return None
        
    # 获取该节点的字符特征张量
    char_tensor = g.nodes[ntype].data['char_feat']
    node_id = int(node_id)
    # 检查node_id是否有效
    if node_id < 0 or node_id >= char_tensor.shape[0]:
        return None
    
    # 获取该节点的字符ID序列
    char_ids = char_tensor[node_id].tolist()
    # 过滤掉填充的0值并转换为字符
    chars = [chr(cid) for cid in char_ids if cid != 0]
    # 组合成原始字符串
    return ''.join(chars)

# 初始化特征
def init_features(g, in_dim, device):
    features_dict = {}
    linear_map = nn.ModuleDict()  # 如果需要映射，记录映射层（可训练）
    print("\nStart graph self-supervised learning:")
    for ntype in g.ntypes:
        if 'feat' in g.nodes[ntype].data:
            feat = g.nodes[ntype].data['feat'].to(device)
            if feat.shape[1] != in_dim:
                # 添加线性映射层使维度匹配
                projector = nn.Linear(feat.shape[1], in_dim).to(device)
                feat = projector(feat.float())
                linear_map[ntype] = projector
                # print(f"✅ Use raw features and map:{ntype} ({feat.shape[1]} → {in_dim})")
            else:
                feat = feat.float()
                print(f"✅ Use the original features:{ntype}")
            features_dict[ntype] = nn.Parameter(feat, requires_grad=True)
        else:
            # 随机初始化
            features_dict[ntype] = nn.Parameter(torch.randn(g.num_nodes(ntype), in_dim, device=device))
            # print(f"⚠️  No original features, using random initialization:{ntype}")

    return features_dict, linear_map  # 第二项可用于模型注册参数

# 从字符串中获取节点名称和ID
def get_node_name_from_string(g, ntype, input_string):
    
    # 检查是否是"Name = xx"格式
    def name_id_match(g, ntype, input_string):
        name_match = re.search(r'Name\s*=\s*(.+)', input_string)
        if name_match:
            name = name_match.group(1).strip()
            # 在图中查找匹配该名称的节点ID
            if 'char_feat' in g.nodes[ntype].data:
                for node_id in range(g.num_nodes(ntype)):
                    char_feat = g.nodes[ntype].data['char_feat'][node_id]
                    original_name = ''.join([chr(int(c)) for c in char_feat if c != 0])
                    if original_name == name:
                        return name, node_id
            return name, None
        
        # 检查是否是"ID=xx"格式
        id_match = re.search(r'ID=(\d+)', input_string)
        if id_match:
            node_id = int(id_match.group(1))
            return fiter_char(g, ntype, node_id), node_id  # 返回节点ID而不是匹配对象
    
        raise ValueError("Invalid input string format, should be 'ID=XX' or 'Name=XX'")

    # 检查是否是复合型
    pattern = r'(?:(ID|Name)=(\d+|[^&]+))&(?:(ID|Name)=(\d+|[^&]+))'
    match = re.search(pattern, input_string)
    if match:
        # 提取两个键值对
        parts = input_string.split('&')
        name1, id1 = name_id_match(g, ntype, parts[0])
        name2, id2 = name_id_match(g, ntype, parts[1])
        return (name1, name2), (id1, id2)

    if input_string=='WAHIN':
        return g, 'WAHIN'

    if input_string=='No Need':
        return 'No Need', 'No Need'

    return name_id_match(g, ntype, input_string)

# 生成图级别的信息查询    
def generate_prompt_fromgraph(input_graph, column="Hacker"):
    from collections import defaultdict

    prompt_lines = []

    feat_ids = input_graph.nodes[column].data["feat_id"].tolist()  # 原始 ID 顺序
    id_to_row = {fid: i for i, fid in enumerate(feat_ids)}     # feat_id → 行号映射
    # 用于记录：{hacker_name: set(neighbor_names)}
    hacker_to_neighbors = defaultdict(set)

    # 遍历所有从 Hacker 出发的边类型
    for srctype, etype, dsttype in input_graph.canonical_etypes:
        if srctype != column:
            continue

        src_ids, dst_ids = input_graph.edges(etype=etype)
        src_ids = src_ids.tolist()
        dst_ids = dst_ids.tolist()

        for src_id, dst_id in zip(src_ids, dst_ids):
            row = id_to_row.get(src_id)
            dst_row = dst_id

            if row is None:
                continue

            hacker_name = fiter_char(input_graph, column, src_id)
            neighbor_name = fiter_char(input_graph, dsttype, dst_id)

            if hacker_name and neighbor_name:
                hacker_to_neighbors[src_id].add(neighbor_name)

    # ✨ 按 ID 顺序遍历输出
    for hacker_id in sorted(feat_ids):
        hacker_name = fiter_char(input_graph, column, hacker_id)
        neighbors = sorted(hacker_to_neighbors.get(hacker_id, []))
        line = f"{hacker_name} : [{', '.join(neighbors)}]"
        prompt_lines.append(line)

    graph_prompt = 'The following is the information of the entire graph, in the format of "hacker name": "associated neighbor node name list".\n'
    graph2words = "\n".join(prompt_lines)

    return graph_prompt+graph2words

# 生成用户指令
def generate_user_instruction(user_input, intelligence_infor, key_infor, g, column="Hacker"):

    if key_infor["LLM task type"] == "open-end chat":
        return user_input

    elif key_infor["LLM task type"] == "label inference":
        # 获取节点名称和ID
        _, node_id = get_node_name_from_string(g, column, key_infor['Target'])
        
        if key_infor["Graph task type"] == "node classification":
            return f"Is {column} node (ID={node_id}) a hacker organization?"
        elif key_infor["Graph task type"] == "edge prediction":
            # 假设target是(src, dst)元组
            _, src_id = get_node_name_from_string(g, column, key_infor['Target'])
            return f"Is there a relationship between {column} nodes (ID={src_id[0]}) and (ID={src_id[1]})? Give th reason no less than 50 words.\n" + "\nThe following is relevant information:\n" + str(intelligence_infor)
        elif key_infor["Graph task type"] == "graph classification":
            return "Is the input graph an APT attack chain graph?" + "\nThe following is relevant information:\n" + str(intelligence_infor)
        else:
            prmopt = user_input + "\nThe following is relevant information:\n" + str(intelligence_infor)
            return prmopt
    else:
        prmopt = user_input + "\nThe following is relevant information:\n" + str(intelligence_infor)
        return prmopt
    
    # 其他任务类型保持原样
    return user_input

# 根据任务类型获取对应的嵌入向量
def get_embedding_by_task_type(task_type, embeddings, target):
    if isinstance(embeddings, np.ndarray):
        # 处理numpy数组格式的嵌入
        if task_type == 'node classification':
            return embeddings[int(target)]
        elif task_type == 'edge prediction':
            src_emb = embeddings[int(target[0])]
            dst_emb = embeddings[int(target[1])]
            return (src_emb + dst_emb) / 2
        elif task_type in {'graph classification', 'graph analysis', 'graph reasoning'}:
            if target=='WAHIN':
                return embeddings.mean(axis=0)
            else:
                hacker_ids = target.nodes['Hacker'].data['feat_id']
                hacker_embeddings = embeddings[hacker_ids] # 从embeddings中获取对应向量
                return hacker_embeddings.mean(axis=0)
    else:
        # 处理字典格式的嵌入
        if task_type == 'node classification':
            return embeddings[int(target)]
        elif task_type == 'edge prediction' and isinstance(target, tuple):
            src_emb = embeddings[int(target[0])]
            dst_emb = embeddings[int(target[1])]
            return (src_emb + dst_emb) / 2
        elif task_type in {'graph classification', 'graph analysis', 'graph reasoning'}:
            if target=='WAHIN':
                node_embeddings = np.array([embeddings[n] for n in target.nodes()])
                return node_embeddings.mean(axis=0)
            else:
                node_embeddings = np.array([embeddings[n] for n in target.nodes()])
                hacker_ids = target.nodes['Hacker'].data['feat_id']
                hacker_embeddings = node_embeddings[hacker_ids] # 从embeddings中获取对应向量
                return node_embeddings.mean(axis=0)

    raise ValueError(f"Unknown task type: {task_type}")

# 处理中文标点符号
def clean_chinese_symbols(text):
    symbol_map = {
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '，': ',',
        '。': '.',
        '；': ';',
        '：': ':',
        '？': '?',
        '！': '!',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>'
    }
    for ch, en in symbol_map.items():
        text = text.replace(ch, en)
    return text

# 格式化指令
def format_instruction(target, instruction):
    escaped_target = re.escape(str(target))
    return re.sub(r'\(ID=(\d+)\)', fr'(ID=\1, name: {escaped_target})', instruction)

# 从gpt的回答中提取json信息
def get_json_info(response):
    # 假设response是一个字符串，包含JSON格式的内容
    try:
        # 1. 清理响应内容
        response = clean_chinese_symbols(response.strip())
        
        # 2. 尝试直接解析
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
            
        # 3. 尝试修复常见JSON格式问题
        # 修复缺失的逗号（在引号后换行的情况）
        response = re.sub(r'\"\s*\n\s*\"', '", "', response)
        # 修复多余的空白行
        response = re.sub(r'\n\s*\n', '\n', response)
        
        # 4. 尝试提取JSON部分
        json_match = re.search(r'\{[\s\S]*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError("Unable to extract JSON content from response")
            
        json_str = json_match.group(0)
        # 清理可能的JSON标记
        json_str = json_str.replace('json', '').strip()
        
        # 5. 再次尝试解析
        return json.loads(json_str)
        
    except Exception as e:
        print(f"JSON parsing failed: {str(e)}")
        # print(f"Original response content:\n{response}")

class GraphInstructionDataset(Dataset):
    def __init__(self, json_path, embed_path, tokenizer, max_length=1024):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)
        self.graph_embeds = np.load(embed_path)
        assert len(self.records) == len(self.graph_embeds), \
            f"Mismatch: {len(self.records)} records vs {len(self.graph_embeds)} embeds"
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        user_content = rec['instruction'] + (rec.get('input') or '')
        prefix = (
            f"<s><|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        instr_tok = self.tokenizer(prefix, add_special_tokens=False)
        resp_tok = self.tokenizer(rec['output'], add_special_tokens=False)
        eos_id = self.tokenizer.eos_token_id

        input_ids = instr_tok['input_ids'] + resp_tok['input_ids'] + [eos_id]
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(instr_tok['input_ids']) + resp_tok['input_ids'] + [eos_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'graph_embeds': torch.tensor(self.graph_embeds[idx], dtype=torch.float32)
        }

class GraphInjectedModel(nn.Module):
    def __init__(self, base_model, scale=0.1):
        super().__init__()
        self.model = base_model
        self.norm = nn.LayerNorm(base_model.config.hidden_size)
        self.scale = scale

    def forward(self, input_ids=None, attention_mask=None, labels=None, graph_embeds=None, **kwargs):
        # Compute token embeddings
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        # Normalize and scale graph embeddings
        graph = graph_embeds.to(inputs_embeds.dtype)
        graph_normed = self.norm(graph)
        graph_scaled = graph_normed * self.scale
        # Clone to avoid in-place operation on leaf variable
        inputs_embeds = inputs_embeds.clone()
        # Inject graph embedding into first token
        inputs_embeds[:, 0, :] = inputs_embeds[:, 0, :] + graph_scaled

        # Forward through base model
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable(**kwargs)

class GradientLoggerCallback(TrainerCallback):
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None: return
        grads = [p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None]
        if grads:
            print(f"Step {state.global_step} - LoRA gradients: min={min(grads):.3e}, max={max(grads):.3e}")