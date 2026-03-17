from .graphlearning import *
import json
import torch
import dgl
from sentence_transformers import SentenceTransformer
import pandas as pd
from datetime import datetime

# 将情报加载于图节点属性中
def load_new_attribute(wahin_mp, json_path):
    # 1. 加载文本嵌入模型
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')  # 输出768维向量
    
    # 2. 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 3. 提取属性文本并生成嵌入
    hacker_ids = []
    texts = []
    for item in data:
        hacker_ids.append(int(item['hacker_id']))
        texts.append(item['attribute_content'])
    
    # 生成文本嵌入向量 (768维)
    embeddings = model.encode(texts, convert_to_tensor=True)
    
    # 4. 更新节点特征
    num_hackers = wahin_mp.num_nodes('Hacker')
    feat_tensor = torch.zeros((num_hackers, embeddings.size(1)), dtype=torch.float32)
    
    for hacker_id, embedding in zip(hacker_ids, embeddings):
        if hacker_id < num_hackers:
            feat_tensor[hacker_id] = embedding
    
    # 添加特征到图中
    wahin_mp.nodes['Hacker'].data['feat'] = feat_tensor
    
    return wahin_mp

# 读取JSON文件并获取元路径，保存在图中
def get_metapath(wahin, metapath_file):
    try:
        with open(metapath_file, 'r', encoding='utf-8') as f:
            metapath_data = json.load(f)
            
        # 预处理：解析元路径并获取实际的边关系
        def parse_metapath(metapath_str, g):
            # 移除方括号和箭头符号
            nodes = metapath_str.strip('[]').replace('→', ' ').split()
            metapath = []
            
            # 遍历节点对，查找它们之间的边关系
            for i in range(len(nodes)-1):
                src_type = nodes[i]
                dst_type = nodes[i+1]
                
                # 在图中查找src_type和dst_type之间的边关系
                found = False
                for rel in g.canonical_etypes:
                    if rel[0] == src_type and rel[2] == dst_type:
                        metapath.append((src_type, rel[1], dst_type))
                        found = True
                        break
                
                if not found:
                    print(f"Warning: No edge relationship found from {src_type} to {dst_type}")
                    return None
            
            return metapath
            
        # 转换所有元路径
        converted_metapaths = {}
        for i, mp_str in enumerate(metapath_data):
            try:
                parsed = parse_metapath(mp_str, wahin)
                if parsed:
                    converted_metapaths[f'metapath_{i}'] = parsed
            except Exception as e:
                print(f"Warning: Unable to resolve meta path '{mp_str}': {e}")
                continue
            
        # 假设wahin有一个metapaths属性来存储元路径
        if not hasattr(wahin, 'metapaths'):
            wahin.metapaths = {}
            
        # 将转换后的元路径数据添加到wahin对象
        wahin.metapaths.update(converted_metapaths)
        
        return wahin
        
    except FileNotFoundError:
        print(f"Error: File {metapath_file} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: File {metapath_file} is not in valid JSON format")
        return None

# 自监督学习
def train_self(g, metapath_dict, in_dim=128, out_dim=4096, epochs=200, lr=1e-3, save_dir="./output"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化每种类型节点特征
    features_dict, map_layers = init_features(g, in_dim, device)
    
    encoder = MetaPathEncoder(g, list(metapath_dict.values()), in_dim, out_dim).to(device)
    model = DGI(encoder, g.ntypes).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(features_dict.values()) + list(map_layers.parameters()),
        lr=lr
    )

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, _ = model(features_dict)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.6f}| Time: {current_time}")

    # 推理
    model.eval()
    with torch.no_grad():
        _, final_embeds = model(features_dict)

    # 保存输出向量
    os.makedirs(save_dir, exist_ok=True)
    embed_dict = {}
    for ntype, tensor in final_embeds.items():
        np_array = tensor.cpu().numpy()
        path = os.path.join(save_dir, f"{ntype}_embedding.npy")
        np.save(path, np_array)
        embed_dict[ntype] = np_array
        print(f"✅ Saved: {path}  shape={np_array.shape}")

    return embed_dict  # 字典，键为节点类型，值为对应嵌入矩阵

# 获取子图（图推理）
def get_subgraph(g, start_node, k=1):
    
    node_type, node_id = start_node
    
    # 1. 获取k跳范围内的所有节点
    nodes = {node_type: {node_id}}
    
    for _ in range(k):
        new_nodes = {ntype: set() for ntype in g.ntypes}
        
        # 遍历当前所有节点
        for ntype in nodes:
            for nid in nodes[ntype]:
                # 获取所有邻居节点
                for etype in g.canonical_etypes:
                    src_ntype, _, dst_ntype = etype
                    
                    # 前向边
                    if ntype == src_ntype:
                        successors = g.successors(nid, etype=etype)
                        new_nodes[dst_ntype].update(successors.tolist())
                    
                    # 反向边
                    if ntype == dst_ntype:
                        predecessors = g.predecessors(nid, etype=etype)
                        new_nodes[src_ntype].update(predecessors.tolist())
        
        # 合并新发现的节点
        for ntype in new_nodes:
            if new_nodes[ntype]:
                if ntype not in nodes:
                    nodes[ntype] = set()
                nodes[ntype].update(new_nodes[ntype])
    
    # 2. 提取子图
    node_dict = {ntype: torch.tensor(list(nodes[ntype])) for ntype in nodes}
    sub_g = dgl.node_subgraph(g, node_dict)
    
    return sub_g