import dgl
import json
import os

# 从文件中获得图
def get_graph_from_file(file_path):
    graphs, metadata = dgl.load_graphs(file_path)
    return graphs[0]

# 获得图信息
def get_graph_info(g, meta_path_file=None):
    if g.is_homogeneous:
        # 同质图
        n_nodes = g.num_nodes()
        n_edges = g.num_edges()
        ndata_keys = list(g.ndata.keys())
        edata_keys = list(g.edata.keys())

        node_attr_desc = "vector" if any(k in ndata_keys for k in ['feat', 'feat_id']) else \
            (', '.join(ndata_keys) if ndata_keys else "none")
        edge_attr_desc = "vector" if 'feat' in edata_keys else \
            (', '.join(edata_keys) if edata_keys else "none")

        info = (
            f"Graph Type: Homogeneous Graph; "
            f"Node: {n_nodes} nodes with attribute: {node_attr_desc}; "
            f"Edge: {n_edges} edges with attribute: {edge_attr_desc};"
        )
    else:
        # 异质图
        node_types = g.ntypes
        node_descriptions = []
        for ntype in node_types:
            attr_keys = list(g.nodes[ntype].data.keys())
            if 'feat' in attr_keys or 'feat_id' in attr_keys:
                desc = "vector"
            elif attr_keys:
                desc = ", ".join(attr_keys)
            else:
                desc = "none"
            node_descriptions.append(f"{ntype}[attribute:{desc}]")

        edge_types = g.canonical_etypes
        base_edge_types = [e for e in edge_types if not e[1].startswith("Rev-")]

        # 判断是否无向图：每种正向边都存在对称反向边
        edge_pair_set = set((s, r, d) for s, r, d in edge_types)
        is_undirected = True
        for s, r, d in base_edge_types:
            if ('Rev-' + r, d, s) not in [(r2, s2, d2) for (s2, r2, d2) in edge_types] and \
               (s, r, d) not in [(d2, r2, s2) for (s2, r2, d2) in edge_types]:
                is_undirected = False
                break
        edge_dir = "Undirected edges" if is_undirected else "Directed edges"

        edge_descriptions = [f"{rel}[{src}→{dst}]" for src, rel, dst in base_edge_types]

        node_stats = ", ".join(f"{ntype}: {g.num_nodes(ntype)}" for ntype in node_types)
        edge_stats = ", ".join(f"{rel}: {g.num_edges((src, rel, dst))}" for (src, rel, dst) in base_edge_types)

        # 元路径导入
        meta_paths = []
        if meta_path_file and os.path.exists(meta_path_file):
            try:
                with open(meta_path_file, "r", encoding="utf-8") as f:
                    meta_paths = json.load(f)
            except Exception as e:
                meta_paths = [f"[error loading meta-paths: {e}]"]

        meta_path_str = f"{len(meta_paths)} Meta-paths for learning: {', '.join(meta_paths)};" if meta_paths else "No meta-paths provided;"

        info = (
            "Graph Type: Heterogeneous Graph; "
            f"Node: {len(node_types)} node types: {', '.join(node_descriptions)}; "
            f"Edge: {edge_dir}, {len(base_edge_types)} Edge types: {', '.join(edge_descriptions)}; "
            f"Node Count: {node_stats}; "
            f"Edge Count: {edge_stats}; "
            f"Meta-path: {meta_path_str}"
        )
    return info

# 获得prompt
def get_manager_prompt(user_input, graph_input):

    prompt_manager_Agent_system="You are a professional security research planner\n"
    prompt_manager_Agent_Background="Your goal is to understand tasks and graphs based on user input and graph data. Task understanding requires interpreting user needs and clarifying user intent, thereby providing a basis for refining LLM task type. Graph understanding requires interpreting graph structure data and clarifying the organizational form of the target graph, thereby providing a basis for refining graph task type.\n"
    prompt_manager_Agent_data = "The following is user input:"+user_input+"\nThe following is the graph structure data:"+graph_input+"\n"
    prompt_manager_Agent_intend = "Based on the above user input and graph data, extract three key pieces of information: Extra Attribute, Graph Task Type, and LLM Task Type.\n"
    prompt_manager_Agent_arr = "The Extra Attribute refers to adding a new attribute to the Hacker node, thereby providing effective semantic reinforcement for the completion of subsequent tasks. After determining the Extra Attribute, please provide a sentence description of this attribute. The optional range of graph task types is (node classification, edge prediction, graph classification, graph analysis, graph reasoning). The optional range of LLM task types is (label inference, text completion, causal reasoning, content summary). The optional range of Extra Attribute is (behavior analysis, ability assessment, attack statistics, personality prediction, habit preference, obvious relationship, graph observation). Target needs to specify the target hacker ID or name, in the format of ID=? or Name=?. When both exist, ID takes precedence. In addition, edge prediction will provide two hacker node, and the two IDs or Names need to be recorded separately in the format of ID=?&ID=? or ID=?&Name=?. If it is a graph-level task, Target is uniformly marked as \"WAHIN\",  the ID and Name are directly ignored."
    prompt_manager_Agent_format = "The output format is strictly following json format:{\n\"Target\":\n\"Extra attributes\":\n\"Extra attribute description\":\n\"Graph task type\":\n\"LLM task type\":}\n"
    prompt_manager_Agent_openchat="\nIn addition, if the user intent does not contain the target object or task information, it can be considered as an open-end chat. In this case, \"LLM task type\" needs to be marked as \"open-end chat\", and other keys are marked as \"No Need\"."

    prompt_manager_Agent_all = prompt_manager_Agent_system + prompt_manager_Agent_Background + prompt_manager_Agent_data + prompt_manager_Agent_intend+ prompt_manager_Agent_arr + prompt_manager_Agent_format + prompt_manager_Agent_openchat
    return prompt_manager_Agent_all

# 获得LLM回复
def get_manager_response(prompt):
    from openai import OpenAI
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    client = OpenAI()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional security research planner.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4o",
    )
    return response.choices[0].message.content
