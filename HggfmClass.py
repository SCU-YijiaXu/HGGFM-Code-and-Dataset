from agents.p_manager import *
from agents.p_intelligencer import *
from agents.p_operator import *
from agents.p_responser import *
from agents.graphlearning import *
import numpy as np

class hggfmclass:
    def __init__(self, graph, metapath, source_data, hacker_intelligence, embedding_data):
        self.results = {
            "manager": '',
            "intelligencer": '',
            "operator":'',
            "responser": ''
        }
        self.graph_path = graph # 图文件路径
        self.metapath_file = metapath # 元路径文件路径
        self.source_data_path = source_data # 原始数据文件路径
        self.hacker_intelligence_path = hacker_intelligence # 黑客情报文件路径
        self.embedding_path = embedding_data
        self.llm_tokenizer = None # 初始化tokenizer
        self.llm_model = None #初始化model
        self.wahin = None # 初始化图
        self.target_name = None
        self.target_id = None
    
    # 加载图
    def load_graph(self):
        self.wahin = get_graph_from_file(self.graph_path)
        return self.wahin

    # 加载模型
    def load_llm_model(self, model_path, lora_dir_n):
        self.llm_tokenizer, self.llm_model = load_fine_tuned(model_path, lora_dir_n)

    # 更新结果
    def updata_results(self, key, value):
        self.results[key] = value

    # 获取结果
    def get_results(self, key):
        return self.results[key]

    # Manager Agent Work
    def manager_agent_process(self, user_input, column="Hacker"):
        graph_info = get_graph_info(self.wahin, self.metapath_file) # 获得图信息input
        manager_prompt = get_manager_prompt(user_input,graph_info) # 组合用户输入和图信息获得prompt
        for i in range(3): # 重试3次
            try:
                response = get_manager_response(manager_prompt) # 获得gpt的回答
                manager_key_info = get_json_info(response) # 解析gpt的回答
                if manager_key_info!= None:
                    break # 成功则退出循环
            except Exception as e:
                print(f"Error: {e}, Retrying...")

        self.updata_results('manager', manager_key_info)
        self.target_name, self.target_id = get_node_name_from_string(self.wahin, column, manager_key_info['Target']) # 获得目标节点的名称
        return manager_key_info
    
    # Intelligencer Agent Work
    def intelligencer_agent_process(self, key_infor, column="Hacker"):
        if self.target_name == 'No Need' and key_infor['LLM task type'] == 'open-end chat':
            print("Intelligence analysis is no need!")
            return {"Attribute Name": 'No Need', 'Attribute Content':'- Attribute Content: No Need', 'Analysis Process':'- Attribute Process: No Need'} # 不需要Intelligencer
        if not os.path.exists(self.hacker_intelligence_path):
            process_hackers_intelligence(key_infor['Extra attributes'], key_infor['Extra attribute description'], get_id2name_dict(self.wahin, column), self.hacker_intelligence_path, self.source_data_path)
        else:
            with open(self.hacker_intelligence_path, 'r', encoding='utf-8') as f:
                if len(f.readlines()) < self.wahin.num_nodes(column):
                    process_hackers_intelligence(key_infor['Extra attributes'], key_infor['Extra attribute description'], get_id2name_dict(self.wahin, column), self.hacker_intelligence_path, self.source_data_path)
        
        records_of_colume = search_and_format_records(column, self.target_name, self.source_data_path) # 根据Hacker名搜索并格式化记录
        intelligencer_prompt = get_intelligencer_prompt(key_infor['Extra attributes'], key_infor['Extra attribute description'], records_of_colume, UID = self.target_id)  # 生成Intelligencer的Prompt
        for i in range(3): # 重试3次
            try:
                intelligencer_response = get_intelligencer_response(intelligencer_prompt) # 获取Intelligencer的响应
                intelligencer_response_json = get_json_info(intelligencer_response)
                if intelligencer_response_json!= None:
                    break # 成功则退出循环
            except Exception as e:
                print(f"Error: {e}, Retrying...")
        self.updata_results('intelligencer', intelligencer_response_json)
        return intelligencer_response_json

    # Operator Agent Work
    def operator_agent_process(self, key_infor, column="Hacker"):
        if key_infor['Graph task type'] == 'No Need' and key_infor['LLM task type'] == 'open-end chat':
            print("Graph task type is no need!")
            self.updata_results('operator', np.zeros(4096))
            return 'No Need' # 不需要Intelligencer
        if not os.path.exists(self.embedding_path):
            wahin_mp = get_metapath(self.wahin, self.metapath_file) # 获取元路径
            load_new_attribute(wahin_mp, self.hacker_intelligence_path)
            embedding_nodes = train_self(wahin_mp, wahin_mp.metapaths)
        else:
            embedding_nodes = {column : np.load(self.embedding_path)}

        if key_infor['Graph task type'] in {'graph classification', 'graph analysis', 'graph reasoning'} and self.target_id != "WAHIN":
            sub_g = get_subgraph(self.wahin, (column, int(self.target_id)), k=2)
            self.target_id = sub_g

        embedding_vector = get_embedding_by_task_type(key_infor['Graph task type'], embedding_nodes[column], self.target_id)
        self.updata_results('operator', embedding_vector)
        return embedding_vector

    # Responser Agent Work
    def responser_agent_process(self, user_input, key_infor, column="Hacker"):
        user_instruct = generate_user_instruction(user_input, self.get_results("intelligencer") ,key_infor, self.wahin, column)
        agent_answer = generate_response_onecase(user_instruct, self.get_results('operator'), self.llm_tokenizer, self.llm_model, self.target_name) # 生成响应(使用图TOKEN和指令)
        self.updata_results('responser', agent_answer)
        return agent_answer