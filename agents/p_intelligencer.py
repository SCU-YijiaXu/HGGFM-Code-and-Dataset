import csv
import json
import os
import time
from .graphlearning import generate_prompt_fromgraph
import dgl

# 定义函数，根据指定列和值搜索并格式化记录
def search_and_format_records(search_column, search_value, path='../dataset/hacked_com_cn_data_complete.csv'):
    def search_and_format(search_column, search_value, path):
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, 1):
                if row[search_column] == search_value:
                    # 提取指定字段
                    record = [
                        f"Time: {row['Time']}",
                        f"Hacker: {row['Hacker']}",
                        f"Domain: {row['Domain']}",
                        f"Page: {row['Page']}",
                        f"IP: {row['IP']}",
                        f"System: {row['System']}",
                        f"Web server: {row['Web server']}"
                    ]
                    records.append(f"[Record {i}]" + ", ".join(record))
        # print("Hacker("+ search_value +") searches for the number of records: ",len(records))
        return ".\n".join(records)

    if isinstance(search_value, tuple):
        records_of_search_one = search_and_format(search_column, search_value[0], path)
        records_of_search_two = search_and_format(search_column, search_value[1], path)
        return records_of_search_one + "\n" + records_of_search_two
    
    elif isinstance(search_value, dgl.DGLHeteroGraph):
        return generate_prompt_fromgraph(search_value)

    records_of_search = search_and_format(search_column, search_value, path)
    return records_of_search

# 生成Intelligencer的Prompt
def get_intelligencer_prompt(hggfm_arr, hggfm_arr_de, intelligence_records, UID=None):

    prompt_intelligencer_Agent_system="You are a professional cyber intelligence analyst.\n"
    prompt_intelligencer_Agent_Background="Your goal is to strengthen the Web Attack Heterogeneous Information Network (WAHIN). WAHIN serves the hacker organization analysis scenario and contains 5 types of nodes, among which the most critical one is the hacker node, which is directly related to most downstream tasks. Now, the hacker node needs to add an attribute named \""+hggfm_arr+"\", and the description of this attribute is \""+hggfm_arr_de+"\". Next, you need to fully leverage your reasoning ability to generate the corresponding content for this new attribute.\n"
    prompt_intelligencer_Agent_data = "The following is a set of security incident logs resulting from a \"Hacker\" attack. From left to right, the content includes: \"Record Time, Hacker Name, Domain, Page, IP Address, Operating System, Web Server\".\n"+intelligence_records+"\n"
    prompt_intelligencer_Agent_intend = "Please conduct a comprehensive analysis based on these records. Your analytical stance should remain objective, neutral and rational. According to the difference model, structure your analytical approach from the following five dimensions:\"Time\", \"IP and Domain\", \"Residual Clues\", \"Technical preference and Attack Processe\", \"Target continuity and Tactical Path\". Form a complete chain of reasoning, and condense your analytical process into a paragraph of no more than 200 words and no less than 100 words.\n"
    prompt_intelligencer_Agent_arr = "Please refer to the analysis process and combine the recorded intelligence to generate the content for the\""+hggfm_arr+"\"attribute of the Hacker node. The content should be no more than 200 words and no less than 100 words."
    prompt_intelligencer_Agent_format = "The output format is strictly following json format:{\n\"Attribute Name\":\n\"Attribute Content\":\n\"Analysis Process\":\n}"
    if UID == "WAHIN":
        prompt_intelligencer_Agent_Background="Your goal is to analyse the Web Attack Heterogeneous Information Network (WAHIN). WAHIN serves the hacker organization analysis scenario and contains 5 types of nodes, among which the most critical one is the hacker node, which is directly related to most downstream tasks. Now, the target of this task is named \""+hggfm_arr+"\", and the description is \""+hggfm_arr_de+"\". Next, you need to give full play to your reasoning ability and generate corresponding content based on the graph information and structure. \n"
        prompt_intelligencer_Agent_all = prompt_intelligencer_Agent_system + prompt_intelligencer_Agent_Background + intelligence_records + '\n' +prompt_intelligencer_Agent_format+"\nPlease note: The value of each Key in the output content should be plain text. There is no need for line breaks or paragraph breaks in Attribute Content and Analysis Process, and the number of words in each should not be less than 1,000."
    else: 
        prompt_intelligencer_Agent_all = prompt_intelligencer_Agent_system + prompt_intelligencer_Agent_Background + prompt_intelligencer_Agent_data + prompt_intelligencer_Agent_intend+ prompt_intelligencer_Agent_arr + prompt_intelligencer_Agent_format
    return prompt_intelligencer_Agent_all

# 搜索并格式化记录
def get_intelligencer_response(prompt):
    from openai import OpenAI
    from dotenv import load_dotenv, find_dotenv

    _ = load_dotenv(find_dotenv())
    client = OpenAI()

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional cyber intelligence analyst.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4.1",
        max_tokens=10000,
    )
    return response.choices[0].message.content

# 批量生成Hacker节点的Intelligence属性
def process_hackers_intelligence(extra_attr, extra_attr_de, id_name_dict, save_path, source_data_path):
    # 1. 从id_to_name_all.json加载黑客名称列表
    # with open('../dataset/map/id_to_name_all.json', 'r', encoding='utf-8') as f:
    #     id_to_name = json.load(f)
    
    # 提取Hacker节点的ID和名称
    # hacker_ids = id_to_name.get("Hacker", {})
    hacker_ids = id_name_dict
    # 2. 创建结果文件路径
    output_file = save_path
    
    # 如果文件已存在，加载已有数据
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            # Fix: Read each line as JSON object
            results = [json.loads(line) for line in f if line.strip()]
    else:
        results = []
    
    print("All number need to process: ", len(hacker_ids.items()))
    # 3. 处理每个黑客
    for hacker_id, hacker_name in hacker_ids.items():
        # 跳过已处理的黑客
        if any(item.get('hacker_id') == hacker_id for item in results):
            continue
            
        try:
            # 搜索记录
            records = search_and_format_records("Hacker", hacker_name, source_data_path)
            
            # 获取prompt
            extra_attract = "Behavior analysis"
            extra_attr_dect = "Analyze patterns and behaviors to determine potential grouping characteristics"
            prompt = get_intelligencer_prompt(extra_attract, extra_attr_dect, records)
            
            # 重试机制 如果出错，重新发送请求
            max_retries = 3
            retry_count = 0
            response_json = None
            
            while retry_count < max_retries:
                try:
                    # 获取响应
                    response = get_intelligencer_response(prompt)
                    # 解析JSON响应
                    response_json = get_json_info(response)
                    if response_json is None:  # 添加对None的检查
                        raise ValueError("The response JSON is empty")
                    break
                except (AttributeError, ValueError, json.JSONDecodeError) as e:  # 捕获更具体的异常
                    retry_count += 1
                    print(f"Processing hacker {hacker_name} (ID: {hacker_id}) retry {retry_count}: {str(e)}")
                    if retry_count >= max_retries:
                        raise
                    time.sleep(1)  # 添加短暂延迟避免频繁请求

            # 提取属性名和内容
            attribute_name = response_json.get("Attribute Name", "")
            attribute_content = response_json.get("Attribute Content", "")
            analysis_process = response_json.get("Analysis Process", "")
            
            # 4. 保存结果
            result = {
                "hacker_id": hacker_id,
                "hacker_name": hacker_name,
                "attribute_name": attribute_name,
                "attribute_content": attribute_content,
                "analysis_process": analysis_process
            }
            
            results.append(result)
            
            # 5. 立即保存到文件
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')  # 添加换行符分隔记录
                
            print(f"Hackers dealt with {hacker_name} (ID: {hacker_id})")
            
        except Exception as e:
            print(f"Error processing hacker {hacker_name} (ID: {hacker_id}): {str(e)}")
            break

    print("All hackers are dealt with!")
