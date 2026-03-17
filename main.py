from HggfmClass import hggfmclass
import os
from dotenv import load_dotenv, find_dotenv
import json
import datetime
import re
_ = load_dotenv(find_dotenv())

def process_agents(hggfm_process, user_input):
    print("[*] Manager Agent is working ... ",end ='')
    key_infor = hggfm_process.manager_agent_process(user_input)
    print(f'Successful! The key information: \n- Graph Task Type: {key_infor["Graph task type"]}\n- LLM Task Type: {key_infor["LLM task type"]}\n- Extra Attributes: {key_infor["Extra attributes"]}')
    print(f"- Extra Attribute Description: {key_infor['Extra attribute description'][:70]}..." if len(key_infor['Extra attribute description']) > 70 else key_infor['Extra attribute description'])
    print("[*] (Extra Attributes)  ---->   Intelligencer Agent is working ... ",end ='')
    intell_data = hggfm_process.intelligencer_agent_process(key_infor)
    print(f'Successful! The extra attributes: \n- Attribute Name: {intell_data["Attribute Name"]}')
    word_pattern = r'\w+|[^\w\s]'
    print(f"- Attribute Content: ({len(re.findall(word_pattern, intell_data['Attribute Content']))} words) ... {intell_data['Attribute Content'][110:180]} ... " if len(intell_data['Attribute Content']) > 180 else intell_data['Attribute Content'])
    print(f"- Analysis Process: ({len(re.findall(word_pattern, intell_data['Analysis Process']))} words) ... {intell_data['Analysis Process'][110:180]} ... " if len(intell_data['Analysis Process']) > 180 else intell_data['Analysis Process'])
    print("[*] (Graph Task Type)   ---->   Operator Agent is working ... ", end ='')
    embeding_data = hggfm_process.operator_agent_process(key_infor)
    print(f'Successful!\n The graph token: {embeding_data}')
    print("[*] (LLM Task Type)     ---->   Responser Agent is working ... ")
    answer = hggfm_process.responser_agent_process(user_input, key_infor)
    print("Successful!\n HGGFM Answer:", answer)
    return answer


def main():
    
    print("\n-----------------------------------------------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        HGGFM - v1.6       <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("-----------------------------------------------------------------------------------------------------------------")

    print("Start to load LLM and Graph...")
    hggfm = hggfmclass(os.getenv('GRAPH_FILE_PATH'),os.getenv('METAPATH_FILE_PATH'), os.getenv('SOURCE_DATA_PATH'), os.getenv('HACKER_INTELLIGENCE_PATH'), os.getenv('EMBEDDING_DATA_PATH'))
    hggfm.load_graph()
    hggfm.load_llm_model(os.getenv('MODEL_PATH'),os.getenv('LORA_PATH'))
    print("Successful!")

    print("-----------------------------------------------------------------------------------------------------------------")
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>        Case for Test        <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("---------------------------------------------------------------------")
    print("In order to help users quickly adapt to the model, we provide the following cases for testing:")
    print("[1] Please analyze whether Hacker Node (ID=4) is a hacker group?")
    print("[2] Please analyze whether Hacker (name is \"AnonymousFox\") and Hacker (name is \"Unknown_Br\") are related?")
    print("[3] I'd like to know the the attack habits presented by Hacker who's name is OteTeam.")
    print("[4] Please comprehensively analyze the attack trends of the hacker group from data (not less than 600 words).")
    print("[5] Hello, i want to know about what is hacker?")
    print("Input \"\033[31mexit\033[0m\" to end the program.")
    print("-----------------------------------------------------------------------------------------------------------------")

    while True:
        user_chat = input("User-admin: ")
        
        if user_chat.lower() == 'exit':
            print("The HGGFM program is over, thanks for the test.")
            break

        result = process_agents(hggfm, user_chat)
       
        print("-----------------------------------------------------------------------------------------------------------------")

        chat_history = {"user_input": user_chat, "answer": result,"timestamp": datetime.datetime.now().isoformat()}
        with open("logs/chat_history.json", 'a', encoding='utf-8') as f:
            f.write(json.dumps(chat_history, ensure_ascii=False) + '\n')
    
if __name__ == '__main__':

    # user_input = "Please analyze whether Hacker Node (ID=4) is a hacker group?" # 用户输入
    user_input = "Please analyze whether Hacker Node (name is \"爱好者 Group\") is a hacker group?" # 用户输入
    # user_input = "Please analyze whether Hacker (name is \"爱好者 Group\") and Hacker (name is \"OteTeam\") are related?" # 用户输入
    # user_input = "I'd like to know the the attack habits presented by Hacker who's name is OteTeam."
    # user_input = "Please comprehensively analyze the attack trends of the hacker organization from data."
    # user_input = "Hello, i want to know about what is hacker?"
    main()
    
