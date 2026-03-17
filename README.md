# HGGFM

The official implementation of paper "Hacker Group Identification based on Multi-Agent Graph Foundation Model".

## Features

- **Multi-Agent Architecture**:
  - Manager Agent: parses user requests and extracts key information.
  - Intelligencer Agent: collects and provides more relevant intelligence and analysis.
  - Operator Agent: performs graph learning and generates graph tokens.
  - Responder Agent: aligns graph tokens with text tokens and generates replies.

- **Graph Foundation Model Capabilities**:
  - Node classification
  - Link prediction
  - Graph analysis / reasoning
  - Open-end chat

## Setup

1. Prepare the virtual environment, and recommend Python version ≥ `3.10`. After the virtual environment is successfully built, you need to install the `dgl` first (choose the appropriate version [here](https://www.dgl.ai/pages/start.html)). Then use the command line to install the dependencies:
```bash
(Example) pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install -r requirements.txt
```

2. Create a `.env` file, which needs to add important paths and API keys:
```
OPENAI_API_KEY=your_api_key_here
```

3. We use `Qwen3-8B` as an example, so it is recommended that GPU memory ≥ `24G`, disk free space ≥ `50G`, CUDA version ≥ `11.8`. Enter the `model` directory and use the command line to download the open source LLM to your local computer:
```
cd model
python download_model.py
```

## Usage

1. Run the command line to start the dialog mode. The program will display `"Case for Test"` to help users get started quickly.
```
python main.py
```

2. For the convenience and quick use of users, we have provided the `"hacker_intelligence.json"` and `"Hacker_embedding.npy"` files in the `output` folder for direct use. If you need to fully experience the intelligence analysis and graph training process, please delete these two files. The program will automatically generate these two files in the next round of execution.


3. Enter the `experiment` folder and use the command line to quickly reproduce the hacker organization node classification experiment.
```
cd experiment
python test.py
```

## Project Structure

```
hggfm/
├── agents/
│   ├── graphlearning.py
│   ├── p_intelligencer.py
│   ├── p_manager.py
│   ├── p_operator.py
│   └── p_responser.py
├── dataset/
│   ├── hacked/
│   │   ├── graph_hacker_com.bin
│   │   └── hacked_com_cn_data_complete.csv
│   ├── hacker/
│   │   ├── graph_hacker_com.bin
│   │   └── haxor_id_data_complete.csv
│   └── metapath.json
├── experiment/
│   ├── Hacker_processed_test.npy
│   ├── labels.txt
│   └── test.py
├── logs/
│   └── chat_history.json
├── model/
│   ├── Lora/
│   │   └── Qwen3_wahin_lora/
│   ├── Qwen/
│   │   └── Qwen3-8B/
│   └── download_model.py
├── output/
│   ├── hacked/
│   │   ├── Hacker_embedding.npy
│   │   └── hacker_intelligence.json
│   └── haxor/
│       ├── Hacker_embedding.npy
│       └── hacker_intelligence.json
├── .env
├── HggfmClass.py
├── main.py
├── README.md
└── requirements.txt

```

## Configuration

The system can be configured through environment variables in the `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MODEL_NAME`: Model version to use (default: "gpt-4o")
- `TEMPERATURE`: Model temperature setting (default: 0.7)
- `MAX_TOKENS`: Maximum tokens for responses (default: 1000)
- `GRAPH_FILE_PATH`: Graph input file path (.bin).
- `METAPATH_FILE_PATH`: Metapath file path (.json)
- `SOURCE_DATA_PATH`: Raw data file path (.csv)
- `EMBEDDING_DATA_PATH`: Graph self-supervised learning embedding output path (.npy)
- `HACKER_INTELLIGENCE_PATH`: Intelligence analysis output path (.json)
- `MODEL_PATH`: Local large language model path (folder)
- `LORA_PATH`: Fine-tuning parameter path (folder)


## License

This project is licensed under the MIT License - see the LICENSE file for details. 
