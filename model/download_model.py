from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-8B', cache_dir='./', revision='master')

print(f"The model is downloaded and saved in the path: {model_dir}")