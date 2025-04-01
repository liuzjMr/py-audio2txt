import os
from huggingface_hub import snapshot_download

os.environ["HTTP_PROXY"] = "http://192.168.2.22:1080"
os.environ["HTTPS_PROXY"] = "http://192.168.2.22:1080"
# 设置Hugging Face Hub的传输方式
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  

model_id = "openai/whisper-large-v3-turbo"
local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "openai", "whisper-large-v3-turbo")
os.makedirs(local_dir, exist_ok=True)  # 自动创建目录
print(f"模型下载目标路径：{local_dir}")

# 下载配置
model_path = snapshot_download(
    repo_id=model_id,
    local_dir=local_dir,
    resume_download=False,  # 是否断点续传
    max_workers=4,  # 最大线程数
)
print(f"模型已下载至：{model_path}")