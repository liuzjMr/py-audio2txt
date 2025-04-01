import os
import re
import magic
import torch
from app import load_args
from funasr import AutoModel


# 加载本地模型，用于说话人分离
def load_sensevoice_model() -> AutoModel:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",  
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=DEVICE,
        dtype="fp16" if DEVICE == "cuda" else "fp32",  # GPU使用半精度加速
    )
    return model

# 语音识别（SenseVoiceSmall）
def speech_recognition(model: AutoModel, audio_path: str) -> list[str]:
    res = model.generate(
        input=audio_path,
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=False,  #
        merge_length_s=15,
    )
    raw_speech = res[0]["text"]
    pattern = r'<\|.*?\|>'
    result = re.split(pattern, raw_speech)  # 分割字符串并自动删除各种标记
    return [s.strip() for s in result if s.strip()]

def is_audio_file(file_path: str) -> bool:
    """
    判断文件是否为音频文件
    :param file_path: 文件路径
    :return: 是否为音频文件
    """    
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        return file_type.startswith('audio/')
    except ImportError:
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg']
        return any(file_path.lower().endswith(ext) for ext in audio_extensions)

# 单音频文件处理
def process_single_audio(model: AutoModel, input_path: str):
    if not is_audio_file(input_path):
        print(f"{input_path} is not a valid audio file.")
        return
    if not os.path.exists(input_path):
        print(f"Audio file {input_path} does not exist.")
        return
    if os.path.getsize(input_path) == 0:
        print(f"Audio file {input_path} is empty.")
        return
    try:
        transcription = speech_recognition(model, input_path)
        output_path = os.path.abspath(input_path) + ".txt"
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"parse {input_path} to:")
        print("\n".join(transcription))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcription))
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def process_directory_audios(model: AutoModel, input_dir: str):
    for root_path, _, file_names in os.walk(input_dir):
        for file_name in file_names:
            file_path = os.path.join(root_path, file_name)
            if is_audio_file(file_path):
                print(f"Processing {file_path}...")
                process_single_audio(model, file_path)            

if __name__ == "__main__":
    try:
        _, params = load_args()
        if len(params) < 1:
            print("Please provide the audio file or directory path.")
            exit(0)

        # 加载模型
        model = load_sensevoice_model()

        for input_path in params:
            if os.path.isdir(input_path):
                process_directory_audios(model, input_path)
            elif os.path.isfile(input_path):
                process_single_audio(model, input_path)
            else:
                print(f"{input_path} is not a valid file or directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)