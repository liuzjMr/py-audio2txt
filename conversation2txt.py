import glob
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
def speech_recognition_batch(model: AutoModel, audio_pathes: list[str]) -> list[str]:
    res = model.generate(
        input=audio_pathes,
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=False,  #
        merge_length_s=15,
    )
    result = []
    pattern = r'<\|.*?\|>'
    for item in res:
        raw_speech = item["text"]
        lines = re.split(pattern, raw_speech)  # 分割字符串并自动删除各种标记
        lines = [s.strip() for s in lines if s.strip()]
        result.append("\n".join(lines))
    return result

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
def process_audio_batch(model: AutoModel, audio_pathes: list[str]):
    try:
        transcription = speech_recognition_batch(model, audio_pathes)
        for i, input_path in enumerate(audio_pathes):
            speech = transcription[i] if len(transcription) > i else ""
            print(f"Transcription for {input_path}:")
            print(speech)
            output_path = os.path.abspath(input_path) + ".txt"
            if os.path.exists(output_path):
                os.remove(output_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(speech)
    except Exception as e:
        print(f"Error processing {audio_pathes}: {e}")


def process_directory_audios(model: AutoModel, input_dir: str, batch_size: int):
    # 获取所有音频文件（支持递归搜索）
    audio_files = glob.glob(os.path.join(input_dir, "**/*.*"), recursive=True)
    audio_files = [f for f in audio_files if is_audio_file(f)]
    
    # 按批次处理（网页1/网页4的批量处理思想）
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size+1} ({len(batch)} files)")
        process_audio_batch(model, batch)
         
def split_list(lst: list[str], batch_size: int) -> list[list[str]]:
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
         
def convert(input_pathes : list[str], batch_size: int):
    # 加载模型
    model = load_sensevoice_model()
    audio_files = []
    for input_path in input_pathes:
        if os.path.isdir(input_path):
            process_directory_audios(model, input_path, batch_size)
        elif os.path.isfile(input_path):
            audio_files.append(input_path)
        else:
            print(f"{input_path} is not a valid file or directory.")
    if len(audio_files) > 0:
        subs = split_list(audio_files, batch_size)
        for sub in subs:
            process_audio_batch(model, sub)

if __name__ == "__main__":
    try:
        options, params = load_args()
        if "v" in options or "version" in options:
            print("audio2txt Version: 1.0.0")
            exit(0)
        if "h" in options or "help" in options:
            print("Usage: audio2txt [options] <audio_file> <audio_dir> ...")
            print("Options:")
            print("  -v, --version   Show version")
            print("  -h, --help      Show this help message")
            print("  -b, --batch     Batch size for processing (default: 10)")
            exit(0)
        batch_size = int(options.get("b", options.get("batch", 10)))
        convert(params, batch_size)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        exit(1)