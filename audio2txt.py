import datetime
import glob
import io
import os
from pathlib import Path
import re
import time
import magic
from pyannote.audio import Pipeline
import torch
from pydub import AudioSegment
from common import get_duration, get_executable_directory, load_args
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from funasr import AutoModel

ROOT_DIR = get_executable_directory()
Model_Config_Pyannote = os.path.join(ROOT_DIR, "models", "pyannote", "speaker-diarization-3.1", "pyannote_diarization_config.yaml")

def load_sensevoice_model() -> AutoModel:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",  
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
        dtype="fp16" if torch.cuda.is_available() else "fp32",  # GPU使用半精度加速
    )
    return model

def speech_recognition_batch(model: AutoModel, audio_pathes: list[io.BytesIO]) -> list[str]:
    res = model.generate(
        input=audio_pathes,
        cache={},
        language="zh",  # "zh", "en", "yue", "ja", "ko", "nospeech"
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


# 加载本地模型，用于说话人分离
def load_local_pyannote_pipeline() -> Pipeline:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    path_to_config = Path(Model_Config_Pyannote)
    pipeline = Pipeline.from_pretrained(path_to_config)
    pipeline.to(device)
    return pipeline

def preprocess_audio(input_path, noise_start=0.1, noise_end=0.5) -> io.BytesIO:
    # 加载音频并强制转换为单声道
    y, sr = librosa.load(input_path, sr=16000, mono=True, res_type='kaiser_fast')
    
    # 提取噪声样本段
    noise_samples = y[int(noise_start*sr):int(noise_end*sr)]
    
    # 执行降噪处理（stationary=True适合电话线路噪声）
    denoised = nr.reduce_noise(
        y=y, 
        y_noise=noise_samples,
        sr=sr,
        stationary=True,
        prop_decrease=0.8  # 降噪强度80%，避免语音失真
    )
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, denoised, sr, subtype='PCM_16', format='WAV')
    audio_buffer.seek(0)  # 重置指针位置以便后续读取
    return audio_buffer

def parse_audio_speaker(pipeline : Pipeline, wav_bytes: io.BytesIO, num_speakers: int) -> list[dict]:
    diarization = pipeline(wav_bytes, num_speakers=num_speakers)
    processed_diarization = [
        {
            "speaker": speaker,
            "start": f"{turn.start:.3f}",
            "end": f"{turn.end:.3f}",
        }
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    return processed_diarization


def split_speaker_audios(wav_bytes: io.BytesIO, segments: list[dict]) -> list[io.BytesIO]:
    audio = AudioSegment.from_wav(wav_bytes)
    speaker_audios = []
    for segment in segments:
        start_time = int(float(segment['start']) * 1000)
        end_time = int(float(segment['end']) * 1000)
        segment_audio = audio[start_time:end_time]
        segment_io = io.BytesIO()
        segment_audio.export(segment_io, format="wav")
        segment_io.seek(0)  # 重置指针位置以便后续读取
        speaker_audios.append(segment_io)
    return speaker_audios

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
    
    
def transcript(audio_path: str, num_speakers : int, pipeline: Pipeline, model: AutoModel) -> list[dict]:
    wav_bytes = None
    try:
        start_time = datetime.datetime.now()
        # 预处理音频
        wav_bytes = preprocess_audio(audio_path)
        # 处理音频文件
        speaker_segments = parse_audio_speaker(pipeline, wav_bytes, num_speakers)
        # 分离说话人音频
        speaker_audios = split_speaker_audios(wav_bytes, speaker_segments)
        # 语音识别
        speeches = speech_recognition_batch(model, speaker_audios)
        # 合并文本
        lines = []
        for i, speech in enumerate(speeches):
            lines.append(f"{speaker_segments[i]['speaker']}: {speech}")
        # 输出文本
        output_path = os.path.abspath(audio_path) + ".txt"
        if os.path.exists(output_path):
            os.remove(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Processed {audio_path} in {get_duration(start_time)}.")
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
    finally:
        if wav_bytes:
            wav_bytes.close()


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
            print("  -s, --speeker   Number of speakers (default: 2)")
            exit(0)
        batch_size = int(options.get("s", options.get("speeker", 2)))
        model = load_sensevoice_model()
        pipeline = load_local_pyannote_pipeline()
        for input_path in params:
            if os.path.isfile(input_path):
                if is_audio_file(input_path):
                    transcript(input_path, batch_size, pipeline, model)
                else:
                    print(f"{input_path} is not a valid audio file.")
            elif os.path.isdir(input_path):
                audio_files = glob.glob(os.path.join(input_path, "**/*.*"), recursive=True)
                audio_files = [f for f in audio_files if is_audio_file(f)]
                for audio_file in audio_files:
                    transcript(audio_file, batch_size, pipeline, model)
            else:
                print(f"{input_path} is not a valid file or directory.")
    except KeyboardInterrupt:
        # 捕获 Ctrl+C 中断
        # 处理并清理资源
        print("\nProcess interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        exit(1)
