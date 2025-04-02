import io
import os
from pathlib import Path
import re
import tempfile
from pyannote.audio import Pipeline, Audio, Inference
import torch
from pydub import AudioSegment
from app import get_executable_directory, load_args
import librosa
import noisereduce as nr
import soundfile as sf
import numpy as np
from funasr import AutoModel

ROOT_DIR = get_executable_directory()
Model_Config_Pyannote = os.path.join(ROOT_DIR, "models", "pyannote", "speaker-diarization-3.1", "pyannote_diarization_config.yaml")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load_sensevoice_model() -> AutoModel:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        trust_remote_code=True,
        remote_code="./model.py",  
        # vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device=DEVICE,
        dtype="fp16" if DEVICE == "cuda" else "fp32",  # GPU使用半精度加速
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
    path_to_config = Path(Model_Config_Pyannote)
    pipeline = Pipeline.from_pretrained(path_to_config)
    pipeline.to(DEVICE)
    return pipeline

def preprocess_audio(input_path, noise_start=0.0, noise_end=0.3) -> io.BytesIO:
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


if __name__ == "__main__":
    wav_bytes = None
    pipeline = None
    try:
        _, params = load_args()
        if len(params) == 0:
            print("Please provide the audio file path.")
            exit(1)
        audio_path = params[0]
        if not os.path.exists(audio_path):
            print(f"Audio file {audio_path} does not exist.")
            exit(1)
        pipeline = load_local_pyannote_pipeline()
        model = load_sensevoice_model()

        wav_bytes = preprocess_audio(audio_path)
        # 处理音频文件
        speaker_segments = parse_audio_speaker(pipeline, wav_bytes, 2)
        # 分离说话人音频
        speaker_audios = split_speaker_audios(wav_bytes, speaker_segments)
        # 语音识别
        speeches = speech_recognition_batch(model, speaker_audios)
        for i, speech in enumerate(speeches):
            print(f"{speaker_segments[i]["speaker"]}: {speech}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
    finally:
        # 释放资源
        if wav_bytes:
            wav_bytes.close()
        if pipeline:
            del pipeline
        torch.cuda.empty_cache()
        exit(0)
