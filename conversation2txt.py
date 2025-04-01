from bisect import bisect_right
import os
import tempfile
from typing import cast
import magic
import torch
from pathlib import Path
from pydub import AudioSegment
from pyannote.audio import Pipeline as DiarizationPipeline
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from app import get_executable_directory, load_args


# 初始化参数
ROOT_DIR = get_executable_directory()
Model_Config_Pyannote = os.path.join(ROOT_DIR, "models", "pyannote", "speaker-diarization-3.1", "pyannote_diarization_config.yaml")
Model_Path_Whisper = os.path.join(ROOT_DIR, "models", "openai", "whisper-large-v3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_TYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# 加载本地模型，用于说话人分离
def load_local_pyannote_pipeline() -> DiarizationPipeline:
    path_to_config = Path(Model_Config_Pyannote)
    pipeline = DiarizationPipeline.from_pretrained(path_to_config)
    return pipeline

def load_local_whisper_pipeline() -> pipeline:
    processor = AutoProcessor.from_pretrained(Model_Path_Whisper)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        Model_Path_Whisper,
        torch_dtype=TORCH_TYPE,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model = cast(WhisperForConditionalGeneration, model)
    model.to(DEVICE)
    model.generation_config.language = "zh"  # 设置语言为中文
    model.generation_config.forced_decoder_ids = None  # 解除强制语言限制
    model.eval()
    
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=TORCH_TYPE,
        device=DEVICE,
    )    
    return asr_pipe

# 1：音频预处理（转换为16kHz单声道wav）
def preprocess_audio(input_path: str) -> str:
    """
    音频预处理
    :param input_path: 输入音频文件路径
    :return: 处理后的音频文件路径
    """
    temp_id = os.path.splitext(os.path.basename(input_path))[0]
    temp_dir = tempfile.gettempdir()
    wav_path = os.path.join(temp_dir, f"temp_{temp_id}.wav")
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(wav_path, format="wav")
    return wav_path


# 2：说话人分离（Pyannote), 返回说话人分离的片段(start, end, speaker)
def speaker_diarization(diarization_pipe: DiarizationPipeline, wav_path: str) -> list[tuple[float, float, str]]:    
    """
    说话人分离
    :param diarization_pipe: 说话人分离模型
    :param wav_path: 输入音频文件路径
    :return: 说话人分离的片段
    """
    # 说话人分离
    diarization = diarization_pipe(wav_path)
    results = sorted(
        [(float(seg.start), float(seg.end), str(speaker)) 
         for seg, _, speaker in diarization.itertracks(yield_label=True)],
        key=lambda x: x[0]
    )
    return results


# 3：语音识别（SenseVoiceSmall）
def speech_recognition(asr_pipe: pipeline, wav_path: str, speaker_segments: list[tuple[float, float, str]]) -> list[dict]:
    """
    语音识别
    :param asr_pipe: 语音识别模型
    :param wav_path: 输入音频文件路径
    :param speaker_segments: 说话人分离的片段
    :return: 识别结果
    """
    # 语音识别
    result = asr_pipe(
        wav_path,
        return_timestamps=True,  # 显式启用时间戳
        generate_kwargs={"task": "transcribe", "language": "zh"},
        chunk_length_s=30,  # 每个片段的长度
        stride_length_s=2,  # 重叠长度
    )

    transcription = []
    if "chunks" in result:
        transcription = result["chunks"]

    # 处理对齐逻辑
    aligned_result = []
    for item in transcription:
        text = item["text"].strip()
        if not text:
            continue
        start = item["timestamp"][0]
        end = item["timestamp"][1]
        
        idx = bisect_right(speaker_segments, (start, float('inf'), '')) - 1
        speaker = speaker_segments[idx][2] if idx >=0 and start <= speaker_segments[idx][1] else "Unknown"
        
        aligned_result.append({
            "speaker": speaker,
            "text": text,
            "start": start,
            "end": end
        })
    return aligned_result


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
def process_single_audio(diarization_pipe: DiarizationPipeline, whisper_pipe: pipeline, input_path: str):
    """
    处理单个音频文件
    :param diarization_pipe: 说话人分离模型
    :param whisper_pipe: 语音识别模型
    :param input_path: 输入音频文件路径
    """
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
        wav_path = preprocess_audio(input_path)
        speaker_segments = speaker_diarization(diarization_pipe, wav_path)
        transcription = speech_recognition(whisper_pipe, wav_path, speaker_segments)
        conversation = ""
        for segment in transcription:
            start = segment["start"]
            end = segment["end"]
            speaker = segment["speaker"]
            text = segment["text"]
            conversation += f"{speaker}({start}, {end}): {text}\n"
        output_path = f"{input_path}.txt"
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"parse {input_path} to:")
        print(conversation)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(conversation)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

def process_directory_audios(diarization_pipe: DiarizationPipeline, whisper_pipe: pipeline, input_dir: str):
    """
    处理目录下的所有音频文件
    :param diarization_pipe: 说话人分离模型
    :param whisper_pipe: 语音识别模型
    :param input_dir: 输入目录路径
    """
    for root_path, _, file_names in os.walk(input_dir):
        for file_name in file_names:
            file_path = os.path.join(root_path, file_name)
            if is_audio_file(file_path):
                print(f"Processing {file_path}...")
                process_single_audio(diarization_pipe, whisper_pipe, file_path)            

if __name__ == "__main__":
    try:
        _, params = load_args()
        if len(params) < 1:
            print("Please provide the audio file or directory path.")
            exit(0)

        diarization_pipe = load_local_pyannote_pipeline()
        whisper_pipe = load_local_whisper_pipeline()

        for param in params:
            if os.path.isdir(param):
                process_directory_audios(diarization_pipe, whisper_pipe, param)
            elif os.path.isfile(param):
                process_single_audio(diarization_pipe, whisper_pipe, param)
            else:
                print(f"{param} is not a valid file or directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)