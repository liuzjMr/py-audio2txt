import os
import tempfile
from modelscope.pipelines import pipeline
from pydub import AudioSegment
from common import load_args
from modelscope.utils.constant import Tasks

# 语音降噪
enhancer_pipe = pipeline(
    Tasks.acoustic_noise_suppression, 
    model='iic/speech_zipenhancer_ans_multiloss_16k_base'
)
# 通话转写
inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn', model_revision='v2.0.4', # 语音识别
    vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', vad_model_revision="v2.0.4",      # 语音端点检测
    punc_model='iic/punc_ct-transformer_cn-en-common-vocab471067-large', punc_model_revision="v2.0.4", # 语音标点
    spk_model="iic/speech_campplus_sv_zh-cn_16k-common", spk_model_revision="v2.0.2", # 说话人分离
)

def preprocess_audio(input_path: str) -> str:
    """
    音频预处理: 统一转换为单声道、16kHz的WAV格式
    :param input_path: 输入音频文件路径
    :return: 处理后的WAV文件路径
    """
    temp_id = os.path.splitext(os.path.basename(input_path))[0]
    temp_dir = tempfile.gettempdir()
    wav_path = os.path.join(temp_dir, f"temp_{temp_id}.wav")
    
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        raise ValueError(f"文件处理失败（可能为非音频文件）: {str(e)}")

def is_audio_file(file_path: str) -> bool:
    """
    判断文件是否为音频文件(通过pydub加载文件头信息来判定)
    """
    try:
        # 尝试加载文件头信息（无需完整加载音频）
        AudioSegment.from_file(file_path).dBsFS
        return True
    except:
        # 补充扩展名验证（兼容pydub不支持但实际有效的格式）
        audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
        return any(file_path.lower().endswith(ext) for ext in audio_extensions)
    
def enhance_wav(input_path: str) -> str:
    """
    音频增强: 降噪处理
    :param input_path: 输入音频文件路径
    :return: 增强后的音频文件路径
    """
    temp_id = os.path.splitext(os.path.basename(input_path))[0]
    temp_dir = tempfile.gettempdir()
    enhanced_path = os.path.join(temp_dir, f"enhanced_{temp_id}.wav")
    
    try:
        enhancer_pipe(input_path, output_path=enhanced_path)
        return enhanced_path
    except Exception as e:
        raise ValueError(f"音频增强失败: {str(e)}")
    
      
def transcript_wavs(wav_files: list[str]) -> list[dict[str, str]]:
    wav_contents = []
    try:
        rec_result = inference_pipeline(wav_files, batch_size_s=300, batch_size_token_threshold_s=40)
        for i, item in enumerate(rec_result):
            if item['sentence_info'] is None:
                wav_contents.append({"wav": wav_files[i], "content": ""})
                print(f"音频文件 {wav_files[i]} 处理失败，可能为非音频文件")
                continue
            info = item['sentence_info']
            pre_speaker = None
            lines = []
            pre_line = ""
            speakers = []
            for item in info:
                speakers.append(str(item['spk']))
                current_speaker = f"Speaker_{item['spk']}"
                current_line = item['text']
                if pre_speaker is None:
                    pre_speaker = current_speaker
                    pre_line = current_line
                elif pre_speaker != current_speaker:
                    lines.append(f"{pre_speaker}: {pre_line}")
                    pre_speaker = current_speaker
                    pre_line = current_line
                else:
                    pre_line += current_line
            if pre_speaker is not None:
                lines.append(f"{pre_speaker}: {pre_line}")
            content = "\n".join(lines)
            if len(set(speakers)) == 1:
                content = content.replace(f"Speaker_{speakers[0]}: ", "")
            wav_contents.append({"wav": wav_files[i], "content": content})
        return wav_contents
    except Exception as e:
        print(f"转写失败: {str(e)}")
        return wav_contents
    

def collect_audio_files(paths: list[str]) -> list[str]:
    """递归收集所有音频文件路径"""
    audio_files = []
    for path in paths:
        if os.path.isfile(path):
            if is_audio_file(path):
                audio_files.append(os.path.abspath(path))
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if is_audio_file(file_path):
                        audio_files.append(os.path.abspath(file_path))
    return sorted(list(set(audio_files)))  # 去重并排序

def save_transcript(original_path: str, content: str):
    """保存转写结果到原路径同级目录"""
    txt_path = f"{original_path}.transcript.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)  # 删除已存在的文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"转写结果已保存至：{txt_path}")

def process_batch(file_batch: list[str]):
    try:
        # 预处理阶段
        preprocessed_files = []
        source_files = dict()
        for file in file_batch:
            try:
                wav_path = preprocess_audio(file)
                enhanced_path = enhance_wav(wav_path)
                preprocessed_files.append(enhanced_path)
                source_files[enhanced_path] = file
                print(f"文件预处理成功: {file} -> {enhanced_path}")
            except Exception as e:
                print(f"文件预处理失败 [{file}]: {str(e)}")
                continue
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
        
        # 批量转写
        transcripts = transcript_wavs(preprocessed_files)
        print(f"批量转写完成，共处理 {len(preprocessed_files)} 个文件")
        print("转写结果：")
        for transcript in transcripts:
            print(f"文件 {transcript['wav']} 转写结果：\n{transcript['content']}")
        
        # 结果保存与清理
        for transcript in transcripts:
            enhanced_path = transcript['wav']
            original_path = source_files.get(enhanced_path)
            audio_content = transcript['content']
            if len(audio_content) == 0:
                print(f"音频文件 {original_path} 转写结果为空")
                continue
            # 保存转写结果
            if original_path:
                save_transcript(original_path, audio_content)
            # 清理临时文件
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)
    except Exception as batch_error:
        print(f"批量处理失败: {str(batch_error)}")


def main(input_paths: list[str], batch_size: int = 10):
    """主处理流程"""
    audio_files = collect_audio_files(input_paths)
    print(f"共发现 {len(audio_files)} 个音频文件")
    
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        print(f"\n正在处理第 {i//batch_size +1} 批文件（{len(batch)}个）")
        process_batch(batch)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("请通过命令行参数指定输入路径")
        sys.exit(1)
        
    main(sys.argv[1:], batch_size=10)
