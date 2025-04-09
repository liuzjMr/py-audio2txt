import datetime
import logging
import os
import sys
import tempfile
from typing import Optional
import jieba
from modelscope.pipelines import pipeline
from pydub import AudioSegment
from common import get_duration, get_executable_directory, load_args
from modelscope.utils.constant import Tasks

# 设置jieba的临时目录，防止出现权限问题
jieba.dt.tmp_dir = os.path.expanduser("~/.cache/")
jieba.initialize()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_hot_words(hot_words_file : str) -> list[str]:
    """加载热词"""
    if not os.path.exists(hot_words_file):
        logger.warning(f"热词文件 {hot_words_file} 不存在，使用默认热词")
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    hot_words = []
    with open(hot_words_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and not line.startswith("#"):
                hot_words.append(line)
    return hot_words

def load_modelscope_pipelines() -> tuple[pipeline, pipeline]:
    # 语音降噪
    enhancer_pipe = pipeline(
        Tasks.acoustic_noise_suppression, 
        model='iic/speech_zipenhancer_ans_multiloss_16k_base'
    )
    # 通话转写
    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        # 改用支持热词的模型
        model='iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', model_revision='v2.0.5', # 语音识别
        # model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch', model_revision='v2.0.5', # 语音识别
        vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', vad_model_revision="v2.0.4",      # 语音端点检测
        punc_model='iic/punc_ct-transformer_cn-en-common-vocab471067-large', punc_model_revision="v2.0.4", # 语音标点
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common", spk_model_revision="v2.0.2", # 说话人分离 
    )    
    return enhancer_pipe, inference_pipeline

def get_temp_path(prefix: str, suffix: str = ".wav") -> str:
    """生成临时文件路径"""
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as tmp_file:
        return tmp_file.name

def preprocess_audio(input_path: str) -> Optional[str]:
    """
    音频预处理: 统一转换为单声道、16kHz的WAV格式
    返回处理后的临时文件路径,失败返回None
    """
    try:
        audio = AudioSegment.from_file(input_path)
        if audio.channels > 1:
            audio = audio.set_channels(1)
        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        
        output_path = get_temp_path("preprocess_")
        audio.export(output_path, format="wav")
        logger.info(f"预处理成功: {input_path} -> {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"预处理失败 [{input_path}]: {str(e)}")
        return None

def is_audio_file(file_path: str) -> bool:
    """验证是否为有效音频文件"""
    try:
        # 快速验证文件头
        AudioSegment.from_file(file_path).duration_seconds
        return True
    except:
        # 扩展名白名单验证
        valid_ext = {'.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a', '.opus'}
        return os.path.splitext(file_path)[1].lower() in valid_ext
    
def safe_remove_file(path: str) -> None:
    """安全删除文件"""
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.debug(f"已清理临时文件: {path}")
    except Exception as e:
        logger.warning(f"文件清理失败 [{path}]: {str(e)}")    
    
def enhance_wav(enhancer_pipe : pipeline, input_path: str) -> str:
    """
    音频增强: 降噪处理
    :param input_path: 输入音频文件路径
    :return: 增强后的音频文件路径
    """
    enhanced_path = get_temp_path(f"enhanced_", suffix=".wav")
    
    try:
        enhancer_pipe(input_path, output_path=enhanced_path)
        return enhanced_path
    except Exception as e:
        logger.warning(f"音频{input_path}降噪失败: {str(e)}")
        os.copy_file_range(input_path, enhanced_path)  # 复制原始文件
        logger.warning(f"使用原始音频文件替代: {input_path} -> {enhanced_path}")
        return input_path
    
def mills2timestr(millseconds: int) -> str:
    remain_mills = int(millseconds % 1000)
    total_seconds = int(millseconds // 1000)
    remain_seconds = int(total_seconds % 60)
    total_minutes = int(total_seconds // 60)
    remain_minutes = int(total_minutes % 60)
    total_hours = int(total_minutes // 60)
    if total_hours > 0:
        return f"{total_hours:02d}:{remain_minutes:02d}:{remain_seconds:02d}.{remain_mills:03d}"
    else:
        return f"{remain_minutes:02d}:{remain_seconds:02d}.{remain_mills:03d}"  
      
def transcript_wavs(inference_pipeline : pipeline, hotword : str, wav_files: list[str], verbose: bool) -> list[dict[str, str]]:
    wav_contents = []
    try:
        rec_result = inference_pipeline(
            wav_files, 
            batch_size_s=300, 
            batch_size_token_threshold_s=40,
            hotword=hotword,
        )
        for i, item in enumerate(rec_result):
            if item['sentence_info'] is None:
                wav_contents.append({"wav": wav_files[i], "content": ""})
                logger.warning(f"音频文件 {wav_files[i]} 处理失败，可能为非音频文件")
                continue
            info = item['sentence_info']
            speakers = set()       
            lines = []
            for item in info:
                spk = item['spk']
                txt = item['text']
                if spk is None or txt is None or len(txt) == 0:
                    continue
                start = int(item['start'])
                lines.append({'spk': spk, 'text': txt, 'start': start})
                speakers.add(spk)    
            
            if len(speakers) < 2:
                wav_contents.append({"wav": wav_files[i], "content": "\n".join([item['text'] for item in lines])})
                logger.warning(f"音频文件 {wav_files[i]} 未检测多多个说话人，可能为单人音频")
                continue

            pre_speaker = None
            pre_line = ""
            pre_start = 0
            contents = []
            for line in lines:
                current_speaker = line['spk']
                current_line = line['text']
                current_start = line['start']
                if pre_speaker is None:
                    pre_speaker = current_speaker
                    pre_line = current_line
                    pre_start = current_start
                elif pre_speaker != current_speaker:
                    contents.append(f"Speaker_{pre_speaker} {mills2timestr(pre_start)}: {pre_line}")
                    pre_speaker = current_speaker
                    pre_line = current_line
                    pre_start = current_start
                else:
                    pre_line += current_line
            if pre_speaker is not None:
                contents.append(f"Speaker_{pre_speaker} {mills2timestr(pre_start)}: {pre_line}")
            wav_contents.append({"wav": wav_files[i], "content": "\n".join(contents)})
            if verbose:
                logger.info(f"音频文件 {wav_files[i]} 转写结果：\n{wav_contents[-1]['content']}")
            else:
                logger.debug(f"音频文件 {wav_files[i]} 转写结果：\n{wav_contents[-1]['content']}")
        return wav_contents
    except Exception as e:
        logger.error(f"转写失败: {str(e)}")
        return wav_contents
    

def collect_audio_files(paths: list[str]) -> list[str]:
    """递归查找有效音频文件"""
    audio_files = []
    for path in paths:
        if os.path.isfile(path):
            if is_audio_file(path):
                audio_files.append(os.path.abspath(path))
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    full_path = os.path.join(root, f)
                    if is_audio_file(full_path):
                        audio_files.append(os.path.abspath(full_path))
    return audio_files

def save_transcript(original_path: str, content: str):
    """保存转写结果到原路径同级目录"""
    txt_path = f"{original_path}.txt"
    if os.path.exists(txt_path):
        os.remove(txt_path)  # 删除已存在的文件
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.debug(f"转写结果已保存至：{txt_path}")

def process_batch(enhancer_pipe, inference_pipeline : pipeline, hotword: str, file_batch: list[str], verbose: bool) -> int:
    count = 0
    try:
        # 预处理阶段
        preprocessed_files = []
        source_files = dict()
        for file in file_batch:
            logger.debug(f"正在预处理文件: {file}")
            try:
                wav_path = preprocess_audio(file)
                if wav_path is None:
                    logger.warning(f"文件预处理失败: {file}")
                    continue
                enhanced_path = enhance_wav(enhancer_pipe, wav_path)
                preprocessed_files.append(enhanced_path)
                source_files[enhanced_path] = file
                logger.debug(f"文件预处理成功: {file} -> {enhanced_path}")
            except Exception as e:
                logger.error(f"文件预处理失败 [{file}]: {str(e)}")
                continue
            finally:
                if wav_path and os.path.exists(wav_path):
                    os.remove(wav_path)
        
        # 批量转写
        transcripts = transcript_wavs(inference_pipeline, hotword, preprocessed_files, verbose)
        logger.debug(f"批量转写完成，共处理 {len(preprocessed_files)} 个文件")
        logger.debug("转写结果：")
        for transcript in transcripts:
            logger.debug(f"文件 {transcript['wav']} 转写结果：\n{transcript['content']}")
        
        # 结果保存与清理
        for transcript in transcripts:
            enhanced_path = transcript['wav']
            original_path = source_files.get(enhanced_path)
            audio_content = transcript['content']
            if len(audio_content) == 0:
                logger.warning(f"音频文件 {original_path} 转写结果为空")
                continue
            # 保存转写结果
            if original_path:
                save_transcript(original_path, audio_content)
                count += 1
            # 清理临时文件
            if os.path.exists(enhanced_path):
                os.remove(enhanced_path)
        return count
    except Exception as batch_error:
        logger.error(f"批量处理失败: {str(batch_error)}")
        return count


def main(input_paths: list[str], batch_size: int, hot_words_file: str, verbose: bool):
    """主处理流程"""
    audio_files = collect_audio_files(input_paths)
    if len(audio_files) == 0:
        logger.warning("未找到有效音频文件")
        return
    
    logger.info(f"共发现 {len(audio_files)} 个音频文件")

    # 加载模型
    hot_words = load_hot_words(hot_words_file)
    hotword = '' if len(hot_words) == 0 else ' '.join(hot_words)
    enhancer_pipe, inference_pipeline = load_modelscope_pipelines()
    if verbose:
        logger.info("hotword: " + hotword)
    logger.info("模型加载完成")

    total = 0
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i+batch_size]
        logger.info(f"\n正在处理第 {i//batch_size +1} 批文件（{len(batch)}个）")
        count = process_batch(enhancer_pipe, inference_pipeline, hotword, batch, verbose)
        total += count
        logger.info(f"第 {i//batch_size +1} 批（{len(batch)}个）文件处理完成, 共转写 {count} 个文件")
    logger.info(f"所有{len(audio_files)}个文件处理完成, 共转写 {total} 个文件")


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    options, params = load_args()
    if "v" in options or "version" in options:
        print("audio2txt Version: 1.1.0")
        print("Author: sssxyd@gmail.com")
        print("Repo: https://github.com/sssxyd/py-audio2txt")
        print("License: Apache-2.0")
        print("Dependency: ffmpeg, libsndfile")
        exit(0)
    if "h" in options or "help" in options or len(params) == 0:
        print("Usage: audio2txt [options] <audio_file> <audio_dir> ...")
        print("Dependency: ffmpeg, libsndfile")
        print("Options:")
        print("  -v, --version   Show version")
        print("  -h, --help      Show this help message")
        print("  -b, --batch     Batch size (default: 10)")
        print("  -l, --log-level Log level (default: INFO)")
        print("  -w, --hot-words Hot words file path (default: hotwords.txt)")
        print("  --verbose   Verbose mode")
        exit(0)    
    batch_size = int(options.get("b", options.get("batch", 10)))
    log_level = options.get("l", options.get("log-level", "INFO")).upper()
    hot_words_file = options.get("w", options.get("hot-words", ""))
    if hot_words_file == "":
        hot_words_file = os.path.join(get_executable_directory(), "hotwords.txt") 

    verbose = options.get("verbose", False)
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logger.info(f"当前日志级别: {log_level}")
    logger.info(f"当前批处理大小: {batch_size}")
    logger.info(f"当前热词文件: {hot_words_file}")
    main(input_paths=params, batch_size=batch_size, hot_words_file=hot_words_file, verbose=verbose)
    logger.info(f"总耗时: {get_duration(start_time)}")
    logger.info("处理完成，感谢使用！")
