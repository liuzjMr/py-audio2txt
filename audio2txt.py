from concurrent.futures import ThreadPoolExecutor
import datetime
import logging
from logging.handlers import QueueListener
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
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

class AudioTranscriber:
    def __init__(self, hotword: str, batch_size: int, verbose: bool = False, overwrite: bool = False):
        self.hotword = hotword
        self.batch_size = batch_size
        self.verbose = verbose
        self.overwrite = overwrite
        self.enhancer_pipe, self.inference_pipeline = self._load_modelscope_pipelines()

    def _load_modelscope_pipelines(self) -> tuple[pipeline, pipeline]:
        # 语音降噪
        noise_suppression = pipeline(
            Tasks.acoustic_noise_suppression, 
            model='iic/speech_zipenhancer_ans_multiloss_16k_base'
        )
        # 通话转写
        speech_recognition = pipeline(
            task=Tasks.auto_speech_recognition,
            # 改用支持热词的模型
            model='iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch', model_revision='v2.0.5', # 语音识别
            # model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch', model_revision='v2.0.5', # 语音识别
            vad_model='iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', vad_model_revision="v2.0.4",      # 语音端点检测
            punc_model='iic/punc_ct-transformer_cn-en-common-vocab471067-large', punc_model_revision="v2.0.4", # 语音标点
            spk_model="iic/speech_campplus_sv_zh-cn_16k-common", spk_model_revision="v2.0.2", # 说话人分离 
        )    
        return noise_suppression, speech_recognition        

    def _truncate_text(self, text: str):
        """智能截断文本以适应模型上下文窗口"""
        tokens = jieba.lcut(text)
        if len(tokens) > 512:
            # 保留首尾重要信息（开头50% + 结尾30%）
            keep_tokens = tokens[:len(tokens)//2] + tokens[-len(tokens)//3 * 2:]
            return " ".join(keep_tokens[:512])
        return text
    
    def _get_temp_path(self, prefix: str, suffix: str = ".wav") -> str:
        """生成临时文件路径"""
        with tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False) as tmp_file:
            return tmp_file.name

    def _preprocess_audio(self, input_path: str) -> Optional[str]:
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
            
            output_path = self._get_temp_path("preprocess_")
            audio.export(output_path, format="wav")
            logger.debug(f"预处理成功: {input_path} -> {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"预处理失败 [{input_path}]: {str(e)}")
            return None
        
    def _get_output_path(self, original_path: str) -> str:
        """获取输出文件路径"""
        return os.path.abspath(original_path) + ".txt"
        
    def _save_transcript(self, output_path: str, content: str):
        """保存转写结果到原路径同级目录"""
        if os.path.exists(output_path):
            os.remove(output_path)  # 删除已存在的文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.debug(f"转写结果已保存至：{output_path}")         
    
    def _enhance_wav(self, input_path: str) -> str:
        """
        音频增强: 降噪处理
        :param input_path: 输入音频文件路径
        :return: 增强后的音频文件路径
        """
        enhanced_path = self._get_temp_path(f"enhanced_", suffix=".wav")
        try:
            self.enhancer_pipe(input_path, output_path=enhanced_path)
            return enhanced_path
        except Exception as e:
            logger.warning(f"音频{input_path}降噪失败: {str(e)}")
            shutil.copyfile(input_path, enhanced_path)  # 复制原始文件
            logger.warning(f"使用原始音频文件替代: {input_path} -> {enhanced_path}")
            return input_path
    
    def _mills2timestr(self, millseconds: int) -> str:
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
      
    def _transcript_wavs(self, wav_files: list[str]) -> list[dict[str, str]]:
        wav_contents = []
        try:
            rec_result = self.inference_pipeline(
                wav_files, 
                batch_size_s=300, 
                batch_size_token_threshold_s=40,
                hotword=self.hotword,
            )
            if not rec_result:
                logger.warning("转写结果为空，可能为非音频文件")
                return wav_contents
            
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
                        contents.append(f"Speaker_{pre_speaker} {self._mills2timestr(pre_start)}: {pre_line}")
                        pre_speaker = current_speaker
                        pre_line = current_line
                        pre_start = current_start
                    else:
                        pre_line += current_line
                if pre_speaker is not None:
                    contents.append(f"Speaker_{pre_speaker} {self._mills2timestr(pre_start)}: {pre_line}")
                wav_contents.append({"wav": wav_files[i], "content": "\n".join(contents)})
                if verbose:
                    logger.info(f"音频文件 {wav_files[i]} 转写结果：\n{wav_contents[-1]['content']}")
                else:
                    logger.debug(f"音频文件 {wav_files[i]} 转写结果：\n{wav_contents[-1]['content']}")
            return wav_contents
        except Exception as e:
            logger.error(f"转写失败: {str(e)}")
            return wav_contents
        
    def _process_batch(self, file_batch: list[str]) -> int:
        count = 0
        try:
            # 预处理阶段
            preprocessed_files = []
            source_files = dict()
            for file in file_batch:
                logger.debug(f"正在预处理文件: {file}")
                output_path = self._get_output_path(file)
                if os.path.exists(output_path) and not self.overwrite:
                    logger.warning(f"文件 {output_path} 已存在，跳过处理")
                    continue
                try:
                    wav_path = self._preprocess_audio(file)
                    if wav_path is None:
                        logger.warning(f"文件预处理失败: {file}")
                        continue
                    enhanced_path = self._enhance_wav(wav_path)
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
            transcripts = self._transcript_wavs(preprocessed_files)
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
                    self._save_transcript(self._get_output_path(original_path), audio_content)
                    count += 1
                # 清理临时文件
                if os.path.exists(enhanced_path):
                    os.remove(enhanced_path)
            return count
        except Exception as batch_error:
            logger.error(f"批量处理失败: {str(batch_error)}")
            return count
        
    def process(self, audio_files: list[str]) -> int:
        """处理音频文件"""
        total = 0
        for i in range(0, len(audio_files), self.batch_size):
            batch = audio_files[i:i+self.batch_size]
            count = self._process_batch(batch)
            total += count
            logger.info(f"第 {i//self.batch_size + 1} 批（{len(batch)}个）文件处理完成，共转写 {count} 个文件")
        return total

def process_single(audio_files : list[str], batch_size, hotword: str, verbose: bool, overwrite: bool):
    """单进程处理"""
    trnascriber = AudioTranscriber(hotword=hotword, batch_size=batch_size, verbose=verbose, overwrite=overwrite)
    total = trnascriber.process(audio_files)
    logger.info(f"所有 {len(audio_files)} 个文件处理完成，共转写 {total} 个文件")   

def process_single_worker(audio_files_chunk : list[str], batch_size : int, hotword : str, verbose : bool, overwrite : bool) -> int:
    """子进程处理函数"""
    try:
        trnascriber = AudioTranscriber(hotword=hotword, batch_size=batch_size, verbose=verbose, overwrite=overwrite)
        total = 0
        for i in range(0, len(audio_files_chunk), batch_size):
            batch = audio_files_chunk[i:i+batch_size]
            count = trnascriber.process(batch)
            total += count
            logger.info(f"子进程处理批次 {i//batch_size + 1}，转写 {count} 个文件")
        return total
    except Exception as e:
        logger.error(f"子进程处理失败: {str(e)}")
        return 0
    
def process_multi(audio_files : list[str], batch_size : int, hotword : str, verbose : bool, overwrite : bool, process : int):
    """多进程处理"""
    # 分割文件列表
    chunk_size = len(audio_files) // process
    chunks = []
    for i in range(process):
        start = i * chunk_size
        end = start + chunk_size if i != process-1 else len(audio_files)
        chunks.append(audio_files[start:end])      
    
    # 多进程日志处理
    log_queue = multiprocessing.Queue()
    queue_listener = QueueListener(log_queue, logging.StreamHandler())
    queue_listener.start()  

    # 启动多进程池
    with multiprocessing.Pool(processes=process) as pool:
        args = [(chunk, batch_size, hotword, verbose, overwrite) for chunk in chunks]
        results = pool.starmap(process_single_worker, args)
    
    total = sum(results)
    logger.info(f"所有 {len(audio_files)} 个文件处理完成，共转写 {total} 个文件")      


def main(input_paths: list[str], process: int, batch_size: int, hot_words_file: str, verbose: bool, overwrite: bool):
    """主处理流程"""
    audio_files = collect_audio_files(input_paths)
    if len(audio_files) == 0:
        logger.warning("未找到有效音频文件")
        return
    
    logger.info(f"共发现 {len(audio_files)} 个音频文件")

    # 加载模型
    hot_words = load_hot_words(hot_words_file)
    hotword = '' if len(hot_words) == 0 else ' '.join(hot_words)
    if process == 1:
        logger.info("单进程处理")
        process_single(audio_files=audio_files, batch_size=batch_size, hotword=hotword, verbose=verbose, overwrite=overwrite)
    else:
        logger.info(f"多进程处理，使用 {process} 个进程")
        process_multi(audio_files=audio_files, batch_size=batch_size, hotword=hotword, verbose=verbose, process=process, overwrite=overwrite)        
        

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

def collect_audio_files(paths: list[str]) -> list[str]:
    file_queue = []
    result_lock = threading.Lock()
    audio_files = []

    if len(paths) == 0:
        logger.error("没有提供文件或目录路径")
        return audio_files

    # 第一阶段：多线程遍历目录结构
    def scan_dirs(path):
        if os.path.isfile(path):
            file_queue.append(path)
        else:
            for root, _, files in os.walk(path):
                file_queue.extend(os.path.join(root, f) for f in files)

    with ThreadPoolExecutor() as dir_executor:
        scan_pathes = []
        for path in paths:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                file_queue.append(path)
            else:
                with os.scandir(path) as entries:
                    for entry in entries:
                        if entry.is_file():
                            file_queue.append(os.path.abspath(entry.path))
                        elif entry.is_dir():
                            scan_pathes.append(os.path.join(path, entry.name))
        if len(scan_pathes) > 0:
            dir_executor.map(scan_dirs, scan_pathes)

    # 第二阶段：多线程检测音频文件
    def check_file(file_path):
        if is_audio_file(file_path):
            with result_lock:
                audio_files.append(os.path.abspath(file_path))

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as file_executor:
        file_executor.map(check_file, file_queue)

    return audio_files

if __name__ == "__main__":
    multiprocessing.freeze_support()
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
        print("  -p, --process   Number of processes (default: 1)")
        print("  -b, --batch     Batch size (default: 10)")
        print("  -l, --log-level Log level (default: INFO)")
        print("  -w, --hot-words Hot words file path (default: hotwords.txt)")
        print("  --overwrite   Overwrite existing files")
        print("  --verbose   Verbose mode")
        exit(0)    
    process = int(options.get("p", options.get("process", 1)))
    batch_size = int(options.get("b", options.get("batch", 10)))
    log_level = options.get("l", options.get("log-level", "INFO")).upper()
    hot_words_file = options.get("w", options.get("hot-words", ""))
    if hot_words_file == "":
        hot_words_file = os.path.join(get_executable_directory(), "hotwords.txt") 

    verbose = options.get("verbose", False)
    overwrite = options.get("overwrite", False)
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
    logger.info(f"当前日志级别: {log_level}")
    logger.info(f"当前批处理大小: {batch_size}")
    logger.info(f"当前热词文件: {hot_words_file}")
    main(input_paths=params, process=process, batch_size=batch_size, hot_words_file=hot_words_file, verbose=verbose, overwrite=overwrite)
    logger.info(f"总耗时: {get_duration(start_time)}")
    logger.info("处理完成，感谢使用！")
