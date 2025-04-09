# 功能点
## 语音转写
1. 语音对话转文字, 如果是多人对话则检测speaker并按speaker输出，否则只输出转换的文字  
2. 扫描给定的语音文件/文件夹，在原语音文件相同位置生成转写文件，如 /path/to/some.mp3, 生成 /path/to/some.mp3.txt
3. 可编辑 hotwords.txt 自定义热词
4. 支持的语音格式有：wav/mp3/flac/aac/ogg/m4a/opus

## 文本摘要
1. 扫描给定的txt文件/文件夹，在原txt文件相同位置生成摘要文件，如 /path/to/some.mp3.txt, 生成 /path/to/some.mp3.txt.md
2. 支持的文本格式为：UTF-8编码的txt
3. 可编辑 template.txt 自定义摘要提示词

# 依赖
1. ffmpeg
    ```shell
    sudo apt-get install ffmpeg
    ```
2. libsndfile
    ```shell
    sudo apt-get install libsndfile1
    ```
# 用法
## 初始化
```shell
cd /path/to/py-audio2txt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
deactivate
```

## 语音转写
大约需要3.6G显存，支持多进程，根据你的显存大小设置进程数
``` shell
chmod a+x /path/to/py-audio2txt/audio2txt.sh
sh /path/to/py-audio2txt/audio2txt.sh -h 
sh /path/to/py-audio2txt/audio2txt.sh --verbose -p4 /path/to/a.mp3 /path/to/b.ogg /path/to/audio_dir
```
```txt
Usage: audio2txt [options] <audio_file> <audio_dir> ...
Dependency: ffmpeg, libsndfile
Options:
  -v, --version   Show version
  -h, --help      Show this help message
  -p, --process   Number of processes (default: 1)  
  -b, --batch     Batch size (default: 10)
  -l, --log-level Log level (default: INFO)
  -w, --hot-words Hot words file path (default: hotwords.txt)
  --verbose   Verbose mode
``` 

## 文本摘要
大约需要16G显存，单进程
``` shell
chmod a+x /path/to/py-audio2txt/summary.sh
sh /path/to/py-audio2txt/summary.sh -h 
sh /path/to/py-audio2txt/summary.sh --verbose /path/to/a.txt /path/to/txt_dir1
```
```txt
Usage: summary [options] <txt_file> <txt_dir> ...
Options:
  -v, --version   Show version
  -h, --help      Show this help message
  -l, --log-level Log level (default: INFO)
  -t, --template   Summary template file (default: template.txt)
  --verbose   Verbose mode
``` 

# Models
本项目依赖以下AI模型  
首次执行会自动下载到 /home/your-user-dir/.cache/modelscope/hub/models/
## 语音转写
1. 降噪增强：[iic/speech_zipenhancer_ans_multiloss_16k_base](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
2. 语音识别：[iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch](https://modelscope.cn/models/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch)
3. 端点检测: [iic/speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
4. 语音标点：[iic/punc_ct-transformer_cn-en-common-vocab471067-large](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large)
5. 说话人分离：[iic/speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)

## 文本摘要
1. 通义千问：[Qwen/Qwen2.5-7B-Instruct](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)