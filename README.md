# py-audio2txt
语音对话转文字, 如果是多人对话则检测speaker并按speaker输出，否则只输出转换的文字  
在原语音文件相同位置生成转写文件，如 /path/to/some.mp3, 生成 /path/to/some.mp3.txt

# 依赖
1. ffmpeg
    ```shell
    sudo apt-get install ffmpeg
    ```
2. libsndfile
    ```shell
    sudo apt-get install libsndfile1
    ```
# Usage
1. 初始化
    ```shell
    cd /path/to/py-audio2txt
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    deactivate
    ```

2. 执行脚本
    ``` shell
    chmod a+x /path/to/py-audio2txt/audio2txt.sh
    sh /path/to/py-audio2txt/audio2txt.sh -h 
    ```
    ```txt
    Usage: audio2txt [options] <audio_file> <audio_dir> ...
    Dependency: ffmpeg, libsndfile
    Options:
    -v, --version   Show version
    -h, --help      Show this help message
    -b, --batch     Batch size (default: 10)
    -l, --log-level Log level (default: INFO)
    ```

# Models
本项目依赖以下AI模型  
首次执行会自动下载到 /home/your-user-dir/.cache/modelscope/hub/models/
1. 降噪增强：[iic/speech_zipenhancer_ans_multiloss_16k_base](https://modelscope.cn/models/iic/speech_zipenhancer_ans_multiloss_16k_base)
2. 语音识别：[iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn](https://modelscope.cn/models/iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn)
3. 端点检测: [iic/speech_fsmn_vad_zh-cn-16k-common-pytorch](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)
4. 语音标点：[iic/punc_ct-transformer_cn-en-common-vocab471067-large](https://modelscope.cn/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large)
5. 说话人分离：[iic/speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)