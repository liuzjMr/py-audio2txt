import datetime
import os
import sys


def get_executable_directory():
    if getattr(sys, 'frozen', False):  # 判断是否为打包后的可执行文件
        executable_path = os.path.realpath(sys.executable)  # 获取实际可执行文件的路径
        directory = os.path.dirname(executable_path)  # 获取实际可执行文件所在的目录
    else:
        directory = os.path.dirname(os.path.realpath(__file__))  # 获取脚本文件所在的目录
    return directory

def get_duration(start_time: datetime) -> str:
    # 获取当前时间
    now = datetime.datetime.now()

    # 计算时间差
    duration = now - start_time

    # 提取天、秒、小时、分钟
    days = duration.days
    day_seconds = duration.seconds
    hours = day_seconds // 3600
    day_seconds = day_seconds % 3600
    minutes = day_seconds // 60
    day_seconds = day_seconds % 60
    seconds = day_seconds // 1

    # 构建人类友好的字符串
    parts = []
    if days > 0:
        parts.append(f"{days} 天")
    if hours > 0:
        parts.append(f"{hours} 小时")
    if minutes > 0:
        parts.append(f"{minutes} 分钟")
    if seconds > 0:
        parts.append(f"{seconds} 秒")

    return ", ".join(parts)


def resolve_path(path):
    """
    判断给定的路径是绝对路径还是相对路径，并将相对路径转换为绝对路径。

    :param path: 待检查的路径字符串
    :return: 绝对路径字符串
    """
    # 判断是否为绝对路径
    if os.path.isabs(path):
        # 如果已经是绝对路径，则直接返回
        return path
    else:
        # 如果是相对路径，则使用 os.getcwd() 获取当前工作目录，并将其与相对路径拼接
        current_dir = os.getcwd()
        absolute_path = os.path.join(current_dir, path)
        return absolute_path
    

def load_args() -> tuple[dict[str, str], list[str]]:
    """
    解析命令行参数，并将其转换为字典和列表。
    :return: (options, params) 其中 options 是一个字典，params 是一个列表
    """
    # 获取命令行参数
    # sys.argv[0] 是脚本名称，sys.argv[1:] 是参数列表
    # sys.argv = ['script.py', '--arg1=value1', '-b2', '-c=2', 'value2', '-a', 'value3']
    options = dict()
    params = []
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        if arg.startswith('--') and len(arg) > 2:
            # 处理长参数（如 --depth=2）
            if '=' in arg:
                # 分割等号后的部分（如 --depth=2 → 值取 "2"）
                key_value = arg[2:].split('=', 1)
                key = key_value[0]
                value = key_value[1].strip()
            else:
                # 如果没有等号，则将值设为 True
                key = arg[2:]
                value = True
            # 将参数名转换为小写并存储
            options[key.lower()] = value
        elif arg.startswith('-') and len(arg) > 1:
            # 处理短参数（如 -d=2、-d2、-d）
            key = arg[1]  # 参数名为第一个字符
            if len(arg) == 2:
                # 如果只有一个字符，则将值设为 True
                value = True
            else:
                value_part = arg[2:].strip()  # 剩余部分为值
                if value_part[0] == '=':
                    # 如果值以等号开头，则取等号后的部分
                    value = value_part[1:].strip()
                else:
                    # 如果没有等号，则取剩余部分
                    # 例如 -d2 → 值取 "2"
                    value = value_part
            options[key.lower()] = value
        else:
            # 如果参数没有前缀，则将其视为位置参数
            params.append(arg)
    return options, params