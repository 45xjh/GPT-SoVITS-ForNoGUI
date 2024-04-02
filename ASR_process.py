"""
    本步：ASR(语音转文字)
    数据处理步骤：uvr-->slice-->asr-->打标主-->一键三连(文本提取、SSL的hubert特征提取、语义token提取)
"""

import sys, os
from tools import my_utils
from subprocess import Popen
from config import python_exec, infer_device, is_half, exp_root, webui_port_main, webui_port_infer_tts, webui_port_uvr5, \
    webui_port_subfix, is_share

python_exec = sys.executable or "python"
p_asr = None

def check_fw_local_models():
    '''
    启动时检查本地是否有 Faster Whisper 模型.
    '''
    model_size_list = [
        "tiny",     "tiny.en",
        "base",     "base.en",
        "small",    "small.en",
        "medium",   "medium.en",
        "large",    "large-v1",
        "large-v2", "large-v3"]
    for i, size in enumerate(model_size_list):
        if os.path.exists(f'tools/asr/models/faster-whisper-{size}'):
            model_size_list[i] = size + '-local'
    return model_size_list
asr_dict = {
    "达摩 ASR (中文)": {
        'lang': ['zh'],
        'size': ['large'],
        'path': 'funasr_asr.py',
    },
    "Faster Whisper (多语种)": {
        'lang': ['auto', 'zh', 'en', 'ja'],
        'size': check_fw_local_models(),
        'path': 'fasterwhisper_asr.py'
    }
}

def open_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang):
    global p_asr
    if (p_asr == None):
        asr_inp_dir = my_utils.clean_path(asr_inp_dir)
        cmd = f'"{python_exec}" tools/asr/{asr_dict[asr_model]["path"]}'
        cmd += f' -i "{asr_inp_dir}"'
        cmd += f' -o "{asr_opt_dir}"'
        cmd += f' -s {asr_model_size}'
        cmd += f' -l {asr_lang}'
        cmd += " -p %s" % ("float16" if is_half == True else "float32")

        yield "ASR任务开启：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        print(cmd)
        p_asr = Popen(cmd, shell=True)
        p_asr.wait()
        p_asr = None
        yield f"ASR任务完成, 查看终端进行下一步", {"__type__": "update", "visible": True}, {"__type__": "update",
                                                                                            "visible": False}
    else:
        yield "已有正在进行的ASR任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}
        # return None

if __name__=="__main__":
    asr_inp_dir = "output/slicer_opt"
    asr_opt_dir = "output/asr_opt"
    asr_model = "达摩 ASR (中文)"
    asr_model_size = "large"
    asr_lang = "zh"


    for result in open_asr(asr_inp_dir,asr_opt_dir,asr_model,asr_model_size,asr_lang):
        print(result)
