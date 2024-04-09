"""
    本步：一键三连，即文本bert特征提取，ssl的hubert特征提取，语义token提取
    数据处理步骤：uvr-->slice-->asr-->打标主-->一键三连(文本提取、SSL的hubert特征提取、语义token提取)
"""

import os,sys
import traceback
import platform
import psutil
import signal
from subprocess import Popen
from tools import my_utils
from config import python_exec, infer_device, is_half, exp_root, webui_port_main, webui_port_infer_tts, webui_port_uvr5, \
    webui_port_subfix, is_share

ps1abc = []


def open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c, bert_pretrained_dir,
             ssl_pretrained_dir, pretrained_s2G_path):
    abcParam = {}
    abcParam["inp_text"] = inp_text
    abcParam["inp_wav_dir"] = inp_wav_dir
    abcParam["exp_name"] = exp_name
    abcParam["gpu_numbers1a"] = gpu_numbers1a
    abcParam["gpu_numbers1Ba"] = gpu_numbers1Ba
    abcParam["gpu_numbers1c"] = gpu_numbers1c
    abcParam["bert_pretrained_dir"] = bert_pretrained_dir
    abcParam["ssl_pretrained_dir"] = ssl_pretrained_dir
    abcParam["pretrained_s2G_path"] = pretrained_s2G_path

    print("----------------执行一键三连函数的参数如下：------------------" + "\n")
    print(abcParam)

    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if (ps1abc == []):
        opt_dir = "%s/%s" % (exp_root, exp_name)
        try:
            #############################1a
            path_text = "%s/2-name2text.txt" % opt_dir
            if (os.path.exists(path_text) == False or (os.path.exists(path_text) == True and len(
                    open(path_text, "r", encoding="utf8").read().strip("\n").split("\n")) < 2)):
                config = {
                    "inp_text": inp_text,
                    "inp_wav_dir": inp_wav_dir,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "bert_pretrained_dir": bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names = gpu_numbers1a.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a-ing", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
                for p in ps1abc: p.wait()

                opt = []
                for i_part in range(all_parts):  # txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    with open(txt_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")

            yield "进度：1a-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            #############################1b
            config = {
                "inp_text": inp_text,
                "inp_wav_dir": inp_wav_dir,
                "exp_name": exp_name,
                "opt_dir": opt_dir,
                "cnhubert_base_dir": ssl_pretrained_dir,
            }
            gpu_names = gpu_numbers1Ba.split("-")
            all_parts = len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py' % python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            yield "进度：1a-done, 1b-ing", {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                     "visible": True}
            for p in ps1abc: p.wait()
            yield "进度：1a1b-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if (os.path.exists(path_semantic) == False or (
                    os.path.exists(path_semantic) == True and os.path.getsize(path_semantic) < 31)):
                config = {
                    "inp_text": inp_text,
                    "exp_name": exp_name,
                    "opt_dir": opt_dir,
                    "pretrained_s2G": pretrained_s2G_path,
                    "s2config_path": "GPT_SoVITS/configs/s2.json",
                }
                gpu_names = gpu_numbers1c.split("-")
                all_parts = len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": gpu_names[i_part],
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py' % python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                yield "进度：1a1b-done, 1cing", {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                          "visible": True}
                for p in ps1abc: p.wait()

                opt = ["item_name\tsemantic_audio"]
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r", encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)
                with open(path_semantic, "w", encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                yield "进度：all-done", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
            ps1abc = []
            yield "一键三连进程结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        except:
            traceback.print_exc()
            close1abc()
            yield "一键三连中途报错", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的一键三连任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc = []
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}


system = platform.system()


def kill_process(pid):
    if (system == "Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


if __name__ == "__main__":
    inp_text = 'output/asr_opt/slicer_opt.list'
    inp_wav_dir = 'output/slicer_opt'
    # exp_name = sys.argv[1]    # 使用：python prepare_dataset.py cxq
    exp_name = 'cxq'  # 实验名/模型名
    gpu_numbers1a = '0-0'
    gpu_numbers1Ba = '0-0'
    gpu_numbers1c = '0-0'
    bert_pretrained_dir = 'GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large'
    ssl_pretrained_dir = 'GPT_SoVITS/pretrained_models/chinese-hubert-base'
    pretrained_s2G_path = 'GPT_SoVITS/pretrained_models/s2G488k.pth'

    for result in open1abc(inp_text, inp_wav_dir, exp_name, gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c, bert_pretrained_dir,
                           ssl_pretrained_dir, pretrained_s2G_path):
        print(result)
