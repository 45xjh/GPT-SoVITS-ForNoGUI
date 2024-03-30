"""
    采集到的人声的预处理第二步，去混响
"""
import os
import traceback, gradio as gr
import logging
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

logger = logging.getLogger(__name__)
import librosa, ffmpeg
import soundfile as sf
import torch
import sys
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho

now_dir = os.getcwd()
sys.path.insert(0, now_dir)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(tmp, exist_ok=True)
os.environ["TEMP"] = tmp


def uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
    weight_uvr5_root = "tools/uvr5/uvr5_weights"
    uvr5_names = []
    for name in os.listdir(weight_uvr5_root):
        if name.endswith(".pth") or "onnx" in name:
            uvr5_names.append(name.replace(".pth", ""))
    # print(uvr5_names)
    device = "cuda"
    is_half = eval("True")

    uvrParam = {}
    uvrParam["model_name"] = model_name
    uvrParam["inp_root"] = inp_root
    uvrParam["save_root_vocal"] = save_root_vocal
    uvrParam["paths"] = paths
    uvrParam["save_root_ins"] = save_root_ins
    uvrParam["agg"] = agg
    uvrParam["format0"] = format0
    print("----------------执行uvr函数的参数如下：------------------" + "\n")
    print(uvrParam)
    infos = []
    try:
        inp_root = inp_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        save_root_vocal = (
            save_root_vocal.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        save_root_ins = (
            save_root_ins.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )
        is_hp3 = "HP3" in model_name
        if model_name == "onnx_dereverb_By_FoxJoy":
            pre_fun = MDXNetDereverb(15)
        else:
            func = AudioPre if "DeEcho" not in model_name else AudioPreDeEcho
            pre_fun = func(
                agg=int(agg),
                model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
                device=device,
                is_half=is_half,
            )
        if inp_root != "":
            paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
        else:
            paths = [path.name for path in paths]
        for path in paths:
            inp_path = os.path.join(inp_root, path)
            if (os.path.isfile(inp_path) == False): continue
            need_reformat = 1
            done = 0
            try:
                info = ffmpeg.probe(inp_path, cmd="ffprobe")
                if (
                        info["streams"][0]["channels"] == 2
                        and info["streams"][0]["sample_rate"] == "44100"
                ):
                    need_reformat = 0
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                    done = 1
            except:
                need_reformat = 1
                traceback.print_exc()
            if need_reformat == 1:
                tmp_path = "%s/%s.reformatted.wav" % (
                    os.path.join(os.environ["TEMP"]),
                    os.path.basename(inp_path),
                )
                os.system(
                    "ffmpeg -i %s -vn -acodec pcm_s16le -ac 2 -ar 44100 %s -y"
                    % (inp_path, tmp_path)
                )
                inp_path = tmp_path
            try:
                if done == 0:
                    pre_fun._path_audio_(
                        inp_path, save_root_ins, save_root_vocal, format0, is_hp3
                    )
                infos.append("%s->Success" % (os.path.basename(inp_path)))
                yield "\n".join(infos)
            except:
                infos.append(
                    "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
                )
                yield "\n".join(infos)
    except:
        infos.append(traceback.format_exc())
        yield "\n".join(infos)
    finally:
        try:
            if model_name == "onnx_dereverb_By_FoxJoy":
                del pre_fun.pred.model
                del pre_fun.pred.model_
            else:
                del pre_fun.model
                del pre_fun
        except:
            traceback.print_exc()
        print("clean_empty_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    yield "\n".join(infos)


if __name__ == "__main__":
    # 采集到的人声数据的处理：1.人声伴奏分离 2.去混响 3.去延迟

    # inp_root = sys.argv[1]表示使用命令行第一个参数，格式如下：
    # python uvr_process_2.py /media/amax/4C76448F76447C28/XiongJianHui/VoiceClone/GPT-SoVITS/output/uvr5_opt
    # inp_root = sys.argv[1]

    # 2.去混响
    model_name = "onnx_dereverb_By_FoxJoy"
    # 这里必须用绝对路径
    inp_root = "/media/amax/4C76448F76447C28/XiongJianHui/VoiceClone/GPT-SoVITS/output/uvr5_opt"
    save_root_vocal = "output/uvr5_opt"
    paths = None
    save_root_ins = "output/uvr5_opt"
    agg = 10
    format0 = 'wav'
    for result in uvr(model_name, inp_root, save_root_vocal, paths, save_root_ins, agg, format0):
        print(result)

    # 得到两个文件，保留以main_vocal结尾的文件
    # vocal_audio.m4a.reformatted.wav_10.wav_main_vocal.wav
    # vocal_audio.m4a.reformatted.wav_10.wav_others.wav
    for file in os.listdir(save_root_vocal):
        # print(file)
        # 检查文件名是否以"main_vocal.wav"结尾
        if not file.endswith("main_vocal.wav"):
            file_path = os.path.join(save_root_vocal, file)
            # 删除文件
            os.remove(file_path)
