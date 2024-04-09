"""
    本步骤：SoVITS训练
    训练：SoVITS训练，GPT训练
"""

import os
import json
from subprocess import Popen

from config import python_exec, infer_device, is_half, exp_root, webui_port_main, webui_port_infer_tts, webui_port_uvr5, \
    webui_port_subfix, is_share

SoVITS_weight_root = "SoVITS_weights"
p_train_SoVITS = None


def open1Ba(batch_size, total_epoch, exp_name, text_low_lr_rate, if_save_latest, if_save_every_weights,
            save_every_epoch, gpu_numbers1Ba, pretrained_s2G, pretrained_s2D):
    Param = {}
    Param["batch_size"] = batch_size
    Param["total_epoch"] = total_epoch
    Param["exp_name"] = exp_name
    Param["text_low_lr_rate"] = text_low_lr_rate
    Param["if_save_latest"] = if_save_latest
    Param["if_save_every_weights"] = if_save_every_weights
    Param["save_every_epoch"] = save_every_epoch
    Param["gpu_numbers1Ba"] = gpu_numbers1Ba
    Param["pretrained_s2G"] = pretrained_s2G
    Param["pretrained_s2D"] = pretrained_s2D

    print("----------------执行SoVITS训练的函数的参数如下：------------------" + "\n")
    print(Param)

    global p_train_SoVITS
    if (p_train_SoVITS == None):
        with open("GPT_SoVITS/configs/s2.json") as f:
            data = f.read()
            data = json.loads(data)
        s2_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s2" % (s2_dir), exist_ok=True)
        if (is_half == False):
            data["train"]["fp16_run"] = False
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["train"]["text_low_lr_rate"] = text_low_lr_rate
        data["train"]["pretrained_s2G"] = pretrained_s2G
        data["train"]["pretrained_s2D"] = pretrained_s2D
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["save_every_epoch"] = save_every_epoch
        data["train"]["gpu_numbers"] = gpu_numbers1Ba
        data["data"]["exp_dir"] = data["s2_ckpt_dir"] = s2_dir
        data["save_weight_dir"] = SoVITS_weight_root
        data["name"] = exp_name
        # 熊
        # tmp_config_path="%s/tmp_s2.json"%tmp
        tmp_config_path = "TEMP/tmp_s2.json"
        with open(tmp_config_path, "w") as f:
            f.write(json.dumps(data))

        # 熊
        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"' % (python_exec, tmp_config_path)
        # cmd = 'python GPT_SoVITS/s2_train.py --config "%s"'%(tmp_config_path)
        yield "SoVITS训练开始：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update",
                                                                                    "visible": True}
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS = None
        yield "SoVITS训练完成", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的SoVITS训练任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


if __name__ == "__main__":

    batch_size = 5
    total_epoch = 8
    # exp_name = sys.argv[1]    # 使用：python SoVITS_train.py cxq
    exp_name = "cxq"
    text_low_lr_rate = 0.4
    if_save_latest = True
    if_save_every_weights = True
    save_every_epoch = 4
    gpu_numbers1Ba = "0"
    pretrained_s2G = "GPT_SoVITS/pretrained_models/s2G488k.pth"
    pretrained_s2D = "GPT_SoVITS/pretrained_models/s2D488k.pth"

    for result in open1Ba(batch_size, total_epoch, exp_name, text_low_lr_rate, if_save_latest, if_save_every_weights,
                          save_every_epoch, gpu_numbers1Ba, pretrained_s2G, pretrained_s2D):
        print(result)