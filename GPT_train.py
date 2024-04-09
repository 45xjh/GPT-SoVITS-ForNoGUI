"""
    本步骤：GPT训练
    训练：sovits训练，GPT训练
"""

import os, yaml
from subprocess import Popen

from config import python_exec, infer_device, is_half, exp_root, webui_port_main, webui_port_infer_tts, webui_port_uvr5, \
    webui_port_subfix, is_share

GPT_weight_root = "GPT_weights"
p_train_GPT = None


def open1Bb(batch_size, total_epoch, exp_name, if_dpo, if_save_latest, if_save_every_weights, save_every_epoch,
            gpu_numbers, pretrained_s1):
    Param = {}
    Param["batch_size"] = batch_size
    Param["total_epoch"] = total_epoch
    Param["exp_name"] = exp_name
    Param["if_dpo"] = if_dpo
    Param["if_save_latest"] = if_save_latest
    Param["if_save_every_weights"] = if_save_every_weights
    Param["save_every_epoch"] = save_every_epoch
    Param["gpu_numbers"] = gpu_numbers
    Param["pretrained_s1"] = pretrained_s1

    print("----------------执行GPT训练的函数的参数如下：------------------" + "\n")
    print(Param)

    global p_train_GPT
    if (p_train_GPT == None):
        with open("GPT_SoVITS/configs/s1longer.yaml") as f:
            data = f.read()
            data = yaml.load(data, Loader=yaml.FullLoader)
        s1_dir = "%s/%s" % (exp_root, exp_name)
        os.makedirs("%s/logs_s1" % (s1_dir), exist_ok=True)
        if (is_half == False):
            data["train"]["precision"] = "32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"] = batch_size
        data["train"]["epochs"] = total_epoch
        data["pretrained_s1"] = pretrained_s1
        data["train"]["save_every_n_epoch"] = save_every_epoch
        data["train"]["if_save_every_weights"] = if_save_every_weights
        data["train"]["if_save_latest"] = if_save_latest
        data["train"]["if_dpo"] = if_dpo
        data["train"]["half_weights_save_dir"] = GPT_weight_root
        data["train"]["exp_name"] = exp_name
        data["train_semantic_path"] = "%s/6-name2semantic.tsv" % s1_dir
        data["train_phoneme_path"] = "%s/2-name2text.txt" % s1_dir
        data["output_dir"] = "%s/logs_s1" % s1_dir

        os.environ["_CUDA_VISIBLE_DEVICES"] = gpu_numbers.replace("-", ",")
        os.environ["hz"] = "25hz"
        # 熊
        # tmp_config_path = "%s/tmp_s1.yaml" % tmp
        tmp_config_path = "TEMP/tmp_s1.yaml"

        with open(tmp_config_path, "w") as f:
            f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" ' % (python_exec, tmp_config_path)
        yield "GPT训练开始：%s" % cmd, {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT = None
        yield "GPT训练完成", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的GPT训练任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


if __name__ == "__main__":

    batch_size = 5
    total_epoch = 15
    # exp_name = sys.argv[1]    # 使用：python GPT_train.py cxq
    exp_name = "cxq"
    if_dpo = False
    if_save_latest = True
    if_save_every_weights = True
    save_every_epoch = 5
    gpu_numbers = "0"
    pretrained_s1 = 'GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt'

    for result in open1Bb(batch_size, total_epoch, exp_name, if_dpo, if_save_latest, if_save_every_weights,
                          save_every_epoch, gpu_numbers, pretrained_s1):
        print(result)
