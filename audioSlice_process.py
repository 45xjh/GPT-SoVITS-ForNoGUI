"""
    本步：切割
    数据处理步骤：uvr-->slice-->asr-->打标主-->一键三连(文本提取、SSL的hubert特征提取、语义token提取)
"""

import sys, os
from tools import my_utils
from subprocess import Popen


python_exec = sys.executable or "python"
ps_slice = []

def open_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha, n_parts):
    global ps_slice
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    if (os.path.exists(inp) == False):
        yield "输入路径不存在", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
        return
    if os.path.isfile(inp):
        n_parts = 1
    elif os.path.isdir(inp):
        pass
    else:
        yield "输入路径存在但既不是文件也不是文件夹", {"__type__": "update", "visible": True}, {"__type__": "update",
                                                                                                "visible": False}
        return
    if (ps_slice == []):
        for i_part in range(n_parts):
            cmd = '"%s" tools/slice_audio.py "%s" "%s" %s %s %s %s %s %s %s %s %s''' % (
            python_exec, inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max, alpha,
            i_part, n_parts)
            print(cmd)
            p = Popen(cmd, shell=True)
            ps_slice.append(p)
        yield "切割执行中", {"__type__": "update", "visible": False}, {"__type__": "update", "visible": True}
        for p in ps_slice:
            p.wait()
        ps_slice = []
        yield "切割结束", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}
    else:
        yield "已有正在进行的切割任务，需先终止才能开启下一次任务", {"__type__": "update", "visible": False}, {
            "__type__": "update", "visible": True}


if __name__=="__main__":
    inp = "output/uvr5_opt"
    opt_root = "output/slicer_opt"
    threshold = -34
    min_length = 4000
    min_interval = 300
    hop_size = 10
    max_sil_kept = 500
    _max = 0.9
    alpha = 0.25
    n_parts = 4

    for result in open_slice(inp,opt_root,threshold,min_length,min_interval,hop_size,max_sil_kept,_max,alpha,n_parts):
        print(result)
