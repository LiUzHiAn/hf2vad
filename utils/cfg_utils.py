import os
from shutil import copyfile


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def preprocess_cfg(config, cfg_file):
    makedir(config["ckpt_root"])
    makedir(config["log_root"])
    makedir(config["eval_root"])

    ckpt_dir = os.path.join(config["ckpt_root"], config["exp_name"])
    makedir(ckpt_dir)

    log_dir = os.path.join(config["log_root"], config["exp_name"])
    makedir(log_dir)

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    makedir(eval_dir)

    copyfile(cfg_file, os.path.join(log_dir, "cfg.yaml"))
    paths = {}
    paths["ckpt_dir"] = ckpt_dir
    paths["log_dir"] = log_dir
    paths["eval_dir"] = eval_dir
    return paths
