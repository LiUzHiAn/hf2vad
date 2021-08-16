import argparse
import os
import torch
import cv2
import joblib
import pickle
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import yaml
from models.mem_cvae import HFVAD
from datasets.dataset import Chunked_sample_dataset
from utils.eval_utils import save_evaluation_curves

METADATA = {
    "ped2": {
        "testing_video_num": 12,
        "testing_frames_cnt": [180, 180, 150, 180, 150, 180, 180, 180, 120, 150,
                               180, 180]
    },
    "avenue": {
        "testing_video_num": 21,
        "testing_frames_cnt": [1439, 1211, 923, 947, 1007, 1283, 605, 36, 1175, 841,
                               472, 1271, 549, 507, 1001, 740, 426, 294, 248, 273,
                               76],
    },
    "shanghaitech": {
        "testing_video_num": 107,
        "testing_frames_cnt": [265, 433, 337, 601, 505, 409, 457, 313, 409, 337,
                               337, 457, 577, 313, 529, 193, 289, 289, 265, 241,
                               337, 289, 265, 217, 433, 409, 529, 313, 217, 241,
                               313, 193, 265, 317, 457, 337, 361, 529, 409, 313,
                               385, 457, 481, 457, 433, 385, 241, 553, 937, 865,
                               505, 313, 361, 361, 529, 337, 433, 481, 649, 649,
                               409, 337, 769, 433, 241, 217, 265, 265, 217, 265,
                               409, 385, 481, 457, 313, 601, 241, 481, 313, 337,
                               457, 217, 241, 289, 337, 313, 337, 265, 265, 337,
                               361, 433, 241, 433, 601, 505, 337, 601, 265, 313,
                               241, 289, 361, 385, 217, 337, 265]
    },

}


def evaluate(config, ckpt_path, testing_chunked_samples_file, training_stats_path, suffix):
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    device = config["device"]
    num_workers = config["num_workers"]

    testset_num_frames = np.sum(METADATA[dataset_name]["testing_frames_cnt"])

    eval_dir = os.path.join(config["eval_root"], config["exp_name"])
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  ).to(device).eval()

    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    # print("load pre-trained success!")

    #  get training stats
    if training_stats_path is not None:
        training_scores_stats = torch.load(training_stats_path)

        of_mean, of_std = np.mean(training_scores_stats["of_training_stats"]), \
                          np.std(training_scores_stats["of_training_stats"])
        frame_mean, frame_std = np.mean(training_scores_stats["frame_training_stats"]), \
                                np.std(training_scores_stats["frame_training_stats"])

    score_func = nn.MSELoss(reduction="none")

    dataset_test = Chunked_sample_dataset(testing_chunked_samples_file)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=128, num_workers=num_workers, shuffle=False)

    # bbox anomaly scores for each frame
    frame_bbox_scores = [{} for i in range(testset_num_frames.item())]
    for test_data in tqdm(dataloader_test, desc="Eval: ", total=len(dataloader_test)):

        sample_frames_test, sample_ofs_test, bbox_test, pred_frame_test, indices_test = test_data
        sample_frames_test = sample_frames_test.to(device)
        sample_ofs_test = sample_ofs_test.to(device)

        out_test = model(sample_frames_test, sample_ofs_test, mode="test")

        loss_of_test = score_func(out_test["of_recon"], out_test["of_target"]).cpu().data.numpy()
        loss_frame_test = score_func(out_test["frame_pred"], out_test["frame_target"]).cpu().data.numpy()

        of_scores = np.sum(np.sum(np.sum(loss_of_test, axis=3), axis=2), axis=1)
        frame_scores = np.sum(np.sum(np.sum(loss_frame_test, axis=3), axis=2), axis=1)

        if training_stats_path is not None:
            # mean-std normalization
            of_scores = (of_scores - of_mean) / of_std
            frame_scores = (frame_scores - frame_mean) / frame_std

        scores = config["w_r"] * of_scores + config["w_p"] * frame_scores

        for i in range(len(scores)):
            frame_bbox_scores[pred_frame_test[i][-1].item()][i] = scores[i]

    del dataset_test

    # joblib.dump(frame_bbox_scores,
    #             os.path.join(config["eval_root"], config["exp_name"], "frame_bbox_scores_%s.json" % suffix))

    # frame_bbox_scores = joblib.load(os.path.join(config["eval_root"], config["exp_name"],
    #                                              "frame_bbox_scores_%s.json" % suffix))

    # frame-level anomaly score
    frame_scores = np.empty(len(frame_bbox_scores))
    for i in range(len(frame_scores)):
        if len(frame_bbox_scores[i].items()) == 0:
            frame_scores[i] = config["w_r"] * (0 - of_mean) / of_std + config["w_p"] * (0 - frame_mean) / frame_std
        else:
            frame_scores[i] = np.max(list(frame_bbox_scores[i].values()))

    joblib.dump(frame_scores,
                os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix))

    # frame_scores = joblib.load(
    #     os.path.join(config["eval_root"], config["exp_name"], "frame_scores_%s.json" % suffix)
    # )

    # ================== Calculate AUC ==============================
    # load gt labels
    gt = pickle.load(
        open(os.path.join(config["dataset_base_dir"], "%s/ground_truth_demo/gt_label.json" % dataset_name), "rb"))
    gt_concat = np.concatenate(list(gt.values()), axis=0)

    new_gt = np.array([])
    new_frame_scores = np.array([])

    start_idx = 0
    for cur_video_id in range(METADATA[dataset_name]["testing_video_num"]):
        gt_each_video = gt_concat[start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]
        scores_each_video = frame_scores[
                            start_idx:start_idx + METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]][4:]

        start_idx += METADATA[dataset_name]["testing_frames_cnt"][cur_video_id]

        new_gt = np.concatenate((new_gt, gt_each_video), axis=0)
        new_frame_scores = np.concatenate((new_frame_scores, scores_each_video), axis=0)

    gt_concat = new_gt
    frame_scores = new_frame_scores

    curves_save_path = os.path.join(config["eval_root"], config["exp_name"], 'anomaly_curves_%s' % suffix)
    auc = save_evaluation_curves(frame_scores, gt_concat, curves_save_path,
                                 np.array(METADATA[dataset_name]["testing_frames_cnt"]) - 4)

    return auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_save_path", type=str,
                        default="./pretrained_ckpts/ped2_HF2VAD_99.31.pth",
                        help='path to pretrained weights')
    parser.add_argument("--cfg_file", type=str,
                        default="./pretrained_ckpts/ped2_HF2VAD_99.31_cfg.yaml",
                        help='path to pretrained model configs')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.cfg_file))
    testing_chunked_samples_file = os.path.join("./data", config["dataset_name"],
                                                "testing/chunked_samples/chunked_samples_00.pkl")

    from train import cal_training_stats

    os.makedirs(os.path.join("./eval", config["exp_name"]), exist_ok=True)
    training_chunked_samples_dir = os.path.join("./data", config["dataset_name"], "training/chunked_samples")
    training_stat_path = os.path.join("./eval", config["exp_name"], "training_stats.npy")
    cal_training_stats(config, args.model_save_path, training_chunked_samples_dir, training_stat_path)

    with torch.no_grad():
        auc = evaluate(config, args.model_save_path,
                       testing_chunked_samples_file,
                       training_stat_path, suffix="best")

        print(auc)
