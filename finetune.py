import gc
import os
import torch
import torch.nn as nn
import numpy as np
import yaml
import shutil
from torch.utils.data import DataLoader
from torch import optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from losses.loss import Gradient_Loss, Intensity_Loss, aggregate_kl_loss
from datasets.dataset import Chunked_sample_dataset, img_batch_tensor2numpy

from models.mem_cvae import HFVAD

from utils.initialization_utils import weights_init_kaiming
from utils.vis_utils import visualize_sequences
from utils.model_utils import loader, saver, only_model_saver
from eval import evaluate


def train(config, training_chunked_samples_dir, testing_chunked_samples_file):
    paths = dict(log_dir="%s/%s" % (config["log_root"], config["exp_name"]),
                 ckpt_dir="%s/%s" % (config["ckpt_root"], config["exp_name"]))
    os.makedirs(paths["ckpt_dir"], exist_ok=True)

    batch_size = config["batchsize"]
    epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    device = config["device"]
    lr = config["lr"]
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    grad_loss = Gradient_Loss(config["alpha"],
                              config["model_paras"]["img_channels"] * config["model_paras"]["clip_pred"],
                              device).to(device)
    intensity_loss = Intensity_Loss(l_num=config["intensity_loss_norm"]).to(device)

    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  finetune=config["model_paras"]["finetune"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-7, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    step = 0
    epoch_last = 0
    if not config["pretrained"]:
        model.apply(weights_init_kaiming)
    else:
        assert (config["pretrained"] is not None)
        model_state_dict = torch.load(config["pretrained"])["model_state_dict"]
        model.load_state_dict(model_state_dict)

    writer = SummaryWriter(paths["log_dir"])
    # copy hyper-params settings
    shutil.copyfile("./cfgs/finetune_cfg.yaml",
                    os.path.join(config["log_root"], config["exp_name"], "finetune_cfg.yaml"))

    best_auc = -1
    for epoch in range(epoch_last, epochs + epoch_last):
        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            for idx, train_data in tqdm(enumerate(dataloader),
                                        desc="Training Epoch %d, Chunked File %d" % (epoch + 1, chunk_file_idx),
                                        total=len(dataloader)):
                model.train()

                sample_frames, sample_ofs, _, _, _ = train_data
                sample_ofs = sample_ofs.to(device)
                sample_frames = sample_frames.to(device)

                out = model(sample_frames, sample_ofs, mode="train")

                # loss of ML-MemAE-SC
                loss_sparsity = out["loss_sparsity"]
                loss_flow_recon = out["loss_recon"]
                # loss of CVAE
                loss_kl = aggregate_kl_loss(out["q_means"], out["p_means"])
                loss_frame = intensity_loss(out["frame_pred"], out["frame_target"])
                loss_grad = grad_loss(out["frame_pred"], out["frame_target"])

                loss_all = config["lam_kl"] * loss_kl + \
                           config["lam_frame"] * loss_frame + \
                           config["lam_grad"] * loss_grad + \
                           config["lam_sparse"] * loss_sparsity + \
                           config["lam_recon"] * loss_flow_recon

                optimizer.zero_grad()
                loss_all.backward()
                optimizer.step()

                if step % config["logevery"] == config["logevery"] - 1:
                    print("[Step: {}/ Epoch: {}]: Loss: {:.4f} ".format(step + 1, epoch + 1, loss_all))

                    writer.add_scalar('loss_total/train', loss_all, global_step=step + 1)
                    writer.add_scalar('loss_frame/train', loss_frame, global_step=step + 1)
                    writer.add_scalar('loss_kl/train', loss_kl, global_step=step + 1)
                    writer.add_scalar('loss_grad/train', loss_grad, global_step=step + 1)
                    writer.add_scalar('loss_sparsity/train', loss_sparsity, global_step=step + 1)
                    writer.add_scalar('loss_flow_recon/train', loss_flow_recon, global_step=step + 1)

                    num_vis = 6
                    writer.add_figure("img/train_sample_frames",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_frames.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_frames.size(1) // 3,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_frame_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          out["frame_pred"].detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=config["model_paras"]["clip_pred"],
                                          return_fig=True),
                                      global_step=step + 1)
                    # memAE输入的光流和重建的光流
                    writer.add_figure("img/train_of_target",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          sample_ofs.cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_ofs.size(1) // 2,
                                          return_fig=True),
                                      global_step=step + 1)
                    writer.add_figure("img/train_of_recon",
                                      visualize_sequences(img_batch_tensor2numpy(
                                          out["of_recon"].detach().cpu()[:num_vis, :, :, :]),
                                          seq_len=sample_ofs.size(1) // 2,
                                          return_fig=True),
                                      global_step=step + 1)

                    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=step + 1)

                step += 1
            del dataset

        scheduler.step()

        if epoch % config["saveevery"] == config["saveevery"] - 1:
            model_save_path = os.path.join(paths["ckpt_dir"], config["model_savename"])
            saver(model.state_dict(), optimizer.state_dict(), model_save_path, epoch + 1, step, max_to_save=5)

            # training stats
            stats_save_path = os.path.join(paths["ckpt_dir"], "training_stats.npy-%d" % (epoch + 1))
            cal_training_stats(config, model_save_path + "-%d" % (epoch + 1), training_chunked_samples_dir,
                               stats_save_path)

            with torch.no_grad():
                auc = evaluate(config, model_save_path + "-%d" % (epoch + 1),
                               testing_chunked_samples_file,
                               stats_save_path,
                               suffix=str(epoch + 1))

                if auc > best_auc:
                    best_auc = auc
                    only_model_saver(model.state_dict(), os.path.join(paths["ckpt_dir"], "best.pth"))

                writer.add_scalar("auc", auc, global_step=epoch + 1)


def cal_training_stats(config, ckpt_path, training_chunked_samples_dir, stats_save_path):
    device = config["device"]
    model = HFVAD(num_hist=config["model_paras"]["clip_hist"],
                  num_pred=config["model_paras"]["clip_pred"],
                  config=config,
                  features_root=config["model_paras"]["feature_root"],
                  num_slots=config["model_paras"]["num_slots"],
                  shrink_thres=config["model_paras"]["shrink_thres"],
                  skip_ops=config["model_paras"]["skip_ops"],
                  mem_usage=config["model_paras"]["mem_usage"],
                  ).to(device).eval()

    model_weights = torch.load(ckpt_path)["model_state_dict"]
    model.load_state_dict(model_weights)
    print("load pre-trained success!")

    score_func = nn.MSELoss(reduction="none")
    training_chunk_samples_files = sorted(os.listdir(training_chunked_samples_dir))

    of_training_stats = []
    frame_training_stats = []

    print("=========Forward pass for training stats ==========")
    with torch.no_grad():

        for chunk_file_idx, chunk_file in enumerate(training_chunk_samples_files):
            dataset = Chunked_sample_dataset(os.path.join(training_chunked_samples_dir, chunk_file))
            dataloader = DataLoader(dataset=dataset, batch_size=128, num_workers=0, shuffle=False)

            for idx, data in tqdm(enumerate(dataloader),
                                  desc="Training stats calculating, Chunked File %02d" % chunk_file_idx,
                                  total=len(dataloader)):
                sample_frames, sample_ofs, _, _, _ = data
                sample_frames = sample_frames.to(device)
                sample_ofs = sample_ofs.to(device)

                out = model(sample_frames, sample_ofs, mode="test")

                loss_frame = score_func(out["frame_pred"], out["frame_target"]).cpu().data.numpy()
                loss_of = score_func(out["of_recon"], out["of_target"]).cpu().data.numpy()

                of_scores = np.sum(np.sum(np.sum(loss_of, axis=3), axis=2), axis=1)
                frame_scores = np.sum(np.sum(np.sum(loss_frame, axis=3), axis=2), axis=1)

                of_training_stats.append(of_scores)
                frame_training_stats.append(frame_scores)
            del dataset
            gc.collect()

    print("=========Forward pass for training stats done!==========")
    of_training_stats = np.concatenate(of_training_stats, axis=0)
    frame_training_stats = np.concatenate(frame_training_stats, axis=0)

    training_stats = dict(of_training_stats=of_training_stats,
                          frame_training_stats=frame_training_stats)
    # save to file
    torch.save(training_stats, stats_save_path)


if __name__ == '__main__':
    config = yaml.safe_load(open("./cfgs/finetune_cfg.yaml"))
    dataset_name = config["dataset_name"]
    dataset_base_dir = config["dataset_base_dir"]
    training_chunked_samples_dir = os.path.join(dataset_base_dir, dataset_name, "training/chunked_samples")
    testing_chunked_samples_file = os.path.join(dataset_base_dir, dataset_name,
                                                "testing/chunked_samples/chunked_samples_00.pkl")

    train(config, training_chunked_samples_dir, testing_chunked_samples_file)
