import argparse
import os
import numpy as np
import joblib
from datasets.dataset import get_dataset, img_batch_tensor2numpy


def samples_extraction(dataset_root, dataset_name, mode, all_bboxes, save_dir):
    num_predicted_frame = 1
    # save samples in chunked file
    if dataset_name == "ped2":
        num_samples_each_chunk = 100000
    elif dataset_name == "avenue":
        num_samples_each_chunk = 200000 if mode == "test" else 20000
    elif dataset_name == "shanghaitech":
        num_samples_each_chunk = 300000 if mode == "test" else 100000
    else:
        raise NotImplementedError("dataset name should be one of ped2,avenue or shanghaitech!")

    # frames dataset
    dataset = get_dataset(
        dataset_name=dataset_name,
        dir=os.path.join(dataset_root, dataset_name),
        context_frame_num=4, mode=mode,
        border_mode="predict", all_bboxes=all_bboxes,
        patch_size=32, of_dataset=False
    )

    # flows dataset
    flow_dataset = get_dataset(
        dataset_name=dataset_name,
        dir=os.path.join(dataset_root, dataset_name),
        context_frame_num=4, mode=mode,
        border_mode="predict", all_bboxes=all_bboxes,
        patch_size=32,
        of_dataset=True)

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    global_sample_id = 0
    cnt = 0
    chunk_id = 0  # chunk file id
    chunked_samples = dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[])

    for idx in range(len(dataset)):
        if idx % 1000 == 0:
            print('Extracting foreground in {}-th frame, {} in total'.format(idx + 1, len(dataset)))

        frameRange = dataset._context_range(idx)

        # [num_bboxes,clip_len,C,patch_size, patch_size]
        batch, _ = dataset.__getitem__(idx)
        flow_batch, _ = flow_dataset.__getitem__(idx)

        # all the bboxes in current frame
        cur_bboxes = all_bboxes[idx]
        if len(cur_bboxes) > 0:
            batch = img_batch_tensor2numpy(batch)
            flow_batch = img_batch_tensor2numpy(flow_batch)

            # each STC treated as a sample
            for idx_box in range(cur_bboxes.shape[0]):
                chunked_samples["sample_id"].append(global_sample_id)
                chunked_samples["appearance"].append(batch[idx_box])
                chunked_samples["motion"].append(flow_batch[idx_box])
                chunked_samples["bbox"].append(cur_bboxes[idx_box])
                chunked_samples["pred_frame"].append(frameRange[-num_predicted_frame:])  # the frame id of last patch
                global_sample_id += 1
                cnt += 1

                if cnt == num_samples_each_chunk:
                    chunked_samples["sample_id"] = np.array(chunked_samples["sample_id"])
                    chunked_samples["appearance"] = np.array(chunked_samples["appearance"])
                    chunked_samples["motion"] = np.array(chunked_samples["motion"])
                    chunked_samples["bbox"] = np.array(chunked_samples["bbox"])
                    chunked_samples["pred_frame"] = np.array(chunked_samples["pred_frame"])
                    joblib.dump(chunked_samples, os.path.join(save_dir, "chunked_samples_%02d.pkl" % chunk_id))
                    print("Chunk %d file saved!" % chunk_id)

                    chunk_id += 1
                    cnt = 0
                    del chunked_samples
                    chunked_samples = dict(sample_id=[], appearance=[], motion=[], bbox=[], pred_frame=[])

    # save the remaining samples
    if len(chunked_samples["sample_id"]) != 0:
        chunked_samples["sample_id"] = np.array(chunked_samples["sample_id"])
        chunked_samples["appearance"] = np.array(chunked_samples["appearance"])
        chunked_samples["motion"] = np.array(chunked_samples["motion"])
        chunked_samples["bbox"] = np.array(chunked_samples["bbox"])
        chunked_samples["pred_frame"] = np.array(chunked_samples["pred_frame"])
        joblib.dump(chunked_samples, os.path.join(save_dir, "chunked_samples_%02d.pkl" % chunk_id))
        print("Chunk %d file saved!" % chunk_id)

    print('All samples have been saved!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_root", type=str, default="/home/liuzhian/hdd4T/code/hf2vad", help='project root path')
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--mode", type=str, default="train", help='train or test data')

    args = parser.parse_args()

    all_bboxes = np.load(
        os.path.join(args.proj_root, "data", args.dataset_name, '%s_bboxes_%s.npy' % (args.dataset_name, args.mode)),
        allow_pickle=True
    )
    if args.mode == "train":
        save_dir = os.path.join(args.proj_root, "data", args.dataset_name, "training", "chunked_samples")
    else:
        save_dir = os.path.join(args.proj_root, "data", args.dataset_name, "testing", "chunked_samples")

    samples_extraction(
        dataset_root=os.path.join(args.proj_root, "data"),
        dataset_name=args.dataset_name,
        mode=args.mode,
        all_bboxes=all_bboxes,
        save_dir=save_dir
    )
