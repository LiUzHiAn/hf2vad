import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import scipy.signal as signal


def draw_roc_curve(fpr, tpr, auc, psnr_dir):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(psnr_dir, "auc.png"))
    plt.close()


def nonzero_intervals(vec):
    '''
    Find islands of non-zeros in the vector vec
    '''
    if len(vec) == 0:
        return []
    elif not isinstance(vec, np.ndarray):
        vec = np.array(vec)

    tmp1 = (vec == 0) * 1
    tmp = np.diff(tmp1)
    edges, = np.nonzero(tmp)
    edge_vec = [edges + 1]

    if vec[0] != 0:
        edge_vec.insert(0, [0])
    if vec[-1] != 0:
        edge_vec.append([len(vec)])
    edges = np.concatenate(edge_vec)
    return zip(edges[::2], edges[1::2])


def save_evaluation_curves(scores, labels, curves_save_path, video_frame_nums):
    """
    Draw anomaly score curves for each video and the overall ROC figure.
    """
    if not os.path.exists(curves_save_path):
        os.mkdir(curves_save_path)

    scores = scores.flatten()
    labels = labels.flatten()

    scores_each_video = {}
    labels_each_video = {}

    start_idx = 0
    for video_id in range(len(video_frame_nums)):
        scores_each_video[video_id] = scores[start_idx:start_idx + video_frame_nums[video_id]]
        scores_each_video[video_id] = signal.medfilt(scores_each_video[video_id], kernel_size=17)
        labels_each_video[video_id] = labels[start_idx:start_idx + video_frame_nums[video_id]]

        start_idx += video_frame_nums[video_id]

    truth = []
    preds = []
    for i in range(len(scores_each_video)):
        truth.append(labels_each_video[i])
        preds.append(scores_each_video[i])

    truth = np.concatenate(truth, axis=0)
    preds = np.concatenate(preds, axis=0)
    fpr, tpr, roc_thresholds = roc_curve(truth, preds, pos_label=1)
    auroc = auc(fpr, tpr)

    # draw ROC figure
    draw_roc_curve(fpr, tpr, auroc, curves_save_path)
    for i in sorted(scores_each_video.keys()):
        plt.figure()

        x = range(0, len(scores_each_video[i]))
        plt.xlim([x[0], x[-1] + 5])

        # anomaly scores
        plt.plot(x, scores_each_video[i], color="blue", lw=2, label="Anomaly Score")

        # abnormal sections
        lb_one_intervals = nonzero_intervals(labels_each_video[i])
        for idx, (start, end) in enumerate(lb_one_intervals):
            plt.axvspan(start, end, alpha=0.5, color='red',
                        label="_" * idx + "Anomaly Intervals")

        plt.xlabel('Frames Sequence')
        plt.title('Test video #%d' % (i + 1))
        plt.legend(loc="upper left")
        plt.savefig(os.path.join(curves_save_path, "anomaly_curve_%d.png" % (i + 1)))
        plt.close()

    return auroc
