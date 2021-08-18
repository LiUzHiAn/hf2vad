from utils.flow_utils import flow2img
import cv2
import matplotlib.pyplot as plt
import numpy as np


# plt.switch_backend('agg')

def visualize_sequences(batch, seq_len, return_fig=True):
    """
    visualize a sequence (imgs or flows)
    """
    sequences = []
    channels_per_frame = batch.shape[-1] // seq_len
    for i in range(batch.shape[0]):
        cur_sample = batch[i]  # [H,W,channels_per_frame * seq_len]
        if channels_per_frame == 2:
            sequence = [flow2img(cur_sample[:, :, j * channels_per_frame:(j + 1) * channels_per_frame])
                        for j in range(seq_len)]
        else:
            # to RGB
            sequence = [cur_sample[:, :, j * channels_per_frame:(j + 1) * channels_per_frame][:, :, ::-1]
                        for j in range(seq_len)]
        sequences.append(np.hstack(sequence))
    sequences = np.vstack(sequences)

    if return_fig:
        fig = plt.figure()
        plt.imshow(sequences)
        return fig
    else:
        return sequences
