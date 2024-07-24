import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os



def heatmap_grads(grad_cos_sim_epochs, save_path, convert_to_angle, xlabel, color_norm=None):
    labelsize = 12
    if convert_to_angle == True:
        ax = sns.heatmap(np.arccos(np.clip(grad_cos_sim_epochs, -1, 1))*180/np.pi, cmap="viridis", cbar_kws={'label': 'Angle (deg)'}, norm=color_norm,
                         yticklabels=np.array([i for i in range(1, 1+grad_cos_sim_epochs.shape[0])]))
    else:
        ax = sns.heatmap(grad_cos_sim_epochs, cmap="viridis", cbar_kws={'label': 'Cosine-sim'}, norm=color_norm,
                         yticklabels=np.array([i for i in range(1, 1+grad_cos_sim_epochs.shape[0])]))
    figsize = np.array([8, 2])

    for (i, label) in enumerate(ax.yaxis.get_ticklabels()):
        if (i+1)%4!=0:
            label.set_visible(False)
    plt.gcf().set_size_inches(figsize[0], figsize[1])
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel("Layer", fontsize=labelsize)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return

cifar100_a_00_b1_hist = "runs/cifar100-alg-dualprop-lagr-ff-alpha-0.0-beta-1.0-nudged-passes-16/2024_02_01_14_18_00/hist.npy"
cifar100_a_00_b1_hist = np.load(cifar100_a_00_b1_hist, allow_pickle=True)[()]

cifar100_a_00_b001_hist = "runs/cifar100-alg-dualprop-lagr-ff-alpha-0.0-beta-0.01-nudged-passes-16/2024_01_31_19_44_54/hist.npy"
cifar100_a_00_b001_hist = np.load(cifar100_a_00_b001_hist, allow_pickle=True)[()]

heatmap_grads(cifar100_a_00_b1_hist["grad_cos_sim_epochs"], "plots/cifar100_a_00_b1_epochs.pdf", True, "Epochs", color_norm=None)
heatmap_grads(cifar100_a_00_b001_hist["grad_cos_sim_epochs"], "plots/cifar100_a_00_b001_epochs.pdf", True, "Epochs", color_norm=None)

heatmap_grads(cifar100_a_00_b1_hist["grad_cos_sim_batches"][:,0:200], "plots/cifar100_a_00_b1_batches.pdf", True, "Batch index", color_norm=None)
heatmap_grads(cifar100_a_00_b001_hist["grad_cos_sim_batches"][:,0:200], "plots/cifar100_a_00_b001_batches.pdf", True, "Batch index", color_norm=None)

