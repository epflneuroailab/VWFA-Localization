"""
Generates binary masks for VWF-sensitive units in Qwen2VL using precomputed activations.

1. Loads activations from "./activations/qwen_activation_*.pkl".
2. Computes t-tests for each hidden unit across words vs non-words.
3. Selects top units by t-values to create a binary mask.
4. Saves the mask (.pkl) and a heatmap (.png) for each activation file.

"""

# ---- Imports ----
import os
import random as rand
import pickle as pkl

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats

# ---- Reproducibility ----
base_seed = 42
rand.seed(base_seed)                     # Python RNG
np.random.seed(base_seed)                # NumPy
torch.manual_seed(base_seed)             # PyTorch (CPU)
torch.cuda.manual_seed(base_seed)        # PyTorch (CUDA - single GPU)
torch.cuda.manual_seed_all(base_seed)    # PyTorch (CUDA - multi-GPU)

os.environ["PYTHONHASHSEED"] = str(base_seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- Plot styling ----
sns.set_theme()
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=sns.color_palette("deep"))

# ---- Helpers ----
def is_topk(a, k=1):
    _, rix = np.unique(-a, return_inverse=True)
    return np.where(rix < k, 1, 0).reshape(a.shape)

# ---- Layer names ----
layer_names = ['model.layers.{}.mlp.gate_proj'.format(i) for i in range(80)]

# ---- Main loop ----
for idx in range(0, 20):

    percentage = 21

    ACT_PATH = "./activations"
    FIG_PATH = "./figures"
    MASK_PATH = "./masks"
    ACT_PATH = os.path.join(ACT_PATH, f'qwen_activation_{idx}.pkl')

    os.makedirs(FIG_PATH, exist_ok=True)
    os.makedirs(MASK_PATH, exist_ok=True)

    FIG_PATH = os.path.join(FIG_PATH, f'mask_{percentage}%_{idx}.png')
    MASK_PATH = os.path.join(MASK_PATH, f'mask_{percentage}%_{idx}.pkl')

    p_values_matrix_all = []
    t_values_matrix_all = []

    with open(ACT_PATH, 'rb') as f:
        final_layer_representations = pkl.load(f)

    for layer_idx, layer_name in enumerate(tqdm(layer_names)):
        words_actv = final_layer_representations["words"][layer_name]
        non_words_actv = final_layer_representations["non-words"][layer_name]
        if words_actv.shape[0] == 0:
            print(layer_name)
        print(words_actv.shape)
        print(non_words_actv.shape)
        num_units = final_layer_representations["words"][layer_names[layer_idx]].shape[-1]
        if len(words_actv.shape) > 2:
            words_actv = np.squeeze(words_actv, axis=1)
            non_words_actv = np.squeeze(non_words_actv, axis=1)
        words_actv_meaned = words_actv.mean(axis=0)
        non_words_actv_meaned = non_words_actv.mean(axis=0)
        p_values_matrix = np.zeros((len(layer_names), num_units))
        t_values_matrix = np.zeros((len(layer_names), num_units))
        t_values_matrix, p_values_matrix = scipy.stats.ttest_ind(
            words_actv, non_words_actv, axis=0, equal_var=False
        )
        p_values_matrix_all.append(p_values_matrix)
        t_values_matrix_all.append(t_values_matrix)

    # NaN diagnostics
    nan_indices = np.argwhere(np.isnan(p_values_matrix_all))
    print("number of Nans:", len(nan_indices))

    nan_indices = np.argwhere(np.isnan(t_values_matrix_all))
    print("number of Nans:", len(nan_indices))

    p_values_matrix_all = np.nan_to_num(p_values_matrix_all, nan=1.0)
    t_values_matrix_all = np.nan_to_num(t_values_matrix_all, nan=0.0)

    total_units = sum(len(arr) for arr in t_values_matrix_all)
    min_t_val = min(np.min(arr) for arr in t_values_matrix_all)
    hidden_dim = max(len(arr) for arr in t_values_matrix_all)
    padded_arrays = []

    for arr in t_values_matrix_all:
        padded_arr = np.full((hidden_dim), min_t_val)
        padded_arr[:len(arr)] = arr
        padded_arrays.append(padded_arr)

    t_values_matrix_all = np.vstack(padded_arrays)

    num_units = int((percentage / 100) * total_units)
    print(f"> Percentage: {percentage}% --> Num Units: {num_units}")
    top_k_units = num_units
    language_mask = is_topk(t_values_matrix_all, k=top_k_units)

    num_active_units = language_mask.sum()
    total_num_units = total_units
    print("Shape of language mask", language_mask.shape)
    desc = f"# of Active Units: {num_active_units:,}/{total_num_units:,} = {(num_active_units/total_num_units)*100:.2f}%"
    print(desc)
    print("This is the true percentage:", num_active_units / 7394816 * 100)

    fig_mask, ax_mask = plt.subplots(figsize=(15, 15))
    sns.heatmap(language_mask, ax=ax_mask, cbar=True, center=0)
    ax_mask.set_title(f'mask_{percentage}%')
    ax_mask.set_xlabel('Hidden Dims')
    ax_mask.set_ylabel('Layer')
    plt.tight_layout()
    plt.savefig(FIG_PATH)
    plt.close(fig_mask) 

    with open(MASK_PATH, 'wb') as f:
        pkl.dump(language_mask, f)
