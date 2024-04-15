#!/usr/bin/env python
# coding: utf-8

# In[8]:


import os
import math
from random import randint

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_from_disk, concatenate_datasets
from brainlm_mae.modeling_brainlm import BrainLMForPretraining
from utils.brainlm_trainer import BrainLMTrainer
from utils.plots import plot_future_timepoint_trends


# In[3]:


# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ## Load model

# In[15]:


model_name = "2023-07-17-19_00_00"
checkpoint_n = "500"


# In[4]:


# model = BrainLMForPretraining.from_pretrained(
#     f"/home/ahf38/palmer_scratch/brainlm/training-runs/{model_name}/checkpoint-{checkpoint_n}").to(device)
# model

model = BrainLMForPretraining.from_pretrained(
    f"C:\yamin\eeg-fmri-DL\BrainLM\pretrained_models\checkpoint").to(device)
model


# In[9]:


print(model.vit.config)


# In[5]:


print(model.vit.embeddings.mask_ratio)
print(model.vit.embeddings.config.mask_ratio)


# In[6]:


# need this for matteo's branch, due to multiple train modes (auto-encoder, causal attention, predict last, etc)
model.config.train_mode = "auto_encode"


# ## Load Entire Dataset

# In[10]:


coords_ds = load_from_disk("C:\\yamin\\eeg-fmri-DL\\BrainLM\\toolkit\\sample_dataset\\a424_fMRI_data\\arrow_form\\Brain_Region_Coordinates")
print(coords_ds)


# In[11]:


dataset_v = "v3"


# In[13]:


# load all data
train_ds = load_from_disk("C:\\yamin\\eeg-fmri-DL\\BrainLM\\toolkit\\sample_dataset\\a424_fMRI_data\\arrow_form\\train")
print(train_ds)
# val_ds = load_from_disk("/home/sr2464/palmer_scratch/datasets/UKB_Large_rsfMRI_and_tffMRI_Arrow_WithRegression_v3_with_metadata/val_ukbiobank")
# print(val_ds)
# test_ds = load_from_disk("/home/sr2464/palmer_scratch/datasets/UKB_Large_rsfMRI_and_tffMRI_Arrow_WithRegression_v3_with_metadata/test_ukbiobank")
# print(test_ds)


# In[14]:


print(train_ds[0]['Filename'])


# In[ ]:

#
# concat_ds = concatenate_datasets([train_ds, val_ds, test_ds])
# concat_ds
#
#
# # In[ ]:
#
#
# example0 = test_ds[500]
# print(example0['Filename'])
# print(example0['Patient ID'])


# ## Saving Directory

# In[16]:


dir_name = f"C:\\yamin\\eeg-fmri-DL\\BrainLM\\inference_plots\\dataset_{dataset_v}/{model_name}_ckpt-{checkpoint_n}/"


# In[17]:


if not os.path.exists(dir_name):
    os.makedirs(dir_name)


# In[ ]:


# dataset_split = {"train": train_ds, "val": val_ds, "test": test_ds, "concat": concat_ds}


# ## Forward Pass Through Model, Pass Whole fMRI Recording

# In[59]:


# variable_of_interest_col_name = "Gender"
variable_of_interest_col_name = ""
recording_col_name = "Voxelwise_RobustScaler_Normalized_Recording"
length = 200
num_timepoints_per_voxel = model.config.num_timepoints_per_voxel


# In[60]:


def preprocess_fmri(examples):
    """
    Preprocessing function for dataset samples. This function is passed into Trainer as
    a preprocessor which takes in one row of the loaded dataset and constructs a model
    input sample according to the arguments which model.forward() expects.

    The reason this function is defined inside on main() function is because we need
    access to arguments such as cell_expression_vector_col_name.
    """
    #
    signal_vector = examples[recording_col_name][0]
    signal_vector = torch.tensor(signal_vector, dtype=torch.float32).T # todo, added T to transpose since there's dimension problem below, be careful for model input

    # Choose random starting index, take window of moving_window_len points for each region
    start_idx = randint(0, signal_vector.shape[0] - num_timepoints_per_voxel)
    end_idx = start_idx + num_timepoints_per_voxel
    signal_window = signal_vector[
        start_idx:end_idx, :
    ]  # [moving_window_len, num_voxels]
    signal_window = torch.movedim(
        signal_window, 0, 1
    )  # --> [num_voxels, moving_window_len]

    # Append signal values and coords
    window_xyz_list = []
    for brain_region_idx in range(signal_window.shape[0]):
        # window_timepoint_list = torch.arange(0.0, 1.0, 1.0 / num_timepoints_per_voxel)

        # Append voxel coordinates
        xyz = torch.tensor(
            [
                coords_ds[brain_region_idx]["X"],
                coords_ds[brain_region_idx]["Y"],
                coords_ds[brain_region_idx]["Z"],
            ],
            dtype=torch.float32,
        )
        window_xyz_list.append(xyz)
    window_xyz_list = torch.stack(window_xyz_list)

    # Add in key-value pairs for model inputs which CellLM is expecting in forward() function:
    #  signal_vectors and xyzt_vectors
    #  These lists will be stacked into torch Tensors by collate() function (defined above).
    examples["signal_vectors"] = [signal_window]
    examples["xyz_vectors"] = [window_xyz_list]
    return examples


# In[61]:


def collate_fn(examples):
    """
    This function tells the dataloader how to stack a batch of examples from the dataset.
    Need to stack gene expression vectors and maintain same argument names for model inputs
    which CellLM is expecting in forward() function:
        expression_vectors, sampled_gene_indices, and cell_indices
    """





    signal_vectors = torch.stack(
        [example["signal_vectors"] for example in examples], dim=0
    )
    xyz_vectors = torch.stack([example["xyz_vectors"] for example in examples])
    # labels = torch.stack([example["label"] for example in examples])
    labels = torch.zeros(len(examples[0]["xyz_vectors"]), dtype=torch.float32)
    
    
    # These inputs will go to model.forward(), names must match
    return {
        "signal_vectors": signal_vectors,
        "xyz_vectors": xyz_vectors,
        "input_ids": signal_vectors,
        "labels": labels
    }


# In[62]:


# concat_ds.set_transform(preprocess_fmri)
# test_ds.set_transform(preprocess_fmri)
train_ds.set_transform(preprocess_fmri)
# val_ds.set_transform(preprocess_fmri)


# In[63]:


dataloader_single = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn)


# In[64]:


print(train_ds)


# In[65]:


print(next(iter(dataloader_single)).keys())


# In[ ]:


next(iter(dataloader_single))["signal_vectors"].shape


# In[ ]:


#--- Forward 1 sample through just the model encoder (model.vit) ---#
with torch.no_grad():
    example1 = next(iter(dataloader_single))
    encoder_output = model.vit(
        signal_vectors=example1["signal_vectors"].to(device),
        xyz_vectors=example1["xyz_vectors"].to(device),
        output_attentions=True,
        output_hidden_states=True
    )


# In[ ]:


print("last_hidden_state:", encoder_output.last_hidden_state.shape)
# [batch_size, num_genes + 1 CLS token, hidden_dim]

cls_token = encoder_output.last_hidden_state[:,0,:]
print(cls_token.shape)


# ## Complete reconstruction of a batch of samples

# In[ ]:


batch_size = 1


# In[ ]:


split = "test"


# In[ ]:


dataloader_batched = DataLoader(train_ds,
                               batch_size=batch_size,
                               collate_fn=collate_fn,
                               drop_last=False,
                               )


# In[ ]:


def construct_noise(x, seq_length, mask_ratio, ids_mask=None):
    """
    Constructs a noise tensor which is used by model.vit.embeddings to mask tokens.
    Giving ids_mask enables that every new call of construct_noise will return a tensor 
    that masks tokens that were not previously masked.

    Args:
        x:              tensor of shape [batch_size, num_voxels, num_timepoints_per_voxel]
        seq_length:     length of tokens
        mask_ratio:     ratio of tokens to mask
        ids_mask:       previously masked tokens
    """
    
    # label dimensions of interest
    batch_size = x.shape[0]
    len_mask = int(mask_ratio * seq_length)
    
    # construct random only for not previously masked tokens
    # add zeros to noise to force keep previously masked tokens
    noise = torch.rand(batch_size, seq_length, device=x.device)
    
    if ids_mask != None:        
        # force keep by setting noise to zero at prior masked indeces
        noise.scatter_(index = ids_mask, dim = 1, value=0)

        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    
        ids_mask = torch.cat((ids_mask, ids_shuffle[:, (-1 * len_mask):]), dim=1)
    else:
        ids_mask = torch.argsort(noise, dim=1)[:, (-1 * len_mask):]  # ascend: small is keep, large is remove
    
    return noise, ids_mask


# In[ ]:


# reconstruct whole recording for one example by repeatedly encoding/decoding
example1 = next(iter(dataloader_batched))
seq_length = (model.config.num_timepoints_per_voxel // model.config.timepoint_patching_size) * model.config.num_brain_voxels

masked_tokens = 0
ids_removed = None
predictions = []
masks = []
x=example1["signal_vectors"].to(device)
while masked_tokens < seq_length:
    
    noise, ids_removed = construct_noise(x, seq_length, model.config.mask_ratio, ids_removed)
    
    # get predictions
    out = model(signal_vectors=x, 
                xyz_vectors=example1["xyz_vectors"].to(device),
                noise=noise,
               )
    
    # store predictions and masks
    predictions.append(out.logits[0].detach())
    print("Masked tokens this run at sample 0 and parcel 0:", torch.nonzero(out.mask[0, 0, :]).tolist())
    masks.append(out.mask)

    
    masked_tokens += out["mask"][0, :].sum()
    


# In[ ]:


def aggregate_predictions(predictions, masks, mode="first"):
    '''
    Aggregates all predictions according to masks.
    Avoids adding a prediction twice if masked twice (will happen if num_tokens % (masked_ratio * num_tokens) != 0) 
    by taking the first prediction if mode = "first", or the average if mode = "mean". 
    '''
    preds = torch.zeros(predictions[0].shape)
    cum_mask = torch.zeros(masks[0].shape) # counts how many times particular token is masked
    
    for idx, p in enumerate(predictions):
        cum_mask += masks[idx].cpu()
        
        if mode == "first":
            masked_once = (cum_mask == 1) # keeps only tokens masked once
            m = torch.eq(masked_once.long(), masks[idx].cpu()) # returns True for unmasked tokens and tokens masked for first time
            m = m.long() * masks[idx].cpu() # returns only tokens masked for first time (mask[idx] will be zero for others)
        else:
            m = masks[idx].cpu()
            
        m = m.unsqueeze(-1).repeat(1, 1, 1, p.shape[-1])

        preds += p * m 

    if mode == "mean":
        preds = preds / cum_mask.unsqueeze(-1).repeat(1, 1, 1, p.shape[-1]) # there should not be a division by zero because all tokens masked at least once
    
    return (preds, cum_mask)


# In[ ]:


preds, cum_mask = aggregate_predictions(predictions, masks, mode="first")


# # Plotting one sample
# 
# Visualize wholly reconstructed recording and ground truth. Plot UMAP and PCA over space as well

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import seaborn as sns


# In[ ]:


# sns.set_style()
sns.reset_orig()


# In[ ]:


def plot_masked_pred_trends_one_sample(
    pred_logits: np.array,
    signal_vectors: np.array,
    mask: np.array,
    sample_idx: int,
    node_idxs: np.array,
    dataset_split: str,
):
    """
    Function to plot timeseries of model predictions as continuation of input data compared to
    ground truth.
    Args:
        pred_logits:    numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        signal_vectors: numpy array of shape [batch_size, num_voxels, num_tokens, time_patch_preds]
        mask:           binary mask of shape [batch_size, num_voxels, num_tokens]
        sample_idx:     index of sample to plot; one per figure
        node_idxs:      indices of voxels to plot; affects how many columns in plot grid there will be
    Returns:
    """
    fig, axes = plt.subplots(nrows=len(node_idxs), ncols=1, sharex=True, sharey=True)
    fig.set_figwidth(25)
    fig.set_figheight(3 * len(node_idxs))

    batch_size, num_voxels, num_tokens, time_patch_preds = pred_logits.shape

    # --- Plot Figure ---#
    for row_idx, node_idx in enumerate(node_idxs):
        ax = axes[row_idx]

        input_data_vals = []
        input_data_timepoints = []
        for token_idx in range(signal_vectors.shape[2]):
            input_data_vals += signal_vectors[sample_idx, node_idx, token_idx].tolist()
            start_timepoint = time_patch_preds * token_idx
            end_timepoint = start_timepoint + time_patch_preds
            input_data_timepoints += list(range(start_timepoint, end_timepoint))

            if mask[sample_idx, node_idx, token_idx] == 1:
                model_pred_vals = pred_logits[sample_idx, node_idx, token_idx].tolist()
                model_pred_timepoints = list(range(start_timepoint, end_timepoint))
                ax.plot(
                    model_pred_timepoints,
                    model_pred_vals,
                    marker=".",
                    markersize=3,
                    label="Masked Predictions",
                    color="orange",
                )

        ax.plot(
            input_data_timepoints,
            input_data_vals,
            marker=".",
            markersize=3,
            label="Input Data",
            color="green",
        )
        ax.set_title("Sample {}, Parcel {}".format(sample_idx, node_idx))
        ax.axhline(y=0.0, color="gray", linestyle="--", markersize=2)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.9, 0.99))
    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
    fig.supxlabel("Timepoint")
    fig.supylabel("Prediction Value")
    plt.suptitle("Ground Truth Signal vs Masked Prediction\n({} Split)".format(dataset_split))
    plt.savefig(f"{dir_name}reconstruct_whole_recording_{dataset_split}split.png", bbox_inches="tight", facecolor="white")
    plt.show()
    plt.close()


# In[ ]:


plot_masked_pred_trends_one_sample(
    preds,
    example1["signal_vectors"].reshape(preds.shape),
    mask=torch.ones(preds.shape[:3]),
    sample_idx=0,
    node_idxs=[0, 100, 200],
    dataset_split=split,
)


# ### Plot UMAPs

# In[ ]:


import umap


# In[ ]:


# choose one recording to map, transpose to do PCA and UMAP over time
raw_rec = example1["signal_vectors"][0].T
pred_rec = preds[0].flatten(-2).T


# In[ ]:


# Apply UMAP to raw recording
reducer = umap.UMAP(random_state=42, verbose = True, n_components=3)
embedding_raw = reducer.fit_transform(raw_rec)


# In[ ]:


embedding_raw.shape


# In[ ]:


# Apply UMAP to reconstructed recording
reducer = umap.UMAP(random_state=42, verbose = True, n_components=3)
embedding_pred = reducer.fit_transform(pred_rec)


# In[ ]:


embedding_pred.shape


# In[ ]:


fig, axes = plt.subplots(nrows=embedding_pred.shape[1], ncols=1, sharex=True, sharey=False)
fig.set_figwidth(25)
fig.set_figheight(3 * embedding_pred.shape[1])

# batch_size, num_voxels, num_tokens, time_patch_preds = pred_logits.shape

# --- Plot Figure ---#
for row_idx in range(embedding_pred.shape[1]):
    ax = axes[row_idx]
    data_parcels = list(range(embedding_pred.shape[0]))

    ax.plot(
        data_parcels,
        embedding_pred[:, row_idx],
        marker=".",
        markersize=3,
        label="Masked Predictions",
        color="orange",
    )

    ax.plot(
        data_parcels,
        embedding_raw[:, row_idx],
        marker=".",
        markersize=3,
        label="Input Data",
        color="green",
    )
    ax.set_title("UMAP Coord {}".format(row_idx))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.9))
plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.95])
fig.supxlabel("Timepoint")
fig.supylabel("UMAP value")
plt.suptitle("Ground Truth Signal vs Masked Prediction\n({} Split)".format(split))
plt.savefig(f"{dir_name}reconstruct_umap_{split}split.png", bbox_inches="tight", facecolor="white")
plt.show()
plt.close()


# It looks like these two representations a translated vertically by different amounts, or flipped along horizontal axis. TODO: figure out a way (if there is one) to have them in the same line.

# ### Plot PCAs

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


n_components = 3
pca_pred = PCA(n_components=n_components)
pca_raw = PCA(n_components=n_components)


# In[ ]:


pca_pred.fit(pred_rec)


# In[ ]:


pca_raw.fit(raw_rec)


# In[ ]:


pred_reduced = pca_pred.transform(pred_rec)
raw_reduced = pca_raw.transform(raw_rec)


# In[ ]:


fig, axes = plt.subplots(nrows=pred_reduced.shape[1], ncols=1, sharex=True, sharey=False)
fig.set_figwidth(25)
fig.set_figheight(3 * pred_reduced.shape[1])

# batch_size, num_voxels, num_tokens, time_patch_preds = pred_logits.shape

# --- Plot Figure ---#
for row_idx in range(pred_reduced.shape[1]):
    ax = axes[row_idx]
    data_parcels = list(range(pred_reduced.shape[0]))

    ax.plot(
        data_parcels,
        pred_reduced[:, row_idx],
        marker=".",
        markersize=3,
        label="Masked Predictions",
        color="orange",
    )

    ax.plot(
        data_parcels,
        raw_reduced[:, row_idx],
        marker=".",
        markersize=3,
        label="Input Data",
        color="green",
    )
    ax.set_title("PCA Coord {}".format(row_idx + 1))

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.9, 0.93))
plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.93])
fig.supxlabel("Timepoint")
fig.supylabel("PCA Value")
plt.suptitle("Ground Truth Signal vs Masked Prediction\n({} Split)".format(split))
plt.savefig(f"{dir_name}reconstruct_pca_whole_recording_{split}split.png", bbox_inches="tight", facecolor="white")
plt.show()
plt.close()


# TODO: 1) Average of a batch of recordings, 2) average of all recordings in a dataset split, 3) average of all recordings.

# In[ ]:




