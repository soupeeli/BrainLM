import os
import math
from random import randint, seed
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import load_from_disk, concatenate_datasets
# from transformers import ViTImageProcessor, ViTMAEConfig
# from brainlm_mae.modeling_vit_mae_with_padding import ViTMAEForPreTraining
# from brainlm_mae.replace_vitmae_attn_with_flash_attn import replace_vitmae_attn_with_flash_attn
from toolkit.BrainLM_Toolkit import convert_fMRIvols_to_A424, convert_to_arrow_datasets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

raw_data_dir = "C:\\yamin\\eeg-fmri\\ACCRE Data\\PROC\\fMRI_sm_nr"
save_data_dir = "./toolkit/sample_dataset/a424_fMRI_data" #Make sure this directory exists.
args = {
    "uk_biobank_dir": "./toolkit/sample_dataset/a424_fMRI_data",     # "Path to directory containing dat files, A424 coordinates file, and A424 excel sheet.",
    "arrow_dataset_save_directory": os.path.join(save_data_dir,"arrow_form"),     # "The directory where you want to save the output arrow datasets."
    "dataset_name": "Test_data_arrow_norm",
}

# convert_fMRIvols_to_A424(data_path=raw_data_dir, output_path=save_data_dir)
convert_to_arrow_datasets(args, args["arrow_dataset_save_directory"])

a = 0
