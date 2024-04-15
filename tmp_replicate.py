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

# from datasets import load_from_disk, concatenate_datasets
from brainlm_mae.modeling_brainlm import BrainLMForPretraining
from utils.brainlm_trainer import BrainLMTrainer
from utils.plots import plot_future_timepoint_trends


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainLMForPretraining.from_pretrained(
    f"C:\\yamin\\eeg-fmri-DL\\BrainLM\\pretrained_models\\checkpoint").to(device)

print(model.vit.config)

a = 0