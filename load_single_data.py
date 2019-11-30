# -*- coding: utf-8 -*-

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from scipy.io import wavfile
import librosa
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import librosa
import csv
import torch
from torch.utils import data
import subprocess
# if 'google_drive_downloader' not in sys.modules:
#   subprocess.call('pip install googledrivedownloader'.split())
# from google_drive_downloader import GoogleDriveDownloader as gdd


# dataset_path = os.path.join(gdrive_root, 'dataset')
dataset_path = "/home/under/ddw02141/Attention-SE.pytorch/dataset"
if not os.path.exists(dataset_path):
  os.makedirs(dataset_path)

# Data Loader Part

# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder=dataset_path, setname='train', data_name="Invalid"):
    assert(setname in ['train', 'val', 'test', 'test2', 'test3', 'test4'])

    dataset = {}
    
    if "test" in setname:
      clean_foldername = folder + '/testset'
    else:
      clean_foldername = folder + '/' + setname + "set"
    noisy_foldername = folder + '/' + setname + "set"
    

    print("Loading files...")
    print("data_name")
    print(data_name)
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    noisy_file = ""
    clean_file = ""
    noisy_filelist = os.listdir("%s_noisy"%(noisy_foldername))
    noisy_filelist.sort()
    for file in noisy_filelist:
        if data_name in file:
            noisy_file = file
            break
    # filelist = [f for f in filelist if f.endswith(".wav")]
    
        
    clean_filelist = os.listdir("%s_clean"%(clean_foldername))
    clean_filelist.sort()
    for file in clean_filelist:
        if data_name in file:
            clean_file = file
            break
    if noisy_file=="" or clean_file=="":
        print("****************************************")
        print("File name with %s does not exist"%(data_name))
    dataset['innames'].append("%s_noisy/%s"%(noisy_foldername,noisy_file))
    dataset['shortnames'].append("%s"%(noisy_file))
    dataset['outnames'].append("%s_clean/%s"%(clean_foldername,clean_file))

    return dataset

# DATA LOADING - LOAD FILE DATA
def load_data(dataset):

    dataset['inaudio']  = [None]*len(dataset['innames'])
    dataset['outaudio'] = [None]*len(dataset['outnames'])

    for id in tqdm(range(len(dataset['innames']))):

        if dataset['inaudio'][id] is None:
            inputData, sr = librosa.load(dataset['innames'][id], sr=None)
            outputData, sr = librosa.load(dataset['outnames'][id], sr=None)

            shape = np.shape(inputData)

            dataset['inaudio'][id]  = np.float32(inputData)
            dataset['outaudio'][id] = np.float32(outputData)

    return dataset

class AudioDataset(data.Dataset):
    """
    Audio sample reader.
    """

    def __init__(self, data_type, data_name):
        dataset = load_data_list(setname=data_type, data_name = data_name)
        self.dataset = load_data(dataset)

        self.file_names = dataset['innames']

    def __getitem__(self, idx):
        mixed = torch.from_numpy(self.dataset['inaudio'][idx]).type(torch.FloatTensor)
        clean = torch.from_numpy(self.dataset['outaudio'][idx]).type(torch.FloatTensor)

        return mixed, clean

    def __len__(self):
        return len(self.file_names)

    def zero_pad_concat(self, inputs):
        max_t = max(inp.shape[0] for inp in inputs)
        shape = (len(inputs), max_t)
        input_mat = np.zeros(shape, dtype=np.float32)
        for e, inp in enumerate(inputs):
            input_mat[e, :inp.shape[0]] = inp
        return input_mat

    def collate(self, inputs):
        mixeds, cleans = zip(*inputs)
        seq_lens = torch.IntTensor([i.shape[0] for i in mixeds])

        x = torch.FloatTensor(self.zero_pad_concat(mixeds))
        y = torch.FloatTensor(self.zero_pad_concat(cleans))

        batch = [x, y, seq_lens]
        return batch

# Below is how to use single data loader

# train_data = AudioDataset(data_type='train', data_name="p287_171")
# train_data_loader = DataLoader(dataset=train_data,
#        collate_fn=train_data.collate, shuffle=True, num_workers=4)

# for train in train_data_loader:
#     print(train)


