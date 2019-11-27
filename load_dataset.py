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

train_tar_path = os.path.join(dataset_path, "train.tar.gz")
valid_tar_path = os.path.join(dataset_path, "valid.tar.gz")
test1_tar_path = os.path.join(dataset_path, "test1.tar.gz")
test2_tar_path = os.path.join(dataset_path, "test2.tar.gz")
test3_tar_path = os.path.join(dataset_path, "test3.tar.gz")
test4_tar_path = os.path.join(dataset_path, "test4.tar.gz")
test5_tar_path = os.path.join(dataset_path, "test5.tar.gz")

# # Download Train set
# if not os.path.exists(train_tar_path):

#   gdd.download_file_from_google_drive(file_id='1CULVCAq0T3wqZTPGIqPja6OtwjYJkGAy',
#                                       dest_path=train_tar_path,
#                                       unzip=False, showsize=True)
# train_clean_path = os.path.join(dataset_path, "train_clean")
# train_noisy_path = os.path.join(dataset_path, "train_noisy")
# if not os.path.exists(train_clean_path) or not os.path.exists(train_noisy_path):
#   tar xvzf train.tar.gz

# # Download Valid set
# if not os.path.exists(valid_tar_path):

#   gdd.download_file_from_google_drive(file_id='1WE229Jt9WV2iZbxY7YjkYIfSZyCHz9Iq',
#                                     dest_path=valid_tar_path,
#                                     unzip=False, showsize=True)
# valid_clean_path = os.path.join(dataset_path, "valid_clean")
# valid_noisy_path = os.path.join(dataset_path, "valid_noisy")
# if not os.path.exists(valid_clean_path) or not os.path.exists(valid_noisy_path):
#   !tar xvzf valid.tar.gz

# # Download Test set 1
# if not os.path.exists(test1_tar_path):

#   gdd.download_file_from_google_drive(file_id='16JIMca-JVXgQd7dltYdTducMek4pA58_',
#                                       dest_path=test_tar_path,
#                                       unzip=False, showsize=True)
# # %cd /gdrive/My\ Drive/dataset
# test1_clean_path = os.path.join(dataset_path, "test_clean")
# test1_noisy_path = os.path.join(dataset_path, "test_noisy")
# if not os.path.exists(test1_clean_path) or not os.path.exists(test1_noisy_path):
#   !tar xvzf /gdrive/My\ Drive/dataset/test1.tar.gz

# # Download Test set 2
# if not os.path.exists(test1_tar_path):

#   gdd.download_file_from_google_drive(file_id='10YSj0u_9ni_sVriiNNs97OrKKSeshvr7',
#                                       dest_path=test_tar_path,
#                                       unzip=False, showsize=True)
# # %cd /gdrive/My\ Drive/dataset
# test1_clean_path = os.path.join(dataset_path, "test_clean")
# test1_noisy_path = os.path.join(dataset_path, "test_noisy")
# if not os.path.exists(test1_clean_path) or not os.path.exists(test1_noisy_path):
#   !tar xvzf /gdrive/My\ Drive/dataset/test1.tar.gz

# # Download Test set 3
# if not os.path.exists(test1_tar_path):

#   gdd.download_file_from_google_drive(file_id='1J-UNPZ9SkJECih9SjQDMm6j51nus5Z4q',
#                                       dest_path=test_tar_path,
#                                       unzip=False, showsize=True)
# # %cd /gdrive/My\ Drive/dataset
# test1_clean_path = os.path.join(dataset_path, "test_clean")
# test1_noisy_path = os.path.join(dataset_path, "test_noisy")
# if not os.path.exists(test1_clean_path) or not os.path.exists(test1_noisy_path):
#   !tar xvzf /gdrive/My\ Drive/dataset/test1.tar.gz

# # Download Test set 4
# if not os.path.exists(test1_tar_path):

#   gdd.download_file_from_google_drive(file_id='1j3lrPTp-2gQudNfJgayK-zUp2x19rwPQ',
#                                       dest_path=test_tar_path,
#                                       unzip=False, showsize=True)
# # %cd /gdrive/My\ Drive/dataset
# test1_clean_path = os.path.join(dataset_path, "test_clean")
# test1_noisy_path = os.path.join(dataset_path, "test_noisy")
# if not os.path.exists(test1_clean_path) or not os.path.exists(test1_noisy_path):
#   !tar xvzf /gdrive/My\ Drive/dataset/test1.tar.gz

# # Download Test set 5
# if not os.path.exists(test1_tar_path):

#   gdd.download_file_from_google_drive(file_id='1HQXw26XOtg186QApNVx8DR85wKrFBjcW',
#                                       dest_path=test_tar_path,
#                                       unzip=False, showsize=True)
# # %cd /gdrive/My\ Drive/dataset
# test1_clean_path = os.path.join(dataset_path, "test_clean")
# test1_noisy_path = os.path.join(dataset_path, "test_noisy")
# if not os.path.exists(test1_clean_path) or not os.path.exists(test1_noisy_path):
#   !tar xvzf /gdrive/My\ Drive/dataset/test1.tar.gz


# Data Loader Part

# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder=dataset_path, setname='train'):
    assert(setname in ['train', 'val', 'test', 'test2', 'test3', 'test4'])

    dataset = {}
    
    if "test" in setname:
      clean_foldername = folder + '/testset'
    else:
      clean_foldername = folder + '/' + setname + "set"
    noisy_foldername = folder + '/' + setname + "set"
    

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    noisy_filelist = os.listdir("%s_noisy"%(noisy_foldername))
    noisy_filelist.sort()
    # filelist = [f for f in filelist if f.endswith(".wav")]
    for i in tqdm(noisy_filelist):
        dataset['innames'].append("%s_noisy/%s"%(noisy_foldername,i))
        dataset['shortnames'].append("%s"%(i))
        
    clean_filelist = os.listdir("%s_clean"%(clean_foldername))
    clean_filelist.sort()
    for i in tqdm(clean_filelist):
        dataset['outnames'].append("%s_clean/%s"%(clean_foldername,i))

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

    def __init__(self, data_type):
        dataset = load_data_list(setname=data_type)
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

# Below is how to use data loader

# train_dataset = AudioDataset(data_type='train')
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=4,
#        collate_fn=train_dataset.collate, shuffle=True, num_workers=4)

# # valid_dataset = AudioDataset(data_type='valid')
# # valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=4,
# #         collate_fn=valid_dataset.collate, shuffle=False, num_workers=4)
# train_bar = tqdm(train_data_loader)

# test_dataset = AudioDataset(data_type='test')
# test_data_loader = DataLoader(dataset=test_dataset, batch_size=4,
#        collate_fn=test_dataset.collate, shuffle=True, num_workers=4)

# test_dataset2 = AudioDataset(data_type='test2')
# test_data_loader2 = DataLoader(dataset=test_dataset2, batch_size=4,
#        collate_fn=test_dataset2.collate, shuffle=True, num_workers=4)

