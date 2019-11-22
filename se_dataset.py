#!/usr/bin/env python
# coding: utf-8

# In[7]:


import argparse
import os

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
import os, csv
import torch
from torch.utils import data


# In[3]:


# Reference
# DATA LOADING - LOAD FILE LISTS
def load_data_list(folder='/home/sangwon/바탕화면/dataset', setname='train'):
    assert(setname in ['train', 'valid'])

    dataset = {}
    # foldername = folder + '/' + setname + 'set'
    foldername = folder + '/' + setname
    # print(foldername) # /home/sangwon/바탕화면/dataset/valid
    

    print("Loading files...")
    dataset['innames'] = []
    dataset['outnames'] = []
    dataset['shortnames'] = []

    noisy_filelist = os.listdir("%s_noisy"%(foldername))
    # filelist = [f for f in filelist if f.endswith(".wav")]
    for i in tqdm(noisy_filelist):
        dataset['innames'].append("%s_noisy/%s"%(foldername,i))
        dataset['shortnames'].append("%s"%(i))
        
    clean_filelist = os.listdir("%s_clean"%(foldername))
    for i in tqdm(clean_filelist):
        dataset['outnames'].append("%s_clean/%s"%(foldername,i))

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


# In[4]:


# train_dataset = AudioDataset(data_type='train')
# train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
#         collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
valid_dataset = AudioDataset(data_type='valid')


# In[5]:


valid_data_loader = DataLoader(dataset=test_dataset, batch_size=4,
        collate_fn=test_dataset.collate, shuffle=False, num_workers=4)


# In[6]:


for tdl in iter(valid_data_loader):
    print(tdl)
    print(len(tdl))
    print(tdl[0].size())
    print(tdl[1].size())
    print(tdl[2].size())
    break


# In[ ]:




