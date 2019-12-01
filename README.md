# Attention-SE.pytorch
AN ATTENTION-BASED NEURAL NETWORK APPROACH FOR SINGLE CHANNEL SPEECH ENHANCEMENT

## Requirements

* PYSTOI 
  * Toolbox : [mpariente/pystoi](https://github.com/mpariente/pystoi)
* PYPESQ
  * ToolBox : [vBaiCai/python-pesq](https://github.com/vBaiCai/python-pesq)
* Other
```python3
torch == 1.2.0
numpy 
librosa
```

## DataSet
Noise dataset uses 'MUSAN' dataset
We generate noisy mixed data from dataset

|Set|Train|Valid|Test|Test2|Test3|Test4|
|---|-----|-----|----|-----|-----|-----|
|Noise|Musan|Musan|Musan|Musan|DEMAND|DEMAND|
|SNR|0-20dB|0-20dB|0-20dB|-5-0dB|0-20dB|-5-0dB|

#### Musan data set
Speech and Noise recording
- URL : http://www.openslr.org/17/
- Cites
```BibTeX
@misc{musan2015,
  author = {David Snyder and Guoguo Chen and Daniel Povey},
  title = {{MUSAN}: {A} {M}usic, {S}peech, and {N}oise {C}orpus},
  year = {2015},
  eprint = {1510.08484},
  note = {arXiv:1510.08484v1}
}
```

#### Data Loading Script
Reference URL : 
- https://github.com/sweetcocoa/DeepComplexUNetPyTorch
- https://github.com/jtkim-kaist/Speech-enhancement
- https://github.com/chanil1218/DCUnet.pytorch

- load_dataset.py
```Python3
# load_dataset
# Below is how to use data loader
# data_type = ['val', 'train', 'test', 'test2', 'test3', 'test4']

import load_dataset from AudioDataset

train_dataset = AudioDataset(data_type='train')
train_data_loader = DataLoader(dataset=train_dataset, batch_size=4, collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
```

- load_single_data.py
```Python3
# load_single_data
# Below is how to use data loader
# data_type = ['val', 'train', 'test', 'test2', 'test3', 'test4']

import load_single_data from AudioDataset

train_data = AudioDataset(data_type='train', data_name="p287_171")
train_data_loader = DataLoader(dataset=train_data, collate_fn=train_data.collate, shuffle=True, num_workers=4)
```



## Attention-based SE Model
Reference : Xiang Hao 'AN ATTENTION-BASED NEURAL NETWORK APPROACH FOR SINGLE CHANNEL
SPEECH ENHANCEMENT', 2019
URL : http://lxie.nwpu-aslp.org/papers/2019ICASSP-XiangHao.pdf


## Train
Arguments : 
- batch_size : Train batch size, default = 128
- dropout_p : Attention model's dropout rate, default = 0
- attn_use : Use Attention model, if it is False, Train with LSTM model.  default = False
- stacked_encoder : Use Stacked attention model, if it is False, Train with Extanded Attention model. default = False
- hidden_size : Size of RNN. default = 0
- num_epochs : Train epochs number. default = 100
- learning_rate : Training Learning rate. default = 5e-4
- ck_name : Name with save/load check point. default = 'SEckpt.pt'

```bash
CUDA_VISIBLE_DEVICES=GPUNUMBERS \
python3 train.py --batch_size 128 \
                 --dropout_p 0.2\
                 --attn_use True \
                 --stacked_encoder True\
                 --attn_len 5\
                 --hidden_size 448\
                 --num_epochs 61\
                 --ck_name '5_448_stacked_dropout0_2.pt'

# You can check other arguments from the source code.                   
```

## Test
Test print mean loss, PESQ, and STOI.
Arguments : 
- batch_size : Train batch size, default = 128
- dropout_p : Attention model's dropout rate, default = 0
- attn_use : Use Attention model, if it is False, Train with LSTM model.  default = False
- stacked_encoder : Use Stacked attention model, if it is False, Train with Extanded Attention model. default = False
- hidden_size : Size of RNN. default = 0
- ck_name : Name with load check point. default = 'SEckpt.pt'
- test_set : Name of data_type

```bash
CUDA_VISIBLE_DEVICES=GPUNUMBERS \
python3 test.py --batch_size 128 \
                --dropout_p 0.2\
                --attn_use True \
                --stacked_encoder True\
                --attn_len 5\
                --hidden_size 448\
                --test_set 'test'\
                --ck_name '5_448_stacked_dropout0_2.pt'

# You can check other arguments from the source code.                   
```

## Single file test
Single file test return sample outputs with .wav files.
- clean.wav : select clean voice data
- mixed.wav : noisy voice data
- out.wav : return output from model
Arguments :
- batch_size : Train batch size, default = 128
- dropout_p : Attention model's dropout rate, default = 0
- attn_use : Use Attention model, if it is False, Train with LSTM model.  default = False
- stacked_encoder : Use Stacked attention model, if it is False, Train with Extanded Attention model. default = False
- hidden_size : Size of RNN. default = 0
- ck_name : Name with load check point. default = 'SEckpt.pt'
- test_set : Name of data_type
- wav : Name of clean data

```bash
CUDA_VISIBLE_DEVICES=GPUNUMBERS \
python3 test_single.py --batch_size 128 \
                       --dropout_p 0.2\
                       --attn_use True \
                       --stacked_encoder True\
                       --attn_len 5\
                       --hidden_size 448\
                       --test_set 'test'\
                       --wav 'p232_238'
                       --ck_name '5_448_stacked_dropout0_2.pt'
                      

# You can check other arguments from the source code.                   
```
