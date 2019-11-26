import os
import argparse
import sys 

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision 
from torchvision import transforms
from torch.optim.lr_scheduler import ExponentialLR

import tensorboardX
from tensorboardX import SummaryWriter


from scipy.io import wavfile
import librosa

import soundfile as sf
from pystoi.stoi import stoi
from pypesq import pesq

from tqdm import tqdm
from models.layers.istft import ISTFT
import train_utils
from load_dataset import AudioDataset
from models.unet import Unet
from models.attention import AttentionModel

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiment/SE_model.json', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default = 0, type=float, help='Attention model drop out rate')
args = parser.parse_args()


n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

def main():
    summary = SummaryWriter()
    #os.system('tensorboard --logdir=path_of_log_file')

    #set Hyper parameter
    json_path = os.path.join(args.model_dir)
    params = train_utils.Params(json_path)

    #data loader
    train_dataset = AudioDataset(data_type='train')
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
    valid_dataset = AudioDataset(data_type='val')
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, collate_fn=valid_dataset.collate, shuffle=False, num_workers=4)
    #model select
    print('Model initializing\n')
    net = AttentionModel(257, 112, dropout_p = args.dropout_p).cuda()
    optimizer = optim.Adam(net.parameters(), lr=5e-4)

    scheduler = ExponentialLR(optimizer, 0.5)

    #check point load
    #Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = 'ckpt_dir'
    if not os.path.exists(ckpt_dir):
    	os.makedirs(ckpt_dir)

    best_PESQ = 0.
    best_STOI = 0.
    ckpt_path = os.path.join(ckpt_dir, 'SEckpt.pt')
    if os.path.exists(ckpt_path):
    	ckpt = torch.load(ckpt_path)
    	try:
       	    net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_PESQ = ckpt['best_PESQ']
            best_STOI = ckpt['best_STOI']
    	except RuntimeError as e:
            print('wrong checkpoint\n')
    else:    
        print('checkpoint is loaded !')
        print('current best PESQ : %.4f' % best_PESQ)
        print('current best STOI : %.4f' % best_STOI)
    
    print('Training Start!')
    #train
    iteration = 0
    train_losses = []
    for epoch in range(args.num_epochs):
        train_bar = tqdm(train_data_loader)
        # train_bar = train_data_loader
        for input in train_bar:
            iteration += 1
            #load data
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
            mixed = stft(train_mixed)
            mixed = mixed.transpose(1,2)
            real, imag = mixed[..., 0], mixed[..., 1]

            mag = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)

            #feed data
            out_mag, attn_wegiht = net(mag)
            out_real = out_mag * torch.cos(phase)
            out_imag = out_mag * torch.sin(phase)
            out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
            out_real = out_real.transpose(1,2)
            out_imag = out_imag.transpose(1,2)

            out_audio = istft(out_real, out_imag, train_mixed.size(1))
            out_audio = torch.squeeze(out_audio, dim=1)
            for i, l in enumerate(seq_len):
                out_audio[i, l:] = 0

            loss = 0
            PESQ = 0
            STOI = 0
            for i in range(len(train_mixed)):
                librosa.output.write_wav('mixed.wav', train_mixed[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                librosa.output.write_wav('clean.wav', train_clean[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                librosa.output.write_wav('out.wav', out_audio[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                out_acc, fs = sf.read('out.wav')
                clean_acc, fs = sf.read('clean.wav')
                out_stft = stft(out_audio[i]).unsqueeze(dim=1)
                clean_stft = stft(train_clean[i]).unsqueeze(dim=1)
                loss += F.mse_loss(out_stft[1], clean_stft[1],True)
                PESQ += pesq(clean_acc, out_acc, fs)
                STOI += stoi(clean_acc, out_acc, fs, extended=False)
        
            loss /= len(train_mixed)
            PESQ /= len(train_mixed)
            STOI /= len(train_mixed)	
            #calculate LOSS
            #loss =  wSDRLoss(train_mixed, train_clean, out_audio)
            #loss = torch.nn.MSELoss(out_audio, train_clean)
           
	    #gradient optimizer
            optimizer.zero_grad()

            #backpropagate LOSS
            loss.backward()

            #update weight
            optimizer.step()

            #calculate accuracy
            #PESQ = pesq('clean.wav', 'out.wav', 16000)

            #STOI = stoi('clean.wav', 'out.wav', 16000)


            #flot tensorboard
            if (iteration % 2000) == 0:
                summary.add_scalar('Train Loss', loss.item(), iteration)
                print('[epoch: {}, iteration: {}] train loss : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(epoch, iteration, loss, PESQ, STOI))
        
        train_losses.append(loss)
        if  (len(train_losses) > 2) and (train_losses[-2] < loss):
            print("Learning rate Decay")
            scheduler.step()

        #test phase
        n = 0
        test_loss = 0
        test_PESQ = 0
        test_STOI = 0
        test_bar = tqdm(valid_data_loader)
        for input in test_bar:
            test_mixed, test_clean, seq_len = map(lambda x: x.cuda(), input)
            mixed = stft(test_mixed)
            mixed = mixed.transpose(1,2)
            real, imag = mixed[..., 0], mixed[..., 1]

            mag = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)

            logits_mag, logits_attn_weight = net(mag)
            logits_real = logits_mag * torch.cos(phase)
            logits_imag = logits_mag * torch.sin(phase)
            logits_real, logits_imag = torch.squeeze(logits_real, 1), torch.squeeze(logits_imag, 1)
            logits_real = logits_real.transpose(1,2)
            logits_imag = logits_imag.transpose(1,2)
                
            logits_audio = istft(logits_real, logits_imag, test_mixed.size(1))
            logits_audio = torch.squeeze(logits_audio, dim=1)
            for i, l in enumerate(seq_len):
                logits_audio[i, l:] = 0
            for i in range(len(test_mixed)):
                librosa.output.write_wav('test_mixed.wav', test_mixed[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                librosa.output.write_wav('test_clean.wav', test_clean[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                librosa.output.write_wav('test_out.wav', logits_audio[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                out_acc, fs = sf.read('test_out.wav')
                clean_acc, fs = sf.read('test_clean.wav')
                out_stft = stft(logits_audio[i]).unsqueeze(dim=1)
                clean_stft = stft(test_clean[i]).unsqueeze(dim=1)
                test_loss += F.mse_loss(out_stft[1], clean_stft[1],True)
                test_PESQ += pesq(clean_acc, out_acc, fs)
                test_STOI += stoi(clean_acc, out_acc, fs, extended=False)
        
            test_loss /= len(test_mixed)
            test_PESQ /= len(test_mixed)
            test_STOI /= len(test_mixed)	

            #test loss
            #test_loss = wSDRLoss(test_mixed, test_clean, out_audio)
            #test_loss = torch.nn.MSELoss(out_audio, test_clean)

            #test accuracy
            #test_pesq = pesq('test_clean.wav', 'test_out.wav', 16000)
            #test_stoi = stoi('test_clean.wav', 'test_out.wav', 16000)

        summary.add_scalar('Test Loss', test_loss.item(), iteration)
        print('[epoch: {}, iteration: {}] test loss : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(epoch, iteration, test_loss, test_PESQ, test_STOI))
        if test_PESQ > best_PESQ or test_STOI > best_STOI:
            best_PESQ = test_PESQ
            best_STOI = test_STOI

            # Note: optimizer also has states ! don't forget to save them as well.
            ckpt = {'model':net.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'best_PESQ':best_PESQ,
                    'best_STOI':best_STOI}
            torch.save(ckpt, ckpt_path)
            print('checkpoint is saved !')

if __name__ == '__main__':
    main()
