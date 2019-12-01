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
from load_single_data import AudioDataset
from models.attention import AttentionModel

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=128, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
parser.add_argument('--dropout_p', default = 0, type=float, help='Attention model drop out rate')
parser.add_argument('--learning_rate', default = 5e-4, type=float, help = 'Learning rate')
parser.add_argument('--stacked_encoder', default = False, type = bool)
parser.add_argument('--attn_len', default = 0, type = int)
parser.add_argument('--hidden_size', default = 112, type = int)
parser.add_argument('--ck_dir', default = 'ckpt_dir', help = 'ck path')
parser.add_argument('--ck_name',  help = 'ck file')
parser.add_argument('--test_set',  help = 'test_set')
parser.add_argument('--attn_use', default = False, type=bool)
parser.add_argument('--wav')
args = parser.parse_args()


n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

def main(): 
    
    test_data_set = AudioDataset(data_type=args.test_set, data_name=args.wav)
    test_data_loader = DataLoader(dataset=test_data_set, collate_fn=test_data_set.collate, num_workers=4)


    
    #model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(AttentionModel(257, hidden_size = args.hidden_size, dropout_p = args.dropout_p, use_attn = args.attn_use, stacked_encoder = args.stacked_encoder, attn_len = args.attn_len))
    #net = AttentionModel(257, 112, dropout_p = args.dropout_p, use_attn = args.attn_use)
    net = net.cuda()

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    #Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = args.ck_dir
    best_PESQ = 0.
    best_STOI = 0.
    ckpt_path = os.path.join(ckpt_dir, args.ck_name)


    if not os.path.exists('Test_wav_stoi'):
        os.makedirs('test_wav_stoi')
    
    test_dir = os.path.join('Test_wav_stoi', args.ck_name)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    test_dir = os.path.join(test_dir, args.test_set)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if os.path.exists(ckpt_path):
    	ckpt = torch.load(ckpt_path)
    	try:
       	    net.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            best_STOI = ckpt['best_STOI']

            print('checkpoint is loaded !')
            print('current best loss : %.4f' % best_STOI)
    	except RuntimeError as e:
            print('wrong checkpoint\n')
    else:    
        print('checkpoint not exist!')
        print('current best loss : %.4f' % best_STOI)

    #test phase
    n = 0
    avg_test_loss = 0

    
    net.eval()
    with torch.no_grad():
        test_bar = tqdm(test_data_loader)
        for input in test_bar:
            test_mixed, test_clean, seq_len = map(lambda x: x.cuda(), input)
            mixed = stft(test_mixed)
            cleaned = stft(test_clean)
            mixed = mixed.transpose(1,2)
            cleaned = cleaned.transpose(1,2)
            real, imag = mixed[..., 0], mixed[..., 1]
            clean_real, clean_imag = cleaned[..., 0], cleaned[..., 1]
            mag = torch.sqrt(real**2 + imag**2)
            clean_mag = torch.sqrt(clean_real**2 + clean_imag**2)
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
        
            test_loss = 0
            test_PESQ = 0
            test_STOI = 0
        
            test_loss = F.mse_loss(logits_mag, clean_mag, True)

            mixed_wav = os.path.join(test_dir, 'mixed.wav')
            
            clean_wav = os.path.join(test_dir, 'clean.wav')

            out_wav = os.path.join(test_dir, 'out.wav')

            for i in range(len(test_mixed)):
                librosa.output.write_wav(mixed_wav, test_mixed[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                librosa.output.write_wav(clean_wav, test_clean[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                librosa.output.write_wav(out_wav, logits_audio[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
                test_PESQ += pesq(test_clean[i].detach().cpu().numpy(), logits_audio[i].detach().cpu().numpy(), 16000)
                test_STOI += stoi(test_clean[i].detach().cpu().numpy(), logits_audio[i].detach().cpu().numpy(), 16000, extended=False)
        
            test_PESQ /= len(test_mixed)
            test_STOI /= len(test_mixed)	
            avg_test_loss += test_loss
            n += 1

            #test accuracy
            #test_pesq = pesq('test_clean.wav', 'test_out.wav', 16000)
            #test_stoi = stoi('test_clean.wav', 'test_out.wav', 16000)

        avg_test_loss /= n
            #summary.add_scalar('Test Loss', avg_test_loss.item(), iteration)
        print('test loss : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(avg_test_loss, test_PESQ, test_STOI))

if __name__ == '__main__':
    main()
