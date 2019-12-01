import os
import argparse

import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import librosa

from pystoi.stoi import stoi
from pypesq import pesq

from models.layers.istft import ISTFT
import train_utils
from models.attention import AttentionModel

parser = argparse.ArgumentParser()
parser.add_argument('--dropout_p', default = 0, type=float, help='Attention model drop out rate')
parser.add_argument('--stacked_encoder', default = False, type = bool)
parser.add_argument('--attn_len', default = 0, type = int)
parser.add_argument('--hidden_size', default = 112, type = int)
parser.add_argument('--ck_dir', default = 'ckpt_dir', help = 'ck path')
parser.add_argument('--ck_name',  help = 'ck file')
parser.add_argument('--test_set',  help = 'test_set')
parser.add_argument('--attn_use', default = False, type=bool)
parser.add_argument('--noisy_wav')
parser.add_argument('--clean_wav')
args = parser.parse_args()


n_fft, hop_length = 512, 128
window = torch.hann_window(n_fft).cuda()
# STFT
stft = lambda x: torch.stft(x, n_fft, hop_length, window=window)
# ISTFT
istft = ISTFT(n_fft, hop_length, window='hanning').cuda()

def main():
    #model select
    print('Model initializing\n')
    net = torch.nn.DataParallel(AttentionModel(257, hidden_size = args.hidden_size, dropout_p = args.dropout_p, use_attn = args.attn_use, stacked_encoder = args.stacked_encoder, attn_len = args.attn_len))
    net = net.cuda()

    #Check point load
    print('Trying Checkpoint Load\n')
    ckpt_dir = args.ck_dir
    best_PESQ = 0.
    best_STOI = 0.
    ckpt_path = os.path.join(ckpt_dir, args.ck_name)


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
    net.eval()
    with torch.no_grad():
        inputData, sr = librosa.load(args.noisy_wav, sr=None)
        outputData, sr = librosa.load(args.clean_wav, sr=None)
        inputData = np.float32(inputData)
        outputData = np.float32(outputData)
        mixed_audio = torch.from_numpy(inputData).type(torch.FloatTensor)
        clean_audio = torch.from_numpy(outputData).type(torch.FloatTensor)

        mixed = stft(mixed_audio)
        cleaned = stft(clean_audio)
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

        logits_audio = istft(logits_real, logits_imag, mixed_audio.size(1)) # TODO - check dim
        logits_audio = torch.squeeze(logits_audio, dim=1)

        librosa.output.write_wav('./out.wav', logits_audio[i].cpu().data.numpy(), 16000)
        test_loss = F.mse_loss(logits_mag, clean_mag, True)
        test_PESQ = pesq(test_clean[i].detach().cpu().numpy(), logits_audio[i].detach().cpu().numpy(), 16000)
        test_STOI = stoi(test_clean[i].detach().cpu().numpy(), logits_audio[i].detach().cpu().numpy(), 16000, extended=False)

        #test accuracy
        #test_pesq = pesq('test_clean.wav', 'test_out.wav', 16000)
        #test_stoi = stoi('test_clean.wav', 'test_out.wav', 16000)
        print('test loss : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(test_loss, test_PESQ, test_STOI))

if __name__ == '__main__':
    main()
