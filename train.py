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

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None, help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'best' or 'train'
parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
parser.add_argument('--num_epochs', default=100, type=int, help='train epochs number')
args = parser.parse_args()

n_fft, hop_length = 400, 160
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
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=4, collate_fn=train_dataset.collate, shuffle=True, num_workers=4)
    valid_dataset = AudioDataset(data_type='val')
    valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=4, collate_fn=valid_dataset.collate, shuffle=False, num_workers=4)
    #model select
    print('Model initializing\n')
    net = Unet(params.model).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    #check point load
    #Check point load

    print('Trying Checkpoint Load\n')
    ckpt_dir = 'ckpt_dir'
    if not os.path.exists(ckpt_dir):
    	os.makedirs(ckpt_dir)

    best_pesq = 0.
    best_stoi = 0.
    ckpt_path = os.path.join(ckpt_dir, 'SEckpt.pt')
    if os.path.exists(ckpt_path):
    	ckpt = torch.load(ckpt_path)
    	try:
       		net.load_state_dict(ckpt['model'])
        	optimizer.load_state_dict(ckpt['optimizer'])
        	best_pesq = ckpt['best_pesq']
    	except RuntimeError as e:
        	print('wrong checkpoint\n')
    else:    
        print('checkpoint is loaded !')
        print('current best pesq : %.4f' % best_pesq)
        print('current best stoi : %.4f' % best_stoi)
    
    print('Training Start!')
    #train
    iteration = 0
    for epoch in range(args.num_epochs):
        train_bar = tqdm(train_data_loader)
        for input in train_bar:
            iteration += 1
            #load data
            train_mixed, train_clean, seq_len = map(lambda x: x.cuda(), input)
            mixed = stft(train_mixed).unsqueeze(dim=1)
            real, imag = mixed[..., 0], mixed[..., 1]

            #feed data
            out_real, out_imag = net(real, imag)
            out_real, out_imag = torch.squeeze(out_real, 1), torch.squeeze(out_imag, 1)
            out_audio = istft(out_real, out_imag, train_mixed.size(1))
            out_audio = torch.squeeze(out_audio, dim=1)
            for i, l in enumerate(seq_len):
                out_audio[i, l:] = 0

	    loss = 0
            PESQ = 0
	    STOI = 0
	    for i in range(args.batch_size):
            	librosa.output.write_wav('mixed.wav', train_mixed[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
            	librosa.output.write_wav('clean.wav', train_clean[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
            	librosa.output.write_wav('out.wav', out_audio[i].cpu().data.numpy()[:seq_len[i].cpu().data.numpy()], 16000)
		out = stft(out_audio[i]).unsqueeze(dim=1)
		clean = stft(train_clean[i]).unsqueeze(dim=1)
		loss += torch.nn.MSELoss(out, clean)
		PESQ += pesq('clean.wav', 'out.wav', 16000)
		STOI += stoi('clean.wav', 'out.wav', 16000)
	
	    loss /= args.batch_size
	    PESQ /= args.batch_size
	    STOI /= agrs.batch_size	
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
            if iteration % "num" == 0:
                summary.add_scalar('Train Loss', loss.item(), iteration)
                print('[epoch: {}, iteration: {}] train loss : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(epoch, iteration, loss, PESQ, STOI))

            #test phase
            n = 0
            test_loss = 0
            test_acc = 0
            test_bar = tqdm(valid_data_loader)
            for input in test_bar:
                test_mixed, text_clean, seq_len
                logits_real, logits_imag = net(input)
                logits_real, logits_imag = torch.squeeze(logits_real, 1), torch.squeeze(logits_imag, 1)
                logits_audio = istft(logits_real, logits_imag, test_mixed.size(1))
                logits_audio = torch.squeeze(logits_audio, dim=1)
                for i, l in enumerate(test_seq_len):
                  logits_audio[i, l:] = 0
                librosa.output.write_wav('test_clean.wav', test_clean[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)
                librosa.output.write_wav('test_out.wav', logits_audio[0].cpu().data.numpy()[:seq_len[0].cpu().data.numpy()], 16000)


                #test loss
                #test_loss = wSDRLoss(test_mixed, test_clean, out_audio)
                test_loss = torch.nn.MSELoss(out_audio, test_clean)

                #test accuracy
                test_pesq = pesq('test_clean.wav', 'test_out.wav', 16000)
                test_stoi = stoi('test_clean.wav', 'test_out.wav', 16000)

            summary.add_scalar('Test Loss', test_loss.item(), iteration)
            print('[epoch: {}, iteration: {}] test loss : {:.4f} PESQ : {:.4f} STOI : {:.4f}'.format(epoch, iteration, test_loss, test_pesq, test_stoi))
            if test_acc > best_acc:
                best_acc = test_acc
                # Note: optimizer also has states ! don't forget to save them as well.
                ckpt = {'my_classifier':net.state_dict(),
                        'optimizer':optimizer.state_dict(),
                        'best_acc':best_acc}
                torch.save(ckpt, ckpt_path)
                print('checkpoint is saved !')

if __name__ == '__main__':
    main()
