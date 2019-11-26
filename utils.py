# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy
import matplotlib.pyplot as plt
import json
import argparse
import librosa
import numpy as np
from PIL import Image
from matplotlib import cm

def plot_head_map(mma, num = 5):
  
    space = len(mma)/num
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(mma, cmap=plt.cm.jet)

    # put the major ticks at the middle of each cell
    ax.set_xticks(numpy.arange(mma.shape[1])*space + 0.5, minor=False)
    ax.set_yticks(numpy.arange(mma.shape[0])*space + 0.5, minor=False)

    # without this I get some extra columns rows
    ax.set_xlim(0, int(mma.shape[1]))
    ax.set_ylim(0, int(mma.shape[0]))

    # want a more natural, table-like display
    #ax.invert_yaxis()
    #ax.xaxis.tick_top()

    # source words -> column labels
    ax.set_xticklabels(numpy.arange(mma.shape[1])*space, minor=False)
    # target words -> row labels
    ax.set_yticklabels(numpy.arange(mma.shape[0])*space, minor=False)

    plt.xticks(rotation=45)

    plt.colorbar(heatmap, ax=ax)
    # plt.tight_layout()
    plt.show()
    #plt.savefig('result.png')


#ntt = numpy.array([[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1],[0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.2],[0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.3],[0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.4],[0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5],[0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6],[0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7],[0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.8],[0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89,0.9],[0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99,1]])
#plot_head_map(ntt)


def get_spectrogram(base):
    S = librosa.amplitude_to_db(librosa.core.magphase(librosa.stft(base, hop_length=128, win_length=512, n_fft=512))[0], ref=np.max)
    S = prepare_spec_image(S)
    return S

def prepare_spec_image(spectrogram):
    spectrogram = (spectrogram - np.min(spectrogram)) / ((np.max(spectrogram)) - np.min(spectrogram))
    spectrogram = np.flip(spectrogram, axis=0)
    return np.uint8(cm.gist_heat(spectrogram) * 255)

def read_raw(input_file_dir):
    data = np.fromfile(input_file_dir, dtype=np.int16)  # (# total frame, feature_size)
    data = np.float32(data) / 32767.
    data = np.squeeze(data)
    return data

def make_spectrogram_array(file_name):
    y = read_raw(file_name)
    y_spec = get_spectrogram(y).transpose(2,0,1)[0:3].transpose(1,2,0)
    return y_spec

def save_spectrogram(spec, file_name):
    img = Image.fromarray(spec, 'RGB')
    img.save('spec_'+file_name+'.png')

#input = "clean4.wav"
#spt_array = make_spectrogram_array(input)
#save_spectrogram(spt_array, input)