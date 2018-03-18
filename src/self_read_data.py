import sys
import os
import pandas as pd
import numpy as np
import librosa as lb

Fs         = 12000
N_FFT      = 512
N_MELS     = 96
N_OVERLAP  = 256
DURA       = 29.12

labels_file  = '../data/labels.csv'
tags = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
labels = pd.read_csv(labels_file,header=0)

def log_scale_melspectrogram(path, plot=False):
    signal, sr = lb.load(path, sr=Fs)
    n_sample = signal.shape[0]
    n_sample_fit = int(DURA*Fs)

    if n_sample < n_sample_fit:
        signal = np.hstack((signal, np.zeros((int(DURA*Fs) - n_sample,))))
    elif n_sample > n_sample_fit:
        signal = signal[int((n_sample-n_sample_fit)/2):int((n_sample+n_sample_fit)/2)]

    #melspect = lb.logamplitude(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2, ref_power=1.0)
    melspect = lb.power_to_db(lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2)
    #S = lb.feature.melspectrogram(y=signal, sr=Fs, hop_length=N_OVERLAP, n_fft=N_FFT, n_mels=N_MELS)**2
    #melspect = 10 * np.log10(S/1.0)
    #print("Returning single spectrogram")

    return melspect

def get_labels(labels_dense=labels['label'], num_classes=10):
    num_labels = labels_dense.shape[0] #1000
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    print("Returning one hot encoded labels")
    #print(labels_one_hot.shape)
    return labels_one_hot

def get_melspectrograms(labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path']])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    return spectrograms
    
def get_melspectrograms_indexed(index, labels_dense=labels, num_classes=10):
    spectrograms = np.asarray([log_scale_melspectrogram(i) for i in labels_dense['path'][index]])
    spectrograms = spectrograms.reshape(spectrograms.shape[0], spectrograms.shape[1], spectrograms.shape[2], 1)
    #print("Returning collection of spectrograms")
    return spectrograms




