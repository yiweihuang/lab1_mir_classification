# numerical processing and scientific libraries
import numpy as np
import scipy

# signal processing
from scipy.io import wavfile
from scipy.fftpack import fft

from scikits.talkbox.features import mfcc

# plotting
import matplotlib.pyplot as plt
from numpy.lib import stride_tricks

# Classification and evaluation
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold, ShuffleSplit
from sklearn.cross_validation import cross_val_score, StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV

import os
import csv
import collections

PLOT_WIDTH = 15
PLOT_HEIGHT = 3.5

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    frameSize = 1024
    hopSize = 512
    win = window(frameSize)
    # hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(np.floor(frameSize/2.0)), sig)

    # cols for windowing
    cols = np.ceil((len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples,
                                      shape=(cols, frameSize),
                                      strides=(samples.strides[0]*hopSize,
                                               samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)

""" scale frequency axis logarithmfrom scipy.io import wavfileically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs

""" plot spectrogram"""


def plotstft(samples, samplerate, binsize=2**10, plotpath=None, colormap="jet",
             ax=None, fig=None):
    s = stft(samples, binsize)
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel(dB)

    timebins, freqbins = np.shape(ims)

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(PLOT_WIDTH, 3.5))

    cax = ax.imshow(np.transpose(ims), origin="lower", aspect="auto",
                    cmap=colormap, interpolation="none")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (hz)")
    ax.set_xlim([0, timebins-1])
    ax.set_ylim([0, freqbins])

    xlocs = np.float32(np.linspace(0, timebins-1, 5))
    ax.set_xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins) +
                                                (0.5*binsize))/samplerate])

    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    ax.set_yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    b = ["%.02f" % l
         for l in ((xlocs*len(samples)/timebins) + (0.5*binsize)) / samplerate]
    return xlocs, b, timebins


def run_test_file(num_to_label, C, Gamma):
    test_music = {}
    test_data = []
    filelist = []
    for filename in os.listdir('feature/test'):
        test_mess = np.load('feature/test/' + filename)
        num_test_mess = len(test_mess)
        filelist.append(filename.split(".")[0])
        test_data.append(np.mean(test_mess[10:298], axis=0))
    array_music = np.array(test_data)
    a = np.array(array_music)
    a = ((np.array(a) - np.mean(np.array(a), axis=0)) / np.std(a, axis=0))
    for i in range(len(array_music)):
        predict = classifier.predict(a[i])[0]
        test_music[filelist[i]] = num_to_label[predict]

    f = open('test_music/C_%i_Gamma_%.014f.csv' % (C, Gamma), 'w')
    writer = csv.writer(f)
    writer.writerow(["music_num", "instruments"])
    for key, value in sorted(test_music.items()):
        writer.writerow([key, value])

audio = 'audio/'
feature = 'feature/'
audiolist = ["guitar", "piano", "violin", "voice", "test"]
music = {}

# first, creat a structure array for iterating thru the audio files
for instruments in audiolist:
    musicdata = []
    for filename in os.listdir(audio + instruments):
        info = {}
        info["name"] = filename
        info["path"] = audio + instruments + '/' + filename
        musicdata.append(info)
    music[instruments] = musicdata

for obj in music.keys():
    for musicInfo in music[obj]:
        samplerate, wavedata = wavfile.read(musicInfo["path"])
        musicInfo["samplerate"] = samplerate
        musicInfo["wavedata"] = wavedata
        musicInfo["number_of_samples"] = wavedata.shape[0]

feature_info = []
data = []
for obj in music.keys():
    for num in range(len(music[obj])):
        ceps, mspec, spec = mfcc(music[obj][num]["wavedata"])
        iname = (music[obj][num]["name"]).split(".")[0]
        np.save(feature + obj + '/' + iname, ceps)
        ceps = np.load(feature + obj + '/' + iname + '.npy')
        num_ceps = len(ceps)
        if obj != "test":
            data.append(np.mean(ceps[10:298], axis=0))
    feature_info.append(obj)

music_num = 0
feature_info.pop()
detail_feature = {}  # tag label
num_labels = []  # only label for every songs
num_to_label = {}  # {0: 'guitar', 1: 'violin', 2: 'piano', 3: 'voice'}
each_labels = {}

for f_name in feature_info:
    labels = feature_info.index(f_name)
    num_to_label[labels] = f_name
    for i in range(200):
        each_labels[music_num] = f_name
        num_labels.append(labels)
        music_num += 1

detail_feature["data"] = np.array(data)
detail_feature["data_n"] = ((detail_feature["data"] -
                             np.mean(detail_feature["data"], axis=0)) /
                             np.std(detail_feature["data"], axis=0))
detail_feature["num_labels"] = np.array(num_labels)
detail_feature["num_to_label"] = np.array(num_to_label.values())
detail_feature["each_labels"] = np.array(each_labels.values())
sp = ShuffleSplit(len(detail_feature["num_labels"]), n_iter=1, test_size=.20)
train_split, test_split = zip(*sp)

classifier = KNeighborsClassifier(n_neighbors=1)
C = [1, 10, 100, 1000]
G = [float(1) / (700), float(1) / (600), float(1) / (500), float(1) / (450)]
for c in C:
    for g in G:
        classifier = svm.SVC(kernel='rbf', gamma=g, C=c)
        classifier.fit(detail_feature["data_n"][train_split],
                       detail_feature["num_labels"][train_split])
        predictions = classifier.predict(detail_feature["data_n"][test_split])
        report = classification_report(detail_feature["num_labels"][test_split],
                                       predictions,
                                       target_names=detail_feature["num_to_label"])
        # the confusion table
        with open('test_music/report.txt', 'a') as outfile:
            outfile.write('C:%i , Gamma:%.014f \n' % (c, g))
            outfile.write(report)
            outfile.write('\n')
            outfile.close()

        run_test_file(detail_feature["num_to_label"], c, g)

# Q1
# plotstft(music["piano"][0]["wavedata"], music["piano"][0]["samplerate"])

# Q2
# plotstft(music["guitar"][0]["wavedata"], music["guitar"][0]["samplerate"])
# plotstft(music["violin"][0]["wavedata"], music["violin"][0]["samplerate"])

# Q4
# c = 1000
# g = float(1) / (450)
# classifier = svm.SVC(kernel='rbf', gamma=(float(1) / (450)), C=1000)
# classifier.fit(detail_feature["data_n"][train_split],
#                detail_feature["num_labels"][train_split])
# predictions = classifier.predict(detail_feature["data_n"][test_split])
# report_n = classification_report(detail_feature["num_labels"][test_split],
#                                  predictions,
#                                  target_names=detail_feature["num_to_label"])
# print report_n
#
# classifier.fit(detail_feature["data"][train_split],
#                detail_feature["num_labels"][train_split])
# predictions = classifier.predict(detail_feature["data"][test_split])
# report = classification_report(detail_feature["num_labels"][test_split],
#                                predictions,
#                                target_names=detail_feature["num_to_label"])
# print report
