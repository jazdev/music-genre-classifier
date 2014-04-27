import sys
import os
import glob

import numpy as np
import scipy
import scipy.io.wavfile

from config import GENRE_DIR, CHART_DIR, GENRE_LIST

import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

def write_fft(fft_features, fn):
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)
    print "Written", data_fn

def create_fft(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    fft_features = abs(scipy.fft(X)[:1000])
    #print fft_features
    write_fft(fft_features, fn)

def read_fft(genre_list, base_dir=GENRE_DIR):
    X = []
    y = []
    for label, genre in enumerate(genre_list):
        genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
        file_list = glob.glob(genre_dir)
        ##assert(file_list), genre_dir
        for fn in file_list:
            fft_features = np.load(fn)
            X.append(fft_features[:2000])
            y.append(label)

    return np.array(X), np.array(y)

def create_fft_test(fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    #print "fn=",fn
    #print "X.shape=",X.shape
    #print "sample_rate=",sample_rate
    Y = np.transpose(X)
    #print "Y.shape=",Y.shape
    #print Y
    
    fft_features = abs(scipy.fft(Y)[:1000])
    #print "fft.shape=",fft_features.shape
    base_fn, ext = os.path.splitext(fn)
    data_fn = base_fn + ".fft"
    np.save(data_fn, fft_features)
    print "Written", data_fn
    #print fft_features
    data_fn += ".npy"
    return data_fn
    
def read_fft_test(test_file):
    X = []
    y = []
    fft_features = np.load(test_file)[0]
    X.append(fft_features[:1000])
    #print X
    #y.append(label)
    #print "fft_features read = ", fft_features
    #print "fft_features read shape = ", fft_features.shape
    #print "X read = ", X
    #print "X read shape = ", X[0].shape
    return np.array(X), np.array(y)

def plot_wav_fft(wav_filename, desc=None):
    plt.clf()
    plt.figure(num=None, figsize=(6, 4))
    sample_rate, X = scipy.io.wavfile.read(wav_filename)
    spectrum = np.fft.fft(X)
    freq = np.fft.fftfreq(len(X), 1.0 / sample_rate)

    plt.subplot(211)
    num_samples = 200.0
    plt.xlim(0, num_samples / sample_rate)
    plt.xlabel("time [s]")
    plt.title(desc or wav_filename)
    plt.plot(np.arange(num_samples) / sample_rate, X[:num_samples])
    plt.grid(True)

    plt.subplot(212)
    plt.xlim(0, 5000)
    plt.xlabel("frequency [Hz]")
    plt.xticks(np.arange(5) * 1000)
    if desc:
        desc = desc.strip()
        fft_desc = desc[0].lower() + desc[1:]
    else:
        fft_desc = wav_filename
    plt.title("FFT of %s" % fft_desc)
    plt.plot(freq, abs(spectrum), linewidth=5)
    plt.grid(True)

    plt.tight_layout()

    rel_filename = os.path.split(wav_filename)[1]
    plt.savefig("%s_wav_fft.png" % os.path.splitext(rel_filename)[0],
                bbox_inches='tight')

    plt.show()

def plot_wav_fft_demo():
    plot_wav_fft("sine_a.wav", "400Hz sine wave")
    plot_wav_fft("sine_b.wav", "3,000Hz sine wave")
    plot_wav_fft("sine_mix.wav", "Mixed sine wave")

def plot_specgram(ax, fn):
    sample_rate, X = scipy.io.wavfile.read(fn)
    ax.specgram(X, Fs=sample_rate, xextent=(0, 30))

def plot_specgrams(base_dir=CHART_DIR):
    """
    Plot a bunch of spectrograms of wav files in different genres
    """
    plt.clf()
    genres = GENRE_LIST
    num_files = 5
    f, axes = plt.subplots(len(genres), num_files)

    for genre_idx, genre in enumerate(genres):
        for idx, fn in enumerate(glob.glob(os.path.join(GENRE_DIR, genre, "*.wav"))):
            if idx == num_files:
                break
            axis = axes[genre_idx, idx]
            axis.yaxis.set_major_formatter(EngFormatter())
            axis.set_title("%s song %i" % (genre, idx + 1))
            plot_specgram(axis, fn)

    #specgram_file = os.path.join(base_dir, "Spectrogram_Genres.png")
    #plt.savefig(specgram_file, bbox_inches="tight")

    plt.show()


if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    #traversed = ".".join(x for x in traverse)
    print "Working with these genres --> ", traverse
    
    t=set()
    for subdir, dirs, files in os.walk(GENRE_DIR):
        #print subdir
        for file in files:
            path = subdir+'/'+file
            if path.endswith("wav"):
                tmp = subdir[subdir.rfind('/',0)+1:]
                if tmp in traverse:
                    pass#create_fft(path)

    stop = timeit.default_timer()

    print "Total FFT generation and feature writing time (s) = ", (stop - start) 

    print "Plotting spectrograms"

    plot_specgrams()
