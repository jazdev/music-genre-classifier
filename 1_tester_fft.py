import numpy as np
from collections import defaultdict

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.cross_validation import ShuffleSplit
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

from config import GENRE_DIR, GENRE_LIST, TEST_DIR

from utils import plot_confusion_matrix, plot_pr, plot_roc

from fft import read_fft, create_fft_test, read_fft_test

genre_list = GENRE_LIST

import os
from pydub import AudioSegment
import timeit


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#          Please run the tester script first
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def test_model_on_single_file(file_path):
    """
        Loads saved model and tests it with the provided test datafile.
        Converts the datafile to WAV format, if required.
    """
##    if not path.endswith("wav"):
##        file_path = convert_any_to_wav(file_path)

    # load saved model from disk
    clf = joblib.load('saved_model_fft/my_model.pkl')

    X, y = read_fft_test(create_fft_test(file_path))
    probs = clf.predict_proba(X)
    probs=probs[0]
    max_prob = max(probs)
    for i,j in enumerate(probs):
        if probs[i] == max_prob:
            max_prob_index=i
    predicted_genre = traverse[max_prob_index]
    print "\n\npredicted genre = ",predicted_genre
    return predicted_genre

if __name__ == "__main__":
    global traverse
    for subdir, dirs, files in os.walk(GENRE_DIR):
        traverse = list(set(dirs).intersection( set(GENRE_LIST) ))
        break
    test_file = "/home/jaz/Desktop/MAJOR_PROJECT/genres_test_set/Firework.wav"
    print "Testimg model on file: ", test_file
    predicted_genre = test_model_on_single_file(test_file)            
