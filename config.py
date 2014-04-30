import os
import sys

from matplotlib import pylab
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")

CHART_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "charts")

for d in [DATA_DIR, CHART_DIR]:
    if not os.path.exists(d):
        os.mkdir(d)

# Directory where the music dataset is located (GTZAN dataset)
GENRE_DIR = "/home/jaz/Desktop/genre-project/genres_dataset"

# Directory where the test music is located
TEST_DIR = "/home/jaz/Desktop/genre-project/genres_test_set"

#Working with these genres
#GENRE_LIST = [ "blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
GENRE_LIST = [ "blues","jazz","metal","pop","rock"]

if GENRE_DIR is None or TEST_DIR is None:
    print("Please set GENRE_DIR and TEST_DIR in config.py") 
    sys.exit(1)

def any_to_wav(file_name):
    pass

def plot_confusion_matrix(cm, genre_list, name, title):
    pylab.clf()
    pylab.matshow(cm, fignum=False, cmap='Blues', vmin=0, vmax=1.0)
    ax = pylab.axes()
    ax.set_xticks(range(len(genre_list)))
    ax.set_xticklabels(genre_list)
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks(range(len(genre_list)))
    ax.set_yticklabels(genre_list)
    pylab.title(title)
    pylab.colorbar()
    pylab.grid(False)
    
    pylab.xlabel('Predicted class', fontsize = 20)
    pylab.ylabel('True class', fontsize = 20)
    pylab.grid(False)
    pylab.show()
    #pylab.savefig(os.path.join(CHART_DIR, "confusion_matrix_%s.png" % name), bbox_inches="tight")

def plot_pr(auc_score, name, precision, recall, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.fill_between(recall, precision, alpha=0.5)
    pylab.plot(recall, precision, lw=1)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('Recall')
    pylab.ylabel('Precision')
    pylab.title('P/R curve (AUC = %0.2f) / %s' % (auc_score, label))
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "pr_" + filename + ".png"), bbox_inches="tight")


def plot_roc(auc_score, name, tpr, fpr, label=None):
    pylab.clf()
    pylab.figure(num=None, figsize=(5, 4))
    pylab.grid(True)
    pylab.plot([0, 1], [0, 1], 'k--')
    pylab.plot(fpr, tpr)
    pylab.fill_between(fpr, tpr, alpha=0.5)
    pylab.xlim([0.0, 1.0])
    pylab.ylim([0.0, 1.0])
    pylab.xlabel('False Positive Rate')
    pylab.ylabel('True Positive Rate')
    pylab.title('ROC curve (AUC = %0.2f) / %s' %
                (auc_score, label), verticalalignment="bottom")
    pylab.legend(loc="lower right")
    filename = name.replace(" ", "_")
    pylab.savefig(
        os.path.join(CHART_DIR, "roc_" + filename + ".png"), bbox_inches="tight")


def show_most_informative_features(vectorizer, clf, n=20):
    c_f = sorted(zip(clf.coef_[0], vectorizer.get_feature_names()))
    top = zip(c_f[:n], c_f[:-(n + 1):-1])
    for (c1, f1), (c2, f2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1, f1, c2, f2)



