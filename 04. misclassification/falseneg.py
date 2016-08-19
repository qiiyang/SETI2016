# Attempt to detect squiggles by summing the y axis

import os
import numpy as np
import scipy.misc
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Global "constants"
TRAIN_DIR = r"pos_training"
VALIDATION_DIR = r"pos_validation"

W = 6
H = 3

WIDTH = 15
THRESHOLD = 1.0027

TRY_WIDTH = np.arange(1, 30)
TRY_THRESHOLD = np.arange(1.0, 1.01, 0.001)

def has_vertical_band(arr, width, threshold): # arr as returned by the function above
    # sum each column, and perform autocorrelation
    sum_y = np.sum(arr, axis=0)
    padding = np.concatenate((sum_y, sum_y))
    cor = np.correlate(sum_y, padding, mode='valid')
    
    # true if band width > 10 pixels
    cor_threshold = threshold * np.min(cor)
    return (cor[width] > cor_threshold)
    

# Calculate the false negative rates as a function of width
def train_width():
    total = 0
    positives = np.zeros_like(TRY_WIDTH, dtype=np.float_)
    for name in os.listdir(TRAIN_DIR):
        if name.endswith(".png"):
            total += 1
            file_path = os.path.join(TRAIN_DIR, name)
            img = scipy.misc.imread(file_path)
            for i in range(len(TRY_WIDTH)):
                if has_vertical_band(img, TRY_WIDTH[i], THRESHOLD):
                    positives[i] += 1
        
    false_negs = 1. - positives / total
    print("total = {}".format(total))
    plt.figure(figsize=(W,H))
    plt.plot(TRY_WIDTH, false_negs, "go")
    plt.xlabel("widths")
    plt.ylabel("false negative rates")
    plt.title("False Negative Against Autocorrelation Width")
    plt.tight_layout()
    plt.show()
            
# Calculate the false negative rates as a function of threshold
def train_threshold():
    total = 0
    positives = np.zeros_like(TRY_THRESHOLD, dtype=np.float_)
    for name in os.listdir(TRAIN_DIR):
        if name.endswith(".png"):
            total += 1
            file_path = os.path.join(TRAIN_DIR, name)
            img = scipy.misc.imread(file_path)
            for i in range(len(TRY_THRESHOLD)):
                if has_vertical_band(img, WIDTH, TRY_THRESHOLD[i]):
                    positives[i] += 1
            
    false_negs = 1. - positives / total
    print("total = {}".format(total))
    plt.figure(figsize=(W,H))
    plt.plot(TRY_THRESHOLD, false_negs, "go")
    plt.xlabel("thresholds")
    plt.ylabel("false negative rates")
    plt.title("False Negative Against Threshold")
    plt.tight_layout()
    plt.show()

def exhaustion():
    total = 0
    x, y = np.meshgrid(TRY_WIDTH, TRY_THRESHOLD, indexing='ij')
    positives = np.zeros_like(x, dtype=np.float_)
    for name in os.listdir(TRAIN_DIR):
        if name.endswith(".png"):
            total += 1
            file_path = os.path.join(TRAIN_DIR, name)
            img = scipy.misc.imread(file_path)
            for i in range(len(TRY_WIDTH)):
                for j in range(len(TRY_THRESHOLD)):
                    if has_vertical_band(img, TRY_WIDTH[i], TRY_THRESHOLD[j]):
                        positives[i,j] += 1    
    false_negs = 1. - positives / total
    print("total = {}".format(total))
    
    fig = plt.figure(figsize=(W,H))
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, false_negs, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel("widths")
    ax.set_ylabel("thresholds")
    ax.set_zlabel("false negative rates")
    plt.title("False Negative Rates")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.show()
        

if __name__ == "__main__":
    #train_width()
    #train_threshold()
    exhaustion()