# Attempt to detect squiggles by summing the y axis

import os
import numpy as np
import scipy.misc
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Global "constants"
fn = r"data\1.png"

W = 6
H = 3

WIDTH = 15
THRESHOLD = 1.0027

TRY_WIDTH = np.arange(1, 30)
TRY_THRESHOLD = np.arange(1.0, 1.01, 0.001)

def cor(): # arr as returned by the function above
    arr = scipy.misc.imread(fn)
    # sum each column, and perform autocorrelation
    sum_y = np.sum(arr, axis=0)
    padding = np.concatenate((sum_y, sum_y))
    cor = np.correlate(sum_y, padding, mode='valid')
    
    plt.figure(figsize=(W,H))
    plt.plot(range(len(cor)), cor, "b-")
    plt.xlabel("$w$")
    plt.ylabel("$cor[I(f)](w)$")
    plt.tight_layout()
    plt.show()
    


        

if __name__ == "__main__":
    cor()