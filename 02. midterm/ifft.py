# Attempt to detect squiggles by fft

# for matplotlib python-2 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import six

import os
import numpy as np
import scipy.misc
import scipy.fftpack as fft
import matplotlib.pyplot as plt

# Global "constants"
DATA_DIR = r"data"

W = 6
H = 3

if __name__ == "__main__":

    for name in os.listdir(DATA_DIR):
        if name.endswith(".png"):
            
            file_path = os.path.join(DATA_DIR, name)
            img = scipy.misc.imread(file_path)
            nrow, ncol = img.shape
            
            # Reconstruct the compamp
            compamp = np.empty(2 * nrow * ncol)
            t = np.arange(2 * nrow * ncol) / ncol / 2
            f = np.arange(nrow*ncol-1) / nrow
            row_comp = np.zeros(ncol * 2)
            for i in range(nrow):
                row_comp[::2] = np.sqrt(img[i])
                row_comp[::2] -= np.mean(row_comp[::2])
                compamp[2*i*ncol: 2*(i+1)*ncol] = fft.irfft(row_comp)
            ft = fft.rfft(compamp)
            ft_square = np.square(ft[1:-1:2]) + np.square(ft[2::2])
            ft_square /= np.max(ft_square)
            """
            plt.figure(figsize=(W,H))
            plt.xlabel("$t$ / s")
            plt.ylabel("$V$ / arb. unit")
            plt.plot(t, compamp, "b-")
            plt.tight_layout()
            plt.show()
            #"""
            plt.figure(figsize=(W,H))
            plt.xlabel("$f - f_0$ / Hz")
            plt.ylabel("$Intensity$ / arb. unit")
            plt.plot(f, ft_square, "b-")
            plt.tight_layout()
            plt.show()