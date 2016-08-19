# Attempt to detect squiggles by tracing maxima

# for matplotlib python-2 compatibility
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
import six

import os
import numpy as np
import scipy.misc
import scipy.fftpack
import matplotlib.pyplot as plt

# Global "constants"
DATA_DIR = r"data"
#FILE = r"data\2014-09-22_00-13-36_UTC.act34763.dx1021.id-1.L.png"
#FILE = r"data\2014-09-02_06-32-05_UTC.act30008.dx1004.id-12.L.png"
#FILE = r"data\2015-10-07_23-55-04_UTC.act40960.dx1017.id-17.L.archive-compamp.png"

W = 6
H = 3



if __name__ == "__main__":

    for name in os.listdir(DATA_DIR):
        if name.endswith(".png"):
            
            file_path = os.path.join(DATA_DIR, name)
            img = scipy.misc.imread(file_path)
            nrow, ncol = img.shape
            
            # compute the maxima for all rows
            maxima = np.argmax(img, axis=1)
            
            plt.figure(figsize=(W,H))
            plt.plot(maxima, "b.")
            plt.xlabel("$t$ / s")
            plt.ylabel("$f_{max}$ / arb. unit")
            plt.tight_layout()
            plt.show()
            
            # FT the above plot
            ft = scipy.fftpack.fft(maxima - np.mean(maxima))
            ft_power = np.absolute(ft) **2
            ft_power /= np.max(ft_power)
            
            plt.figure(figsize=(W,H))
            plt.plot(ft_power[:nrow/2], "b-")
            plt.xlabel("$wave number$ / arb. unit")
            plt.ylabel("$intensity$ / arb. unit")
            plt.tight_layout()
            plt.show()