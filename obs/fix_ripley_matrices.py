import csv
import os
import glob
import time
import numpy as np
from matplotlib import pyplot as plt

ROOT = '/home/rjonnal/data/Dropbox/Private/figures/src/opsins_voronoi/'

flist = glob.glob(os.path.join(ROOT,'ripley_matrices_s/*/*mm*.csv'))

for f in flist:
    mat_mm = np.loadtxt(f,delimiter=',')
    print mat_mm
    

