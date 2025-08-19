from __future__ import division, print_function
import os
import matplotlib.pyplot as plt
import pyzdde.zdde as pyz
import scipy.io
import numpy as np
from tqdm import tqdm

l1 = pyz.createLink()
print(l1.zGetPath())

zfile = os.path.join(l1.zGetPath()[1],'grid_8_10.zmx')
# zfile = l1.zGetPath()[1]

l1.zLoadFile(zfile)

PSF = np.zeros((80,64,64))

for wave in range(7):
    for i in tqdm(range(1,81)):
    #  l1.zSetSurfaceParameter(2,25,i*1.0114)
        cfgfile = l1.zSetHuygensPSFSettings(imgDelta=12,pupilSample=3,imgSample=2,wave=wave+1,field=i+1)
        psfdata = l1.zGetPSF(which='huygens',settingsFile = cfgfile)
        psfdata = np.array(psfdata[1])
        PSF[i-1,:,:] = PSF[i-1,:,:] + psfdata * 0.2

scipy.io.savemat('8_10.mat', mdict={'psfdata': PSF})