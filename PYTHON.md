# Reading-LAS
My first repository
# Standard library import

import sys

# External library import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Loading logs without header

exe_log = np.genfromtxt('Input\exercise_logs.txt', skip_header=1, dtype=np.float32)
#print(exe_log)

# Allocate different columns to different variables (logs)
depth = exe_log[:, 0]
bs = exe_log[:, 1]
cs = exe_log[:, 2]
tens = exe_log[:, 3]
dtco = exe_log[:, 4]
dtsm = exe_log[:, 5]
gr = exe_log[:, 6]
ecgr = exe_log[:, 7]
nphi = exe_log[:, 8]
npl = exe_log[:, 9]
tnph = exe_log[:, 10]
gdev = exe_log[:, 11]
ah10 = exe_log[:, 12]
ah20 = exe_log[:, 13]
ah30 = exe_log[:, 14]
ah60 = exe_log[:, 15]
ah90 = exe_log[:, 16]
ahtca = exe_log[:, 17]
rxoi = exe_log[:, 18]
hcal = exe_log[:, 19]
hdra = exe_log[:, 20]
pefz = exe_log[:, 21]
rhoz = exe_log[:, 22]
hmin = exe_log[:, 23]
hmno = exe_log[:, 24]
rxoz = exe_log[:, 25]
hgr = exe_log[:, 26]

# Data samples and range (necessary for loops)

samples = len(exe_log)
interval = depth[samples - 1] - depth[0]
step = depth[2] - depth[1]
top_interval = depth[0]
base_interval = depth[samples - 1]

# Depth variable verification

print('depth:', depth)

depth_range = np.max(depth) - np.min(depth)

print('samples     =', len(depth))
print('range       =', depth_range)
print('min         =', np.min(depth))
print('p33         =', np.percentile(depth,33))
print('mean        =', np.mean(depth))
print('p66         =', np.percentile(depth,66))
print('max         =', np.max(depth))
print('std         =', np.std(depth))
print('mean + 3std =', (np.mean(depth) + 3*np.std(depth)))

# GR variable verification
print()
print('gr:', gr)

gr_range = np.max(gr) - np.min(gr)

print('samples     =', len(gr))
print('range       =', gr_range)
print('min         =', np.min(gr))
print('p33         =', np.percentile(gr,33))
print('mean        =', np.mean(gr))
print('p66         =', np.percentile(gr,66))
print('max         =', np.max(gr))
print('std         =', np.std(gr))
print('mean + 3std =', (np.mean(gr) + 3*np.std(gr)))

# NPHI variable verification
print()
print('nphi:', nphi)

nphi_range = np.max(nphi) - np.min(nphi)

print('samples     =', len(nphi))
print('range       =', nphi_range)
print('min         =', np.min(nphi))
print('p33         =', np.percentile(gr,nphi))
print('mean        =', np.mean(nphi))
print('p66         =', np.percentile(nphi,66))
print('max         =', np.max(nphi))
print('std         =', np.std(nphi))
print('mean + 3std =', (np.mean(nphi) + 3*np.std(nphi)))

# DTCO variable verification
print()
print('dtco:', dtco)

dtco_range = np.max(dtco) - np.min(dtco)

print('samples     =', len(dtco))
print('range       =', dtco_range)
print('min         =', np.min(dtco))
print('p33         =', np.percentile(dtco,33))
print('mean        =', np.mean(dtco))
print('p66         =', np.percentile(dtco,66))
print('max         =', np.max(dtco))
print('std         =', np.std(dtco))
print('mean + 3std =', (np.mean(dtco) + 3*np.std(dtco)))

# NPL variable verification
print()
print('npl:', npl)

npl_range = np.max(npl) - np.min(npl)

print('samples     =', len(npl))
print('range       =', npl_range)
print('min         =', np.nanmin(npl))
print('p33         =', np.percentile(npl,33))
print('mean        =', np.mean(npl))
print('p66         =', np.percentile(npl,66))
print('max         =', np.max(npl))
print('std         =', np.std(npl))
print('mean + 3std =', (np.mean(npl) + 3*np.std(npl)))

# RXO variable verification
print()
print('rxoz:', rxoz)

rxoz_range = np.max(rxoz) - np.min(rxoz)

print('samples     =', len(rxoz))
print('range       =', rxoz_range)
print('min         =', np.nanmin(rxoz))
print('p33         =', np.percentile(rxoz,33))
print('mean        =', np.mean(rxoz))
print('p66         =', np.percentile(rxoz,66))
print('max         =', np.max(rxoz))
print('std         =', np.std(rxoz))
print('mean + 3std =', (np.mean(rxoz) + 3*np.std(rxoz)))

# Single plot of gamma ray

plt.figure(2, figsize=(5,20))
plt.plot(gr, depth)
plt.ylim(13310, 7672)
plt.xlabel("GR API")
plt.ylabel("Depth (ft)")

# Multiplot of the input logs 
top_display = 12400
base_display = 13200

plt.figure(2, figsize=(13, 8))

# Track 1
plt.subplot(1, 6, 1)
plt.plot(bs, depth)
plt.axis([7, 15, base_display, top_display])
plt.plot(bs, depth, 'r')
plt.title('BIT SIZE (inch)')
plt.grid(True)

# Track 2
plt.subplot(1, 6, 2)
plt.plot(gr, depth)
plt.axis([0, 200, base_display, top_display])
plt.title('GR (api)')
plt.plot(gr, depth, 'b')
plt.grid(True)

# Track 3
plt.subplot(1, 6, 3)
plt.semilogx(hmin, depth)
plt.axis([0.2, 2000, base_display, top_display])
plt.plot(hmin, depth, 'k')
plt.title('RES (ohmm)')
plt.grid(True)

# Track 4
plt.subplot(1, 6, 4)
plt.plot(nphi, depth)
plt.axis([1.2, 0.15, base_display, top_display])
plt.plot(nphi, depth, 'c')
plt.title('NPHI (v/V)')
plt.grid(True)

# Track 5
plt.subplot(1, 6, 5)
plt.plot(dtco, depth)
plt.axis([140, 50, base_display, top_display])
plt.title('DT (US/ft)')
plt.grid(True)

# Track 6
plt.subplot(1, 6, 6)
plt.plot(rhoz, depth)
plt.axis([1.9, 2.9, base_display, top_display])
plt.plot(rhoz, depth, 'r')
plt.title('DEN (gr/cm3)')
plt.grid(True)

# plt.tight_layout(1)
# plt.show()
