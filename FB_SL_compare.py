# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:30:24 2024

@author: Linne
"""
from SLclass import SL
from FBclass import FB
import time 
import matplotlib.pyplot as plt

start = time.time()
day = 24*60*60
ndays = 80
N = int(day*ndays/160)+1
# semi = SL(1e6, 25e3, 160, N)
# semi.run()
# semi.plot_uva()

FBtime = FB(1e6, 25e3, 160, N)
FBtime.run()
#%%
FBtime.plot_uva()
print(time.time()-start)

#%%
from Analytical import plot_analytical
plot_analytical(1e6, 25e3)
