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
ndays = 40
N = int(day*ndays/160)+1
# N=5
# semi = SL(1e6, 25e3, 160, N, method = 'cubic')
semi.run()
semi.plot_uva()
plt.savefig('SL_cubic_SS.pdf', bbox_inches = 'tight')
print(semi.calc_Ediff())
print(time.time()-start)
semi.plot_solution_diff()
plt.savefig('SL_cubic_SS_Diffs.pdf', bbox_inches = 'tight')

# FBtime = FB(1e6, 25e3, 160, N)
# FBtime.run()


#%%
FBtime.plot_uva()
print(time.time()-start)

#%%
from Analytical import plot_analytical
plot_analytical(1e6, 25e3)
