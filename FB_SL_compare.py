# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:30:24 2024

@author: Linne
"""
from SLclass import SL
from FBclass import FB

semi = SL(1e6, 25e3, 160, 540*40)
semi.run()
semi.plot_uva()

FBtime = FB(1e6, 25e3, 160, 540*40)
FBtime.run()
FBtime.plot_uva()

#%%
from Analytical import plot_analytical

plot_analytical(1e6, 25e3)