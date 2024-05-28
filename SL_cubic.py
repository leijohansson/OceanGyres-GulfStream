# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:30:24 2024

@author: Linne
"""
from SLclass import SL
import numpy as np
if __name__ == '__main__':
    day = 24*60*60
    ndays = 40
    N = int(day*ndays/160)+1
    semi = SL(1e6, 25e3, 160, N, method = 'cubic')
    semi.run()
    
    np.save('SL_cubic_u', semi.u)
    np.save('SL_cubic_v', semi.v)
    np.save('SL_cubic_eta', semi.eta)
