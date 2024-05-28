# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 01:24:24 2024

@author: Linne
"""

from FBclass import FB
from Params import *
from functions import calc_energy
test = FB(1e6, 25e3, 160, int(40*day/160))
test.run()
energy1 = test.calcE()

interpu = (test.u[:, 1:] + test.u[:, :-1])/2
interpv = (test.v[1:, :] + test.v[:-1, :])/2

energy2 = calc_energy(interpu, interpv, test.eta, 25e3)
energy1 = calc_energy(test.u, test.v, test.eta, 25e3)



