# -*- coding: utf-8 -*-
"""
Parameters kept constant across all models
Created on Thu Feb 22 13:35:38 2024

@author: Linne
"""
# import numpy as np
f0 = 1e-4 #s^-1 #coriolis parameter
beta =  1e-11 #m^-1s^-1 #beta for beta plane approximation
g = 10 #ms^-2 #gravitational accerlation
gamma = 1e-6 #s^-1 #linear drag coefficient
rho = 1000 #kgm^-3 #uniform density
H = 1000 #m #resting depth of fluid
tau0 = 0.2 #Nm^-2 #constant for wind stress
day = 24*60*60 #s #one day in seconds