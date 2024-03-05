# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:29:48 2024

@author: Linne
"""
import numpy as np
from Params import *
from numba import jit
from functions import *

def ddx(arr, d):
    f_x = (arr[:, 1:] - arr[:, :-1])/d
    return f_x

def ddy(arr, d):
    f_y = (arr[1:, :] - arr[:-1, :])/d
    return f_y

# @jit(nopython=True)
def FB_time(eta, u, v, y_u, y_v, taux, tauy, dt, d, f_v, f_u, energy = True):
    #moves two timesteps
    eta -=  H*dt*(ddx(u, d)+ddy(v, d))
    eta_x, eta_y = ddx(eta, d), ddy(eta, d)
    u[:, 1:-1] += dt*(crop_x(f_u)*vu_interp(v) - g*eta_x - gamma*crop_x(u)\
                      + crop_x(taux)/(rho*H))
    v[1:-1, :] += dt*(-crop_y(f_v)*vu_interp(u) - g*eta_y - gamma*crop_y(v)\
                      + tauy/(rho*H))
    if energy:
        E1 = calc_energy(u, v, eta, d)
    
    eta -= H*dt*(ddx(u, d)+ddy(v, d))
    eta_x, eta_y = ddx(eta, d), ddy(eta, d)
    v[1:-1, :] += dt*(-crop_y(f_v)*vu_interp(u) - g*eta_y - gamma*crop_y(v)\
                      + tauy/(rho*H))    
    u[:, 1:-1] +=+ dt*(crop_x(f_u)*vu_interp(v) - g*eta_x - gamma*crop_x(u)\
                       + crop_x(taux)/(rho*H))
    if energy:
        E2 = calc_energy(u, v, eta, d)
        return eta, u, v, E1, E2
    else:
        return eta, u, v


    