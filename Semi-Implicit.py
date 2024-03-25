# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:46:43 2024

@author: Linne
"""

import numpy as np
from model import model
from functions import *
from Params import *
from FBclass import FB

dt = 150
# day = 24*60**2
# ndays = 1
nt = 10000
grid = FB(1e6, 25e3, dt, nt)
n = int(grid.L/grid.d)
#whether or not to use FB for first timestep
FB_onestep = True

dt = grid.dt
d = grid.d

ngrids = n*n
L = np.zeros((ngrids,ngrids))
mu = (dt/d)**2*g*H

#one above the diagonal
one_above_1d = [-mu]*(n-1)
one_above_1d.append(0) #next column point doesnt exist for last row

one_above = np.diag(np.array(one_above_1d*n)[:-1], k = 1)
#one below the diagonal
one_below = one_above.T

L += one_above+one_below
right = np.diag(np.array([-mu]*(n*(n-1))), k=n)
left = right.T
L += right+left

#each row needs to sum to one
diag_vals = 1-np.sum(L, axis = 1)
diag = np.diag(diag_vals, k=0)
L += diag
L_inv = np.linalg.inv(L)

def midpoints(arr1d, var):
    ave = (arr1d[1:] + arr1d[:-1])/2
    if var == 'v':
        return ave[:, None]
    if var == 'u':
        return ave[None, :]
def flat(arr):
    '''
    flattens array into 1D array like stacking columns on top of each other

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    return arr.T.flatten()
def unflat(arr, n):
    return arr.reshape(n, n).T

def calcABF(u, v, u_old, v_old, eta_old, d, dt):
    #setting up v on u grid, assume v beyond x boundaries is equal to v at the boundary
    v_on_u = vu_interp(v)
    v_left, v_right = v[:, 0], v[:, -1]                                 
    v_on_u = np.concatenate((midpoints(v_left, 'v'), v_on_u, midpoints(v_right, 'v')), axis = 1)

    #setting up deta/dx on u grid
    eta_ext = np.pad(eta_old, ((0,0),(1,1)), mode = 'edge')
    deta_dx = ddx(eta_ext, d)
    
    #calculating A, dAdx
    A = -dt*g*deta_dx + 2*dt*(f0*v_on_u - gamma*u + taux/(rho*H))
    dAdx = ddx(A, d)
    
    #setting up u on v grid, assume u beyond y boundaries is equal to u at the boundary
    u_on_v = vu_interp(u)
    u_top, u_bottom = u[0, :], u[-1, :]                                 
    u_on_v = np.concatenate((midpoints(u_top, 'u'), u_on_v, midpoints(u_bottom, 'u')), axis = 0)

    #setting up deta/dx on u grid
    eta_ext = np.pad(eta_old, ((1,1),(0,0)), mode = 'edge')
    deta_dy = ddy(eta_ext, grid.d)

    #calculating B, dBdy
    B = -dt*g*deta_dy + 2*dt*(-f0*u_on_v - gamma*v + tauy/(rho*H))
    dBdy = ddy(B, d)
    
    #calculating C using u, v at time n-1
    C = eta + dt*H*(ddx(u_old, d) + ddy(v_old, d))

    F = C - dt*H*(dAdx + dBdy)
    
    return A, B, F


taux = tau0*-np.cos(np.pi*grid.y_u/grid.L)
tauy = 0

if FB_onestep:
    u_old, v_old, eta_old = grid.u, grid.v, grid.eta
    grid.FB_time_one()
    u, v, eta = grid.u, grid.v, grid.eta    

else:
    u, v, eta = grid.u, grid.v, grid.eta    
    # first timestep, assume -1 values are the same as initial values
    A, B, F = calcABF(u, v, u, v, eta, d, dt)
    F_1d = flat(F)
    
    #saving n-1 values for next timestep
    u_old, v_old, eta_old = u, v, eta
    
    #updating eta
    eta = unflat(np.matmul(L_inv, F_1d), n)
    #updating u, v
    u[:, 1:-1] = crop_x(A) - dt*g*ddx(eta, d)
    v[1:-1, :] = crop_y(B) - dt*g*ddy(eta, d)


for i in range(nt):
    A, B, F = calcABF(u, v, u_old, v_old, eta_old, d, dt)
    F_1d = flat(F)
    u_old, v_old, eta_old = u, v, eta
    #updating eta
    eta = unflat(np.matmul(L_inv, F_1d), n)
    #updating u, v
    u[:, 1:-1] = crop_x(A) - dt*g*ddx(eta, d)
    v[1:-1, :] = crop_y(B) - dt*g*ddy(eta, d)

grid.u, grid.v, grid.eta = u, v, eta
grid.plot_uva()

