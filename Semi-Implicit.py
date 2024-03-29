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
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def make_L(n, dt, d):
    #calculating mu term (d in the notes)
    ngrids = n**2
    mu = (dt/d)**2*g*H

    #setting up L grid
    L = np.zeros((ngrids,ngrids))

    #one above the diagonal 
    #same column, row after 
    #(i, j+1) contribution
    one_above_1d = [-mu]*(n-1)
    #next column point doesnt exist for last row
    one_above_1d.append(0)
    one_above = np.diag(np.array(one_above_1d*n)[:-1], k = 1)
    
    #one below the diagonal 
    #same column, row before 
    #(i, j-1) contribution
    one_below = one_above.T #symmetrical about the diagonal to one_above


    #adding j-1, j+1 contributions to L
    L += one_above+one_below
    
    #n to the right of the diagonal
    #same row, next column 
    #(i+1, j) contribution
    right = np.diag(np.array([-mu]*(n*(n-1))), k=n)
    
    #n to the left of the diagonal
    #same row, previous column 
    #(i-1, j) contribution
    left = right.T
    
    #adding i-1, i+1 contributions to L
    L += right+left

    #calculating term on the diagonal
    #each row needs to sum to one
    diag_vals = 1-np.sum(L, axis = 1)
    diag = np.diag(diag_vals, k=0)
    
    #adding diagonal term to L
    L += diag
    
    #finding inverse of L
    L_inv = np.linalg.inv(L)
    
    return L_inv



def midpoints(arr1d, var):
    '''
    Finding midpoints of 1D array for interpolation of u onto v grid and vice
    versa (on the boundary)

    Parameters
    ----------
    arr1d : arr
        1d array for which to find the midpoints.
    var : 'u' or 'v'
        whether u or v is being interpolated (sets shape of returned array).

    Returns
    -------
    array
        midpoints of input array.

    '''
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
    arr : array
        array to flatten.

    Returns
    -------
    1d array
        flattened array.

    '''
    #transpose so that it flattens like stacked columns
    return arr.T.flatten()



def unflat(arr, n):
    '''
    Makes 1D array into n by n array

    Parameters
    ----------
    arr : 1d array
        array to unflatten.
    n : int
        n where shape of returned array is nxn.

    Returns
    -------
    n x n array
        unflattened array.

    '''
    return arr.reshape(n, n).T



def calcABF(u, v, u_old, v_old, eta_old, d, dt, f_u, f_v, gravity_wave = False):
    '''
    Calculates A, B and F

    Parameters
    ----------
    u : 2D array
        u on u grid at current timestep.
    v : 2D array
        v on v grid at current timestep.
    u_old : 2D array
        u on u grid at previous timestep.
    v_old : 2D array
        v on v grid at previous timestep.
    eta_old : 2D  array
        eta on eta grid at previous timestep.
    d : float
        grid spacing.
    dt : float
        time step.
    f_u : array
        coriolis force on u grid
    f_v : array
        coriolis force on u grid 
    gravity_wave : bool, optional
        Whether or not to turn on forcings. The default is False.

    Returns
    -------
    A : 
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.
    F : TYPE
        DESCRIPTION.

    '''

    #setting up v on u grid
    v_on_u = vu_interp(v)
    #assume v beyond x boundaries is equal to v at the boundary
    v_left, v_right = v[:, 0], v[:, -1]                                 
    v_on_u = np.concatenate((midpoints(v_left, 'v'), v_on_u, 
                             midpoints(v_right, 'v')), axis = 1)

    #setting up deta/dx on u grid
    eta_ext = np.pad(eta_old, ((0,0),(1,1)), mode = 'edge')
    deta_dx = ddx(eta_ext, d)
    
    #calculating A, dAdx
    A = -dt*g*deta_dx 
    if not gravity_wave:
        A += 2*dt*(f_u*v_on_u - gamma*u + taux/(rho*H))
    dAdx = ddx(A, d)
    
    #setting up u on v grid
    u_on_v = vu_interp(u)
    #assume u beyond y boundaries is equal to u at the boundary
    u_top, u_bottom = u[0, :], u[-1, :]                                 
    u_on_v = np.concatenate((midpoints(u_top, 'u'), u_on_v, 
                             midpoints(u_bottom, 'u')), axis = 0)

    #setting up deta/dx on u grid
    eta_ext = np.pad(eta_old, ((1,1),(0,0)), mode = 'edge')
    deta_dy = ddy(eta_ext, grid.d)

    #calculating B, dBdy
    B = -dt*g*deta_dy 
    if not gravity_wave:
        B += 2*dt*(-f_v*u_on_v - gamma*v + tauy/(rho*H))
    dBdy = ddy(B, d)
    
    #calculating C using u, v at time n-1
    C = eta + dt*H*(ddx(u_old, d) + ddy(v_old, d))

    F = C - dt*H*(dAdx + dBdy)
    
    return A, B, F


def blobby(grid):
    '''
    Updates eta field to a multivariate gaussian (for gravity wave)

    Parameters
    ----------
    grid : model object
        model object for which to change eta.

    Returns
    -------
    None.

    '''
    var = ((1*grid.d)**2*np.array([2,2])**2)
    height = 100
    pos = np.empty(grid.x_eta.shape + (2,))
    pos[..., 0] = grid.x_eta
    pos[..., 1] = grid.y_eta
    pdf = multivariate_normal(np.array([grid.L/2, grid.L/2]),
                              [[var[0], 0], [0, var[1]]]).pdf(pos)
    grid.eta = pdf * (height/pdf.max())

#just a gravity wave without any forcings/coriolis if true
gravity_wave = True
dt = 5000
d = 20e3
nt = 10000
grid = model(1e6, d, dt, nt)
if gravity_wave:
    #set up initial condition for gravity wave
    blobby(grid)

n = int(grid.L/grid.d)


#calculating tau
taux = tau0*-np.cos(np.pi*grid.y_u/grid.L)
tauy = 0

#coriolis force from grid only defined on points that were being updated
#define a version for all points on f and u grids
f_u = np.concatenate((grid.f_u, grid.f_u[:, :2]), axis = 1)
f_v = np.concatenate((grid.f_v, grid.f_v[:2, :]), axis = 0)

#making L and finding the inverse
L_inv = make_L(n, grid.dt, grid.d)

#setting old eta values
u_old, v_old, eta_old = grid.u, grid.v, grid.eta
#use FBtime to go one timestep
#update values
u, v, eta = grid.u, grid.v, grid.eta
grid.plot_uva()

#loop through time
for ts in range(nt):
    #calculated A, B, F
    A, B, F = calcABF(u, v, u_old, v_old, eta_old, d, dt, f_u, f_v,
                      gravity_wave)
    #1D version of F 
    F_1d = flat(F)
    #store old values of u, v, eta before updating
    u_old, v_old, eta_old = u, v, eta
    
    #updating eta
    #matrix multiplication of L_inv and F
    eta1d = np.matmul(L_inv, F_1d)
    eta = unflat(eta1d, n)
    
    #updating u, v
    u[:, 1:-1] = crop_x(A) - dt*g*ddx(eta, d)
    v[1:-1, :] = crop_y(B) - dt*g*ddy(eta, d)

#updating u, v, eta in class so that I can use the class plotting function
grid.u, grid.v, grid.eta = u, v, eta
#plots u, v, and eta fields
grid.plot_uva()

fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
ax.plot_surface(grid.x_eta, grid.y_eta, eta, cmap = 'viridis')    

