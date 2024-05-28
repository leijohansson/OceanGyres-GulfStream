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
    '''
    make the L matrix for semi implicit

    Parameters
    ----------
    n : int
        length of eta grid (how many grid points in 1d).
    dt : float
        timestep in seconds.
    d : float
        grid spacing.

    Returns
    -------
    L : array
        n^2 by n^2 matrix.

    '''
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
        
    return L



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


class SemiImplicit(model):
    def __init__(self, L, d, dt, nt):
        #initialise model
        super().__init__(L, d, dt, nt)
        L = make_L(self.n-1, self.dt, self.d)
        #finding inverse of L
        self.L_inv = np.linalg.inv(L)


    def calcABF(self):
        '''
        Calculates A, B and F
    
        '''
    
        #setting up v on u grid
        v_on_u = vu_interp(self.v)
            
        #calculating A, dAdx
        self.A = self.u.copy()
        self.A[:, 1:-1] += self.dt*(self.f_u*v_on_u - gamma*crop_x(self.u) + self.taux/(rho*H))
        dAdx = ddx(self.A, self.d)
        
        #setting up u on v grid
        u_on_v = vu_interp(self.u)
        
        #calculating B, dBdy
        # B = -dt*g*deta_dy 
        self.B = self.v.copy()
        self.B[1:-1,:] += self.dt*(-self.f_v*u_on_v - gamma*crop_y(self.v) + self.tauy/(rho*H))
        dBdy = ddy(self.B, self.d)
        
        #calculating C using u, v at time n-1
        # C = eta + dt*H*(ddx(u, d) + ddy(v, d))
        C = self.eta.copy()
    
        self.F = C - self.dt*H*(dAdx + dBdy)
    
    def one_step(self):
        '''
        Moves model one timestep

        Returns
        -------
        None.

        '''
        self.calcABF()
        #1D version of F 
        F_1d = flat(self.F)
        #updating eta
        #matrix multiplication of L_inv and F
        eta1d = np.matmul(self.L_inv, F_1d)
        self.eta = unflat(eta1d, self.n-1)
        
        #updating u, v
        self.u[:, 1:-1] = crop_x(self.A) - self.dt*g*ddx(self.eta, self.d)
        self.v[1:-1, :] = crop_y(self.B) - self.dt*g*ddy(self.eta, self.d)
        self.nt_count += 1
        if self.energy:
            self.E_ts[self.nt_count] = self.calcE()

    def run(self):
        '''
        Runs model to nt timesteps

        Returns
        -------
        None.

        '''
        while self.nt_count < self.nt:
            self.one_step()

