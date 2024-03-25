# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 11:08:56 2024

@author: Linne
"""
import numpy as np
# import jax.numpy as np
from Params import *
from numba import njit

def crop_x(arr):
    '''
    Removes columns corresponding to first and last x value (x-values at the 
    edge of the grid)

    Parameters
    ----------
    arr : array
        array to be cropped.

    Returns
    -------
    array
        cropped array.

    '''
    return arr[:, 1:-1]

def crop_y(arr):
    '''
    Removes columns corresponding to first and last y value (y-values at the 
    edge of the grid)

    Parameters
    ----------
    arr : array
        array to be cropped.

    Returns
    -------
    array
        cropped array.

    '''
    return arr[1:-1, :]

def vu_interp(v):
    '''
    Calculates interpolated v values on u grid or vice versa

    Parameters
    ----------
    v : array
        v or u array to interpolate.

    Returns
    -------
    v_u : array
        interpolated v or u array.

    '''
    v1 = v[:-1, :-1]
    v2 = v[1:, :-1]
    v3 = v[:-1, 1:]
    v4 = v[1:, 1:]
    v_u = 1/4*(v1+v2+v3+v4)
    return v_u

def ddx(arr, d):
    '''
    Calculates d/dx values of arrays. Output array has one less column than
    input array. (d/dx values calculated at midpoints of x values in the
    input array)

    Parameters
    ----------
    arr : array
        array for which d/dx is calculated.
    d : float
        x spacing between values in the array.

    Returns
    -------
    f_x : array
        d/dx values calculated at midpoints of x values in the input array. Has
        one less column than the input array

    '''
    f_x = (arr[:, 1:] - arr[:, :-1])/d
    return f_x

def ddy(arr, d):
    '''
    Calculates d/dy values of arrays. Output array has one less row than
    input array. (d/dy values calculated at midpoints of y values in the
    input array)

    Parameters
    ----------
    arr : array
        array for which d/dy is calculated.
    d : float
        y spacing between values in the array.

    Returns
    -------
    f_y : array
        d/dy values calculated at midpoints of y values in the input array. Has
        one less row than the input array

    '''
    f_y = -(arr[1:, :] - arr[:-1, :])/d
    return f_y

def calc_energy(u, v, eta, d):
    '''
    Calculates energy

    Parameters
    ----------
    u : 2D array
        array of u values.
    v : 2D array
        array of v values.
    eta : array
        array of eta values.
    d : float
        grid spacing.

    Returns
    -------
    E : float
        Energy.

    '''
    uv_term = H*(np.sum(u**2) + np.sum(v**2))
    eta_term = g*np.sum(eta**2)
    E = 0.5*rho*d**2*(uv_term+eta_term)
    return E

        

