# -*- coding: utf-8 -*-
"""
Analytical solution
Created on Thu Feb 22 13:35:12 2024

@author: Linne
"""

import numpy as np
import matplotlib.pyplot as plt
from Params import *

def calc_epsilon(L):
    '''
    Function for calculating epsilon for analytical solution

    Parameters
    ----------
    L : float
        length of domain.

    Returns
    -------
    epsilon: float
        value of epsilon for analytical solution.

    '''
    return gamma/(L*beta)

def calc_ab(L):
    '''
    Function to calculate a and b for analytical solution

    Parameters
    ----------
    L : float
        length of domain.

    Returns
    -------
    a: float
        variable a for calculation of analytical solution
    b: float
        variable b for calculation of analytical solution

    '''
    epsilon = calc_epsilon(L)
    a =  (-1-np.sqrt(1+(2*np.pi*epsilon)**2))/(2*epsilon)
    b =  (-1+np.sqrt(1+(2*np.pi*epsilon)**2))/(2*epsilon)
    return a, b

def f1(x, a, b):    
    '''
    Function f_1 for the analytical solution

    Parameters
    ----------
    x : float or array
        x values.
    a : float 
        constant a.
    b : float
        constant b.

    Returns
    -------
    f : float or array
        result of function f1.

    '''
    f = np.pi*(1+((np.exp(a)-1)*np.exp(b*x)+(1-np.exp(b))*np.exp(a*x))/\
               (np.exp(b)-np.exp(a)))
    return f

def f2(x, a, b):
    '''
    Function f_2 for the analytical solution

    Parameters
    ----------
    x : float or array
        x values.
    a : float 
        constant a.
    b : float
        constant b.

    Returns
    -------
    f : float or array
        result of function f2.

    '''
    f = ((np.exp(a)-1)*b*np.exp(b*x)+(1-np.exp(b))*a*np.exp(a*x))/\
        (np.exp(b)-np.exp(a))
    return f

def u_ss(x, y, L, a, b):
    '''
    Analytical solution for u in the steady state

    Parameters
    ----------
    x : 2D array
        meshgrid of x values.
    y : 2D array
        meshgrid of y values.
    L : float
        length of domain.
    a : float
        constant a.
    b : float
        constant b.

    Returns
    -------
    u : 2D array
        u values for each x and y.

    '''
    u = -tau0/(np.pi*gamma*rho*H)*f1(x/L, a, b)*np.cos(np.pi*y/L)
    return u
def v_ss(x, y, L, a, b):
    '''
    Analytical solution for v in the steady state

    Parameters
    ----------
    x : 2D array
        meshgrid of x values.
    y : 2D array
        meshgrid of y values.
    L : float
        length of domain.
    a : float
        constant a.
    b : float
        constant b.

    Returns
    -------
    v : 2D array
        v values for each x and y.

    '''
    v = tau0/(np.pi*gamma*rho*H)*f2(x/L, a, b)*np.sin(np.pi*y/L)
    return v


def eta_ss(x, y, L, a, b, eta0 = 0):
    '''
    Analytical solution for u in the steady state

    Parameters
    ----------
    x : 2D array
        meshgrid of x values.
    y : 2D array
        meshgrid of y values.
    L : float
        length of domain.
    a : float
        constant a.
    b : float
        constant b.
    eta0: float
        Steady-state eta from model. Optional. The default value is 0.

    Returns
    -------
    u : 2D array
        u values for each x and y.

    '''
    term1 = gamma/(f0*np.pi) * f2(x/L, a, b) * np.cos(np.pi*y/L)
    term2 = 1/np.pi*f1(x/L, a, b)*(np.sin(np.pi*y/L)*(1+beta*y/f0)+beta*L/\
                                   (f0*np.pi)*np.cos(np.pi*y/L))
    eta = eta0 + tau0*f0*L/(np.pi*gamma*rho*H*g)*(term1+term2)
    return eta
    

def plot_analytical(L, d):
    x_1D = np.arange(0, L+0.1, d)
    y_1D = np.arange(L, 0-0.1, -d)
    #setting up 2D arrays for X and Y
    XX, YY = np.meshgrid(x_1D, y_1D)
    
    #calculating analytical solution
    a, b = calc_ab(L)
    U = u_ss(XX, YY, L, a, b)
    V = v_ss(XX, YY, L, a, b)
    eta = eta_ss(XX, YY, L, a, b)
    
    #plottung analytical solutions
    labels = ['u', 'v', '$\eta$']
    data = [U, V, eta]
    
    fig, axs = plt.subplots(1, 3, figsize = (16, 7), sharey = True)
    extent = [0, L, 0, L]
    axs[0].set_ylabel('Y')
    
    for i in range(3):
        axs[i].set_title(labels[i])
        axs[i].set_xlabel('X')
        plot = axs[i].imshow(data[i], extent = extent, cmap = 'plasma')
        plt.colorbar(plot, location = 'bottom', ax= axs[i])
    streams = axs[2].streamplot(XX, np.flip(YY, axis = 0), 
                             np.flip(U, axis = 0), 
                             np.flip(V, axis = 0), density = 0.5,
                             color = 'white')
    axs[2].set_ylim(0, L)
    axs[2].set_xlim(0, L)
