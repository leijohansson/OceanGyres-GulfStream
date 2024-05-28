# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 17:23:47 2024

@author: Linne
Functions to run in the report
"""


import numpy as np
from Params import *
from FBclass import FB
from RK4class import RK4
from SemiImplicit import SemiImplicit as SI
import matplotlib.pyplot as plt
from Analytical import *
from functions import *
import time
from scipy.optimize import curve_fit
import seaborn as sns
from SLclass import SL
import pandas as pd


L = 1e6



def linear(x, m, c):
    '''
    linear function

    Parameters
    ----------
    x : array
        array of x values.
    m : float
        gradient.
    c : float
        y intersect.

    Returns
    -------
    array
        linear function: m*x + c.

    '''
    return m*x + c


def FB_1day():
    '''
    Runs model using Forward-Backward time scheme to 1 day and plots 
    variable profiles and eta field

    Returns
    -------
    None.

    '''
    d = 25e3
    dt = 160
    endtime = day
    nt = int(endtime/dt)
    sim = FB(L, d, dt, nt)
    sim.run()
    sim.plot_1d(plot_analytical = False)


def steadystate_energy():
    '''
    Runs model using Forward-Backward time scheme to steady-state day and plots 
    the energy time series

    Returns
    -------
    None.

    '''
    endtime = 100*day
    d = 25e3
    dt = 160
    nt = int(endtime/dt)
    long = FB(L, d, dt, nt, energy = True)
    long.run_SS()
    long.plot_energy()
    
def FB_steadystate():
    '''
    Runs model using Forward-Backward time scheme to 40 days (steady state)
    and plots variable profiles and eta field

    Returns
    -------
    FB model at steady state

    '''
    d = 25e3
    dt = 160
    nt = int(40*day/dt + 1/2)
    long = FB(L, d, dt, nt, energy = True)
    long.run()
    long.plot_1d()
    return long

def FB_Ediff():
    '''
    Runs model using Forward-Backward time scheme to 40 day using a range of 
    d (and thus dt), and calculates E'. Plots E' against d

    Returns
    -------
    None.

    '''
    endtime = 40*day
    d0, dt0 = 25e3, 160
    ratio = dt0/d0
    Es = []
    ds = np.array([10, 12.5, 15, 17.5, 20, 22.5, 25])
    times = []
    for d in ds:
        d = d*1e3
        dt = ratio*d
        nt = int(endtime/dt)
        varyd = FB(L, d, dt, nt)
        start = time.time()
        varyd.run()
        times.append(time.time()-start)
        Ediff = varyd.calc_Ediff()
        Es.append(Ediff)
    sns.set_style('white')
    # res = curve_fit(linear, ds, Es, p0 = [1e10, 1e12])[0]
    fig, axs = plt.subplots(1,2, figsize = (10, 4))
    ax = axs[0]
    ax.scatter(ds, Es, marker = 'x', color = 'purple')
    # ax.plot(ds, linear(ds, res[0],res[1]), label = 'Linear Fit', linestyle = '--', color = 'black')
    ax.set_ylabel('$E\prime, J$')
    ax.set_xlabel('d, km')

    ax1 = axs[1]
    ax1.scatter(np.log(ds), np.log(times), color = 'purple', marker = 'd')
    res1 = curve_fit(linear, np.log(ds), np.log(times))[0]
    ax1.plot(np.log(ds), linear(np.log(ds), res1[0],res1[1]), label = 'Linear Fit', linestyle = '--', color = 'black')
    ax1.set_ylabel('log(Runtime)')
    ax1.set_xlabel('log(d)')
    ax1.legend()


def SL_1day():
    '''
    Runs model using Semi-Lagrangian time scheme to 1 day and plots 
    variablel profiles and eta field

    Returns
    -------
    None.

    '''

    N = int(1*day/160)
    semi = SL(1e6, 25e3, 160, N)
    semi.run()
    semi.plot_1d(scheme = 'Semi-Lagrangian', plot_analytical = False)

def SL_SS():
    '''
    Runs model using Semi-Lagrangian time scheme to 40 days (steady state)
    and plots variable profiles and eta field

    Returns
    -------
    SL model at steady state

    '''
    N = int(40*day/160)
    semi = SL(1e6, 25e3, 160, N)
    semi.run()
    semi.plot_1d(scheme = 'Semi-Lagrangian')
    return semi

def cubic_SL():
    '''
    Imports variable fields for cubic Semi-Lagrangian time scheme 
    at 40 days (steady state) and plots variable profiles and eta field

    Returns
    -------
    Cubic SL model at steady state

    '''
    d = 25e3
    cubic = SL(L, d, 175, int(40*day/175), method = 'cubic')
    cubic.nt_count = cubic.nt
    cubic.u = np.load('SL_cubic_u.npy')
    cubic.v = np.load('SL_cubic_v.npy')
    cubic.eta = np.load('SL_cubic_eta.npy')
    cubic.plot_1d()
    return cubic

def RK4_SS():
    '''
    Runs model using RK4 time scheme to 40 days (steady state) with dt = 160s
    and plots variable profiles and eta field

    Returns
    -------
    RK4 model at steady state

    '''
    
    RK = RK4(1e6, 25e3, 160, int(40*day/160))
    RK.run()
    RK.plot_1d(scheme = 'Runge-Kutta 4')
    return RK

def RK4_Ediffs():
    '''
    Runs model using RK4 time scheme to 40 day using a range of 
    dt, and calculates E'. Plots E' against d

    Returns
    -------
    None.

    '''
    Ediffs = []
    dts = [160, 190, 220, 250]
    for dt in dts:
        RK = RK4(1e6, 25e3, dt, int(40*day/dt))
        RK.run()
        Ediffs.append(RK.calc_Ediff())
    plt.figure(figsize = (10, 2))
    plt.scatter(dts, Ediffs)
    plt.ylabel('$E\prime$, J')
    plt.xlabel('$\Delta t, s$')
    plt.grid()
    
    
def SI_SS():
    '''
    Runs model using RK4 time scheme to 40 days (steady state) with dt = 160s
    and plots variable profiles and eta field

    Returns
    -------
    RK4 model at steady state

    '''
    SemiI = SI(1e6, 25e3, 160, int(40*day/160))
    SemiI.run()
    SemiI.plot_1d(scheme = 'Semi-Implicit')
    return SemiI

def semi_ediffs():
    '''
    Runs model using SemiImplicit time scheme to 40 day using a range of 
    dt, and calculates E'. Plots E' against d

    Returns
    -------
    None.

    '''
    Ediffs = []
    dts_d = [160/day, 2000/day, 4000/day, 1/4, 1/2, 1, 2, 3, 4, 5]
    for dt in dts_d:
        dt = dt*day
        SemiI = SI(1e6, 25e3, dt, int(40*day/dt))
        SemiI.run()
        Ediffs.append(SemiI.calc_Ediff())
    plt.figure(figsize = (10, 2))
    plt.scatter(dts_d, Ediffs)
    plt.ylabel('$E\prime$, J')
    plt.xlabel('$\Delta t, days$')

def max_timesteps():
    '''
    Runs model using the various time scheme to 40 day with their maximum dt,
    and calculates E'. Plots a table with dt, E' and runtime.

    Returns
    -------
    None.

    '''
    maxtimes = ['173s', '176s', '250s', '3 days']
    names = ['FB', 'SL', 'RK4', 'SI']
    df = pd.DataFrame(maxtimes, columns=['Maximum Timestep'], index = names)

    Ediffs = []
    times = []
    dts = [173, 176, 250, 3*day]
    schemes = [FB, SL, RK4, SI]
    for i in range(4):
        scheme = schemes[i]
        start = time.time()
        dt = dts[i]
        runscheme = scheme(1e6, 25e3, dt, int(40*day/dt))
        runscheme.run()
        Ediffs.append(runscheme.calc_Ediff())
        times.append(time.time()-start)
    df['Runtime, s'] = times
    df['Energy Difference'] = Ediffs
    return df

def FB_unstable():
    '''
    Runs model to 40 days with dt=174 using Forward Backward.

    Returns
    -------
    None.

    '''
    dt = 174
    test = FB(L, 25e3, dt, int(40/dt*day))
    test.run()
    test.plot_uva()