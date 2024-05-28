# -*- coding: utf-8 -*-
"""
Base model class
Created on Wed Feb 28 14:08:02 2024

@author: Linne
"""
import numpy as np
from Params import *
from functions import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from Analytical import *
from scipy.interpolate import lagrange
sns.set_context('notebook')
class model:
    def __init__(self, L, d, dt, nt, energy = False):
        #setting model parameters
        self.L = L
        self.d = d
        self.dt = dt
        self.nt = nt
        #ensuring that nt is even 
        if nt % 2 != 0:
            self.nt += 1
        self.n = int(L/d)+1
        
        #setting up initial conditions
        self.u = np.zeros((self.n-1, self.n))
        self.v = self.u.copy().T
        self.eta = np.zeros((self.n-1, self.n-1))
                
        #setting up x and y values for plotting and for calculation of coriolis
        #parameter for the different grids
        
        self.y_v1d = np.linspace(L, 0, self.n, endpoint = True)
        self.x_u1d = np.linspace(0, L, self.n, endpoint = True)
        self.y_u1d = (self.y_v1d[1:] + self.y_v1d[:-1])/2
        self.x_v1d = (self.x_u1d[1:] + self.x_u1d[:-1])/2
        self.x_v, self.y_v = np.meshgrid(self.x_v1d, self.y_v1d)
        self.x_u, self.y_u = np.meshgrid(self.x_u1d, self.y_u1d)
        self.x_eta, self.y_eta = np.meshgrid(self.x_v1d, self.y_u1d)
        
        #calculation of coriolis parameter on the v and u grids
        self.f_v = crop_y(f0+beta*self.y_v)
        self.f_u = crop_x(f0+beta*self.y_u)
        
        #setting values of taux and tauy
        self.taux = crop_x(tau0*-np.cos(np.pi*self.y_u/L))
        self.tauy = 0
        
        #setting counter for time
        self.nt_count = 0
        #setting bool for whether to calculate energy at each step
        self.energy = energy
        if self.energy:
            self.E_ts = np.zeros((self.nt+1))
    
    def calcE(self):
        '''
        Calculates energy 
        Returns
        -------
        E : float
            energy in joules.

        '''
        interpu = (self.u[:, 1:] + self.u[:, :-1])/2
        interpv = (self.v[1:, :] + self.v[:-1, :])/2

        E = calc_energy(interpu, interpv, self.eta, self.d)

        # E = calc_energy(self.u, self.v, self.eta, self.d)
        return E
    
    def eta0(self):
        '''
        Finds eta0 with 3rd order lagrange interpolation
        Returns
        -------
        float
            eta0 value in steady state.

        '''
        id_mid = int(self.eta.shape[0]/2)
        if self.eta.shape[0]%2 == 0:
            etamids = 0.5*(self.eta[id_mid-1, :] + self.eta[id_mid, :])
        else:
            etamids = self.eta[id_mid, :]

        f = lagrange(self.x_v1d[:2], etamids[:2])
        return f(0)
    
    def plot_uva(self):
        '''
        2D Plots of U, V and eta

        Returns
        -------
        None.

        '''
        labels = ['u', 'v', '$\eta$']
        data = [self.u, self.v, self.eta]
        
        fig, axs = plt.subplots(1, 3, figsize = (16, 7), sharey = True)
        extent = [0, self.L, 0, self.L]
        axs[0].set_ylabel('Y')
        
        for i in range(3):
            axs[i].set_title(labels[i])
            axs[i].set_xlabel('X')
            plot = axs[i].imshow(data[i], extent = extent, cmap = 'plasma')
            plt.colorbar(plot, location = 'bottom', ax= axs[i])
        fig.suptitle(f'''Time = {np.round(self.nt*self.dt/day,0)} days
                     $\Delta t = {self.dt}$s, d = {self.d}m''')
            
    def plot_1d(self, scheme = 'Forward-Backward', plot_analytical = True):
        eta0 = self.eta0()
            
        fig = plt.figure(figsize = (15, 7))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, width_ratios = (2,2))
        axu = fig.add_subplot(spec[0, 0])
        axv = fig.add_subplot(spec[1, 0])
        axeta = fig.add_subplot(spec[2, 0])
        axc = fig.add_subplot(spec[:, 1])


        a, b = calc_ab(self.L)
        U = u_ss(self.x_u, self.y_u, self.L, a, b)
        V = v_ss(self.x_v, self.y_v, self.L, a, b)
        ETA = eta_ss(self.x_eta, self.y_eta, self.L, a, b, eta0 = eta0)
    
        axu.plot(self.x_u1d, self.u[-1, :], label = scheme)
        axu.set_title('u at southern boundary')
        axu.set_ylabel('$u, ms^{-1}$')
        axu.set_xlabel('x, m')
        
        id_mid = int(self.eta.shape[0]/2)
        if self.eta.shape[0]%2 == 0:
            etamiddle = (self.eta[id_mid, :] + self.eta[id_mid-1, :])/2
            ETAmid = (ETA[id_mid, :] + ETA[id_mid-1,:])/2
                
        else:
            etamiddle = self.eta[int(self.n/2+1), :]
            ETAmid = ETA[int(self.n/2+1), :]
        
        if plot_analytical:        
            axu.plot(self.x_u1d, U[-1, :], linestyle = '--', color = 'black', label = 'Analytical')
            axv.plot(self.y_v1d, V[:, 0], linestyle = '--', color = 'black')
            axeta.plot(self.x_v1d, ETAmid, linestyle = '--', color = 'black')
            axu.legend()
        
        
        axv.plot(self.y_v1d, self.v[:, 0])
        axv.set_title('v at western boundary')
        axv.set_ylabel('$v, ms^{-1}$')
        axv.set_xlabel('y, m')

       

        axeta.plot(self.x_v1d, etamiddle)
        axeta.set_title('$\eta$ in the middle of the gyre')
        axeta.set_ylabel('$\eta, m$')
        axeta.set_xlabel('x, m')

        extent = [0, self.L, 0, self.L]
        plot = axc.imshow(self.eta, extent = extent, cmap = 'plasma')
        u_eta = (self.u[:, 1:] + self.u[:, :-1])/2
        v_eta = (self.v[1:, :] + self.v[:-1, :])/2

        streams = axc.streamplot(self.x_eta, np.flip(self.y_eta, axis = 0), 
                                 np.flip(u_eta, axis = 0), 
                                 np.flip(v_eta, axis = 0), density = 0.5,
                                 color = 'white')
                                 # , 
                                 # color = np.sqrt(u_eta**2 + v_eta**2),
                                 # cmap = 'Pastel2')

        axc.set_title('$\eta$')
        axc.set_xlabel('x, m')
        axc.set_ylabel('y, m')
        # cbar2 = plt.colorbar(streams.lines, ax=axc, location = 'bottom', shrink = 0.65)
        # cbar2.set_label('$|\mathbf{u}|, ms^{-1}$')
        cbar = plt.colorbar(plot, ax=axc, location = 'right')
        cbar.set_label('$\eta, m$')

        fig.tight_layout()
        
        
    def plot_energy(self):
        plt.figure(figsize = (8, 3))
        plt.ylabel('Energy, J')
        plt.xlabel('Time, days')
        plt.plot(np.arange(self.nt_count)*self.dt/(24*60*60), self.E_ts[:self.nt_count])
    
    def calc_Ediff(self):
        eta0 = self.eta0()
        a, b = calc_ab(self.L)
        U = u_ss(self.x_u, self.y_u, self.L, a, b)
        V = v_ss(self.x_v, self.y_v, self.L, a, b)
        # U = u_ss(self.x_eta, self.y_eta, self.L, a, b)
        # V = v_ss(self.x_eta, self.y_eta, self.L, a, b)
        ETA = eta_ss(self.x_eta, self.y_eta, self.L, a, b, eta0 = eta0)
        
        
        # interpu = (self.u[:, 1:] + self.u[:, :-1])/2
        # interpv = (self.v[1:, :] + self.v[:-1, :])/2
        # udiff, vdiff, etadiff = interpu-U, interpv-V, self.eta-ETA
        udiff, vdiff, etadiff = self.u-U, self.v-V, self.eta-ETA
       
        
        Ediff = calc_energy(udiff, vdiff, etadiff, self.d)
        return np.abs(Ediff)
    
    def plot_solution_diff(self):
        a, b = calc_ab(self.L)
        U = u_ss(self.x_u, self.y_u, self.L, a, b)
        V = v_ss(self.x_v, self.y_v, self.L, a, b)
        ETA = eta_ss(self.x_eta, self.y_eta, self.L, a, b, eta0 = self.eta0())
        
        udiff, vdiff, etadiff = self.u-U, self.v-V, self.eta-ETA
        
        labels = ['$\Delta u$', '$\Delta v$', '$\Delta \eta$']
        data = [udiff, vdiff, etadiff]
        
        fig, axs = plt.subplots(1, 3, figsize = (16, 7), sharey = True)
        extent = [0, self.L, 0, self.L]
        axs[0].set_ylabel('Y')
        
        for i in range(3):
            axs[i].set_title(labels[i])
            axs[i].set_xlabel('X')
            plot = axs[i].contourf(data[i], extent = extent, cmap = 'plasma')
            cbar = plt.colorbar(plot, location = 'bottom', ax= axs[i])
            cbar.formatter.set_powerlimits((0, 0))
        fig.suptitle(f'''Time = {np.round(self.nt_count*self.dt/day,0)} days
                     $\Delta t = {self.dt}$s, d = {self.d}m''')
                     
    def calc_KEdiff(self):
        # eta0 = self.eta0()
        a, b = calc_ab(self.L)
        U = u_ss(self.x_u, self.y_u, self.L, a, b)
        V = v_ss(self.x_v, self.y_v, self.L, a, b)
        # ETA = eta_ss(self.x_eta, self.y_eta, self.L, a, b, eta0 = eta0)
        
        udiff, vdiff, etadiff = self.u-U, self.v-V, 0
        
        Ediff = calc_energy(udiff, vdiff, etadiff, self.d)
        return np.abs(Ediff)
            