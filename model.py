# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 14:08:02 2024

@author: Linne
"""
import numpy as np
from Params import *
from functions import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
sns.set_context('notebook')

class model:
    def __init__(self, L, d, dt, nt, scheme = 'FB', energy = False):
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
        
        ##Perturbed initial condition for eta
        # halfeta = int(self.eta.shape[0]/2+0.5)
        # self.eta[halfeta, halfeta] = 10
        
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

    def run(self):
        for i in range(int(self.nt/2+0.5)):
            self.FB_time()
            
    def FB_time(self, energy = False):
        #ddx, ddy calculate the x and y derivatives
        
        #vu_interp interpolates u onto the v grid and vice versa
        
        #crop_x and crop_y remove the columns and rows on the edges 
        #respectively
        
        #moves two timesteps
        self.eta -=  H*self.dt*(ddx(self.u, self.d)+ddy(self.v, self.d))
        eta_x, eta_y = ddx(self.eta, self.d), ddy(self.eta, self.d)
        self.u[:, 1:-1] += self.dt*(self.f_u*vu_interp(self.v) - g*eta_x - 
                                    gamma*crop_x(self.u) + self.taux/(rho*H))
        self.v[1:-1, :] += self.dt*(-self.f_v*vu_interp(self.u) - g*eta_y - 
                                    gamma*crop_y(self.v) + self.tauy/(rho*H))
        self.nt_count += 1
        if self.energy: #would it be more efficient to define a different 
                        #function to avoid if statements
            self.E_ts[self.nt_count] = self.calcE()
        
        self.eta -=  H*self.dt*(ddx(self.u, self.d)+ddy(self.v, self.d))
        eta_x, eta_y = ddx(self.eta, self.d), ddy(self.eta, self.d)
        self.v[1:-1, :] += self.dt*(-self.f_v*vu_interp(self.u) - g*eta_y - 
                                    gamma*crop_y(self.v) + self.tauy/(rho*H))
        self.u[:, 1:-1] += self.dt*(self.f_u*vu_interp(self.v) - g*eta_x - 
                                    gamma*crop_x(self.u) + self.taux/(rho*H))
        self.nt_count += 1
        
        if self.energy:
            self.E_ts[self.nt_count] = self.calcE()
    def FB_time_one(self, energy = False):
        #ddx, ddy calculate the x and y derivatives
        
        #vu_interp interpolates u onto the v grid and vice versa
        
        #crop_x and crop_y remove the columns and rows on the edges 
        #respectively
        
        #moves two timesteps
        if self.nt_count%2 == 0:
            self.eta -=  H*self.dt*(ddx(self.u, self.d)+ddy(self.v, self.d))
            eta_x, eta_y = ddx(self.eta, self.d), ddy(self.eta, self.d)
            self.u[:, 1:-1] += self.dt*(self.f_u*vu_interp(self.v) - g*eta_x - 
                                        gamma*crop_x(self.u) + self.taux/(rho*H))            
            
            self.v[1:-1, :] += self.dt*(-self.f_v*vu_interp(self.u) - g*eta_y - 
                                        gamma*crop_y(self.v) + self.tauy/(rho*H))
            print(-(self.f_v*vu_interp(self.u))[0,0])
            print(- g*eta_y[0,0])
            print(-gamma*crop_y(self.v)[0,0])
            
            self.nt_count += 1
        
        else:
            self.eta -=  H*self.dt*(ddx(self.u, self.d)+ddy(self.v, self.d))
            eta_x, eta_y = ddx(self.eta, self.d), ddy(self.eta, self.d)
            self.v[1:-1, :] += self.dt*(-self.f_v*vu_interp(self.u) - g*eta_y - 
                                        gamma*crop_y(self.v) + self.tauy/(rho*H))
            self.u[:, 1:-1] += self.dt*(self.f_u*vu_interp(self.v) - g*eta_x - 
                                        gamma*crop_x(self.u) + self.taux/(rho*H))
            self.nt_count += 1
             
        
    def SL(self):
        xstar_u = self.x_u - self.u*self.dt/2
        #find u at xstar_u? at the last time step?
        #interpolator - LinearNDInterpolator (Scipy)
        
        
        #interpolate value of u at x_star at that value of y? gridy or ystar?
        # ystar = y_v - self.v*self.dt/2
        #interpolate value of v at y_star at that value of x? gridx or xstar from above?
        #im confused
    
    def calcE(self):
        '''
        Calculates energy 
        Returns
        -------
        E : float
            energy in joules.

        '''
        E = calc_energy(self.u, self.v, self.eta, self.d)
        return E
    
    def eta0(self):
        '''

        Returns
        -------
        float
            eta0 value in steady state.

        '''
        return self.eta[-1, int(self.eta.shape[0]/2+0.5)]
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
        extent = [0, self.L, self.L, 0]
        axs[0].set_ylabel('Y')
        
        for i in range(3):
            axs[i].set_title(labels[i])
            axs[i].set_xlabel('X')
            plot = axs[i].imshow(data[i], extent = extent, cmap = 'plasma')
            plt.colorbar(plot, location = 'bottom', ax= axs[i])
        fig.suptitle(f'''Time = {np.round(self.nt*self.dt/day,0)} days
                     $\Delta t = {self.dt}$s, d = {self.d}m''')
            
    def plot_1d(self):
        fig = plt.figure(figsize = (15, 7))
        spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig, width_ratios = (2,2))
        axu = fig.add_subplot(spec[0, 0])
        axv = fig.add_subplot(spec[1, 0])
        axeta = fig.add_subplot(spec[2, 0])
        axc = fig.add_subplot(spec[:, 1])

    
        axu.plot(self.x_u1d, self.u[-1, :])
        axu.set_title('u at southern boundary')
        axu.set_ylabel('$u, ms^{-1}$')
        axu.set_xlabel('x, m')
        
        
        axv.plot(self.y_v1d, self.v[:, 0])
        axv.set_title('v at western boundary')
        axv.set_ylabel('$v, ms^{-1}$')
        axv.set_xlabel('y, m')

        if self.eta.shape[0]%2 == 0:
            etamiddle = (self.eta[int(self.eta.shape[0]/2)] + \
                         self.eta[int(self.eta.shape[0]/2)+1])/2
        else:
            etamiddle = self.eta[int(self.n/2+1), :]        

        axeta.plot(self.x_v1d, etamiddle)
        axeta.set_title('$\eta$ in the middle of the gyre')
        axeta.set_ylabel('$\eta, m$')
        axeta.set_xlabel('x, m')

        extent = [self.d, self.L, self.L, self.d]
        plot = axc.imshow(self.eta, extent = extent)
        axc.set_title('$\eta$')
        axc.set_xlabel('x, m')
        axc.set_ylabel('y, m')
        plt.colorbar(plot, ax=axc, location = 'bottom', shrink = 0.65)
        fig.tight_layout()
        
        
    def plot_energy(self):
        plt.figure()
        plt.plot(np.arange(self.nt+1)*self.dt/(24*60*60), self.E_ts)


