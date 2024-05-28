# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:28:51 2024

@author: Linne

Class for RK4
"""

from model import model
from Params import *
from functions import *

class RK4(model):
    def __init__(self, L, d, dt, nt, energy = False):
        '''
        Initialising model

        Parameters
        ----------
        L : float
            Length of domain.
        d : float
            gridsize.
        dt : float
            timestep.
        nt : int
            number of timesteps.
        energy : Bool, optional
            whether to record energy at each timestep. The default is False.

        Returns
        -------
        None.

        '''
        #initialise model
        super().__init__(L, d, dt, nt)
        
        self.Fs = [self.F_u, self.F_v, self.F_eta]

        
    def F_eta(self, u, v, eta):
        '''
        Calculates deta/dt

        Parameters
        ----------
        u : array
            array of u values on u grid.
        v : array
            array of v values on v grid.
        eta : array
            array of eta values on eta grid.

        Returns
        -------
        F : array
            deta/dt on eta grid.

        '''
        dudx = ddx(u, self.d)
        dvdy = ddy(v, self.d)
        F = -H*(dudx+dvdy)
        return F
    
    def F_u(self, u, v, eta):
        '''
        Calculates du/dt

        Parameters
        ----------
        u : array
            array of u values on u grid.
        v : array
            array of v values on v grid.
        eta : array
            array of eta values on eta grid.

        Returns
        -------
        F : array
            du/dt on eta grid.

        '''
        eta_x = ddx(eta, self.d)
        F = self.f_u*vu_interp(v) - g*eta_x - gamma*crop_x(u) \
            + self.taux/(rho*H)
        return F
    
    def F_v(self, u, v, eta):
        '''
        Calculates dv/dt

        Parameters
        ----------
        u : array
            array of u values on u grid.
        v : array
            array of v values on v grid.
        eta : array
            array of eta values on eta grid.

        Returns
        -------
        F : array
            dv/dt on eta grid.

        '''
        eta_y = ddy(eta, self.d)
        F = -self.f_v*vu_interp(u) - g*eta_y - gamma*crop_y(v) \
            + self.tauy/(rho*H)
        return F
    
    
    def one_step(self):
        '''
        Moves model one timestep

        Returns
        -------
        None.

        '''
        k = [0]*4
        l = [0]*4
        m = [0]*4
        
        klm = [k, l, m]
        #first k
        for i in range(4):
            if i == 0:
                u, v, eta = self.u, self.v, self.eta
            elif i == 3:
                u = self.u.copy()
                u[:, 1:-1] += self.dt*klm[0][i-1]
                
                v = self.v.copy()
                v[1:-1, :] += self.dt*klm[1][i-1]
                
                eta = self.eta + self.dt*klm[2][i-1]
            else:
                u = self.u.copy()
                u[:, 1:-1] += self.dt/2*klm[0][i-1]
                
                v = self.v.copy()
                v[1:-1, :] += self.dt/2*klm[1][i-1]
                
                eta = self.eta + self.dt/2*klm[2][i-1]
            
            for j in range(3):
                F = self.Fs[j]
                klm[j][i] = F(u, v, eta)
        
        
        self.u[:, 1:-1] += self.dt/6*(k[0] + 2*k[1] + 2*k[2] + k[3])
        self.v[1:-1, :] += self.dt/6*(l[0] + 2*l[1] + 2*l[2] + l[3])
        self.eta += self.dt/6*(m[0] + 2*m[1] + 2*m[2] + m[3])
        self.nt_count+=1
        if self.energy:
            self.E_ts[self.nt_count] = self.calcE()

        
        
    def run(self):
        '''
        Runs model for nt timesteps

        Returns
        -------
        None.

        '''
        for i in range(self.nt):
            self.one_step()
        
            
        
    
