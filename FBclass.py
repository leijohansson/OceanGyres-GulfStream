# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:24:02 2024

@author: Linne
"""
from model import *
from scipy.interpolate import RegularGridInterpolator as interpolater
from functions import *
from Params import *

class FB(model):
    def __init__(self, L, d, dt, nt, energy = False):
        #initialise model
        super().__init__(L, d, dt, nt, energy = energy)
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