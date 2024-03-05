# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:10:11 2024

@author: Linne
"""
from model import *
# import scipy.interpolate.LinearNDInterpolator as interpolater
from scipy.interpolate import RegularGridInterpolator as interpolater
from functions import *
#%%
#start with eta
def SL_eta(test):
    X = test.x_eta
    Y = test.y_eta
    grid = (Y,X)
    u= find_u(grid)
    v= find_v(grid)

    #FINDING XSTAR
    
    xstar = X - u*test.dt/2
    ystar = Y - v*test.dt/2
    pos_star = (ystar, xstar)
    #finding u at xstar
    ustar = find_u(pos_star)
    vstar = find_v(pos_star)
    
    #finding x tilde
    xtilde= X - ustar*test.dt
    ytilde= Y - vstar*test.dt
    pos_tilde = (ytilde, xtilde)
    
    eta_int = find_eta(pos_tilde)
    dudx_tilde = find_dudx(pos_tilde)
    dvdy_tilde = find_dvdy(pos_tilde)
    
    neweta = eta_int - 0.5*test.dt*H*(dudx+dvdy+dudx_tilde+dvdy_tilde)
    #update eta
    return neweta


def SL_u(test):
    X = crop_x(test.x_u)
    Y = crop_x(test.y_u)
    grid = (Y,X)
    u= find_u(grid)
    v= find_v(grid)

    #FINDING XSTAR
    
    xstar = X - u*test.dt/2
    ystar = Y - v*test.dt/2
    pos_star = (ystar, xstar)
    #finding u at xstar
    ustar = find_u(pos_star)
    vstar = find_v(pos_star)
    
    #finding x tilde
    xtilde= X - ustar*test.dt
    ytilde= Y - vstar*test.dt
    pos_tilde = (ytilde, xtilde)
    
    u_tilde = find_u(pos_tilde)
    v_tilde = find_v(pos_tilde)
    detadx_tilde = find_detadx(pos_tilde)
    f_tilde = f0+beta*ytilde
    taux_tilde = tau0*-np.cos(np.pi*ytilde/test.L)
    forcing_tilde = f_tilde*v_tilde - g*detadx_tilde - gamma*u_tilde + taux_tilde/(rho*H)
    forcing_grid = test.f_u*v - g*detadx - gamma*u + test.taux/(rho*H)
    
    unew = u_tilde + 0.5*test.dt*(forcing_tilde + forcing_grid)
    
    #update eta
    return unew


def SL_v(test):
    X = crop_y(test.x_v)
    Y = crop_y(test.y_v)
    grid = (Y,X)
    u= find_u(grid)
    v= find_v(grid)

    #FINDING XSTAR
    xstar = X - u*test.dt/2
    ystar = Y - v*test.dt/2
    pos_star = (ystar, xstar)
    #finding u at xstar
    ustar = find_u(pos_star)
    vstar = find_v(pos_star)
    
    #finding x tilde
    xtilde= X - ustar*test.dt
    ytilde= Y - vstar*test.dt
    pos_tilde = (ytilde, xtilde)
    
    u_tilde = find_u(pos_tilde)
    v_tilde = find_v(pos_tilde)
    detady_tilde = find_detady(pos_tilde)
    f_tilde = f0+beta*ytilde
    taux_tilde = tau0*-np.cos(np.pi*ytilde/test.L)
    forcing_tilde = - f_tilde*u_tilde - g*detady_tilde - gamma*v_tilde + test.tauy/(rho*H)
    forcing_grid = - test.f_v*u - g*detady - gamma*v + test.tauy/(rho*H)
    
    vnew = v_tilde + 0.5*test.dt*(forcing_tilde + forcing_grid)
    
    #update eta
    return vnew

test = model(1e6, 25e3, 160, 40*540)
compare = model(1e6, 25e3, 160, 40*540)
compare.run()

find_eta = interpolater([test.y_u1d, test.x_v1d], test.eta, bounds_error = False, fill_value = None)
detady= ddy(test.eta, test.d)
detadx= ddx(test.eta, test.d)
find_detady = interpolater((test.y_v1d[1:-1], test.x_v1d), detady, bounds_error=False, fill_value = None)
find_detadx = interpolater((test.y_u1d, test.x_u1d[1:-1]), detadx, bounds_error=False, fill_value = None)

for i in range(540):
    find_u = interpolater([test.y_u1d, test.x_u1d], test.u, bounds_error=False, fill_value = None)
    find_v = interpolater([test.y_v1d, test.x_v1d], test.v, bounds_error=False, fill_value = None)
    # find_eta = interpolater([test.y_u1d, test.x_v1d], test.eta, bounds_error = False, fill_value = None)
    dudx = ddx(test.u,  test.d)
    dvdy = ddy(test.v, test.d)
    find_dudx = interpolater((test.y_u1d, test.x_v1d), dudx, bounds_error=False, fill_value = None)
    find_dvdy = interpolater((test.y_u1d, test.x_v1d), dvdy, bounds_error=False, fill_value = None)
    # detady= ddy(test.eta, test.d)
    # detadx= ddx(test.eta, test.d)
    # find_detady = interpolater((test.y_v1d[1:-1], test.x_v1d), detady, bounds_error=False, fill_value = None)
    # find_detadx = interpolater((test.y_u1d, test.x_u1d[1:-1]), detadx, bounds_error=False, fill_value = None)

    test.eta = SL_eta(test)
    
    find_eta = interpolater([test.y_u1d, test.x_v1d], test.eta, bounds_error = False, fill_value = None)
    detady= ddy(test.eta, test.d)
    detadx= ddx(test.eta, test.d)
    find_detady = interpolater((test.y_v1d[1:-1], test.x_v1d), detady, bounds_error=False, fill_value = None)
    find_detadx = interpolater((test.y_u1d, test.x_u1d[1:-1]), detadx, bounds_error=False, fill_value = None)
    test.u[:, 1:-1] = SL_u(test)
    test.v[1:-1,:] = SL_v(test)


test.plot_uva()
compare.plot_uva()