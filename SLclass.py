from scipy.interpolate import RegularGridInterpolator as interpolater
from Params import *
from model import model
from functions import *
# import jax.numpy as np

'''
Class for semi lagrange
Running the file runs the model for one day
Parameters can be changed at the bottom of the file
'''

class SL(model):
    def __init__(self, L, d, dt, nt, method = 'linear'):
        #initialise model
        super().__init__(L, d, dt, nt)
        #setup interp functions here
        self.method = method
        self.setup_interpolators()
        
        
    def setup_interpolators(self):
        '''
        Set up interpolators for all variables needed

        '''
        self.update_eta_interps()
        self.update_u_interps()
        self.update_v_interps()
    
    def update_eta_interps(self):
        '''
        Updates all interpolators that depend on eta

        '''
        self.find_eta = interpolater([self.y_u1d, self.x_v1d], self.eta, 
                                     bounds_error = False, fill_value = np.nan,
                                     method = self.method)
        self.detady= ddy(self.eta, self.d)
        self.detadx= ddx(self.eta, self.d)
        # self.find_detady = interpolater((self.y_v1d[1:-1], self.x_v1d), 
        #                                 self.detady, bounds_error=False, 
        #                                 fill_value = np.nan,
        #                                 method = self.method)
        # self.find_detadx = interpolater((self.y_u1d, self.x_u1d[1:-1]), 
        #                                 self.detadx, bounds_error=False, 
        #                                 fill_value = np.nan,
        #                                 method = self.method)
    def update_u_interps(self):
        '''
        Updates all interpolatoors that depend on u

        '''
        self.find_u = interpolater([self.y_u1d, self.x_u1d], self.u, 
                                   bounds_error=False, fill_value = np.nan,
                                   method = self.method)
        self.dudx = ddx(self.u, self.d)
        # self.find_dudx = interpolater((self.y_u1d, self.x_v1d), self.dudx, 
        #                                bounds_error=False, fill_value = np.nan,
        #                                method = self.method)

    def update_v_interps(self):
        '''
        Updates all interpolatoors that depend on v

        '''
        self.find_v = interpolater([self.y_v1d, self.x_v1d], self.v, 
                                   bounds_error=False, fill_value = np.nan,
                                   method = self.method)
        self.dvdy = ddy(self.v, self.d)        
        # self.find_dvdy = interpolater((self.y_u1d, self.x_v1d), self.dvdy, 
        #                               bounds_error=False, fill_value = np.nan,
        #                               method = self.method)

    def reset_oldwind(self):
        '''
        Set old u and v interpolators to current u and v interpolators
        (Updating u, v at timestep n)

        '''
        self.find_oldv = self.find_v
        self.find_oldu = self.find_u
        self.old_v = self.v.copy()
        self.old_u = self.u.copy()

    def patch_old_u(self, patch_arr):
        # oldu_temp = False
        if patch_arr.shape == self.old_u.shape:
            #u grid
            oldu_temp = self.old_u
        elif patch_arr.shape == crop_x(self.old_u).shape:
            #cropped u grid
            oldu_temp = crop_x(self.old_u)
            
        elif patch_arr.shape == self.v.shape:
            #v grid
            oldu_temp = vu_interp(self.old_u)
            top = oldu_temp[0, :].reshape(1, oldu_temp.shape[1])
            bot = oldu_temp[-1, :].reshape(1, oldu_temp.shape[1])
            oldu_temp = np.concatenate((top, oldu_temp, bot), axis = 0)
        elif patch_arr.shape == crop_y(self.v).shape:
            oldu_temp = vu_interp(self.old_u)
        
        elif patch_arr.shape == self.eta.shape:
            #eta grid
            oldu_temp = (self.old_u[:, :-1]+self.old_u[:, 1:])/2
            
        patch_arr[np.isnan(patch_arr)] = oldu_temp[np.isnan(patch_arr)]
        # if not oldu_temp:
            # raise Exception(f'{patch_arr.shape}')           
        return patch_arr

        return patch_arr
    
    def patch_old_v(self, patch_arr):
        # oldv_temp = False
        if patch_arr.shape == crop_y(self.old_v).shape:
            #v grid
            oldv_temp = crop_y(self.old_v)
            
        elif patch_arr.shape == self.old_v.shape:
            #v grid
            oldv_temp = self.old_v
        
        elif patch_arr.shape == self.u.shape:
            oldv_temp = vu_interp(self.old_v)
            left = oldv_temp[:, 0].reshape(oldv_temp.shape[0],1)
            right = oldv_temp[:, -1].reshape(oldv_temp.shape[0],1)
            oldv_temp = np.concatenate((left, oldv_temp, right), axis = 1)
            
        elif patch_arr.shape == crop_x(self.old_u).shape:
            oldv_temp = vu_interp(self.old_v)
        
        elif patch_arr.shape == self.eta.shape:
            oldv_temp = (self.old_v[:-1, :]+self.old_v[1:, :])/2
        # print(patch_arr.shape)
    
        # if not oldv_temp:
            # raise Exception(f'{patch_arr.shape}')
        patch_arr[np.isnan(patch_arr)] = oldv_temp[np.isnan(patch_arr)]            
        return patch_arr

    
        
            
        

    def find_departure_points(self, X, Y):
        '''
        Finding the departure points using a two step method

        Parameters
        ----------
        X : Array
            2D array of X values at each grid point.
        Y : Array
            2D array of X values at each grid point.

        Returns
        -------
        xtilde : Array
            2D array of X values at each departure point.
        ytilde : Array
            2D array of Y values at each departure point.

        '''
        grid = (Y, X)
        #using wind at timestep n to calculate departure points
        u = self.find_oldu(grid)
        u = self.patch_old_u(u)
        
        v = self.find_oldv(grid)
        v = self.patch_old_v(v)
        
        #first step
        xstar = X - u*self.dt/2
        ystar = Y - v*self.dt/2
        pos_star = (ystar, xstar)
        #finding u at xstar
        ustar = self.find_oldu(pos_star)
        ustar = self.patch_old_u(ustar)
        
        vstar = self.find_oldv(pos_star)
        vstar = self.patch_old_v(vstar)
        
        #finding departure points
        xtilde= X - ustar*self.dt
        ytilde= Y - vstar*self.dt
        return xtilde, ytilde
    
    def update_eta(self):
        '''
        Updates eta using semi lagrangian method with averaging over sources/
        sinks

        '''
        x_d, y_d = self.find_departure_points(self.x_eta, self.y_eta)
        pos_d = (y_d, x_d)
        eta_d = self.find_eta(pos_d)
        eta_d[np.isnan(eta_d)] = self.eta[np.isnan(eta_d)]

        # dudx_d = self.find_dudx(pos_d)
        # dudx_d[np.isnan(dudx_d)] = self.dudx[np.isnan(dudx_d)]
        # dvdy_d = self.find_dvdy(pos_d)
        # dvdy_d[np.isnan(dvdy_d)] = self.dvdy[np.isnan(dvdy_d)]
        # forcing_d = dudx_d + dvdy_d

        forcing_grid = self.dudx + self.dvdy
        
        # self.eta = eta_d - 0.5*self.dt*H*(forcing_d + forcing_grid)
        
        self.eta = eta_d - self.dt*H*forcing_grid
        self.update_eta_interps()
        
        
    def update_u(self):
        '''
        Updates u using semi lagrangian method. Forcings are 
        0.5*Forcings_grid + 0.5*Forcings_departurepoint

        '''
        x_d, y_d = self.find_departure_points(crop_x(self.x_u), 
                                              crop_x(self.y_u))
        pos_d = (y_d, x_d)
        
        u_d = self.find_oldu(pos_d)
        u_d = self.patch_old_u(u_d)
        
        # v_d = self.find_oldv(pos_d)
        # v_d = self.patch_old_v(v_d)

        # detadx_d = self.find_detadx(pos_d)
        # detadx_d[np.isnan(detadx_d)] = self.detadx[np.isnan(detadx_d)]

        # f_d= f0+beta*y_d
        # taux_d= tau0*-np.cos(np.pi*y_d/self.L)
        # forcing_d = f_d*v_d - g*detadx_d - gamma*u_d + taux_d/(rho*H)
        
        forcing_grid = self.f_u*vu_interp(self.v) - g*self.detadx - \
            gamma*crop_x(self.u) + self.taux/(rho*H)
        
        # self.u[:, 1:-1] = u_d + 0.5*self.dt*(forcing_d + forcing_grid)
        self.u[:, 1:-1] = u_d + self.dt*forcing_grid
        self.update_u_interps()
        
    def update_v(self):
        '''
        Updates v using semi lagrangian method. Forcings are 
        0.5*Forcings_grid + 0.5*Forcings_departurepoint

        '''
        x_d, y_d = self.find_departure_points(crop_y(self.x_v), 
                                              crop_y(self.y_v))
        pos_d = (y_d, x_d)
        
        # u_d = self.find_oldu(pos_d)
        # u_d = self.patch_old_u(u_d)
        v_d = self.find_oldv(pos_d)
        v_d = self.patch_old_v(v_d)
        # detady_d= self.find_detady(pos_d)
        # detady_d[np.isnan(detady_d)] = self.detady[np.isnan(detady_d)]

        # f_d= f0+beta*y_d
        # forcing_d =- f_d*u_d - g*detady_d- gamma*v_d+ self.tauy/(rho*H)
        forcing_grid =- self.f_v*vu_interp(self.u) - g*self.detady - \
            gamma*crop_y(self.v) + self.tauy/(rho*H)
        
        # self.v[1:-1, :] = v_d + 0.5*self.dt*(forcing_d + forcing_grid)
        self.v[1:-1, :] = v_d + self.dt*forcing_grid

        self.update_v_interps()
    
    def twosteps(self):
        '''
        Runs the model for two timesteps

        '''
        self.reset_oldwind()
        self.update_eta()
        self.update_u()
        self.update_v()
        self.nt_count +=1
        
        
        self.reset_oldwind()
        self.update_eta()
        self.update_v()
        self.update_u()
        self.nt_count +=1

    
    def run(self):
        '''
        Runs the model for all timesteps

        '''
        while self.nt_count < self.nt:
        # for i in range(self.nt):
            self.twosteps()

