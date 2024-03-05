from scipy.interpolate import RegularGridInterpolator as interpolater
from Params import *
from model import model
from functions import *
'''
Class for semi lagrange
Running the file runs the model for one day
Parameters can be changed at the bottom of the file
'''

class SL(model):
    def __init__(self, L, d, dt, nt):\
        #initialise model
        super().__init__(L, d, dt, nt)
        #setup interp functions here
        self.setup_interpolators()
        
        
    def setup_interpolators(self):
        '''
        Set up interpolators for all variables needed

        '''
        self.find_eta = interpolater([self.y_u1d, self.x_v1d], self.eta, 
                                     bounds_error = False, fill_value = None)
        self.find_u = interpolater([self.y_u1d, self.x_u1d], self.u, 
                                   bounds_error=False, fill_value = None)
        self.find_v = interpolater([self.y_v1d, self.x_v1d], self.v, 
                                   bounds_error=False, fill_value = None)
        self.detady= ddy(self.eta, self.d)
        self.detadx= ddx(self.eta, self.d)
        self.find_detady = interpolater((self.y_v1d[1:-1], self.x_v1d), 
                                        self.detady, bounds_error=False, 
                                        fill_value = None)
        self.find_detadx = interpolater((self.y_u1d, self.x_u1d[1:-1]), 
                                        self.detadx, bounds_error=False, 
                                        fill_value = None)
        self.update_eta_interps()
        self.update_u_interps()
        self.update_v_interps()
        
    def update_eta_interps(self):
        '''
        Updates all interpolators that depend on eta

        '''
        self.find_eta = interpolater([self.y_u1d, self.x_v1d], self.eta, 
                                     bounds_error = False, fill_value = None)
        self.detady= ddy(self.eta, self.d)
        self.detadx= ddx(self.eta, self.d)
        self.find_detady = interpolater((self.y_v1d[1:-1], self.x_v1d), 
                                        self.detady, bounds_error=False, 
                                        fill_value = None)
        self.find_detadx = interpolater((self.y_u1d, self.x_u1d[1:-1]), 
                                        self.detadx, bounds_error=False, 
                                        fill_value = None)
    def update_u_interps(self):
        '''
        Updates all interpolatoors that depend on u

        '''
        self.find_u = interpolater([self.y_u1d, self.x_u1d], self.u, 
                                   bounds_error=False, fill_value = None)
        self.dudx = ddx(self.u, self.d)
        self.find_dudx = interpolater((self.y_u1d, self.x_v1d), self.dudx, 
                                       bounds_error=False, fill_value = None)
        # self.find_detadx = interpolater((self.y_u1d, self.x_u1d[1:-1]), 
                                        # self.detadx, bounds_error=False, 
                                        # fill_value = None)
    def update_v_interps(self):
        '''
        Updates all interpolatoors that depend on v

        '''
        self.find_v = interpolater([self.y_v1d, self.x_v1d], self.v, 
                                   bounds_error=False, fill_value = None)
        self.dvdy = ddy(self.v, self.d)        
        self.find_dvdy = interpolater((self.y_u1d, self.x_v1d), self.dvdy, 
                                      bounds_error=False, fill_value = None)
        # self.find_detady = interpolater((self.y_v1d[1:-1], self.x_v1d), 
                                        # self.detady, bounds_error=False, 
                                        # fill_value = None)
    def reset_oldwind(self):
        '''
        Set old u and v interpolators to current u and v interpolators
        (Updating u, v at timestep n)

        '''
        self.find_oldv = self.find_v
        self.find_oldu = self.find_u


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
        v = self.find_oldv(grid)
        #first step
        xstar = X - u*self.dt/2
        ystar = Y - v*self.dt/2
        pos_star = (ystar, xstar)
        #finding u at xstar
        ustar = self.find_oldu(pos_star)
        vstar = self.find_oldv(pos_star)
        
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
        dudx_d = self.find_dudx(pos_d)
        dvdy_d = self.find_dvdy(pos_d)
        forcing_d = dudx_d + dvdy_d
        forcing_grid = self.dudx + self.dvdy
        
        self.eta = eta_d - 0.5*self.dt*H*(forcing_d + forcing_grid)
        # self.eta = eta_d - self.dt*H*forcing_grid
        self.update_eta_interps()
        
        
    def update_u(self):
        '''
        Updates u using semi lagrangian method. Forcings are 
        0.5*Forcings_grid + 0.5*Forcings_departurepoint

        '''
        x_d, y_d = self.find_departure_points(crop_x(self.x_u), 
                                              crop_x(self.y_u))
        pos_d = (y_d, x_d)
        u_d = self.find_u(pos_d)
        v_d = self.find_v(pos_d)

        detadx_d = self.find_detadx(pos_d)
        f_d= f0+beta*y_d
        taux_d= tau0*-np.cos(np.pi*y_d/self.L)
        forcing_d = f_d*v_d - g*detadx_d - gamma*u_d + taux_d/(rho*H)
        forcing_grid = self.f_u*vu_interp(self.v) - g*self.detadx - \
            gamma*crop_x(self.u) + self.taux/(rho*H)
        
        self.u[:, 1:-1] = u_d + 0.5*self.dt*(forcing_d + forcing_grid)
        # self.u[:, 1:-1] = u_d + self.dt*forcing_grid
        self.update_u_interps()
        
    def update_v(self):
        '''
        Updates v using semi lagrangian method. Forcings are 
        0.5*Forcings_grid + 0.5*Forcings_departurepoint

        '''
        x_d, y_d = self.find_departure_points(crop_y(self.x_v), 
                                              crop_y(self.y_v))
        pos_d = (y_d, x_d)
        u_d = self.find_u(pos_d)
        v_d = self.find_v(pos_d)

        detady_d= self.find_detady(pos_d)
        f_d= f0+beta*y_d
        forcing_d =- f_d*u_d - g*detady_d- gamma*v_d+ self.tauy/(rho*H)
        forcing_grid =- self.f_v*vu_interp(self.u) - g*self.detady - \
            gamma*crop_y(self.v) + self.tauy/(rho*H)
        
        self.v[1:-1, :] = v_d + 0.5*self.dt*(forcing_d + forcing_grid)
        # self.v[1:-1, :] = v_d + self.dt*forcing_grid

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
        self.update_u()
        self.update_v()
        self.nt_count +=1

    
    def run(self):
        '''
        Runs the model for all timesteps

        '''
        for i in range(int(self.nt/2+0.5)):
        # for i in range(self.nt):
            self.twosteps()

if __name__ == 'main':
    semi = SL(1e6, 25e3, 160, 540)
    semi.run()
    semi.plot_uva()