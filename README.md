Simplified Model of Ocean Gyres and the Gulf Stream.ipynb:
	Jupyter notebook with report

Analytical.py: 
	Code for analytical solution

report_funcs.py: 
	Functions to run for plots in the report

SL_cubic.py: 
	Code to run semi lagrange with cubic interpolation to steady-state. Takes about 20 hours, saves the u, v and eta fields
at steady state.

SL_cubic_eta.npy
SL_cubic_u.npy
SL_cubic_v.npy
	Saved cubic semi lagrange variable fields

functions.py:
	array functions that the time schemes use

Params.py:
	Parameters kept constant across all time schemes

model.py:
	Class for base model object

RK4class.py:
	RK4 time scheme

SemiImplicit.py:
	Semi Implicit time scheme

SLclass.py:
	Semi Lagrangian time scheme

ArakawaC.png:
	Arakawa C diagram
	
Lmatrix.png:
	Semi Implicit L matrix diagram
