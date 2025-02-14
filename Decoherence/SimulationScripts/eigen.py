import qutip
import sympy as sp
import numpy as np

p=sp.Symbol("p")
rho=0.25*sp.Matrix([[2-p+2*sp.sqrt(1-p),0,p,0],[0,p,0,p],[p,0,2-p-2*sp.sqrt(1-p),0],[0,p,0,p]])
sigma_y=sp.Matrix([[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]])
rhos=sigma_y@rho@sigma_y
R=rho@rhos
print(R.eigenvals())