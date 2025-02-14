import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
from qutip import Bloch
from typing import List
import math, cmath
from scipy.special import factorial

G=1 #coupling strength
ALPHA=5 #coherent state \alpha
N_TH=50 #threshold in Fock state superposition
DETUNING=0.0
N_BAR=abs(ALPHA)**2
DELTA_2=DETUNING**2
TIME_STOP=50
TIME_STEP=1e-02
RABI=[0 for _ in range(N_TH+1)]
C10=[0 for _ in range(N_TH+1)]

def c1tn(n: int,t: float) -> complex:
    g=RABI[n]
    t1=np.exp(0.5j*t*(DETUNING+g))
    t2=np.exp(0.5j*t*(DETUNING-g))
    return 0.5*(t1+t2)
        
def initialize() -> None:
    global RABI,C10
    for n in range(0,N_TH+1):
        RABI[n]=np.sqrt(DELTA_2+4*(G**2)*(n+1))
        C10[n]=np.exp(-N_BAR)*((N_BAR**n)/factorial(n))

def coherent_sup(t: float) -> tuple[complex,complex]:
    c1t=0
    for n in range(0,N_TH+1):
        c1=C10[n]*abs(c1tn(n,t))**2
        c1t+=c1
    return c1t,1-c1t