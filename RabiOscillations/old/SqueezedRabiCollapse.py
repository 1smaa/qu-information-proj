import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial
from functools import lru_cache
import math

A=4 #Coherent state amplitude
PHI=np.pi/4 #Coherent state phase
ALPHA=A*np.exp(1j*PHI) #Coherent state parameter
PHI_SQUEEZE=PHI #Squeezing phase
R=1e-3 #Squeezing amplitude
EPS=R*np.exp(1j*PHI_SQUEEZE) #Squeezing parameter
N_BAR=A**2 #Coherent state average photon number
N_TH=100 #Photon number threshold for superposition
DETUNING=0 #Detuning in the driving photon state
DELTA_2=DETUNING**2 #Detuning^2 (for ease of calculation)
TIME_STOP=100 #Stop time for the siMUlation
TIME_STEP=0.1 #Time step for evolution
RABI=[0 for _ in range(N_TH)] #Generalized rabi frequencies for each fock basis state
C10=[0 for _ in range(N_TH)] #Starting point for each coefficient of the fock basis (t=0)
G=1 #Coupling constant
MU=np.cosh(R)
V=np.sinh(R)*np.exp(1j*PHI_SQUEEZE)

def c1tn(n: int,t: float) -> complex:
    g=RABI[n]
    t1=np.exp(0.5j*t*(DETUNING+g))
    t2=np.exp(0.5j*t*(DETUNING-g))
    return 0.5*(t1+t2)

@lru_cache(maxsize=None)
def cached_factorial(n: int) -> int:
    return factorial(n,exact=False) 

@lru_cache(maxsize=None)
def cached_hermite(n: int,z: complex):
    if(n==0): return 1
    elif(n==1): return 2*z
    else: return 2*z*cached_hermite(n-1,z)-2*(n-1)*cached_hermite(n-2,z)

def photon_distribution(n: int) -> float:
    lg=-np.log(MU)-np.log(cached_factorial(n))+n*np.log(abs(V/(2*MU)))+2*np.log(abs(cached_hermite(n,ALPHA/np.sqrt(2*MU*V))))
    return np.exp(lg)
    
def squeezed_sup(t: float) -> complex:
    c1t=0
    for n in range(N_TH):
        c1t+=C10[n]*abs(c1tn(n,t))**2
    return c1t,1-c1t
    
def initialize() -> None:
    global RABI,C10
    for n in range(N_TH):
        RABI[n]=np.sqrt(DELTA_2+4*(G**2)*(n+1))
        C10[n]=photon_distribution(n)
    n=0
    for c in C10: n+=c
    for i in range(N_TH): C10[i]/=n
            
def main() -> None:
    print("Starting...")
    initialize()
    print("Initialized.")
    e_p=abs(MU*np.conj(ALPHA)-np.conj(V)*ALPHA)**2+abs(V)**2
    avg_p=0
    for n in range(N_TH):
        avg_p+=C10[n]*n
    print("Average Photon Number Fidelty",1-abs((e_p-avg_p)/e_p))
    steps=math.ceil(TIME_STOP/TIME_STEP)
    tsp=np.linspace(0,TIME_STOP,steps)
    w=[0 for _ in range(steps)]
    for t in range(steps):
        c1t,c2t=squeezed_sup(t*TIME_STEP)
        w[t]=c2t-c1t
    nsp=np.linspace(0,N_TH,N_TH)
    fig,axes=plt.subplots(2,2)
    fig.suptitle("Population Dynamics and Photon Distributions in Driven Squeezed Coherent States")
    plt.plot(nsp,C10)
    plt.title("Photon distribution")
    plt.grid(True)
    plt.show()
    plt.plot(tsp,w)
    plt.title("Inversion Dynamics of Squeezed State")
    plt.grid(True)
    plt.show()
    
        
        

if __name__=="__main__":
    main()