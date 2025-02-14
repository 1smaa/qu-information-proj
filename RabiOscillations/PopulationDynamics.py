from pkg.cached import *
from pkg.data import *
from pkg.distribution import *

import numpy as np
import os, math,sys,time
import matplotlib.pyplot as plt
from qutip import wigner, squeeze, displace, basis


PARAM_FILE=os.path.join(os.getcwd(),"parameters.json")

def envelope(signal: list[int]) -> tuple[list,list]:
    positive=False
    top_envelope=[1]
    bottom_envelope=[-1]
    for i in range(1,len(signal)):
        if signal[i]-signal[i-1]>0 and not positive:
            bottom_envelope.append(signal[i-1])
            top_envelope.append(top_envelope[-1])
            positive=not positive
        elif signal[i]-signal[i-1]<0 and positive:
            top_envelope.append(signal[i-1])
            bottom_envelope.append(bottom_envelope[-1])
            positive=not positive
        else:
            bottom_envelope.append(bottom_envelope[-1])
            top_envelope.append(top_envelope[-1])
    return top_envelope,bottom_envelope



def main() -> None:
    s=Simulation(PARAM_FILE,True)
    s.load()
    T_STOP,T_STEP,T_START=s.get_params("TIME_STOP","TIME_STEP","TIME_START")
    tspace=np.linspace(T_START,T_STOP,math.ceil(T_STOP/T_STEP))
    N_TH,C20=s.get_params("N_TH","C20")
    nspace=np.linspace(0,N_TH,N_TH)
    w=[0 for _ in range(len(tspace))]
    start=time.time()
    sys.stdout.write("Starting Squeezed Simulation...\n\n")
    for j,t in enumerate(tspace):
        c1t,c2t=s.evolve(t)
        w[j]=c2t-c1t
        e=time.time()
        if j%10==0:     
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            sys.stdout.write("Current Simulated Time: {sm} s - Approximated Remaining Time : {rm} s\n".format(sm=round(t,4),rm=round((e-start)*(len(tspace)-j),2)))
        start=e
        
    c=Simulation(PARAM_FILE,False)
    c.load()
    wc=[0 for _ in range(len(tspace))]
    start=time.time()
    sys.stdout.write("Starting Coherent Simulation...\n\n")
    for j,t in enumerate(tspace):
        c1t,c2t=c.evolve(t)
        wc[j]=c2t-c1t
        e=time.time()
        if j%10==0:     
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            sys.stdout.write("Current Simulated Time: {sm} s - Approximated Remaining Time : {rm} s\n".format(sm=round(t,4),rm=round((e-start)*(len(tspace)-j),2)))
        start=e
    top_env,bottom_env=envelope(wc)
    fig,axes=plt.subplots(2,2)
    plt.suptitle("Inversion Dynamics of a Two-Level Atom Interacting with a Squeezed Coherent State")
    
    axes[0][0].plot(tspace,top_env,label="Coherent Oscillations Envelope",color="red")
    axes[0][0].plot(tspace,bottom_env,color="red")
    axes[0][0].set_title("Inversion")
    axes[0][0].plot(tspace,w,label="Squeezed Oscillation Dynamics")
    axes[0][0].set_xlabel("t")
    axes[0][0].set_ylabel("w")
    axes[0][0].grid(True)

    axes[0][1].plot(nspace,s.get_param("RABI"))
    axes[0][1].set_title("Generalized Rabi Frequency")
    axes[0][1].set_xlabel("n")
    axes[0][1].set_ylabel(r"$\Omega^\Delta_n$")
    
    axes[1][0].plot(nspace,C20)
    axes[1][0].plot(nspace,c.get_param("C20"),color="red")
    axes[1][0].set_xlabel("n")
    axes[1][0].set_ylabel("Probability")
    axes[1][0].set_title("Photon Number Distribution")
    axes[1][0].grid(True)
    
    XI,A,ALPHA=s.get_params("XI","A","ALPHA")
    psi=displace(N_TH,ALPHA)*squeeze(N_TH,XI)*basis(N_TH,0)
    x1=np.linspace(-A*2,A*2,200)
    x2=np.linspace(-A*2,A*2,200)
    X1,X2=np.meshgrid(x1,x2)
    wigner_data=wigner(psi,x1,x2)
    axes[1][1].contourf(X1, X2, wigner_data, 100, cmap='RdBu_r')
    axes[1][1].set_title("Wigner Function")
    axes[1][1].set_xlabel(r"$X_1$")
    axes[1][1].set_ylabel(r"$X_2$")
    axes[1][1].grid(alpha=(A*4/200))
    axes[1][1].axhline(0,color="white",linewidth=1)
    axes[1][1].axvline(0,color="white",linewidth=1)
    plt.tight_layout()
    plt.show()
    
if __name__=="__main__":
    main()