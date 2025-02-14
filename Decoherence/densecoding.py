import os,json,math
import qutip
import numpy as np
import matplotlib.pyplot as plt

from pkg.KrausOperators import KrausOperators,evolve

BELL=(qutip.basis(4,0)+qutip.basis(4,3)).unit()
RHO=(BELL@BELL.dag()).unit()
P_STEP=0.01
I=qutip.identity(2)
U=qutip.Qobj([[1,0,0,1],[0,1,1,0],[1,0,0,-1],[0,1,-1,0]]).unit()
OP=[I,qutip.sigmax(),qutip.sigmay(),qutip.sigmaz()]
for i in range(4):
    OP[i]=qutip.tensor(OP[i],I)
SQ_KRAUS=KrausOperators.SingleQubit.phase_flip_ch
OBJ=qutip.basis(4,0)
T=np.asarray([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

def concurrence(rho: qutip.Qobj) -> float:
    sigma_y=qutip.sigmay().full()
    flip=(np.kron(sigma_y,sigma_y))
    rho_tilde=np.dot(np.dot(flip,rho.conj().full()),flip)
    R=rho.full()@rho_tilde
    e=(np.linalg.eigvals(R))
    s=np.sort(np.sqrt(e))
    return max(0,s[3]-s[2]-s[1]-s[0])
    

def main() -> None:
    pspace=np.linspace(0,1,math.ceil(1/P_STEP))
    c_s=[0 for _ in range(len(pspace))]
    f=[0 for _ in range(len(pspace))]
    for k,p in enumerate(pspace):
        sqk=SQ_KRAUS(p)
        bsk = [ qutip.Qobj(qutip.tensor(sqk[j],I).full()) for j in range(len(sqk))]
        rhop=evolve(RHO,bsk)
        print(rhop)
        c_s[k]=concurrence(rhop)
        f[k]=(U.full()@rhop.full()@U.dag().full())[0][0]
    plt.plot(pspace,c_s,color="red")
    plt.plot(pspace,f,color="green")
    plt.xlabel("p")
    plt.ylabel("C")
    plt.grid(True)
    plt.title("Concurrence (bit-flip channel)")
    plt.show()

if __name__=="__main__":
    main()