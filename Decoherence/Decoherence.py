import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import qutip
import numpy as np

from pkg.KrausOperators import KrausOperators,evolve

MAP={
    "bit_flip_ch":KrausOperators.SingleQubit.bit_flip_ch,
    "phase_flip_ch":KrausOperators.SingleQubit.phase_flip_ch,
    "bit_phase_flip_ch":KrausOperators.SingleQubit.bit_phase_flip_ch,
    "depolarizing_ch":KrausOperators.SingleQubit.depolarizing_ch,
    "amplitude_damping_ch":KrausOperators.SingleQubit.amplitude_damping_ch
}

PARAM_FILE=os.path.join(os.getcwd(),"Decoherence","parameters.json")
INTERVAL=50
R=None
BLOCH=None

def bloch_sphere(rho: qutip.Qobj) -> tuple:
    return np.real((rho*qutip.sigmax()).tr()),np.real((rho*qutip.sigmay()).tr()),np.real((rho*qutip.sigmaz()).tr())

def update(frame: int)-> None:
    BLOCH.clear()
    BLOCH.add_vectors(R[frame])
    BLOCH.render()

def main() -> None:
    global R,BLOCH
    with open(PARAM_FILE,mode="r",encoding="utf-8") as f:
        PARAM=json.load(f)
    KRAUS=MAP[PARAM["type"]](PARAM["p"])
    steps=PARAM["steps"]
    frames=[qutip.Qobj(PARAM["rho"]).unit(inplace=False)]+[None for _ in range(steps-1)]
    R=[bloch_sphere(frames[0])]+[None for _ in range(steps-1)]
    for i in range(1,steps):
        frames[i]=evolve(frames[i-1],KRAUS)
        R[i]=bloch_sphere(frames[i])
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    BLOCH=qutip.Bloch(fig=fig,axes=ax)
    BLOCH.vector_color='green'
    ani=FuncAnimation(fig,update,frames=PARAM["steps"],interval=INTERVAL)
    plt.show()
    
if __name__=="__main__":
    main()
    
#BIT-FLIP
#C=|2p-1|
#<00|\rho|00>=1-p
#E_F=H_B(0.5+sqrt(p)sqrt(1-p))

#DEPOLARIZING
#C=(2/3)p^2-2p+1 for p<(3-sqrt(3))/2, 0 otherwise
#<00|\rho|00>=1-p
#E_F=H_B((1+sqrt(2p)sqrt(1-p/3))/2)

#AMPLTIUDE_DAMPING
#C=1-p
#<00|\rho|00>=(2-p+2sqrt(1-p))/4
#E_F di conseguenza 