from qutip import Bloch,basis
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import pi
from typing import List
TIME_STEP=0.01
DETUNING=0
RABI_FREQ=4*pi
B=Bloch()
B.fig=plt.figure()

def normalize(v: List) -> List:
    n=0
    for i in v:
        n+=i**2
    n=n**0.5
    for i in range(len(v)):
        v[i]/=n
    return v

def compute_step(b_v: List) -> List:
    u,v,w=b_v
    nu=-TIME_STEP*DETUNING*v+u
    nv=TIME_STEP*(DETUNING*u-RABI_FREQ*w)+v
    nw=TIME_STEP*RABI_FREQ*u+w
    return normalize([nu,nv,nw])
    
def update(frame) -> None:
    B.clear()
    B.add_vectors(frame)
    B.render()

def generate(STEPS:int): 
    b_v=[0,0,1]
    for _ in range(STEPS):
        b_v=compute_step(b_v)
        yield b_v
        
def main() -> None:
    fig=B.fig
    anim=FuncAnimation(fig,update,frames=generate(1000),interval=5)
    plt.show()
    
if __name__=="__main__":
    main()