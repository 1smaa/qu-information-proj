import matplotlib.pyplot as plt
import numpy as np

def binary_entropy(p: float) -> float:
    return -p*np.log2(p)-(1-p)*np.log2(1-p)

def to_plot(p: float) -> float:
    return 1-p

pspace=np.linspace(0,1,1000)
vspace=[to_plot(p) for p in pspace]
plt.plot(pspace,vspace,color="blue",label="Concurrence")
plt.grid(True)
h1space=[binary_entropy(0.5*(1+np.sqrt(1-p**2))) for p in vspace]
plt.plot(pspace,h1space,color="Green",label="EoF")
plt.xlabel("p",fontdict={"size":12})
plt.ylabel(r"$C/E_F$",fontdict={"size":12})
plt.legend(loc="upper right")
plt.legend(fontsize=18)
plt.show()