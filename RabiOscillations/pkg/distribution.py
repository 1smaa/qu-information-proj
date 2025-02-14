import numpy as np
from mpmath import mp,sqrt,exp,ln,factorial

from pkg.cached import *
from pkg.data import PRECISION

mp.dps=PRECISION

class PhotonDistribution(object):
    @staticmethod
    def coherent(params: dict) -> None:
        N_TH,DELTA_2,N_BAR,G=params["N_TH"],params["DELTA_2"],params["N_BAR"],params["G"]
        RABI=[0 for _ in range(N_TH)]
        C20=[0 for _ in range(N_TH)]
        for n in range(0,N_TH):
            RABI[n]=sqrt(DELTA_2+4*(G**2)*(n))
            C20[n]=exp(-N_BAR)*((N_BAR**(n))/factorial(n,exact=True))
        params["RABI"]=RABI
        params["C20"]=C20
        
    @staticmethod
    def squeezed(params: dict) -> None:
        N_TH,MU,ALPHA,V,DELTA_2,G=params["N_TH"],params["MU"],params["ALPHA"],params["V"],params["DELTA_2"],params["G"]
        RABI=[0 for _ in range(N_TH)]
        C20=[0 for _ in range(N_TH)]
        for n in range(N_TH):
            lg=-ln(MU)-ln(cached_factorial(n))+n*ln(abs(V/(2*MU)))+2*ln(abs(cached_hermite(n,ALPHA/sqrt(2*MU*V))))
            C20[n]=exp(lg)
            RABI[n]=sqrt(DELTA_2+4*(G**2)*(n+1))
        params["RABI"]=RABI
        params["C20"]=C20
    
    @staticmethod
    def normalize(distribution: list) -> None:
        c=sum(distribution)
        for i in range(len(distribution)): distribution[i]/=c
        
        
    