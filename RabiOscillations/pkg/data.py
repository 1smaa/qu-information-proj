import json
from typing import Any
import numpy as np

from mpmath import exp,cosh,sinh,mp,cos

PRECISION=25

mp.dps=PRECISION

from pkg.distribution import PhotonDistribution

class Simulation(object):
    def __init__(self, p_file: str, squeeze: bool) -> None:
        self.__file=p_file
        self.__parameters={}
        self.__squeeze=squeeze
        
    def get_param(self,param: str) -> Any:
        if param not in self.__parameters.keys():
            return None
        else:
            return self.__parameters[param]
        
    def get_params(*args,**kwargs) -> list:
        obj=args[0]
        return [obj.__parameters[arg] for arg in args[1:] if arg in obj.__parameters]
        
    def load(self) -> None:
        with open(self.__file,mode="r") as f:
            self.__parameters=json.load(f)
        self.__fill_in()
        
    def __fill_in(self) -> None:
        self.__parameters["PHI"]*=np.pi
        self.__parameters["PHI_SQUEEZE"]*=np.pi
        for key in self.__parameters.keys():
            if isinstance(self.__parameters[key],complex):
                self.__parameters[key]=mp.mpc(np.real(self.__parameters[key]),np.imag(self.__parameters[key]))
            elif isinstance(self.__parameters[key],float):
                self.__parameters[key]=mp.mpf(self.__parameters[key])
        self.__parameters["ALPHA"]=self.__parameters["A"]*exp(mp.mpc(0,self.__parameters["PHI"]))
        self.__parameters["XI"]=self.__parameters["R"]*exp(mp.mpc(0,self.__parameters["PHI_SQUEEZE"]))
        self.__parameters["N_BAR"]=self.__parameters["A"]**2
        self.__parameters["DELTA_2"]=self.__parameters["DETUNING"]**2
        self.__parameters["MU"]=cosh(self.__parameters["R"])
        self.__parameters["V"]=sinh(self.__parameters["R"])*exp(mp.mpc(0,self.__parameters["PHI_SQUEEZE"]))
        if self.__squeeze: 
            PhotonDistribution.squeezed(self.__parameters)
            PhotonDistribution.normalize(self.__parameters["C20"])
        else: PhotonDistribution.coherent(self.__parameters)
        
        
    def evolve(self,t: float) -> tuple[float,float]:
        c2t=0
        for n in range(self.__parameters["N_TH"]):
            c2t+=self.__parameters["C20"][n]*cos(self.__parameters["RABI"][n]*t*0.5)**2
        return 1-c2t,c2t