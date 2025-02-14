import qutip
import numpy as np

class KrausOperators(object):
    class SingleQubit(object):
        @staticmethod
        def bit_flip_ch(p: float) -> list:
            '''
            Returns the Kraus operators for modeling a bit-flip channel for a single qubit
            
            p: bit-flip probability
            '''
            if p<0 or p>1: raise ValueError("Probability must be a real number 0<=p<=1")
            return [np.sqrt(1-p)*qutip.identity(2),np.sqrt(p)*qutip.sigmax()]
        
        @staticmethod
        def phase_flip_ch(p: float) -> list:
            '''
            Returns the Kraus operators for modeling a phase-flip channel for a single qubit
            
            p: phase-flip probability
            '''
            if p<0 or p>1: raise ValueError("Probability must be a real number 0<=p<=1")
            return [np.sqrt(1-p)*qutip.identity(2),np.sqrt(p)*qutip.sigmaz()]
        
        @staticmethod
        def bit_phase_flip_ch(p: float) -> list:
            '''
            Returns the Kraus operators for modeling a bit-phase-flip channel for a single qubit
            
            p: bit-phase-flip probability
            '''
            if p<0 or p>1: raise ValueError("Probability must be a real number 0<=p<=1")
            return [np.sqrt(1-p)*qutip.identity(2),np.sqrt(p)*qutip.sigmay()]
        
        @staticmethod
        def depolarizing_ch(p: float) -> list:
            '''
            Returns the Kraus operators for modeling a bit-phase-flip channel for a single qubit
            
            p: depolarizing coefficient
            '''
            if p<0 or p>1: raise ValueError("Depolarizing coefficient must be a real number 0<=p<=1")
            n=np.sqrt(p/3)
            return [np.sqrt(1-p)*qutip.identity(2),n*qutip.sigmax(),n*qutip.sigmay(),n*qutip.sigmaz()]
        
        @staticmethod
        def amplitude_damping_ch(p: float) -> list:
            '''
            Returns the Kraus operators for modeling an amplitude damping channel for a single qubit
            
            p: damping probability
            '''
            if p<0 or p>1: raise ValueError("Probability must be a real number 0<=p<=1")
            return [qutip.Qobj([[1,0],[0,np.sqrt(1-p)]]),qutip.Qobj([[0,np.sqrt(p)],[0,0]])]
        
def evolve(rho: qutip.Qobj,kraus_operators: list[qutip.Qobj]) -> qutip.Qobj:
    start=qutip.Qobj(np.zeros((4,4)))
    for i in range(0,len(kraus_operators)):
        c=kraus_operators[i]@rho@(kraus_operators[i].dag())
        start+=kraus_operators[i]@rho@(kraus_operators[i].dag())
    return start