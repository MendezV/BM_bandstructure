import numpy as np
import MoireLattice

class Ham():
    def __init__(self, hvkd, alpha, kx, ky):

        self.hvkd = hvkd
        self.alpha=alpha
        
        self.pauli0=np.eye(2,2)
        self.paulix=np.array([[0,1],[1,0]])
        self.pauliy=np.array([[0,-1j],[1j,0]])
        self.pauliz=np.array([[1,0],[0,-1]])

    def __repr__(self):
        return "Hamiltonian at {kx} {ky} with parameter {alpha}".format(kx=self.kx, ky=self.ky, alpha =self.alpha)
