import numpy as np
import scipy
from scipy import linalg as la
import time
import sys
import MoireLattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import concurrent.futures
import os
import pandas as pd



class MF_Lattice_M1_phonon:

    def __init__(self, latt):
        
        self.latt=latt
        
        self.kx_max=latt.M1[0]/2.0
        self.ky_max=latt.M2[1]
        
        IkMbz=[]
        for k in range(latt.Npoi1bz):
            if (self.latt.KX1bz[k]<=self.kx_max) and (self.latt.KX1bz[k]>-self.kx_max):
                if (self.latt.KY1bz[k]<=self.ky_max) and (self.latt.KY1bz[k]>-self.ky_max):
                    IkMbz.append(k)
                    # plt.scatter([self.latt.KX1bz[k]],[self.latt.KY1bz[k]],c='r', marker='x')
                    
        self.IkMbz=np.array(IkMbz)
        self.IkMbz_0=self.latt.insertion_index( self.latt.KX1bz[self.IkMbz],self.latt.KY1bz[self.IkMbz], self.latt.KQX,self.latt.KQY)
        self.IkMbz_1=self.latt.insertion_index( self.latt.KX1bz[self.IkMbz]+latt.M1[0],self.latt.KY1bz[self.IkMbz]+latt.M1[1], self.latt.KQX,self.latt.KQY)
        
        # plt.scatter(self.latt.KX1bz,self.latt.KY1bz)
        # # plt.scatter(self.latt.KX1bz[self.IkMbz],self.latt.KY1bz[self.IkMbz],c='r', marker='x')
        # plt.scatter(self.latt.KQX[self.IkMbz_0],self.latt.KQY[self.IkMbz_0],c='r', marker='x')
        # plt.scatter(self.latt.KQX[self.IkMbz_1],self.latt.KQY[self.IkMbz_1],c='g', marker='x')
        # plt.show()

    


def main() -> int:
    print("\n \n")
    print("lattice sampling...")
    #Lattice parameters 
    #lattices with different normalizations
    modulation_theta=1.05
    Nsamp=18
    theta=modulation_theta*np.pi/180  # magic angle
    c6sym=True
    umkl=2 #the number of umklaps where we calculate an observable ie Pi(q), for momentum transfers we need umkl+1 umklapps when scattering from the 1bz
    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0,c6sym,umkl)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2,c6sym,umkl) #this one is normalized
    [q1,q2,q3]=l.q
    q=np.sqrt(q1@q1)
    print(f"taking {umkl} umklapps")
    VV=lq.boundary()
    
    lmf=MF_Lattice_M1_phonon(lq)
    print('size of the OG lattice',np.size(lq.KX1bz))
    print('size of the mean field lattice',np.size(lmf.IkMbz),np.size(lmf.IkMbz_0),np.size(lmf.IkMbz_1))


if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
