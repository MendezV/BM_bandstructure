import numpy as np
import Hamiltonian
import MoireLattice
import matplotlib.pyplot as plt
import sys
import numpy.linalg as la

##########################################
#parameters energy calculation
##########################################
a_graphene=2.46*(1e-10) #in meters
hbvf=0.003404*0.1973269804*1e-6 /a_graphene #ev*m
# hbvf = 2.1354; # eV
theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
nbands=8 #Number of bands 
Nsamp=int(sys.argv[1])
kappa_p=0.0797/0.0975;
kappa=kappa_p;
up = 0.0975; # eV
u = kappa*up; # eV

l=MoireLattice.MoireTriangLattice(Nsamp,theta)

[q1,q1,q3]=l.qvect()
q=la.norm(q1)

hvkd=hbvf*q
Kvec=(2*l.b[0,:]+l.b[1,:])/3 
K=la.norm(Kvec)
GG=la.norm(l.b[0,:])
print(q , 2*K*np.sin(theta/2))

w=0.110 #ev
hvfK_andrei=19.81

#andreis params

hvfkd_andrei=hvfK_andrei*np.sin(theta/2) #wrong missing 1/sqrt3
alpha_andrei=w/hvfkd_andrei
alpha=w/hvkd 
alpha_andrei_corrected=(np.sqrt(3)/2)*w/hvfkd_andrei


#will use alpha andrei, if it fails, use alphacorrected, maybe their mistake was only in the plot
print(alpha_andrei,alpha_andrei_corrected, alpha)
alpha=w/hvkd
h=Hamiltonian.Ham(hvkd, alpha, 0,0)
print(h.alpha, alpha)

