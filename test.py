import numpy as np
import Hamiltonian
import MoireLattice
import matplotlib.pyplot as plt
import sys
import numpy.linalg as la


fillings = np.array([0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
mu_values = np.array([0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])


filling_index=int(sys.argv[1]) #0-25
##########################################
#parameters energy calculation
##########################################
a_graphene=2.46*(1e-10) #in meters
hbvf=0.003404*0.1973269804*1e-6 /a_graphene #ev*m
# hbvf = 2.1354; # eV
theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
nbands=8 #Number of bands 
Nsamp=int(sys.argv[2])
kappa_p=0.0797/0.0975;
kappa=kappa_p;
up = 0.0975; # eV
u = kappa*up; # eV

l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
ln=MoireLattice.MoireTriangLattice(Nsamp,theta,1)
lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2)
[KX,KY]=lq.Generate_lattice()
plt.scatter(KX,KY)
plt.show()

[q1,q1,q3]=l.q
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

xi=1


Numklpx=30
Numklpy=30
gridpx=np.arange(-int(Numklpx/2),int(Numklpx/2),1) #grid to calculate wavefunct
gridpy=np.arange(-int(Numklpy/2),int(Numklpy/2),1) #grid to calculate wavefunct
n1,n2=np.meshgrid(gridpx,gridpy) #grid to calculate wavefunct
#will use alpha andrei, if it fails, use alphacorrected, maybe their mistake was only in the plot
print(alpha_andrei,alpha_andrei_corrected, alpha)
alpha=w/hvkd
h=Hamiltonian.Ham(hvkd, alpha, xi, 0, 0,n1,n2, l)
print(h)

