#implements the Koshino continuum model
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys
import scipy
np.set_printoptions(threshold=sys.maxsize)

####################################################################################
####################################################################################
####################################################################################
#
#       DEFINING PARAMETERS THAT SPECIFY THE GEOMETRY OF THE MOIRE RECIPROCAL VECTOR
#
####################################################################################
####################################################################################
####################################################################################

##########################################
#parameters energy calculation
##########################################
hv = 2.1354; # eV
theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
mu=0.289/1000 #chemical potential
u = 0.0797; # eV
up = 0.0975; # eV
w = np.exp(1j*2*np.pi/3); # enters into the hamiltonian
en0 = 1.6598/1000; # band energy of the flat bands, eV
en_shift = 10.0/1000; #10/1000; # to sort energies correctly, eV
art_gap = 0.0/1000; # artificial gap
paulix=np.array([[0,1],[1,0]])
pauliy=np.array([[0,-1j],[1j,0]])
##########################################
#lattice vectors
##########################################
#i vectror j coordinate
a=np.array([[1,0],[1/2,np.sqrt(3)/2]])
astar=(2*np.pi)*np.array([[1,-1/np.sqrt(3)],[0,2/np.sqrt(3)]])
Rotth1=np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]) #rotation matrix +thet
Rotth2=np.array([[np.cos(theta/2),np.sin(theta/2)],[-np.sin(theta/2),np.cos(theta/2)]]) #rotation matrix -thet

a1=np.dot(a,Rotth1) #top layer
a2=np.dot(a,Rotth2) #bottom layer
astar1=np.dot(astar,Rotth1) #top layer
astar2=np.dot(astar,Rotth2)  #bottom layer

GM1=astar1[0,:]-astar2[0,:] #see Koshino-liang fu paper for further illustration (going off towards negative quadrant in the middle of the edge of the MBZ)
GM2=astar1[1,:]-astar2[1,:] #see Koshino-liang fu paper for further illustration (Going horizontal towards middle of the edge of the MBZ)
LM1=2*np.pi*np.array([GM2[1],-GM2[0]])/la.det(np.array([GM1,GM2]))
LM2=2*np.pi*np.array([-GM1[1],GM1[0]])/la.det(np.array([GM1,GM2]))

LM=np.sqrt(np.dot(LM1,LM1)) #moire period 2/np.sin(theta/2)
GM=np.sqrt(np.dot(GM1,GM1)) #reciprocal space period  8*np.pi/np.sqrt(3)*np.sin(theta/2)
#print(np.sqrt(np.dot(LM1,LM1)),0.5/np.sin(theta/2)) #verify the relation with the moire period

##########################################
#Valley locations
##########################################
#1,2 label layer and  + - labels valley from the  original graphene dispersion
Kplus1=-(2*astar1[0,:]+astar1[1,:])/3
Kplus2=-(2*astar2[0,:]+astar2[1,:])/3
Kmin1=+(2*astar1[0,:]+astar1[1,:])/3
Kmin2=+(2*astar2[0,:]+astar2[1,:])/3
Ktopplus=Kplus2-GM1
Kbottommin=Kmin2+GM1
Mpoint=(Kplus1+Kplus2)/2 #position of the M point in the - valley
Mpointneg=-(Kplus1+Kplus2)/2 #position of the M point in the - valley

##########################################
#definitions of the BZ grid
##########################################
nbands=4 #Number of bands
k_window_sizex = GM2[0]/2 #4*(np.pi/np.sqrt(3))*np.sin(theta/2) (half size of  MBZ from edge to edge)
k_window_sizey = Ktopplus[1]   #(1/np.sqrt(3))*GM2[0] ( half size of MBZ from k to k' along a diagonal)
kn_pointsx = 61
kn_pointsy = 61



gridp=np.arange(-50,50,1) #grid to calculate wavefunct
n1,n2=np.meshgrid(gridp,gridp) #grid to calculate wavefunct


####################################################################################
####################################################################################
####################################################################################
#
#      ENERGY DISPERSION IN THE CONTINUUM MODEL
#
####################################################################################
####################################################################################
####################################################################################


def eigsystem(kx, ky, xi, nbands, n1, n2):
    #we diagonalize a matrix made up
    qx_dif = kx+GM1[0]*n1+GM2[0]*n2-xi*Mpoint[0]
    qy_dif = ky+GM1[1]*n1+GM2[1]*n2-xi*Mpoint[1]
    vals = np.sqrt(qx_dif**2+qy_dif**2)
    ind_to_sum = np.where(vals <= 4*GM) #Finding the indices where the difference of  q lies inside threshold
    n1_val = n1[ind_to_sum]
    n2_val = n2[ind_to_sum]
    N = np.shape(ind_to_sum)[1]

    qx = kx+GM1[0]*n1_val+GM2[0]*n2_val
    qy = ky+GM1[1]*n1_val+GM2[1]*n2_val

    if xi<0:
        qx_1= (+(qx-Kmin1[0])*np.cos(theta/2) - (qy-Kmin1[1])*np.sin(theta/2) ) #rotated momenta
        qy_1= (+(qx-Kmin1[0])*np.sin(theta/2) + (qy-Kmin1[1])*np.cos(theta/2) ) #rotated momenta
        qx_2= (+(qx-Kmin2[0])*np.cos(theta/2) + (qy-Kmin2[1])*np.sin(theta/2) ) #rotated momenta
        qy_2= (-(qx-Kmin2[0])*np.sin(theta/2) + (qy-Kmin2[1])*np.cos(theta/2) ) #rotated momenta
    else:
        qx_1= (+(qx-Kplus1[0])*np.cos(theta/2) - (qy-Kplus1[1])*np.sin(theta/2) ) #rotated momenta
        qy_1= (+(qx-Kplus1[0])*np.sin(theta/2) + (qy-Kplus1[1])*np.cos(theta/2) ) #rotated momenta
        qx_2= (+(qx-Kplus2[0])*np.cos(theta/2) + (qy-Kplus2[1])*np.sin(theta/2) ) #rotated momenta
        qy_2= (-(qx-Kplus2[0])*np.sin(theta/2) + (qy-Kplus2[1])*np.cos(theta/2) ) #rotated moment

    #the following block generates the momentum space rep of the interlayer coupling matrix
    matG2=np.zeros([N,N])
    matG4=np.zeros([N,N])
    for i in range(N):
        indi1=np.where((n1_val==n1_val[i]-xi)*(n2_val==n2_val[i]))
        if np.size(indi1)>0:
            matG2[i,indi1]=1
        indi1=np.where((n1_val==n1_val[i]-xi)*(n2_val==n2_val[i]-xi))
        if np.size(indi1)>0:
            matG4[i,indi1]=1

    #Matrices that  appeared as coefficients of the real space ops
    #full product is the kronecker product of both matrices
    matu1=np.array([[u,up],[up,u]])
    matu2=np.array([[u,up*(w**(-xi))],[up*(w**(xi)),u]])

    #assembling the matrix to be diagonalized
    U=(np.kron(matu1,np.eye(N,N))+np.kron(matu2,matG2)+np.kron(matu2.T,matG4)) #interlayer coupling
    H1=-hv*(np.kron(xi*paulix,np.diag(qx_1))+np.kron(pauliy,np.diag(qy_1)))
    H2=-hv*(np.kron(xi*paulix,np.diag(qx_2))+np.kron(pauliy,np.diag(qy_2)))
    Hxi=np.bmat([[H1, (U.conj()).T], [U, H2]]) #Full matrix
    #a= np.linalg.eigvalsh(Hxi) - en_shift
    (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues
    return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]


print(eigsystem(0,0, 1, nbands, n1, n2)[0])
####################################################################################
####################################################################################
####################################################################################
#
#      GEOMETRY OF THE FBZ
#
####################################################################################
####################################################################################
####################################################################################



Nsamp=15
n1=np.arange(0,Nsamp,1)
n2=np.arange(0,Nsamp,1)
k1=np.array([1,0])
k2=np.array([1/2,np.sqrt(3)/2])
n_1,n_2=np.meshgrid(n1,n2)
G1=k1*Nsamp
G2=k2*Nsamp

XsLatt=(GM/Nsamp)*(k1[0]*n_1+k2[0]*n_2).T
YsLatt=(GM/Nsamp)*(k1[1]*n_1+k2[1]*n_2).T
S=np.empty((0))
for l in range(Nsamp*Nsamp):
    i=int(l%Nsamp)
    j=int((l-i)/Nsamp)
    print(XsLatt[i,j],YsLatt[i,j])
    E=eigsystem(XsLatt[i,j],YsLatt[i,j], 1, nbands, n1, n2)[0]
    print(E)
    S=np.append(S,E)

plt.scatter(XsLatt,YsLatt, c=S)

plt.show()