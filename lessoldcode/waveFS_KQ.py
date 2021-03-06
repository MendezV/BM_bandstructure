#implements the Koshino continuum model
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys
import scipy
np.set_printoptions(threshold=sys.maxsize)
import time

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

fillings = np.array([0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
mu_values = np.array([0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])


filling_index=int(sys.argv[1]) #0-25

mu=mu_values[filling_index]/1000 #chemical potential in eV
filling=fillings[filling_index]

print("FILLING FOR THE CALCULATION IS .... ",filling)
print("CHEMICAL POTENTIAL FOR THAT FILLING IS .... ",mu)

u = 0.0797; # eV
up = 0.0975; # eV
w = np.exp(1j*2*np.pi/3); # enters into the hamiltonian
en0 = 1.6598/1000; # Sets the zero of energy to the middle of flat bands, ev
paulix=np.array([[0,1],[1,0]])
pauliy=np.array([[0,-1j],[1j,0]])
pauliz=np.array([[1,0],[0,-1]])

hbarc=0.1973269804*1e-6 #ev*m
alpha=137.0359895 #fine structure constant
a_graphene=2.46*(1e-10) #in meters
ee2=(hbarc/a_graphene)/alpha
kappa_di=3.03
T=10/1000 #in ev for finite temp calc

print("COULOMB CONSTANT...",ee2)
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

##volume of the MBZ
zhat=np.array([0,0,1])
b_1=np.array([GM1[0],GM1[1],0])
b_2=np.array([GM2[0],GM2[1],0])
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)
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


Numklpx=30
Numklpy=100
gridpx=np.arange(-int(Numklpx/2),int(Numklpx/2),1) #grid to calculate wavefunct
gridpy=np.arange(-int(Numklpy/2),int(Numklpy/2),1) #grid to calculate wavefunct
n1,n2=np.meshgrid(gridpx,gridpy) #grid to calculate wavefunct


####################################################################################
####################################################################################
####################################################################################
#
#      GEOMETRY AND RESHAPING ARRAYS
#
####################################################################################
####################################################################################
####################################################################################


#######calculating the boundaries and high symmetry points of the FBZ using a voronoi decomposition
from scipy.spatial import Voronoi, voronoi_plot_2d
def FBZ_points(b_1,b_2):
    #creating reciprocal lattice
    Np=4
    n1=np.arange(-Np,Np+1)
    n2=np.arange(-Np,Np+1)
    Recip_lat=[]
    for i in n1:
        for j in n2:
            point=b_1*i+b_2*j
            Recip_lat.append(point)

    #getting the nearest neighbours to the gamma point
    Recip_lat_arr=np.array(Recip_lat)
    dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=10)
    sorted_dist=np.sort(list(set(dist)) )
    points=Recip_lat_arr[np.where(dist<sorted_dist[2])[0]]

    #getting the voronoi decomposition of the gamma point and the nearest neighbours
    vor = Voronoi(points)
    Vertices=(vor.vertices)

    #ordering the points counterclockwise in the -pi,pi range
    angles_list=list(np.arctan2(Vertices[:,1],Vertices[:,0]))
    Vertices_list=list(Vertices)

    #joint sorting the two lists for angles and vertices for convenience later.
    # the linear plot routine requires the points to be in order
    # atan2 takes into acount quadrant to get the sign of the angle
    angles_list, Vertices_list = (list(t) for t in zip(*sorted(zip(angles_list, Vertices_list))))

    ##getting the M points as the average of consecutive K- Kp points
    Edges_list=[]
    for i in range(len(Vertices_list)):
        Edges_list.append([(Vertices_list[i][0]+Vertices_list[i-1][0])/2,(Vertices_list[i][1]+Vertices_list[i-1][1])/2])

    Gamma=[0,0]
    K=Vertices_list[0::2]
    Kp=Vertices_list[1::2]
    M=Edges_list[0::2]
    Mp=Edges_list[1::2]

    return Vertices_list, Gamma, K, Kp, M, Mp

Vertices_list, Gamma, K, Kp, M, Mp=FBZ_points(GM1,GM2) #DESCRIBING THE MBZ 

print("size of FBZ, maximum momentum trans",np.sqrt(np.sum(np.array(K[0])**2))/GM )



MGS=[[0,1],[1,0],[0,-1],[-1,0],[-1,-1],[1,1],[-1,-2],[-2,-1],[-1,1],[1,2],[2,1],[1,-1]]
def kwrap_FBZ(kx,ky):
    dmin=kx**2+ky**2
    G=[0,0]
    for MG in MGS:
        d=(kx-MG[0]*GM1[0]-MG[1]*GM2[0])**2+(ky-MG[0]*GM1[1]-MG[1]*GM2[1])**2
        if d<dmin+1e-10: #slightly more stringent to keep terms fixed if they differ by machine precission 
            dmin=d
            G=MG
    kxp=kx-G[0]*GM1[0]-G[1]*GM2[0]
    kyp=ky-G[0]*GM1[1]-G[1]*GM2[1]
    return kxp,kyp
    
####################################################################################
####################################################################################
####################################################################################
#
#      ENERGY DISPERSION IN THE CONTINUUM MODEL
#
####################################################################################
####################################################################################
####################################################################################
tp=1 

# c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
# plt.show()

def eigsystem(kx, ky, xi, nbands, n1, n2):
    #we diagonalize a matrix made up
    qx_dif = kx+GM1[0]*n1+GM2[0]*n2-xi*Mpoint[0]
    qy_dif = ky+GM1[1]*n1+GM2[1]*n2-xi*Mpoint[1]
    vals = np.sqrt(qx_dif**2+qy_dif**2)
    ind_to_sum = np.where(vals+GM*1e-5 < 5*GM) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
    n1_val = n1[ind_to_sum] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
    n2_val = n2[ind_to_sum] #
    N = np.shape(ind_to_sum)[1] ##number of indices for which the condition above is satisfied
    # print(kx,ky)
    # plt.scatter(n1,n2)
    # plt.scatter(n1_val,n2_val)
    # plt.scatter(-n1_val,-n2_val)
    # plt.show()
    qx = kx+GM1[0]*n1_val+GM2[0]*n2_val
    qy = ky+GM1[1]*n1_val+GM2[1]*n2_val

    if xi<0:
        qx_1= (+(qx-Kmin1[0])*np.cos(theta/2) - (qy-Kmin1[1])*np.sin(theta/2) ) #rotated momenta
        qy_1= (+(qx-Kmin1[0])*np.sin(theta/2) + (qy-Kmin1[1])*np.cos(theta/2) ) #rotated momenta
        qx_2= (+(qx-Kmin2[0])*np.cos(theta/2) + (qy-Kmin2[1])*np.sin(theta/2) ) #rotated momenta
        qy_2= (-(qx-Kmin2[0])*np.sin(theta/2) + (qy-Kmin2[1])*np.cos(theta/2) ) #rotated momenta
    else:
        qx_1= (+(qx-Kplus1[0])*np.cos(theta/2) - (qy-Kplus1[1])*np.sin(theta/2) ) #rotated momenta
        qy_1= (+(qx-Kplus1[0])*np.sin(theta/2) + (qy-Kplus1[1])*np.cos(theta/2) ) #rotated momenta7/89+-78
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

    #######HANDLING WITH RESHAPE
    #umklp,umklp, layer, sublattice
    psi_p=np.zeros([Numklpy,Numklpx,2,2]) +0*1j
    # psi=np.zeros([Numklp*Numklp*4,nbands])+0*1j
    psi=np.zeros([Numklpy, Numklpx, 2,2, nbands]) +0*1j

    for nband in range(nbands):
        # print(np.shape(ind_to_sum), np.shape(psi_p), np.shape(psi_p[ind_to_sum]))
        psi_p[ind_to_sum] = np.reshape(  np.array(np.reshape(Eigvect[:,2*N-int(nbands/2)+nband] , [4, N])  ).T, [N, 2, 2] )    
        ##GAUGE FIXING by making the 30 30 1 1 component real
        # phas=np.angle(psi_p[30,30,1,1])
        # phas=0 ## not fixing the phase
        maxisind = np.unravel_index(np.argmax(psi_p, axis=None), psi_p.shape)
        phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
        
        psi[:,:,:,:,nband] = psi_p*np.exp(-1j*phas)    


    return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0,psi , N
    #  return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]



############################################################
###denser grid for integration
############################################################
k_window_sizey = K[2][1] 
k_window_sizex = K[1][0] 

Radius_inscribed_hex=1.0000005*k_window_sizey
def hexagon(pos):
    y,x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge



print("starting sampling in reciprocal space....")
s=time.time()
LP=int(sys.argv[2])
nn1=np.arange(-LP,LP+1,1)
nn2=np.arange(-LP,LP+1,1)
nn_1,nn_2=np.meshgrid(nn1,nn2)

nn_1p=[]
nn_2p=[]
for x in nn1:
    for y in nn2:
        kx=(2*np.pi*x/LP)/LM
        ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))/LM
        if hexagon(( kx, ky)):
            nn_1p.append(x)
            nn_2p.append(y)


KXX=2*np.pi*nn_1/LP
KYY= 2*(2*np.pi*nn_2/LP - np.pi*nn_1/LP)/np.sqrt(3)


nn_1pp=np.array(nn_1p)
nn_2pp=np.array(nn_2p)

KX=(2*np.pi*nn_1pp/LP)/LM
KY= (2*(2*np.pi*nn_2pp/LP - np.pi*nn_1pp/LP)/np.sqrt(3))/LM

#Making the sampling lattice commensurate with the MBZ
fact=K[2][1]/np.max(KY)
KX=KX*fact
KY=KY*fact
Npoi=np.size(KX)
print("effective number of points...", Npoi)

KQ=[]
for i in range(Npoi):
    for j in range(Npoi):
         KQ.append([round(KX[j]+KX[i], 8),round(KY[j]+KY[i], 8)])
# plt.scatter(KX,KY)
# plt.show()
KQarr=np.array(KQ)
print("Kq non unique grid ",np.shape(KQarr))

#unique_data =np.array( [list(x) for x in set(tuple(x) for x in KQ)])
unique_data =np.array( [list(i) for i in set(tuple(i) for i in KQ)])
print("Kq grid unique",np.shape(unique_data))
print("K grid ",np.shape(KX))
KQX=unique_data[:,0]
KQY=unique_data[:,1]


Npoi_Q=np.shape(KQY)[0]
#for debugging purposes

#c2 rotated grid
th1=np.pi
Rotth=np.array([[np.cos(th1),-np.sin(th1)],[np.sin(th1),np.cos(th1)]]) #rotation matrix +thet
KXc2=KX*Rotth[0,0]+KY*Rotth[0,1]
KYc2=KX*Rotth[1,0]+KY*Rotth[1,1]
n1c2=np.zeros(Npoi)
for i in range(Npoi):
    #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
    n1c2[i]=np.argmin( (KX-KXc2[i])**2 +(KY-KYc2[i])**2)


#c2 rotated grid
th1=np.pi
Rotth=np.array([[np.cos(th1),-np.sin(th1)],[np.sin(th1),np.cos(th1)]]) #rotation matrix +thet
KQXc2=KQX*Rotth[0,0]+KQY*Rotth[0,1]
KQYc2=KQX*Rotth[1,0]+KQY*Rotth[1,1]
n1Qc2=np.zeros(Npoi)
for i in range(Npoi):
    #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
    n1Qc2[i]=np.argmin( (KQX-KQXc2[i])**2 +(KQY-KQYc2[i])**2)

#c3 rotated grid
th1=2*np.pi/3
Rotth=np.array([[np.cos(th1),-np.sin(th1)],[np.sin(th1),np.cos(th1)]]) #rotation matrix +thet
KXc3=KX*Rotth[0,0]+KY*Rotth[0,1]
KYc3=KX*Rotth[1,0]+KY*Rotth[1,1]
n1c3=np.zeros(Npoi)
for i in range(Npoi):
    #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
    n1c3[i]=np.argmin( (KX-KXc3[i])**2 +(KY-KYc3[i])**2)

#print(eigsystem(GM,GM, 1, nbands, n1, n2)[0][0])

e=time.time()
print("finished sampling in reciprocal space....", np.shape(KY))
print("time for sampling was...",e-s)


Ik=[]
for j in range(Npoi):
    indmin=np.argmin(np.sqrt((KQX-KX[j])**2+(KQY-KY[j])**2))
    Ik.append(indmin)



Ikc2=[]
for j in range(Npoi):
    indmin=np.argmin(np.sqrt((KQXc2-KXc2[j])**2+(KQYc2-KYc2[j])**2))
    Ikc2.append(indmin)


####################################################################################
####################################################################################
####################################################################################
#
#     EVALUATING IN THE FBZ
#
####################################################################################
####################################################################################
####################################################################################

s=time.time()

Nsamp=int(sys.argv[2])

Ene_valley_plus_a=np.empty((0))
Ene_valley_min_a=np.empty((0))
psi_plus_a=[]
psi_min_a=[]

print("starting dispersion ..........")
# for l in range(Nsamp*Nsamp):
Nsms=[]
for l in range(Npoi_Q):

    E1,wave1, N1=eigsystem(KQX[l],KQY[l], 1, nbands, n1, n2)
    E2,wave2, Nm1=eigsystem(KQXc2[l],KQYc2[l], -1, nbands, n1, n2)
    Nsms.append([N1,Nm1])

    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
    Ene_valley_min_a=np.append(Ene_valley_min_a,E2)

    psi_plus_a.append(wave1)
    psi_min_a.append(wave2)

##relevant wavefunctions and energies for the + valley
psi_plus=np.array(psi_plus_a)
psi_plus_conj=np.conj(np.array(psi_plus_a))
Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi_Q,nbands])

#modified wavefunctions
# muz_sgx_psi_plus=np.zeros(np.shape(psi_plus)) +1j
muz_psi_plus=np.zeros(np.shape(psi_plus)) +1j*1e-30
sgx_muz_psi_plus=np.zeros(np.shape(psi_plus)) +1j*1e-30
sgy_muz_psi_plus=np.zeros(np.shape(psi_plus)) +1j*1e-30

muz_psi_plus[:,:,:,0,:,:]=psi_plus[:,:,:,0,:,:]
muz_psi_plus[:,:,:,1,:,:]=-psi_plus[:,:,:,1,:,:]

sgx_muz_psi_plus[:,:,:,:,0,:]=muz_psi_plus[:,:,:,:,1,:]
sgx_muz_psi_plus[:,:,:,:,1,:]=muz_psi_plus[:,:,:,:,0,:]

sgy_muz_psi_plus[:,:,:,:,0,:]=-1j*muz_psi_plus[:,:,:,:,1,:]
sgy_muz_psi_plus[:,:,:,:,1,:]=1j*muz_psi_plus[:,:,:,:,0,:]

# muz_sgx_psi_plus[:,:,:,:,0,:]=psi_plus[:,:,:,:,1,:]
# muz_sgx_psi_plus[:,:,:,:,1,:]=psi_plus[:,:,:,:,0,:]
# muz_sgx_psi_plus[:,:,:,0,:,:]=muz_sgx_psi_plus[:,:,:,0,:,:]
# muz_sgx_psi_plus[:,:,:,1,:,:]=-muz_sgx_psi_plus[:,:,:,1,:,:]
# print("testing order of prod... ",np.mean(muz_sgx_psi_plus- sgx_muz_psi_plus))

#relevant wavefunctions and energies for the - valley
psi_min=np.array(psi_min_a)
psi_min_conj=np.conj(np.array(psi_min_a))
Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi_Q,nbands])


#modified wavefunction
muz_psi_min=np.zeros(np.shape(psi_min)) +1j*1e-30
sgx_muz_psi_min=np.zeros(np.shape(psi_min)) +1j*1e-30
sgy_muz_psi_min=np.zeros(np.shape(psi_min)) +1j*1e-30

muz_psi_min[:,:,:,0,:,:]=psi_min[:,:,:,0,:,:]
muz_psi_min[:,:,:,1,:,:]=-psi_min[:,:,:,1,:,:]

sgx_muz_psi_min[:,:,:,:,0,:]=muz_psi_min[:,:,:,:,1,:]
sgx_muz_psi_min[:,:,:,:,1,:]=muz_psi_min[:,:,:,:,0,:]

sgy_muz_psi_min[:,:,:,:,0,:]=-1j*muz_psi_min[:,:,:,:,1,:]
sgy_muz_psi_min[:,:,:,:,1,:]=1j*muz_psi_min[:,:,:,:,0,:]

e=time.time()
print("finished dispersion ..........")
print("time elapsed for dispersion ..........", e-s)
print("shape of the wavefunctions...", np.shape(psi_plus))
# plt.scatter(XsLatt,YsLatt, c=Z[:,:,1])
# plt.show()
s=time.time()
print("calculating tensor that stores the overlaps........")
#for the plus valley
Lambda_Tens_plus=np.tensordot(psi_plus_conj,psi_plus, axes=([1,2,3,4],[1,2,3,4])) 
# Lambda_Tens_plus_muz=np.tensordot(psi_plus_conj,muz_psi_plus, axes=([1,2,3,4],[1,2,3,4])) 
# Lambda_Tens_plus_sgx_muz=np.tensordot(psi_plus_conj,sgx_muz_psi_plus, axes=([1,2,3,4],[1,2,3,4])) 
# Lambda_Tens_plus_sgy_muz=np.tensordot(psi_plus_conj,sgy_muz_psi_plus, axes=([1,2,3,4],[1,2,3,4])) 

Lambda_Tens_min=np.tensordot(psi_min_conj,psi_min, axes=([1,2,3,4],[1,2,3,4])) 
# Lambda_Tens_min_muz=np.tensordot(psi_min_conj,muz_psi_min, axes=([1,2,3,4],[1,2,3,4])) 
# Lambda_Tens_min_sgx_muz=np.tensordot(psi_min_conj,sgx_muz_psi_min, axes=([1,2,3,4],[1,2,3,4])) 
# Lambda_Tens_min_sgy_muz=np.tensordot(psi_min_conj,sgy_muz_psi_min, axes=([1,2,3,4],[1,2,3,4])) 

VV=np.array(Vertices_list+[Vertices_list[0]])

for i in range(nbands):
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX,KY, s=30, c=Ene_valley_plus[Ik,i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()
for i in range(nbands):
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KXc2,KYc2, s=30, c=Ene_valley_min[Ik,i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()

print( "tensorshape",np.shape(Lambda_Tens_plus) )

Ene_valley_plus_a=np.empty((0))
Ene_valley_min_a=np.empty((0))
psi_plus_a=[]
psi_min_a=[]
print("starting dispersion ..........")

# GM1p=-GM1
# GM2p=-GM2

# GM1=GM1p
# GM2=GM2p

# for l in range(Nsamp*Nsamp):
for l in range(Npoi_Q):

    E1,wave1,N1=eigsystem(KQXc2[l],KQYc2[l], 1, nbands, -n1, -n2)
    E2,wave2,Nm1=eigsystem(KQX[l],KQY[l], -1, nbands, -n1, -n2)

    if( Nsms[l][1]!=N1 or Nsms[l][0]!=Nm1):
        print("fd")

    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
    Ene_valley_min_a=np.append(Ene_valley_min_a,E2)

    psi_plus_a.append(wave1)
    psi_min_a.append(wave2)
# GM1p=-GM1
# GM2p=-GM2

# GM1=GM1p
# GM2=GM2p
##relevant wavefunctions and energies for the + valley
psi_plusc2=np.array(psi_plus_a)
psi_plus_conjc2=np.conj(np.array(psi_plus_a))
Ene_valley_plusc2= np.reshape(Ene_valley_plus_a,[Npoi_Q,nbands])

#relevant wavefunctions and energies for the - valley
psi_minc2=np.array(psi_min_a)
psi_min_conjc2=np.conj(np.array(psi_min_a))
Ene_valley_minc2= np.reshape(Ene_valley_min_a,[Npoi_Q,nbands])
e=time.time()
print("finished dispersion ..........")
print("time elapsed for dispersion ..........", e-s)

print("testing complex conjugation wavefunction, total plus... ",np.mean(psi_plus_conj[Ik,:,:,:,:,:]- psi_minc2[Ikc2,:,:,:,:,:]) )
print("testing complex conjugation wavefunction, total minus... ",np.mean(psi_min_conj[Ik,:,:,:,:,:]- psi_plusc2[Ikc2,:,:,:,:,:]) )

print("testing complex conjugation wavefunction, index one plus... ",np.mean(psi_plus_conj[Ik,:,:,:,:,1]- psi_minc2[Ikc2,:,:,:,:,1])  )
print("testing complex conjugation wavefunction, index one minus... ",np.mean(psi_min_conj[Ik,:,:,:,:,1]- psi_plusc2[Ikc2,:,:,:,:,1]) )

print("testing complex conjugation wavefunction, index two plus... ",np.mean(psi_plus_conj[Ik,:,:,:,:,2]- psi_minc2[Ikc2,:,:,:,:,2])  )
print("testing complex conjugation wavefunction, index two minus... ",np.mean(psi_min_conj[Ik,:,:,:,:,2]- psi_plusc2[Ikc2,:,:,:,:,2]) )


print("inner product test")

# print("testing normalization wavefunction, index one plus... ",np.mean(np.abs(np.tensordot( np.conj(psi_plus) , psi_plus,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,1,Ik,1]))**2)
# print("testing normalization wavefunction, index one plus... ",np.mean(np.abs(np.tensordot( np.conj(psi_min) , psi_min,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,1,Ik,1]))**2)
# print("testing normalization wavefunction, index one plus... ",np.mean((np.tensordot( np.conj(psi_plus) , psi_plus,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,1,Ik,1])))
# print("testing normalization wavefunction, index one plus... ",np.mean((np.tensordot( np.conj(psi_min) , psi_min,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,1,Ik,1]))) 



print("testing complex conjugation wavefunction, index one plus... ",np.mean(np.tensordot( np.conj(psi_plus_conj) , psi_minc2,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,1,Ikc2,1]))
print("testing complex conjugation wavefunction, index one minus... ",np.mean(np.tensordot( np.conj(psi_min_conj) , psi_plusc2,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,1,Ikc2,1]))


print("testing complex conjugation wavefunction, index one plus... ",np.mean(np.tensordot( np.conj(psi_plus_conj) , psi_minc2,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,2,Ikc2,2]))
print("testing complex conjugation wavefunction, index one minus... ",np.mean(np.tensordot( np.conj(psi_min_conj) , psi_plusc2,  axes=([1,2,3,4],[1,2,3,4]) )[Ik,2,Ikc2,2]))



print("todos los mimso")

print("testing complex conjugation wavefunction, total plus... ",np.mean(psi_plus_conj) )
print("testing complex conjugation wavefunction, total plus... ",np.mean( psi_minc2) )
print("testing complex conjugation wavefunction, total minus... ",np.mean(psi_min_conj) )
print("testing complex conjugation wavefunction, total minus... ",np.mean(psi_plusc2) )

print("testing complex conjugation wavefunction, index one plus... ",np.mean(psi_plus_conj[Ik,:,:,:,:,1])  )
print("testing complex conjugation wavefunction, index one plus... ",np.mean(psi_minc2[Ikc2,:,:,:,:,1])  )
print("testing complex conjugation wavefunction, index one minus... ",np.mean(psi_min_conj[Ik,:,:,:,:,1]) )
print("testing complex conjugation wavefunction, index one minus... ",np.mean(psi_plusc2[Ikc2,:,:,:,:,1]) )

print("testing complex conjugation wavefunction, index two plus... ",np.mean(psi_plus_conj[Ik,:,:,:,:,2])  )
print("testing complex conjugation wavefunction, index two plus... ",np.mean( psi_minc2[Ikc2,:,:,:,:,2])  )
print("testing complex conjugation wavefunction, index two minus... ",np.mean(psi_min_conj[Ik,:,:,:,:,2]) )
print("testing complex conjugation wavefunction, index two minus... ",np.mean(psi_plusc2[Ikc2,:,:,:,:,2]) )


print("shape of the wavefunctions...", np.shape(psi_plus))
# plt.scatter(XsLatt,YsLatt, c=Z[:,:,1])
# plt.show()
s=time.time()
print("calculating tensor that stores the overlaps........")
Lambda_Tens_plusc2=np.tensordot(psi_plus_conjc2,psi_plusc2, axes=([1,2,3,4],[1,2,3,4])) 
Lambda_Tens_minc2=np.tensordot(psi_min_conjc2,psi_minc2, axes=([1,2,3,4],[1,2,3,4]))
print( "tensorshape 2",np.shape(Lambda_Tens_plus) )

print("testing form factors, complex conjugation ", np.mean(np.conj(Lambda_Tens_plus)-Lambda_Tens_minc2) )


SigLminconjSig=np.zeros(np.shape(Lambda_Tens_minc2)) +1j*1e-30
LminconjSig=np.zeros(np.shape(Lambda_Tens_minc2))+1j*1e-30

LminconjSig[:,:,:,0]=np.conj(Lambda_Tens_minc2[:,:,:,0])
LminconjSig[:,:,:,1]=-np.conj(Lambda_Tens_minc2[:,:,:,1])
SigLminconjSig[:,0,:,:]=LminconjSig[:,0,:,:]
SigLminconjSig[:,1,:,:]=-LminconjSig[:,1,:,:]

print("calculating tensor that stores the overlaps........")



SigLminconjSig=np.zeros(np.shape(Lambda_Tens_min)) +1j*1e-30
LminconjSig=np.zeros(np.shape(Lambda_Tens_min))+1j*1e-30
LminconjSig[:,:,:,1]=(Lambda_Tens_min)[:,:,:,0]
LminconjSig[:,:,:,0]=(Lambda_Tens_min)[:,:,:,1]
SigLminconjSig[:,1,:,:]=LminconjSig[:,0,:,:]
SigLminconjSig[:,0,:,:]=LminconjSig[:,1,:,:]


print("calculating tensor that stores the overlaps........")



with open('Energies_plus_'+str(Nsamp)+'.npy', 'wb') as f:
    np.save(f, Ene_valley_plus)
with open('Energies_min_'+str(Nsamp)+'.npy', 'wb') as f:
    np.save(f, Ene_valley_min)
with open('Overlap_plus_'+str(Nsamp)+'.npy', 'wb') as f:
    np.save(f, Lambda_Tens_plus)
with open('Overlap_min_'+str(Nsamp)+'.npy', 'wb') as f:
    np.save(f, Lambda_Tens_min)



print("Loading tensor ..........")
with open('Energies_plus_'+str(Nsamp)+'.npy', 'rb') as f:
    Ene_valley_plus=np.load(f)
with open('Energies_min_'+str(Nsamp)+'.npy', 'rb') as f:
    Ene_valley_min=np.load(f)
with open('Overlap_plus_'+str(Nsamp)+'.npy', 'rb') as f:
    Lambda_Tens_plus=np.load(f)
with open('Overlap_min_'+str(Nsamp)+'.npy', 'rb') as f:
    Lambda_Tens_min=np.load(f)


Z= Ene_valley_plus


band_max=np.max(Z)
band_min=np.min(Z)
band_max_FB=np.max(Z[:,1:3])
band_min_FB=np.min(Z[:,1:3])

###infinitesimal to be added to the bubble calculation
# eta=np.mean( np.abs( np.diff( Ene_valley_plus[:,:,2].flatten() )  ) )/2
eta=np.mean( np.abs( np.diff( Ene_valley_plus[:,2].flatten() )  ) )/2
print("minimum energy and maximum energy between all the bands considered........",band_min, band_max)
print("minimum energy and maximum energy for flat bands........",band_min_FB, band_max_FB)



VV=np.array(Vertices_list+[Vertices_list[0]])

# for i in range(nbands):
#     plt.plot(VV[:,0],VV[:,1])
#     plt.scatter(KX,KY, s=10, c=Ene_valley_plus[:,i])
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()
# for i in range(nbands):
#     plt.plot(VV[:,0],VV[:,1])
#     plt.scatter(KX,KY, s=10, c=Ene_valley_min[:,i])
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()


