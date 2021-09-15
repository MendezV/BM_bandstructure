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
tp=1 
def eigsystem2(kx, ky, xi, nbands, n1, n2):
    ed=-tp*(np.cos(LM1[0]*kx+LM1[1]*ky)+np.cos(LM2[0]*kx+LM2[1]*ky)+np.cos((LM2[0]-LM1[0])*kx+(LM2[1]-LM1[1])*ky))-mu
    return [ed,ed,ed,ed],ed


# c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
# plt.show()

def eigsystem(kx, ky, xi, nbands, n1, n2):
    #we diagonalize a matrix made up
    qx_dif = kx+GM1[0]*n1+GM2[0]*n2-xi*Mpoint[0]
    qy_dif = ky+GM1[1]*n1+GM2[1]*n2-xi*Mpoint[1]
    vals = np.sqrt(qx_dif**2+qy_dif**2)
    ind_to_sum = np.where(vals <= 4*GM) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
    n1_val = n1[ind_to_sum] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
    n2_val = n2[ind_to_sum] #
    N = np.shape(ind_to_sum)[1] ##number of indices for which the condition above is satisfied

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
    # psi=np.zeros([np.shape(n1)[0],np.shape(n1)[1],nbands ,4], dtype=np.cdouble) #4 corresponds to layer and sublattice indices and shape n1 corresponds to the Q processes taken 
    # psi[ind_to_sum]=np.reshape(np.array(Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]) , [N, nbands,4 ])
    #MANUAL ARRAY CASTING
    
    # psi_p=np.zeros([np.shape(n1)[0],np.shape(n1)[1],4], dtype=np.cdouble)
    # psi=np.zeros([nbands,np.shape(n1)[0]*np.shape(n1)[1]*4], dtype=np.cdouble)
    # # print(np.shape(Eigvect), "shapeeig")
    # for nband in range(nbands):
    #     psi_p[ind_to_sum] = np.reshape(np.array(Eigvect[:,2*N-int(nbands/2)+nband]) , [N, 4 ], order='F')
    #     psi[nband,:]=psi_p.flatten()

    psi_p=np.zeros([np.shape(n1)[0],np.shape(n1)[1],2,2]) +0*1j
    psi=np.zeros([nbands,np.shape(n1)[0]*np.shape(n1)[1]*4], dtype=np.cdouble)

    for nband in range(nbands):
        psi_p[ind_to_sum] = np.reshape(        np.array(np.reshape(Eigvect[:,2*N-int(nbands/2)+nband] , [4, N])  ).T, [N, 2, 2] )    
        psi[nband,:]=np.reshape(psi_p,[np.shape(n1)[0]*np.shape(n1)[1]*4]).flatten()


    return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0,psi
    #  return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]


#print(eigsystem(GM,GM, 1, nbands, n1, n2)[0][0])

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
n_1=np.arange(0,Nsamp,1)
n_2=np.arange(0,Nsamp,1)
k1=np.array([1,0])
k2=np.array([1/2,np.sqrt(3)/2])
n_1p,n_2p=np.meshgrid(n_1,n_2)
n_1p_flat=n_1p.flatten()
n_2p_flat=n_2p.flatten()
G1=k1*(Nsamp)
G2=k2*(Nsamp)

XsLatt=(GM/Nsamp)*(k1[0]*n_1p+k2[0]*n_2p).T
YsLatt=(GM/Nsamp)*(k1[1]*n_1p+k2[1]*n_2p).T
Ene_valley_plus_a=np.empty((0))
Ene_valley_min_a=np.empty((0))
psi_plus_a=[]
psi_min_a=[]

print("starting dispersion ..........")
for l in range(Nsamp*Nsamp):
    i=int(l%Nsamp)
    j=int((l-i)/Nsamp)
    #print(XsLatt[i,j],YsLatt[i,j])
    
    E1,wave1=eigsystem(XsLatt[i,j],YsLatt[i,j], 1, nbands, n1, n2)
    E2,wave2=eigsystem(XsLatt[i,j],YsLatt[i,j], -1, nbands, n1, n2)

    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
    Ene_valley_min_a=np.append(Ene_valley_min_a,E2)

    psi_plus_a.append(wave1)
    psi_min_a.append(wave2)

##relevant wavefunctions and energies for the + valley
psi_plus=np.array(psi_plus_a)
psi_plus_r=np.reshape(np.array(psi_plus_a),[Nsamp,Nsamp,nbands,np.shape(n1)[0]*np.shape(n1)[1]*4])
psi_plus_conj=np.conj(np.array(psi_plus_a))
psi_plus_r_conj=np.conj(np.array(psi_plus_r))
Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Nsamp,Nsamp,nbands])

#relevant wavefunctions and energies for the - valley
psi_min=np.array(psi_min_a)
psi_min_r=np.reshape(np.array(psi_min_a),[Nsamp,Nsamp,nbands,np.shape(n1)[0]*np.shape(n1)[1]*4])
psi_min_conj=np.conj(np.array(psi_min_a))
psi_min_r_conj=np.conj(np.array(psi_min_r))
Ene_valley_min= np.reshape(Ene_valley_min_a,[Nsamp,Nsamp,nbands])

 ## for testing different things
Z= np.reshape(Ene_valley_plus_a,[Nsamp,Nsamp,nbands])

e=time.time()
print("finished dispersion ..........")
print("time elapsed for dispersion ..........", e-s)
print("shape of the wavefunctions...", np.shape(psi_plus))
# plt.scatter(XsLatt,YsLatt, c=Z[:,:,1])
# plt.show()

s=time.time()
# print("calculating tensor that stores the overlaps........")
# Lambda_Tens_plus=np.tensordot(psi_plus_conj,psi_plus_r, axes=(2,3))
# Lambda_Tens_min=np.tensordot(psi_min_conj,psi_min_r, axes=(2,3))

print("calculating tensor that stores the overlaps........")
Lambda_Tens_p_plus=np.tensordot(psi_plus_r_conj,psi_plus_r, axes=(3,3))
Lambda_Tens_p_min=np.tensordot(psi_min_r_conj,psi_min_r, axes=(3,3))
# Lambda_Tens_pm_plus=np.tensordot(psi_plus_r,psi_plus_r_conj, axes=(3,3))
# Lambda_Tens_pm_plus=np.zeros([Nsamp , Nsamp, nbands,Nsamp, Nsamp, nbands])+0*1j
# Lambda_Tens_pm_min=np.zeros([Nsamp , Nsamp, nbands,Nsamp, Nsamp, nbands])+0*1j

# for l in range(nbands):
#     for k in range(nbands):
#         for s in range(Nsamp):
#             for r in range(Nsamp):
#                 for m in range(Nsamp):
#                     for y in range(Nsamp):
#                         for j in range(np.shape(n1)[0]*np.shape(n1)[1]*4):
#                             Lambda_Tens_pm_plus[y,m,k,r,s,l]=Lambda_Tens_pm_plus[y,m,k,r,s,l]+psi_plus_r_conj[y,m,k,j]*psi_plus_r[r,s,l,j]
#                             Lambda_Tens_pm_min[y,m,k,r,s,l]=Lambda_Tens_pm_min[y,m,k,r,s,l]+psi_min_r_conj[y,m,k,j]*psi_min_r[r,s,l,j]


# Lambda_Tens_pm_min=np.tensordot(psi_min_r,psi_min_r_conj, axes=(3,3))
print( "tensorshape",np.shape(Lambda_Tens_p_plus) )
e=time.time()
print("time elapsed for tensor ..........", e-s)

print("Saving tensor ..........")
with open('Energies_plus_'+str(Nsamp)+'.npy', 'wb') as file1:
    np.save(file1, Ene_valley_plus)

with open('Energies_min_'+str(Nsamp)+'.npy', 'wb') as file1:
    np.save(file1, Ene_valley_min)

# with open('Overlap_plus_'+str(Nsamp)+'.npy', 'wb') as file2:
#     np.save(file2, Lambda_Tens_plus)

# with open('Overlap_min_'+str(Nsamp)+'.npy', 'wb') as file2:
#     np.save(file2, Lambda_Tens_min)

with open('Overlap_p_plus_'+str(Nsamp)+'.npy', 'wb') as file2:
    np.save(file2, Lambda_Tens_p_plus)

with open('Overlap_p_min_'+str(Nsamp)+'.npy', 'wb') as file2:
    np.save(file2, Lambda_Tens_p_min)

# with open('Overlap_pm_plus_'+str(Nsamp)+'.npy', 'wb') as file2:
#     np.save(file2, Lambda_Tens_pm_plus)

# with open('Overlap_pm_min_'+str(Nsamp)+'.npy', 'wb') as file2:
#     np.save(file2, Lambda_Tens_pm_min)

print("Loading tensor ..........")
Ene_valley_plus=np.load('Energies_plus_'+str(Nsamp)+'.npy')
Ene_valley_min=np.load('Energies_min_'+str(Nsamp)+'.npy')
# Lambda_Tens_plus=np.load('Overlap_plus_'+str(Nsamp)+'.npy')
# Lambda_Tens_min=np.load('Overlap_min_'+str(Nsamp)+'.npy')
Lambda_Tens_p_plus=np.load('Overlap_p_plus_'+str(Nsamp)+'.npy')
Lambda_Tens_p_min=np.load('Overlap_p_min_'+str(Nsamp)+'.npy')
# Lambda_Tens_pm_plus=np.load('Overlap_pm_plus_'+str(Nsamp)+'.npy')
# Lambda_Tens_pm_min=np.load('Overlap_pm_min_'+str(Nsamp)+'.npy')

Z= Ene_valley_plus


band_max=np.max(Z)
band_min=np.min(Z)
band_max_FB=np.max(Z[:,:,1:3])
band_min_FB=np.min(Z[:,:,1:3])

###infinitesimal to be added to the bubble calculation
eta=np.mean( np.abs( np.diff( Ene_valley_plus[:,:,2].flatten() )  ) )/2

print("minimum energy and maximum energy between all the bands considered........",band_min, band_max)
print("minimum energy and maximum energy for flat bands........",band_min_FB, band_max_FB)


##for testing shifts
n_1pp=(n_1p+int(Nsamp/2))%Nsamp
n_2pp=(n_2p+int(Nsamp/2))%Nsamp

Z1=Z[:,:,1]
Z2=Z[n_1pp,n_2pp,1]
#####
# print(band_max, band_min)
# plt.scatter(XsLatt,YsLatt, c=Z2)
# plt.show()



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

Vertices_list, Gamma, K, Kp, M, Mp=FBZ_points(G1,G2)  #FOR RESHAPING THE STORED ARRAY
Vertices_list_MBZ, Gamma_MBZ, K_MBZ, Kp_MBZ, M_MBZ, Mp_MBZ=FBZ_points(GM1,GM2) #DESCRIBING THE MBZ 



##HIGH SYMMETRY POINTS FOR THE CONVENTIONAL UNIT CELL
Gamma_CBZ=[[0,0],G1,G2,G1+G2]

vor = Voronoi(Gamma_CBZ)

K_CBZ=(GM/Nsamp)*np.array(vor.vertices)[1,:]
Kp_CBZ=(GM/Nsamp)*np.array(vor.vertices)[0,:]
M_CBZ=(GM/Nsamp)*np.array(Gamma_CBZ)[3,:]/2
Mp_li=[]
Mp_li.append(np.array(Gamma_CBZ)[1,:]/2)
Mp_li.append(np.array(Gamma_CBZ)[2,:]/2)
Mp_li.append(np.array(Gamma_CBZ)[2,:]+np.array(Gamma_CBZ)[1,:]/2)
Mp_li.append(np.array(Gamma_CBZ)[1,:]+np.array(Gamma_CBZ)[2,:]/2)
Mp_CBZ=(GM/Nsamp)*np.array(Mp_li)
G_CBZ=(GM/Nsamp)*np.array(Gamma_CBZ)



##DIFFERENT HEXAGONS USED TO DELIMIT THE REGIONS 
## THAT ARE GOING TO BE REARANGED IN THE CONVENTIONAL UNIT CELL
VL=np.array(Vertices_list)

VL2=np.array(VL)
VL2[:,0]=VL[:,0]+G1[0]
VL2[:,1]=VL[:,1]+G1[1]
Vertices_list2=list(VL2)

VL3=np.array(VL)
VL3[:,0]=VL[:,0]+G2[0]
VL3[:,1]=VL[:,1]+G2[1]
Vertices_list3=list(VL3)

VL4=np.array(VL)
VL4[:,0]=VL[:,0]+G2[0]+G1[0]
VL4[:,1]=VL[:,1]+G2[1]+G1[1]
Vertices_list4=list(VL4)


##METHOD THAT CHECKS WHETHER A POINT IS INSIDE A GIVEN CONVEX POLYGON
def is_within_polygon(polygon, point):
    A = []
    B = []
    C = []  
    for i in range(len(polygon)):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % len(polygon)]

        # calculate A, B and C
        a = -(p2[1] - p1[1])
        b = p2[0] - p1[0]
        c = -(a * p1[0] + b * p1[1])
        
        A.append(a)
        B.append(b)
        C.append(c)

    D = []
    for i in range(len(A)):
        d = A[i] * point[0] + B[i] * point[1] + C[i]
        D.append(d)

    t1 = np.all([d >= 0 for d in D])
    t2 = np.all([d <= 0 for d in D])
    x=t1 or t2
    return x



XsLatt_hex=(k1[0]*n_1p+k2[0]*n_2p).T
YsLatt_hex=(k1[1]*n_1p+k2[1]*n_2p).T
for l in range(Nsamp*Nsamp):
    i=int(l%Nsamp)
    j=int((l-i)/Nsamp)
    point=[ XsLatt_hex[i,j],YsLatt_hex[i,j] ]
    if is_within_polygon(Vertices_list, point ):
        XsLatt_hex[i,j]=XsLatt_hex[i,j]
        YsLatt_hex[i,j]=YsLatt_hex[i,j]
    elif is_within_polygon(Vertices_list2,  point ):
        XsLatt_hex[i,j]=XsLatt_hex[i,j]-G1[0]
        YsLatt_hex[i,j]=YsLatt_hex[i,j]-G1[1]
    elif is_within_polygon(Vertices_list3, point):
        XsLatt_hex[i,j]=XsLatt_hex[i,j]-G2[0]
        YsLatt_hex[i,j]=YsLatt_hex[i,j]-G2[1]
    elif is_within_polygon(Vertices_list4,  point):
        XsLatt_hex[i,j]=XsLatt_hex[i,j]-G2[0]-G1[0]
        YsLatt_hex[i,j]=YsLatt_hex[i,j]-G2[1]-G1[1]
    else:
        XsLatt_hex[i,j]=XsLatt_hex[i,j]-G1[0]
        YsLatt_hex[i,j]=YsLatt_hex[i,j]-G1[1]

VV=np.array(Vertices_list_MBZ+[Vertices_list_MBZ[0]])
KX_in=(GM/Nsamp)*XsLatt_hex
KY_in=(GM/Nsamp)*YsLatt_hex

### saving fermi surfaces of the bands that where considered
for i in range(nbands):
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX_in,KY_in, s=10, c=Ene_valley_plus[:,:,i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()
for i in range(nbands):
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX_in,KY_in, s=10, c=Ene_valley_min[:,:,i])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()


KX_rhomb=XsLatt
KY_rhomb=YsLatt
####################################################################################
####################################################################################
####################################################################################
#
#      PATH IN RECIPROCAL SPACE
#
####################################################################################
####################################################################################
####################################################################################



#linear parametrization accross different points in the BZ
def linpam(Kps,Npoints_q):
    Npoints=len(Kps)
    t=np.linspace(0, 1, Npoints_q)
    linparam=np.zeros([Npoints_q*(Npoints-1),2])
    for i in range(Npoints-1):
        linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
        linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

    return linparam


Nt=1000
kpath=linpam(VV,Nt)

L=[]
L=L+[Gamma_MBZ]+[K_MBZ[1]]
Nt=25
kpath2=linpam(L,Nt)

# plt.plot(kpath[:,0],kpath[:,1])
# plt.plot(kpath2[:,0],kpath2[:,1])
# plt.show()


# def findpath(Kps,KX,KY):

#     path=np.empty((0))
#     path_ij=np.empty((0))
#     pthK=[]
#     HSP_index=[]
#     counter_path=0
#     HSP_index.append(counter_path)
    
    
#     l=np.argmin(  (Kps[0][0]-KX)**2 + (Kps[0][1]-KY)**2 )
#     j=int(l%Nsamp)
#     i=int((l-j)/Nsamp)

    
#     path=np.append(path,int(l)) 
#     path_ij=np.append(path,np.array([i,j]))
#     pthK.append([KX[i,j],KY[i,j]])
    
    
#     amin=GM/Nsamp
#     k1=(GM/Nsamp)*np.array([1,0])
#     k2=(GM/Nsamp)*np.array([1/2,np.sqrt(3)/2])
#     nnlist=[[0,1],[1,0],[0,-1],[-1,0],[1,-1],[-1,1]]

#     NHSpoints=np.shape(Kps)[0]

    

#     for indhs in range(NHSpoints-1):

#         c=0
#         c2=0
        
        
#         dist=np.sqrt( (Kps[indhs+1][0]-KX[i,j])**2 + (Kps[indhs+1][1]-KY[i,j])**2)
#         while ( c2<1  and  dist>=0.8*amin):
#             dists=[]
#             KXnn=[]
#             KYnn=[]

#             dist_pre=dist
        
#             for nn in range(6): #coordination number is 6
#                 kxnn= KX[i,j]+nnlist[nn][0]*k1[0]+nnlist[nn][1]*k2[0]
#                 kynn= KY[i,j]+nnlist[nn][0]*k1[1]+nnlist[nn][1]*k2[1]
#                 di=np.sqrt( (Kps[indhs+1][0]-kxnn)**2 + (Kps[indhs+1][1]-kynn)**2)
#                 dists.append(di)
#                 KXnn.append(kxnn)
#                 KYnn.append(kynn)
            
#             dist=np.min(np.array(dists))
#             ind_min=np.argmin(np.array(dists))
            
            
#             l=np.argmin(  (KXnn[ind_min]-KX)**2 + (KYnn[ind_min]-KY)**2 )
#             j=int(l%Nsamp)
#             i=int((l-j)/Nsamp)

#             if dist_pre==dist:
#                 c2=c2+1
            

#             path=np.append(path,int(l))
#             path_ij=np.append(path,np.array([i,j]))
#             pthK.append([KX[i,j],KY[i,j]])
#             # print([KX[i,j],KY[i,j]],[Kps[indhs+1][0],Kps[indhs+1][1]], dist)

#             c=c+1
#             counter_path=counter_path+1
    
#         HSP_index.append(counter_path)
        
        
#     return path,np.array(pthK),HSP_index

# L2=[]
# L2=L2+[G_CBZ[0,:]]+[Kp_CBZ]


# L3=[]
# L3=L3+[Kp_CBZ]+[G_CBZ[0,:]]+[Mp_CBZ[0,:]]+[Kp_CBZ]

# L4=[]
# L4=L4+[K_CBZ]+[G_CBZ[3,:]]

# path,kpath,HSpoints=findpath(L3,XsLatt,YsLatt)
# Npath=np.size(path)
# # print(Npath,HSpoints)
# # print(Npath,HSpoints, np.shape(kpath),np.shape(path))
# plt.scatter(XsLatt,YsLatt, s=30, c='r' )
# plt.scatter(kpath[:,0],kpath[:,1], s=30, c='g' )
# plt.gca().set_aspect('equal')
# # plt.show()
# plt.savefig("path.png")
# plt.close()

# print("calculated path accross the FBZ.......")


# L4=[]
# L4=L4+[K_CBZ]+[G_CBZ[3,:]]

# path1,kpath1,HSpoints1=findpath(L4,XsLatt,YsLatt)
# Npath1=np.size(path1)
# # print(Npath1,HSpoints1)
# plt.scatter(XsLatt,YsLatt, s=30, c='r' )
# plt.scatter(kpath1[:,0],kpath1[:,1], s=30, c='g' )
# plt.gca().set_aspect('equal')

# L4=[]
# L4=L4+[G_CBZ[2,:]]+[M_CBZ]+[Kp_CBZ]
# path2,kpath2,HSpoints2=findpath(L4,XsLatt,YsLatt)
# Npath2=np.size(path2)
# # print(Npath2,HSpoints2)
# #plt.scatter(XsLatt,YsLatt, s=30, c='r' )
# plt.scatter(kpath2[:,0],kpath2[:,1], s=30, c='g' )
# plt.gca().set_aspect('equal')
# # plt.show()
# plt.savefig("path.png")
# plt.close()

# path=np.append(path1,path2)
# kpath=np.vstack((kpath1,kpath2))
# HSpoints=np.append(np.array(HSpoints1),np.array(HSpoints2)+Npath1-1)
# Npath=np.size(path)
# print("size of the path in BZ and index of HSP...",Npath,HSpoints)
# # print(kpath)
# plt.scatter(XsLatt,YsLatt, s=30, c='r' )
# plt.scatter(kpath[:,0],kpath[:,1], s=30, c='g' )
# plt.gca().set_aspect('equal')
# # plt.show()
# plt.savefig("path.png")
# plt.close()



####################################################################################
####################################################################################
####################################################################################
#
#      INTEGRALS
#
####################################################################################
####################################################################################
####################################################################################


#####INTEGRAND FUNCTION
dS_in=Vol_rec/((Nsamp)*(Nsamp)) #some points are repeated in my scheme
xi=1
def integrand(nx,ny,ekn,ekm,w,mu,T):
    edk=ekn-mu
    edkq=ekm[nx,ny]-mu

    #finite temp
    #nfk= 1/(np.exp(edk/T)+1)
    #nfkq= 1/(np.exp(edkq/T)+1)

    #zero temp
    nfk=np.heaviside(-edk,1.0) # at zero its 1
    nfkq=np.heaviside(-edkq,1.0) #at zero is 1
    eps=eta ##SENSITIVE TO DISPERSION

    fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
    return (fac_p)




####### COMPUTING BUBBLE FOR ALL MOMENTA AND FREQUENCIES SAMPLED


# maxomeg=(band_max-band_min)*1.5
# minomeg=band_min*(np.sign(band_min)*(-2))*0+0.00005
# maxomeg=10/1000#(band_max_FB-band_min_FB)*2.5
# minomeg=1e-14
# omegas=np.linspace(minomeg,maxomeg,Nomegs)
integ=[]
sb=time.time()

print("starting bubble.......")
omegas=[1e-14]
n_1pp_flat=(n_1p_flat+n_1p_flat[1])%Nsamp
n_2pp_flat=(n_2p_flat+n_2p_flat[1])%Nsamp
Lambda_Tens_plus_kq_k=np.array([Lambda_Tens_p_plus[n_1pp_flat[ss],n_2pp_flat[ss],:,n_1p_flat[ss],n_2p_flat[ss],:] for ss in range(Nsamp**2)])

plt.scatter(KX_rhomb,KY_rhomb,c=np.abs(Lambda_Tens_plus_kq_k[:,1,0])**2)
plt.colorbar()
plt.show()

plt.scatter(KX_rhomb,KY_rhomb,c=np.abs(Lambda_Tens_plus_kq_k[:,1,1])**2)
plt.colorbar()
plt.show()


path=np.arange(0,Nsamp*Nsamp)
for omegas_m_i in omegas:
    sd=[]
    sp=time.time()
    for l in path:  #for calculating only along path in FBZ
        bub=0
        j=int(l%Nsamp)
        i=int((l-j)/Nsamp)

        #for the integrand
        n_1pp=(n_1p+n_1p[i,j])%Nsamp
        n_2pp=(n_2p+n_2p[i,j])%Nsamp

        #for the form factors
        n_1pp_flat=(n_1p_flat+n_1p_flat[l])%Nsamp
        n_2pp_flat=(n_2p_flat+n_2p_flat[l])%Nsamp

        integrand_var=0
        #save one reshape below by reshaping n's
    
        # s=time.time()
       
        #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
        # Lambda_Tens_plus_kq_k=np.array([Lambda_Tens_plus[ss,:,n_1pp_flat[ss],n_2pp_flat[ss],:] for ss in range(Nsamp**2)])
        # Lambda_Tens_min_kq_k=np.array([Lambda_Tens_min[ss,:,n_1pp_flat[ss],n_2pp_flat[ss],:] for ss in range(Nsamp**2)])
        
        
        Lambda_Tens_plus_kq_k=np.array([Lambda_Tens_p_plus[n_1p_flat[ss],n_2p_flat[ss],:,n_1p_flat[ss],n_2p_flat[ss],:] for ss in range(Nsamp**2)])
        Lambda_Tens_min_kq_k=np.array([Lambda_Tens_p_min[n_1p_flat[ss],n_2p_flat[ss],:,n_1p_flat[ss],n_2p_flat[ss],:] for ss in range(Nsamp**2)])
        # Lambda_Tens_plus_k_kq=np.array([Lambda_Tens_pm_plus[n_1pp_flat[ss],n_2pp_flat[ss],:,n_1p_flat[ss],n_2p_flat[ss],:] for ss in range(Nsamp**2)])
        # Lambda_Tens_min_k_kq=np.array([Lambda_Tens_pm_min[n_1pp_flat[ss],n_2pp_flat[ss],:,n_1p_flat[ss],n_2p_flat[ss],:] for ss in range(Nsamp**2)])
        
        # print(np.shape(Lambda_Tens_plus_kq_k))
        e=time.time()
        # print("time to op: evaluation of kq shift",e-s)
        s=time.time()
        #####all bands for the + and - valley
        for nband in range(nbands):
            for mband in range(nbands):
                
                ek_n=Ene_valley_plus[:,:,nband]
                ek_m=Ene_valley_plus[:,:,mband]
                Lambda_Tens_plus_kq_k_nm=np.reshape(Lambda_Tens_plus_kq_k[:,nband,mband], [Nsamp,Nsamp])
                # Lambda_Tens_plus_k_kq_mn=np.reshape(Lambda_Tens_plus_k_kq[:,mband,nband], [Nsamp,Nsamp])
                # Lambda_Tens_plus_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                integrand_var=integrand_var+np.abs( Lambda_Tens_plus_kq_k_nm )**2 # *integrand(n_1pp,n_2pp,ek_n,ek_m,omegas_m_i,mu,T)
                # integrand_var=integrand_var+np.conj(Lambda_Tens_plus_k_kq_mn)-(Lambda_Tens_plus_kq_k_nm) #*integrand(n_1pp,n_2pp,ek_n,ek_m,omegas_m_i,mu,T)
                

                ek_n=Ene_valley_min[:,:,nband]
                ek_m=Ene_valley_min[:,:,mband]
                Lambda_Tens_min_kq_k_nm=np.reshape(Lambda_Tens_min_kq_k[:,nband,mband], [Nsamp,Nsamp])
                # Lambda_Tens_min_k_kq_mn=np.reshape(Lambda_Tens_min_k_kq[:,mband,nband], [Nsamp,Nsamp])
                # Lambda_Tens_min_kq_k_nm=int(nband==mband)   #TO SWITCH OFF THE FORM FACTORS
                integrand_var=integrand_var+np.abs( Lambda_Tens_min_kq_k_nm )**2 #*integrand(n_1pp,n_2pp,ek_n,ek_m,omegas_m_i,mu,T)
                # integrand_var=integrand_var+np.conj(Lambda_Tens_min_k_kq_mn)  -(Lambda_Tens_min_kq_k_nm) #*integrand(n_1pp,n_2pp,ek_n,ek_m,omegas_m_i,mu,T)
                

        e=time.time()
       
        bub=bub+np.sum(np.sum(integrand_var))*dS_in

        sd.append( bub )

    integ.append(sd)
    
integ_arr_no_reshape=np.array(integ)#/(8*Vol_rec) #8= 4bands x 2valleys

e=time.time()
print("finished bubble.......")
print("Time for bubble",e-sb)


plt.plot(VV[:,0],VV[:,1])
plt.scatter(KX_in,KY_in, s=10, c=np.log10(np.real(integ_arr_no_reshape)))
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.savefig("energycuttestreal.png")
plt.close()


plt.plot(VV[:,0],VV[:,1])
plt.scatter(KX_in,KY_in, s=10, c=np.abs(np.imag(integ_arr_no_reshape)))
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.savefig("energycuttestimag.png")
plt.close()


plt.scatter(KX_rhomb,KY_rhomb, s=10, c=np.log10(np.real(integ_arr_no_reshape)))
plt.gca().set_aspect('equal', adjustable='box')
plt.colorbar()
plt.savefig("energycuttestreal2.png")
plt.close()

