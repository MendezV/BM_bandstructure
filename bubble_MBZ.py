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
T=0.01*tp
mu=-1.5

def eigsystem(kx, ky, xi, nbands, n1, n2):
    ed=-tp*(np.cos(LM1[0]*kx+LM1[1]*ky)+np.cos(LM2[0]*kx+LM2[1]*ky)+np.cos((LM2[0]-LM1[0])*kx+(LM2[1]-LM1[1])*ky))-mu
    return [ed,ed],ed



def Disp(kx, ky,mu):
    ed=-tp*(np.cos(LM1[0]*kx+LM1[1]*ky)+np.cos(LM2[0]*kx+LM2[1]*ky)+np.cos((LM2[0]-LM1[0])*kx+(LM2[1]-LM1[1])*ky))
    return ed-mu

x = np.linspace(-GM, GM, 300)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)

band_max=np.max(Z)
band_min=np.min(Z)

# c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
# plt.show()

def eigsystem2(kx, ky, xi, nbands, n1, n2):
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
    return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]


print(eigsystem(GM,GM, 1, nbands, n1, n2)[0][0])
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

Nsamp=60
n_1=np.arange(0,Nsamp,1)
n_2=np.arange(0,Nsamp,1)
k1=np.array([1,0])
k2=np.array([1/2,np.sqrt(3)/2])
n_1p,n_2p=np.meshgrid(n_1,n_2)
G1=k1*Nsamp
G2=k2*Nsamp

XsLatt=(GM/Nsamp)*(k1[0]*n_1p+k2[0]*n_2p).T
YsLatt=(GM/Nsamp)*(k1[1]*n_1p+k2[1]*n_2p).T
S=np.empty((0))
for l in range(Nsamp*Nsamp):
    i=int(l%Nsamp)
    j=int((l-i)/Nsamp)
    #print(XsLatt[i,j],YsLatt[i,j])
    E=eigsystem(XsLatt[i,j],YsLatt[i,j], 1, nbands, n1, n2)[0][1]
    #print(E)
    S=np.append(S,E)

Z1= S.flatten() 
print(np.shape(XsLatt),np.shape(YsLatt),np.shape(Z1))
plt.scatter(XsLatt,YsLatt, c=np.reshape(Z1, np.shape(XsLatt)))
plt.show()

e=time.time()
print("time for dispersion", e-s)
####################################################################################
####################################################################################
####################################################################################
#
#      GEOMETRY AND RESHAPING ARRAYS
#
####################################################################################
####################################################################################
####################################################################################

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
plt.plot(VV[:,0],VV[:,1])
plt.scatter(KX_in,KY_in, s=30, c=np.reshape(Z1, np.shape(KX_in)))
plt.show()

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

####################################################################################
####################################################################################
####################################################################################
#
#      INTEGRALS
#
####################################################################################
####################################################################################
####################################################################################

dS_in=Vol_rec/(Nsamp*Nsamp)
xi=1
def integrand(qx,qy,kx,ky,w,mu,T):
    edk=eigsystem(kx, ky, xi, nbands, n1, n2)[0][1]
    edkq=eigsystem(kx+qx,ky+qy, xi, nbands, n1, n2)[0][1]
    nfk= 1/(np.exp(edk/T)+1)
    nfkq= 1/(np.exp(edkq/T)+1)
    eps=1e-1

    fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
    return (fac_p)


# plt.scatter(KX_in,KY_in, c=np.log10(abs(integrand(0.01,0.01,KX_in,KY_in,0.0001,mu,T) +1e-17)), s=30)
# plt.colorbar()
# plt.show()


# plt.scatter(KX_in,KY_in, c=np.abs(integrand(0.01,0.01,KX_in,KY_in,0.0001,mu,T)) , s=30)
# plt.colorbar()
# plt.show()


Nomegs=50
maxomeg=band_max*2
minomeg=0
omegas=np.linspace(0.01,maxomeg,Nomegs)
#omegas=np.logspace(-5,1,Nomegs)
t=np.arange(0,len(kpath2),1)
t_m,omegas_m=np.meshgrid(t,omegas)
integ=[]
for t_m_i in t:
    sd=[]
    print(t_m_i,"/",np.size(t))
    for omegas_m_i in omegas:
        sd.append( np.sum((integrand(kpath2[t_m_i,0],kpath2[t_m_i,1],KX_in,KY_in,omegas_m_i,mu,T)))*dS_in )
        #print(omegas_m_i,t_m_i)
    integ.append(sd)
integ_arr=np.array(integ)

limits_X=1
limits_Y=maxomeg
N_X=len(kpath2)
N_Y=Nomegs


plt.imshow(np.imag(integ_arr.T), origin='lower', aspect='auto')


ticks_X=5
ticks_Y=5
Npl_X=np.arange(0,N_X+1,int(N_X/ticks_X))
Npl_Y=np.arange(0,N_Y+1,int(N_Y/ticks_Y))
xl=np.round(np.linspace(0,limits_X,ticks_X+1),3)
yl=np.round(np.linspace(0,limits_Y,ticks_Y+1),3)

plt.xticks(Npl_X,xl)
plt.yticks(Npl_Y,yl)
plt.xlabel(r"$q_x$",size=16)
plt.ylabel(r"$\omega$",size=16)
#axhline(N_Y/2 -7, c='r')
#print(omegas[int(N_Y/2 -7)])
plt.colorbar()
plt.show()


plt.imshow(np.real(integ_arr.T), origin='lower', aspect='auto')


ticks_X=5
ticks_Y=5
Npl_X=np.arange(0,N_X+1,int(N_X/ticks_X))
Npl_Y=np.arange(0,N_Y+1,int(N_Y/ticks_Y))
xl=np.round(np.linspace(0,limits_X,ticks_X+1),3)
yl=np.round(np.linspace(0,limits_Y,ticks_Y+1),3)

plt.xticks(Npl_X,xl)
plt.yticks(Npl_Y,yl)
plt.xlabel(r"$q_x$",size=16)
plt.ylabel(r"$\omega$",size=16)
#axhline(N_Y/2 -7, c='r')
#print(omegas[int(N_Y/2 -7)])
plt.colorbar()
plt.show()


plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , np.imag(integ_arr[:,0]), s=30)
plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , np.imag(integ_arr[:,0]))
plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , np.real(integ_arr[:,0]), s=30)
plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , np.real(integ_arr[:,0]))


# plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,-1])
# plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,-1])

plt.show()

