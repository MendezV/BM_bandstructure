#integrates a dispersion to get the polarization bubble

import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys
import scipy
np.set_printoptions(threshold=sys.maxsize)


#implements the Koshino continuum model
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys
import scipy
np.set_printoptions(threshold=sys.maxsize)
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
kn_pointsx = 60
kn_pointsy = 60
kx_rangex = np.linspace(-k_window_sizex,k_window_sizex,kn_pointsx)
ky_rangey = np.linspace(-k_window_sizey,k_window_sizey,kn_pointsy)
step_x = kx_rangex[1]-kx_rangex[0]
step_y = ky_rangey[2]-ky_rangey[1]

tot_kpoints=kn_pointsx*kn_pointsy
step = min([ step_x, step_y])
bz = np.zeros([kn_pointsx,kn_pointsy])

##########################################
#check if a point is inside of an Hexagon inscribed in a circle of radius Radius_inscribed_hex
##########################################
Radius_inscribed_hex=1.00001*k_window_sizey
def hexagon(pos):
    y , x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
    return y < (3**0.5) * min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

##########################################
#Setting up kpoint arrays
##########################################
#Number of relevant kpoints
for x in range(kn_pointsx):
    for y in range(kn_pointsy):
        if hexagon((kx_rangex[x],ky_rangey[y])):
            bz[x,y]=1
num_kpoints=int(np.sum(bz))

#x axis of the BZ along Y axis of the plot
#plt.imshow(bz)
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()

#kpoint arrays
k_points = np.zeros([num_kpoints,2]) #matrix of brillion zone points (inside hexagon)
k_points_all = np.zeros([tot_kpoints,2]) #positions of all  the kpoints
k_points_closest = np.zeros([num_kpoints]) #index of the point in the  original array (polarization bubble calc)
k_points_closest_valley_flipped = np.zeros([num_kpoints]) #index of the point in the  original array flipping kx_>-kx and ky->-ky (polarization bubble calc)

#filling kpoint arrays
count1=0 #counting kpoints in the Hexagon
count2=0 #counting kpoints in the original grid
for x in range(kn_pointsx):
    for y in range(kn_pointsy):
        pos=[kx_rangex[x],ky_rangey[y]] #position of the kpoint
        k_points_all[count2,:]=pos #saving the position to the larger grid
        if hexagon((kx_rangex[x],ky_rangey[y])):
            k_points[count1,:]=pos #saving the kpoint in the hexagon only
            k_points_closest[count1]=count2 #saving the kpoint index in the larger square array
            count1=count1+1
        count2=count2+1

k_points_closest_valley_flipped=np.array(k_points_closest[::-1]) #index of the point in the  original array flipping kx_>-kx and ky->-ky (polarization bubble calc)

##########################################
#Diagonalizing continuum model
##########################################

gridp=np.arange(-50,50,1) #grid to calculate wavefunct
n1,n2=np.meshgrid(gridp,gridp) #grid to calculate wavefunct

def eigsystem(kx, ky, xi, nbands, n1, n2, nb):
    #we diagonalize a matrix made up
    qx_dif = kx+GM1[0]*n1+GM2[0]*n2-xi*Mpoint[0]
    qy_dif = ky+GM1[1]*n1+GM2[1]*n2-xi*Mpoint[1]
    vals = np.sqrt(qx_dif**2+qy_dif**2)
    ind_to_sum = np.where(vals <= 3*GM) #Finding the indices where the difference of  q lies inside threshold should be 4G
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
    Ener=Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0
    #return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]
    return Ener[nb]


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

################################
################################
################################ddd
################################
#defining triangular lattice


def hexagon_a(pos):
    Radius_inscribed_hex=1.000000000000001*4*np.pi/3
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


#getting the first brilloin zone from the Voronoi decomp of the recipprocal lattice
#input: reciprocal lattice vectors
#output: Points that delimit the FBZ -
#high symmetry points (for now just the triangular lattice will be implemented)
from scipy.spatial import Voronoi, voronoi_plot_2d
def FBZ_points(GM1,GM2):
    #creating reciprocal lattice
    Np=4
    n1=np.arange(-Np,Np+1)
    n2=np.arange(-Np,Np+1)
    Recip_lat=[]
    for i in n1:
        for j in n2:
            point=GM1*i+GM2*j
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

Vertices_list, Gamma, K, Kp, M, Mp=FBZ_points(GM1,GM2)

#linear parametrization accross different points in the BZ
def linpam(Kps,Npoints_q):
    Npoints=len(Kps)
    t=np.linspace(0, 1, Npoints_q)
    linparam=np.zeros([Npoints_q*(Npoints-1),2])
    for i in range(Npoints-1):
        linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
        linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

    return linparam

VV=Vertices_list+[Vertices_list[0]]
Nt=1000
kpath=linpam(VV,Nt)

def hexsamp2(npoints_x,npoints_y):
    ##########################################
    #definitions of the BZ grid
    ##########################################
    k_window_sizey = K[2][1] #4*(np.pi/np.sqrt(3))*np.sin(theta/2) (half size of  MBZ from edge to edge)
    k_window_sizex = K[1][0]   #(1/np.sqrt(3))*GM2[0] ( half size of MBZ from k to k' along a diagonal)
    kn_pointsx = npoints_x
    kn_pointsy = npoints_y
    kx_rangex = np.linspace(-k_window_sizex,k_window_sizex,kn_pointsx)
    ky_rangey = np.linspace(-k_window_sizey,k_window_sizey,kn_pointsy)
    step_x = kx_rangex[1]-kx_rangex[0]
    step_y = ky_rangey[2]-ky_rangey[1]

    tot_kpoints=kn_pointsx*kn_pointsy
    step = min([ step_x, step_y])
    bz = np.zeros([kn_pointsx,kn_pointsy])

    ##########################################
    #check if a point is inside of an Hexagon inscribed in a circle of radius Radius_inscribed_hex
    ##########################################
    Radius_inscribed_hex=1.00001*k_window_sizex
    def hexagon(pos):
        y, x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
        return y < (3**0.5) * min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

    ##########################################
    #Setting up kpoint arrays
    ##########################################
    #Number of relevant kpoints
    for x in range(kn_pointsx):
        for y in range(kn_pointsy):
            if hexagon((kx_rangex[x],ky_rangey[y])):
                bz[x,y]=1
    num_kpoints=int(np.sum(bz))

    #x axis of the BZ along Y axis of the plot
    #plt.imshow(bz)
    #plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()


    #filling kpoint arrays

    KX2=[]
    KY2=[]
    for x in range(kn_pointsx):
        for y in range(kn_pointsy):

            if hexagon((kx_rangex[x],ky_rangey[y])):
                #plt.scatter(kx_rangex[x],ky_rangey[y])
                KX2.append(kx_rangex[x])
                KY2.append(ky_rangey[y])

    return np.array(KX2),np.array(KY2),step_x*step_y
ni=10
KX_in, KY_in, dS_in=hexsamp2(ni,ni)
print(np.shape(KX_in))
# plt.scatter(KX_in,KY_in)
# plt.show()

tp=1 
T=0.05*tp


mu=0
xi=-1
nb=1
x = np.linspace(-2*GM/3,2*GM/3, 20)
X, Y = np.meshgrid(x, x)
print(np.shape(X))
import time 

s=time.time()
Z = [[eigsystem(X[i,j], Y[i,j], xi, nbands, n1, n2, nb)-mu  for i in range(np.shape(X)[0])]  for j in range(np.shape(X)[1])]
e=time.time()
print("time ", e-s)
band_max=np.max(Z)
band_min=np.min(Z)

spect=[]
funcs=[]
for kkk in k_points_all:
    cois=eigsystem2(kkk[0], kkk[1], 1, nbands, n1, n2)
    spect.append(np.real(cois[0]))
    
    funcs.append(cois[1])

Edisp=np.array(spect)
Psis=np.array(funcs)
energy_cut2 = np.zeros([kn_pointsx,kn_pointsy]);

for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        if hexagon((kx_rangex[k_x_i],ky_rangey[k_y_i])):
            ind = np.where(((k_points_all[:,0] == kx_rangex[k_x_i])*(k_points_all[:,1] == ky_rangey[k_y_i]))>0);
            energy_cut2[k_x_i,k_y_i] =Edisp[ind,1];

plt.imshow(energy_cut2)
plt.show()

c= plt.contour(X, Y, Z, linewidths=3, cmap='summer');
# v = c.collections[0].get_paths()[0].vertices
# xFS2 = v[::10,0]
# yFS2 = v[::10,1]
# KFx2=xFS2[0]
# KFy2=yFS2[0]
# plt.scatter(KFx2,KFy2)
plt.colorbar()
plt.plot(kpath[:,0],kpath[:,1])
plt.show()


def integrand(qx,qy,kx,ky,w,mu,T):
    #edk=Disp(kx,ky,mu)
    #edkq=Disp(kx+qx,ky+qy,mu)
    
    edk=eigsystem(kx, ky, xi, nbands, n1, n2, nb)-mu
    edkq=eigsystem(kx+qx, ky+qy, xi, nbands, n1, n2, nb)-mu
    nfk= 1/(np.exp(edk/T)+1)
    nfkq= 1/(np.exp(edkq/T)+1)
    eps=1e-1

    fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
    return np.imag(fac_p)


s=time.time()
cs= [ integrand(0.01,0.01,KX_in[i],KY_in[i],-0.000059,mu,T)  for i in range( np.size(KX_in) ) ]  
e=time.time()
print("time colors", e-s)

plt.scatter(KX_in,KY_in, c=cs, s=1)
plt.colorbar()
plt.show()


# L=[]
# L=L+[K[1]]+[Gamma]+[Mp[1]]+[[np.pi,0]]
# Nt=50
# kpath2=linpam(L,Nt)

L=[]
L=L+[Gamma]+[K[1]]
Nt=20
kpath2=linpam(L,Nt)

Nomegs=7
maxomeg=band_max
minomeg=band_min
omegas=np.linspace(minomeg,maxomeg,Nomegs)
#omegas=np.logspace(-5,1,Nomegs)
t=np.arange(0,len(kpath2),1)
t_m,omegas_m=np.meshgrid(t,omegas)
integ=[]
for t_m_i in t:
    sd=[]
    print(t_m_i,"/",np.size(t))
    for omegas_m_i in omegas:
        cs= [ integrand(kpath2[t_m_i,0],kpath2[t_m_i,1],KX_in[i],KY_in[i],omegas_m_i,mu,T)  for i in range( np.size(KX_in) ) ]  
        sd.append( np.sum(cs)*dS_in )
        #print(omegas_m_i,t_m_i)
    integ.append(sd)
integ_arr=np.array(integ)

limits_X=1
limits_Y=maxomeg
N_X=len(kpath2)
N_Y=Nomegs


plt.imshow(integ_arr.T, origin='lower')


ticks_X=5
ticks_Y=5
Npl_X=np.arange(0,N_X+1,int(N_X/ticks_X))
Npl_Y=np.arange(0,N_Y+1,int(N_Y/ticks_Y))
xl=np.round(np.linspace(0,limits_X,ticks_X+1),3)
yl=np.round(np.linspace(0,limits_Y,ticks_Y+1),5)

plt.xticks(Npl_X,xl)
plt.yticks(Npl_Y,yl)
plt.xlabel(r"$q_x$",size=16)
plt.ylabel(r"$\omega$",size=16)
#axhline(N_Y/2 -7, c='r')
#print(omegas[int(N_Y/2 -7)])
plt.colorbar()
plt.show()


plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,0])
plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,0])


# plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,-1])
# plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,-1])

plt.show()
