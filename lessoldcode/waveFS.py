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


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


##########################################
#parameters energy calculation
##########################################
hv = 2.1354; # eV
theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
nbands=8 #Number of bands

fillings = np.array([0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
mu_values = np.array([0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])


filling_index=int(sys.argv[1]) #0-25

mu=mu_values[filling_index]/1000 #chemical potential in eV
filling=fillings[filling_index]

print("FILLING FOR THE CALCULATION IS .... ",filling)
print("CHEMICAL POTENTIAL FOR THAT FILLING IS .... ",mu)


kappa_p=0.0797/0.0975;
kappa=kappa_p;
up = 0.0975; # eV
u = kappa*up; # eV

print("kappa is..",u/up)
w = np.exp(1j*2*np.pi/3); # enters into the hamiltonian
en0 = 0; # Sets the zero of energy to the middle of flat bands, ev
gap = 0#.001/1000 # in ev
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
# Useful predefined matrices 
##########################################
paulix=np.array([[0,1],[1,0]])
pauliy=np.array([[0,-1j],[1j,0]])
pauliz=np.array([[1,0],[0,-1]])

Rotth1=np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]) #rotation matrix +thet
Rotth2=np.array([[np.cos(theta/2),np.sin(theta/2)],[-np.sin(theta/2),np.cos(theta/2)]]) #rotation matrix -thet
#C2
th1=np.pi
C2=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix +thet
#C2
th1=2*np.pi/3
C3=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix +thet



##########################################
# Constructing lattice vectors 
##########################################
#i vectror j coordinate
a=np.array([[1,0],[1/2,np.sqrt(3)/2]])  #original graphene lattice vectors
astar=(2*np.pi)*np.array([[1,-1/np.sqrt(3)],[0,2/np.sqrt(3)]]) # original graphene reciprocal lattice vectors

#rotated vectors
a1=a@Rotth1 #top layer
a2=a@Rotth2 #bottom layer
astar1=astar@Rotth1 #top layer
astar2=astar@Rotth2  #bottom layer

GM1=astar1[0,:]-astar2[0,:] #see Koshino-liang fu paper for further illustration (going off towards negative quadrant in the middle of the edge of the MBZ)
GM2=astar1[1,:]-astar2[1,:] #see Koshino-liang fu paper for further illustration (Going horizontal towards middle of the edge of the MBZ)
LM1=2*np.pi*np.array([GM2[1],-GM2[0]])/la.det(np.array([GM1,GM2]))
LM2=2*np.pi*np.array([-GM1[1],GM1[0]])/la.det(np.array([GM1,GM2]))

LM=np.sqrt(LM1@LM1) #moire period 2/np.sin(theta/2)
GM=np.sqrt(GM1@GM1) #reciprocal space period  8*np.pi/np.sqrt(3)*np.sin(theta/2)

##volume of the MBZ
zhat=np.array([0,0,1])
b_1=np.array([GM1[0],GM1[1],0]) # Moire reciprocal lattice vect extended
b_2=np.array([GM2[0],GM2[1],0]) # Moire reciprocal lattice vect extended
Vol_rec=np.cross(b_1,b_2)@zhat
#print(np.sqrt(np.dot(LM1,LM1)),0.5/np.sin(theta/2)) #verify the relation with the moire period

##########################################
#Valley locations
##########################################
#1,2 label layer and  + - labels valley from the  original graphene dispersion, see liang fu-koshino
Kplus1=-(2*astar1[0,:]+astar1[1,:])/3
Kplus2=-(2*astar2[0,:]+astar2[1,:])/3
Kmin1=+(2*astar1[0,:]+astar1[1,:])/3
Kmin2=+(2*astar2[0,:]+astar2[1,:])/3

#K points for the MBZ
Ktopplus=Kplus2-GM1 
Kbottommin=Kmin2+GM1

q1=Kplus1-Kplus2
q2=C3@q1
q3=C3@q2

#M points for the MBZ's
Mpoint=(Kplus1+Kplus2)/2 #position of the M point in the + valley
Mpointneg=-(Kplus1+Kplus2)/2 #position of the M point in the - valley

##########################################
#definitions of the BZ grid
##########################################


Numklpx=30
Numklpy=100
gridpx=np.arange(-int(Numklpx/2),int(Numklpx/2),1) #grid to calculate wavefunct
gridpy=np.arange(-int(Numklpy/2),int(Numklpy/2),1) #grid to calculate wavefunct
n1,n2=np.meshgrid(gridpx,gridpy) #grid to calculate wavefunct


####################################################################################
####################################################################################
####################################################################################
#
#      High symmetry points, k-point grids and symmetry operations on those grids
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


############################################################
### function that wraps around the FBZ
############################################################
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
    


############################################################
###grid for integration
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

#grids for Checking symmetries

#c2z rotated grid
KXc2=KX*C2[0,0]+KY*C2[0,1]
KYc2=KX*C2[1,0]+KY*C2[1,1]
#indices of the c2 rotated grid within the original grid
n1c2=np.zeros(Npoi)
Indc2=[]
for i in range(Npoi):
    #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
    n1c2[i]=int(np.argmin( (KX-KXc2[i])**2 +(KY-KYc2[i])**2))
    Indc2.append(int(np.argmin( (KX-KXc2[i])**2 +(KY-KYc2[i])**2)))

#c3 rotated grid
KXc3=KX*C3[0,0]+KY*C3[0,1]
KYc3=KX*C3[1,0]+KY*C3[1,1]
#indices of the c3 rotated grid within the original grid
n1c3=np.zeros(Npoi)
Indc3=[]
for i in range(Npoi):
    #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
    n1c3[i]=int(np.argmin( (KX-KXc3[i])**2 +(KY-KYc3[i])**2))
    Indc3.append(int(np.argmin( (KX-KXc3[i])**2 +(KY-KYc3[i])**2)))

#c2x rotated grid
KXc2x=KX
KYc2x=-KY
#indices of the c2 rotated grid within the original grid
n1c2x=np.zeros(Npoi)
Indc2x=[]
for i in range(Npoi):
    #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
    n1c2[i]=int(np.argmin( (KX-KXc2x[i])**2 +(KY-KYc2x[i])**2))
    Indc2x.append(int(np.argmin( (KX-KXc2x[i])**2 +(KY-KYc2x[i])**2)))


e=time.time()
print("finished sampling in reciprocal space....", np.shape(KY))
print("time for sampling was...",e-s)

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
    ind_to_sum = np.where(vals <= 5*GM) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
    n1_val = n1[ind_to_sum] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
    n2_val = n2[ind_to_sum] #
    # print(kx,ky)
    # plt.scatter(n1,n2)
    # plt.scatter(n1_val,n2_val)
    # plt.show()
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
    U=0*(np.kron(matu1,np.eye(N,N))+np.kron(matu2,matG2)+np.kron(matu2.T,matG4)) #interlayer coupling
    Udag=(U.conj()).T
    H1=-hv*(np.kron(xi*paulix,np.diag(qx_1))+np.kron(pauliy,np.diag(qy_1)))#+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
    H2=-hv*(np.kron(xi*paulix,np.diag(qx_2))+np.kron(pauliy,np.diag(qy_2)))#+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
    Hxi=np.bmat([[H1, Udag], [U, H2]]) #Full matrix
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
        # phas=np.angle(psi_p[50-int(xi*25),15,0,0])
        # phas=0 ## not fixing the phase
        maxisind = np.unravel_index(np.argmax(psi_p, axis=None), psi_p.shape)
        phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
        psi[:,:,:,:,nband] = psi_p*np.exp(-1j*phas)  
        # psi[:,nband]=np.reshape(psi_p,[np.shape(n1)[0]*np.shape(n1)[1]*4]).flatten()


    return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0,psi, ind_to_sum
    #  return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]


#fixing the arbitrary shift in the spectrum so that zero energy is at dirac point
KP=(-2*GM1 -GM2)/3 #dirac point

en03,ww, iss=eigsystem(KP[0], KP[1], 1, nbands, n1, n2)
en0=(en03[int(nbands/2)-1]+en03[int(nbands/2)])/2
print("reference energy as the energy at the dirac point",en0)

# en0=0



def eigsystem2(kx, ky, xi, nbands, n1arg, n2arg):
    #we diagonalize a matrix made up
    Numklpyw=np.shape(n1arg)[0]
    Numklpxw=np.shape(n1arg)[1]
    # print(Numklpyw,Numklpxw)
    qx_dif = kx + GM1[0]*n1arg+GM2[0]*n2arg
    qy_dif = ky + GM1[1]*n1arg+GM2[1]*n2arg
    vals = np.sqrt(qx_dif**2+qy_dif**2)
    ind_to_sum = np.where(vals <= 5*GM) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
    n1_val = n1arg[ind_to_sum] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
    n2_val = n2arg[ind_to_sum] #
    # print(kx,ky)
    # plt.scatter(n1,n2)
    # plt.scatter(n1_val,n2_val)
    # plt.show()
    N = np.shape(ind_to_sum)[1] ##number of indices for which the condition above is satisfied
    
    G0x= GM1[0]*n1_val+GM2[0]*n2_val
    G0y= GM1[1]*n1_val+GM2[1]*n2_val
    Qplusx = G0x + xi*q1[0]
    Qplusy = G0y + xi*q1[1]
    Qminx = G0x - xi*q1[0]
    Qminy = G0y - xi*q1[1]

    # #top layer
    qx_1 = kx + G0x - xi*q1[0]
    qy_1 = ky + G0y - xi*q1[1]

    # #bottom layer
    # q_2=Rotth1@np.array([kx,ky])
    qx_2 = kx + G0x+xi*q1[0]
    qy_2 = ky + G0y+xi*q1[1]

    
    # the following block generates the momentum space rep of the interlayer coupling matrix
    matGq1=np.zeros([N,N])
    matGq2=np.zeros([N,N])
    matGq3=np.zeros([N,N])
    tres=(1e-5)*np.sqrt(q1[0]**2 +q1[1]**2)
    for i in range(N):
        indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + xi*q1[0])**2+(Qplusy-Qminy[i] + xi*q1[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qplusx-Qminx[i] - xi*q1[0])**2+(Qplusy-Qminy[i] - xi*q1[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qplusx-Qplusx[i] + xi*q1[0])**2+(Qplusy-Qplusy[i] + xi*q1[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qminx-Qminx[i] + xi*q1[0])**2+(Qminy-Qminy[i] + xi*q1[1])**2  )<tres)
        if np.size(indi1)>0:
            matGq1[indi1,i]=1
        # if np.size(indi2)>0:
        #     matGq1[indi2,i]=1
        #     print("u")
        indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + xi*q2[0])**2+(Qplusy-Qminy[i] + xi*q2[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qplusx-Qminx[i] - xi*q2[0])**2+(Qplusy-Qminy[i] - xi*q2[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qplusx-Qplusx[i] + xi*q2[0])**2+(Qplusy-Qplusy[i] + xi*q2[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qminx-Qminx[i] + xi*q2[0])**2+(Qminy-Qminy[i] + xi*q2[1])**2  )<tres)
        if np.size(indi1)>0:
            matGq2[indi1,i]=1 #indi1+1=i
        # if np.size(indi2)>0:
        #     matGq2[indi2,i]=1
        #     print("u")
            
        indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + xi*q3[0])**2+(Qplusy-Qminy[i] + xi*q3[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qplusx-Qminx[i] - xi*q3[0])**2+(Qplusy-Qminy[i] - xi*q3[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qplusx-Qplusx[i] + xi*q3[0])**2+(Qplusy-Qplusy[i] + xi*q3[1])**2  )<tres)
        # indi2=np.where(np.sqrt(  (Qminx-Qminx[i] + xi*q3[0])**2+(Qminy-Qminy[i] + xi*q3[1])**2  )<tres)
        if np.size(indi1)>0:
            matGq3[indi1,i]=1
        # if np.size(indi2)>0:
        #     matGq3[indi2,i]=1
        #     print("u")

    # matG2=np.zeros([N,N])
    # matG4=np.zeros([N,N])
    # for i in range(N):
    #     indi1=np.where((n1_val==n1_val[i]-xi)*(n2_val==n2_val[i]))
    #     if np.size(indi1)>0:
    #         matG2[i,indi1]=1
    #     indi1=np.where((n1_val==n1_val[i]-xi)*(n2_val==n2_val[i]-xi))
    #     if np.size(indi1)>0:
    #         matG4[i,indi1]=1
    # print(np.mean(matG2-matGq2), np.mean(matG4-matGq3),np.mean(matG2-matGq3))
        

    #Matrices that  appeared as coefficients of the real space ops
    #full product is the kronecker product of both matrices
    matu1=np.array([[u,up],[up,u]])
    matu2=np.array([[u,up*(w**(-xi))],[up*(w**(xi)),u]])
    matu3=matu2.T

    Mdelt1=matGq1
    Mdelt2=matGq2
    Mdelt3=matGq3

    #assembling the matrix to be diagonalized
    U=0*(np.kron(matu1,Mdelt1)+np.kron(matu2,Mdelt2)+np.kron(matu3,Mdelt3)) #interlayer coupling
    Udag=(U.conj()).T
    H1=hv*(np.kron(xi*paulix,np.diag(qx_1))+np.kron(pauliy,np.diag(qy_1)))#+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
    H2=hv*(np.kron(xi*paulix,np.diag(qx_2))+np.kron(pauliy,np.diag(qy_2)))#+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
    Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
    #a= np.linalg.eigvalsh(Hxi) - en_shift
    (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

    #######HANDLING WITH RESHAPE
    #umklp,umklp, layer, sublattice
    psi_p=np.zeros([Numklpyw,Numklpxw,2,2]) +0*1j
    # psi=np.zeros([Numklp*Numklp*4,nbands])+0*1j
    psi=np.zeros([Numklpyw, Numklpxw, 2,2, nbands]) +0*1j

    for nband in range(nbands):
        # print(np.shape(ind_to_sum), np.shape(psi_p), np.shape(psi_p[ind_to_sum]))
        psi_p[ind_to_sum] = np.reshape(  np.array( np.reshape(Eigvect[:,2*N-int(nbands/2)+nband] , [4, N])  ).T, [N, 2, 2] )    

        ##GAUGE FIXING by making the 30 30 1 1 component real
        # phas=np.angle(psi_p[50-int(xi*25),15,0,0])
        # phas=0 ## not fixing the phase
        maxisind = np.unravel_index(np.argmax(psi_p, axis=None), psi_p.shape)
        phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
        psi[:,:,:,:,nband] = psi_p*np.exp(-1j*phas)
        # psi[:,nband]=np.reshape(psi_p,[np.shape(n1)[0]*np.shape(n1)[1]*4]).flatten()


    return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, psi, ind_to_sum
    #  return Eigvals[2*N-int(nbands/2):2*N+int(nbands/2)]-en0, Eigvect[:,2*N-int(nbands/2):2*N+int(nbands/2)]



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
for l in range(Npoi):

    # E1,wave1,ind_to_sum1=eigsystem(KXc2[l],KYc2[l], 1, nbands, -n1, -n2)
    # E2,wave2, ind_to_sum2=eigsystem(KX[l],KY[l], -1, nbands, n1, n2)
    E1,wave1,ind_to_sum1=eigsystem(KX[l],KY[l], 1, nbands, n1, n2)
    E2,wave2, ind_to_sum2=eigsystem(KX[l],KY[l], -1, nbands, n1, n2)

    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
    Ene_valley_min_a=np.append(Ene_valley_min_a,E2)

    psi_plus_a.append(wave1)
    psi_min_a.append(wave2)
    printProgressBar(l + 1, Npoi, prefix = 'Progress Diag1:', suffix = 'Complete', length = 50)

##relevant wavefunctions and energies for the + valley
psi_plus=np.array(psi_plus_a)
psi_plus_conj=np.conj(np.array(psi_plus_a))
Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])


#relevant wavefunctions and energies for the - valley
psi_min=np.array(psi_min_a)
psi_min_conj=np.conj(np.array(psi_min_a))
Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])
print(np.shape(ind_to_sum1))
print(ind_to_sum1)
# plt.scatter(n1,n2)
# plt.scatter(n1[ind_to_sum1],n2[ind_to_sum1])
# plt.scatter(n1[ind_to_sum2],n2[ind_to_sum2])
# plt.show()

psi_minp=np.sum(np.sum(np.sum(psi_min,axis=5),axis=4),axis=3)
psi_plsp=np.sum(np.sum(np.sum(psi_plus,axis=5),axis=4),axis=3)
# subs=np.sum(np.sum(np.sum(psi_plus-psi_min,axis=5),axis=4),axis=3)
subs=np.sum(np.sum(np.sum(psi_plus_conj*psi_min,axis=5),axis=4),axis=3)
ids=np.where(np.abs(psi_minp[5,:,:])>0)
print("sizeids1,",np.shape(ids))
ids2=np.where(np.abs(psi_plsp[5,:,:])>0)
print("sizeids2,",np.shape(ids2))
idsbs=np.where(np.abs(subs[5,:,:])>0)
print("sizeidssub,",np.shape(idsbs))
# plt.scatter(n1,n2)
# plt.scatter(n1[ids],n2[ids])
# plt.scatter(n1[ids2],n2[ids2])
# plt.show()
# plt.scatter(n1,n2)
# plt.scatter(n1[idsbs],n2[idsbs])
# plt.show()

psipre1=psi_min
psipre2=psi_plus_conj[:, :,:,:,:,:]
psi_minp=np.sum(np.sum(np.sum(psipre1,axis=5),axis=4),axis=3)
psi_plsp=np.sum(np.sum(np.sum(psipre2,axis=5),axis=4),axis=3)
# subs=np.sum(np.sum(np.sum(psi_plus-psi_min,axis=5),axis=4),axis=3)
subs=np.sum(np.sum(np.sum(psipre1*psipre2,axis=5),axis=4),axis=3)
ids=np.where(np.abs(psi_minp[5,:,:])>0)
print("sizeids1,",np.shape(ids))
ids2=np.where(np.abs(psi_plsp[5,:,:])>0)
print("sizeids2,",np.shape(ids2))
idsbs=np.where(np.abs(subs[5,:,:])>0)
print("sizeidssub,",np.shape(idsbs))
# plt.scatter(n1,n2)
# plt.scatter(n1[ids],n2[ids])
# plt.scatter(n1[ids2],n2[ids2])
# plt.show()
# plt.scatter(n1,n2)
# plt.scatter(n1[ids],n2[ids])
# plt.scatter(n1[ids2],n2[ids2])
# plt.scatter(n1[idsbs],n2[idsbs])
# plt.show()


testover1=np.tensordot(np.conj(psipre2) , psipre1,  axes=([1,2,3,4],[1,2,3,4]))
for i in range(4):  
    print("testing complex conjugation wavefunction, index one plus... ",np.mean(np.diag(np.abs(testover1[:,i,:,i])**2))  )
# print("ASDFASDFA... ", np.mean( np.abs(psi_plus)**2  ))
# print("ASDFASDFA... ", np.mean( np.abs(psi_min)**2   ))
# print("ASDFASDFA... ", np.mean( np.abs(psi_plus)**2 + np.abs(psi_min)**2 ))
# print("ASDFASDFA... ", np.mean( np.abs(psi_plus)**2 - np.abs(psi_min)**2 ))


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
Lambda_Tens_min=np.tensordot(psi_min_conj,psi_min, axes=([1,2,3,4],[1,2,3,4])) 

print("normalization ",  np.abs(Lambda_Tens_plus[0,0,0,0])**2  )
print("normalization ",  np.mean(np.diag(np.abs(Lambda_Tens_plus[:,1,:,1])**2)) )
print("normalization ",  np.mean(np.diag(np.abs(Lambda_Tens_plus[:,2,:,2])**2)) )

print("normalization ",  np.mean(np.diag(np.abs(Lambda_Tens_plus[:,1,:,1])**2)) )
print("normalization ",  np.mean(np.diag(np.abs(Lambda_Tens_plus[:,2,:,2])**2)) )
# plt.plot( np.diag(np.abs(Lambda_Tens_plus[:,0,:,0])**2) )
# plt.plot( np.diag(np.abs(Lambda_Tens_plus[:,1,:,1])**2) )
# plt.plot( np.diag(np.abs(Lambda_Tens_plus[:,2,:,2])**2) )
# plt.plot( np.diag(np.abs(Lambda_Tens_plus[:,1,:,2])**2) )
# plt.show()
# plt.plot( np.diag(np.abs(Lambda_Tens_min[:,0,:,0])**2) )
# plt.plot( np.diag(np.abs(Lambda_Tens_min[:,1,:,1])**2) )
# plt.plot( np.diag(np.abs(Lambda_Tens_min[:,2,:,2])**2) )
# plt.plot( np.diag(np.abs(Lambda_Tens_min[:,1,:,2])**2) )
# plt.show()
print( "tensorshape",np.shape(Lambda_Tens_plus) )



print("fffenergy",np.mean(Ene_valley_plus[Indc2,:]-Ene_valley_min[:,:]))

VV=np.array(Vertices_list+[Vertices_list[0]])
bds1=[]
for i in range(nbands):
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX,KY, s=30, c=Ene_valley_plus[:,i])
    bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
    print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("1plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()
print("minimum bw_plus was:",np.min(np.array(bds1)))
bds2=[]
for i in range(nbands):
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX,KY, s=30, c=Ene_valley_min[:,i])
    print("bandwidth min,",int(i),np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]) )
    bds2.append( np.max(Ene_valley_min[:,i]) - np.min(Ene_valley_min[:,i]) )
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("1minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()
print("minimum bw_plus was:",np.min(np.array(bds2)))

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]


# # print("starting dispersion ..........")
# # # for l in range(Nsamp*Nsamp):
# # for l in range(Npoi):

# #     E1,wave1,ind_to_sum1=eigsystem(KXc2[l],KYc2[l], 1, nbands, -n1, -n2)
# #     E2,wave2,ind_to_sum2=eigsystem(KXc2[l],KYc2[l], 1, nbands, -n1, -n2)

# #     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
# #     Ene_valley_min_a=np.append(Ene_valley_min_a,E2)

# #     psi_plus_a.append(wave1)
# #     psi_min_a.append(wave2)

# # ##relevant wavefunctions and energies for the + valley
# # psi_plus=np.array(psi_plus_a)
# # psi_plus_conj=np.conj(np.array(psi_plus_a))
# # Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])


# # #relevant wavefunctions and energies for the - valley
# # psi_min=np.array(psi_min_a)
# # psi_min_conj=np.conj(np.array(psi_min_a))
# # Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])
# # print(np.shape(ind_to_sum1))
# # print(ind_to_sum1)
# # # plt.scatter(n1,n2)
# # # plt.scatter(n1[ind_to_sum1],n2[ind_to_sum1])
# # # plt.scatter(n1[ind_to_sum2],n2[ind_to_sum2])
# # # plt.show()

# # psi_minp=np.sum(np.sum(np.sum(psi_min,axis=5),axis=4),axis=3)
# # psi_plsp=np.sum(np.sum(np.sum(psi_plus,axis=5),axis=4),axis=3)
# # # subs=np.sum(np.sum(np.sum(psi_plus-psi_min,axis=5),axis=4),axis=3)
# # subs=np.sum(np.sum(np.sum(psi_plus_conj*psi_min,axis=5),axis=4),axis=3)
# # ids=np.where(np.abs(psi_minp[5,:,:])>0)
# # print("sizeids1,",np.shape(ids))
# # ids2=np.where(np.abs(psi_plsp[5,:,:])>0)
# # print("sizeids2,",np.shape(ids2))
# # idsbs=np.where(np.abs(subs[5,:,:])>0)
# # print("sizeidssub,",np.shape(idsbs))
# # # plt.scatter(n1,n2)
# # # plt.scatter(n1[ids],n2[ids])
# # # plt.scatter(n1[ids2],n2[ids2])
# # # plt.show()
# # # plt.scatter(n1,n2)
# # # plt.scatter(n1[idsbs],n2[idsbs])
# # # plt.show()

# Numklpxw=30
# Numklpyw=30
# gridpxw=np.arange(-int(Numklpxw/2),int(Numklpxw/2),1) #grid to calculate wavefunct
# gridpyw=np.arange(-int(Numklpyw/2),int(Numklpyw/2),1) #grid to calculate wavefunct
# n1w,n2w=np.meshgrid(gridpxw,gridpyw) #grid to calculate wavefunct

# en0=0
# KP=(-2*GM1 -GM2)/3 #dirac point
# en03,ww, iss=eigsystem2(KP[0], KP[1], 1, nbands, n1w, n2w)
# en0=(en03[int(nbands/2)-1]+en03[int(nbands/2)])/2
# print("reference energy as the energy at the dirac point",en0)


# print("starting dispersion ..........")
# # for l in range(Nsamp*Nsamp):
# for l in range(Npoi):

#     E1,wave1,ind_to_sum1=eigsystem2(KX[l],KY[l], 1, nbands,n1w, n2w)
#     E2,wave2,ind_to_sum2=eigsystem2(KX[l],KY[l], -1, nbands, n1w,n2w)

#     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
#     Ene_valley_min_a=np.append(Ene_valley_min_a,E2)

#     psi_plus_a.append(wave1)
#     psi_min_a.append(wave2)
#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# ##relevant wavefunctions and energies for the + valley
# psi_plus=np.array(psi_plus_a)
# psi_plus_conj=np.conj(np.array(psi_plus_a))
# Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])


# #relevant wavefunctions and energies for the - valley
# psi_min=np.array(psi_min_a)
# psi_min_conj=np.conj(np.array(psi_min_a))
# Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])
# print(np.shape(ind_to_sum1))
# print(ind_to_sum1)
# # plt.scatter(n1,n2)
# # plt.scatter(n1[ind_to_sum1],n2[ind_to_sum1])
# # plt.scatter(n1[ind_to_sum2],n2[ind_to_sum2])
# # plt.show()

# psi_minp=np.sum(np.sum(np.sum(psi_min,axis=5),axis=4),axis=3)
# psi_plsp=np.sum(np.sum(np.sum(psi_plus,axis=5),axis=4),axis=3)
# # subs=np.sum(np.sum(np.sum(psi_plus-psi_min,axis=5),axis=4),axis=3)
# subs=np.sum(np.sum(np.sum(psi_plus_conj*psi_min,axis=5),axis=4),axis=3)
# ids=np.where(np.abs(psi_minp[5,:,:])>0)
# print("sizeids1,",np.shape(ids))
# ids2=np.where(np.abs(psi_plsp[5,:,:])>0)
# print("sizeids2,",np.shape(ids2))
# idsbs=np.where(np.abs(subs[5,:,:])>0)
# print("sizeidssub,",np.shape(idsbs))
# # plt.scatter(n1w,n2w)
# # plt.scatter(n1w[ids],n2w[ids])
# # plt.scatter(n1w[ids2],n2w[ids2])
# # plt.show()
# # plt.scatter(n1w,n2w)
# # plt.scatter(n1w[idsbs],n2w[idsbs])
# # plt.show()

# bds1=[]
# for i in range(nbands):
#     plt.plot(VV[:,0],VV[:,1])
#     plt.scatter(KX,KY, s=30, c=Ene_valley_plus[:,i])
#     bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
#     print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("2plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()
# print("minimum bw_plus was:",np.min(np.array(bds1)))
# bds2=[]
# for i in range(nbands):
#     plt.plot(VV[:,0],VV[:,1])
#     plt.scatter(KX,KY, s=30, c=Ene_valley_min[:,i])
#     print("bandwidth min,",int(i),np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]) )
#     bds2.append( np.max(Ene_valley_min[:,i]) - np.min(Ene_valley_min[:,i]) )
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("2minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()
# print("minimum bw_plus was:",np.min(np.array(bds2)))