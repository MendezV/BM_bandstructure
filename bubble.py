#integrates a dispersion to get the polarization bubble

import concurrent.futures
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys
import scipy


################################
################################
################################ddd
################################
#defining triangular lattice
a=1
a_1=a*np.array([1,0,0])
a_2=a*np.array([1/2,np.sqrt(3)/2,0])
zhat=np.array([0,0,1])

Vol_real=np.dot(np.cross(a_1,a_2),zhat)
b_1=np.cross(a_2,zhat)*(2*np.pi)/Vol_real
b_2=np.cross(zhat,a_1)*(2*np.pi)/Vol_real
Vol_rec=np.dot(np.cross(b_1,b_2),zhat)

a_1=a_1[0:2]
a_2=a_2[0:2]
b_1=b_1[0:2]
b_2=b_2[0:2]

def hexagon_a(pos):
    Radius_inscribed_hex=1.000000000000001*4*np.pi/3
    x, y = map(abs, pos) #only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge


#getting the first brilloin zone from the Voronoi decomp of the recipprocal lattice
#input: reciprocal lattice vectors
#output: Points that delimit the FBZ -
#high symmetry points (for now just the triangular lattice will be implemented)
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

Vertices_list, Gamma, K, Kp, M, Mp=FBZ_points(b_1,b_2)

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
        x, y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
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
ni=200
KX_in, KY_in, dS_in=hexsamp2(ni,ni)
print(np.shape(KX_in))
# plt.scatter(KX_in,KY_in)
# plt.show()

tp=1 
T=0.05*tp


mu=-5.8
def Disp(kx,ky,mu):
    ed=-tp*(2*np.cos(kx)+4*np.cos((kx)/2)*np.cos(np.sqrt(3)*(ky)/2))
    ed=ed-mu
    return ed

x = np.linspace(-3.8, 3.8, 300)
X, Y = np.meshgrid(x, x)
Z = Disp(X, Y, mu)

band_max=np.max(Z)
band_min=np.min(Z)

c= plt.contour(X, Y, Z, levels=[0],linewidths=3, cmap='summer');
v = c.collections[0].get_paths()[0].vertices
xFS2 = v[::10,0]
yFS2 = v[::10,1]
KFx2=xFS2[0]
KFy2=yFS2[0]
plt.scatter(KFx2,KFy2)
plt.plot(kpath[:,0],kpath[:,1])
plt.show()


def integrand(qx,qy,kx,ky,w,mu,T):
    edk=Disp(kx,ky,mu)
    edkq=Disp(kx+qx,ky+qy,mu)
    nfk= 1/(np.exp(edk/T)+1)
    nfkq= 1/(np.exp(edkq/T)+1)
    eps=1e-1

    fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
    return np.imag(fac_p)


plt.scatter(KX_in,KY_in, c=((integrand(0.1,0.1,KX_in,KY_in,0.059,mu,T))), s=1)
plt.colorbar()
plt.show()


# L=[]
# L=L+[K[1]]+[Gamma]+[Mp[1]]+[[np.pi,0]]
# Nt=50
# kpath2=linpam(L,Nt)

L=[]
L=L+[Gamma]+[K[1]]
Nt=200
kpath2=linpam(L,Nt)

Nomegs=70
maxomeg=band_max
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


plt.imshow(integ_arr.T, origin='lower')


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


plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,0])
plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,0])


# plt.scatter(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,-1])
# plt.plot(np.sqrt(kpath2[t,0]**2+kpath2[t,1]**2) , integ_arr[:,-1])

plt.show()