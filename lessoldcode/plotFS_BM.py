#implements the Koshino continuum model
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as la
import sys
import scipy
import Hamiltonian
import MoireLattice
np.set_printoptions(threshold=sys.maxsize)


fillings = np.array([0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
mu_values = np.array([0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])
filling_index=int(sys.argv[1]) #0-25
Nsamp=int(sys.argv[2]) 
##########################################
#parameters energy calculation
##########################################


en0 = -0.001665258323557075 # in ev #1.6598/1000; # band energy of the flat bands, eV

############relevant params
theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
mu=mu_values[filling_index]/1000 #chemical potential in eV
print("chemical potential in mev ", mu*1000)
print("Filling at this chemical potential is given by ", fillings[filling_index])
l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one
[q1,q1,q3]=l.q
q=la.norm(q1)
xi=1
#kosh params realistic  -- this is the closest to the actual Band Struct used in the paper
hbvf = 2.1354; # eV
hvkd=hbvf*q
kappa_p=0.0797/0.0975
kappa=kappa_p
up = 0.0975; # eV
u = kappa*up; # eV
alpha=up/hvkd
alph=alpha

PH=False

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
kn_pointsx = Nsamp
kn_pointsy = Nsamp
kx_rangex = np.linspace(-k_window_sizex,k_window_sizex,kn_pointsx)/q #normalization q
ky_rangey = np.linspace(-k_window_sizey,k_window_sizey,kn_pointsy)/q #normalization q
step_x = kx_rangex[1]-kx_rangex[0]
step_y = ky_rangey[2]-ky_rangey[1]

tot_kpoints=kn_pointsx*kn_pointsy
step = min([ step_x, step_y])
bz = np.zeros([kn_pointsx,kn_pointsy])

##########################################
#check if a point is inside of an Hexagon inscribed in a circle of radius Radius_inscribed_hex
##########################################
Radius_inscribed_hex=1.0000005*k_window_sizey
def hexagon(pos):
    y,x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
    return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

##########################################
#Setting up kpoint arrays
##########################################
#Number of relevant kpoints
for x in range(kn_pointsx):
    for y in range(kn_pointsy):
        if hexagon((kx_rangex[x],ky_rangey[y])):
            bz[x,y]=1
num_kpoints=int(np.sum(bz))

# x axis of the BZ along Y axis of the plot
# plt.imshow(bz.T)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

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
VV=np.array(Vertices_list+[Vertices_list[0]])/q #normalization q
print("size of FBZ, maximum momentum trans",np.sqrt(np.sum(np.array(K[0])**2))/GM )


spect=[]
funcs=[]
spect_min=[]
funcs_min=[]
hpl=Hamiltonian.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
hmin=Hamiltonian.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
for kkk in k_points_all:
    E1,wave1=hpl.eigens(kkk[0], kkk[1],nbands)
    cois=[E1,wave1]
    spect.append(np.real(cois[0])-en0)
    funcs.append(cois[1])

    E1,wave1=hmin.eigens(kkk[0], kkk[1],nbands)
    cois_min=[E1,wave1]
    spect_min.append(np.real(cois_min[0])-en0)
    funcs_min.append(cois_min[1])


Edisp=np.array(spect)
Psis=np.array(funcs)
Edisp_min=np.array(spect_min)
Psis_min=np.array(funcs_min) 
with open('Edisp_'+str(Nsamp)+'.npy', 'wb') as f:
    np.save(f, Edisp)
# with open('Psis_plus_'+str(Nsamp)+'.npy', 'wb') as f:
#     np.save(f, Psis)
with open('Edisp_min_'+str(Nsamp)+'.npy', 'wb') as f:
    np.save(f, Edisp_min)
# with open('Psis_min_'+str(Nsamp)+'.npy', 'wb') as f:
#     np.save(f, Psis_min)



print("Loading  ..........")
with open('Edisp_'+str(Nsamp)+'.npy', 'rb') as f:
    Edisp=np.load(f)
with open('Edisp_min_'+str(Nsamp)+'.npy', 'rb') as f:
    Edisp_min=np.load(f)
# with open('Psis_plus_'+str(Nsamp)+'.npy', 'rb') as f:
#     Psis_plus=np.load(f, allow_pickle=True)
# with open('Psis_min_'+str(Nsamp)+'.npy', 'rb') as f:
#     Psis_min=np.load(f, allow_pickle=True)


energy_cut2 = np.zeros([kn_pointsx,kn_pointsy]);


energy_cut = np.zeros([kn_pointsx,kn_pointsy]);
for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        ind = np.where(((k_points_all[:,0] == (kx_rangex[k_x_i]))*(k_points_all[:,1] == (ky_rangey[k_y_i]) ))>0);
        energy_cut[k_x_i,k_y_i] =Edisp[ind,2];

print(np.max(energy_cut), np.min(energy_cut))
plt.imshow(energy_cut.T) #transpose since x and y coordinates dont match the i j indices displayed in imshow
plt.colorbar()
plt.show()

energy_cut_min = np.zeros([kn_pointsx,kn_pointsy]);
for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        ind = np.where(((k_points_all[:,0] == (kx_rangex[k_x_i]))*(k_points_all[:,1] == (ky_rangey[k_y_i]) ))>0);
        energy_cut_min[k_x_i,k_y_i] =Edisp_min[ind,2];

print(np.max(energy_cut_min), np.min(energy_cut_min))

plt.imshow(energy_cut_min.T ) #transpose since x and y coordinates dont match the i j indices displayed in imshow
plt.colorbar()
plt.show()



import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
x_edges, y_edges = np.mgrid[-1:1:21j, -1:1:21j]
x = x_edges[:-1, :-1] + np.diff(x_edges[:2, 0])[0] / 2.
y = y_edges[:-1, :-1] + np.diff(y_edges[0, :2])[0] / 2.
z = (x+y) * np.exp(-6.0*(x*x+y*y))




k1,k2= np.meshgrid(kx_rangex,ky_rangey) #grid to calculate wavefunct
kx_rangexp = np.linspace(-k_window_sizex,k_window_sizex,kn_pointsx)/q #normalization q
ky_rangeyp = np.linspace(-k_window_sizey,k_window_sizey,kn_pointsy)/q #normalization q
k1p,k2p= np.meshgrid(kx_rangexp,ky_rangeyp) #grid to calculate wavefunct




f = interpolate.interp2d(k1,k2, energy_cut.T, kind='linear')
# f_x = interpolate.bisplev(kx_rangex,ky_rangey, f.tck, dx=1, dy=0)

f_min = interpolate.interp2d(k1,k2, energy_cut_min.T, kind='linear')
# f_x_min = interpolate1.bisplev(kx_rangex,ky_rangey, f_min.tck, dx=1, dy=0)


print(mu)
plt.plot(VV[:,0],VV[:,1])
plt.contour(k1p, k2p, f(kx_rangexp,ky_rangeyp),[mu],cmap='RdYlBu')
plt.contour(k1p, k2p, f_min(kx_rangexp,ky_rangeyp),[mu])
plt.show()





