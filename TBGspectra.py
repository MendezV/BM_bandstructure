
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
kn_pointsx = 61
kn_pointsy = 61
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

    #return a[2*N-int(nbands/2):2*N+int(nbands/2)] + en_shift - en0

spect=[]
funcs=[]
for kkk in k_points_all:
    cois=eigsystem(kkk[0], kkk[1], 1, nbands, n1, n2)
    spect.append(np.real(cois[0]))
    #print(np.shape(cois[1]))
    funcs.append(cois[1])

Edisp=np.array(spect)
Psis=np.array(funcs)

print(np.shape(Psis))
#following blocks compute heatmaps of the fermi surfaces


"""
energy_cut = np.zeros([kn_pointsx,kn_pointsy]);
for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        ind = np.where(((k_points_all[:,0] == kx_rangex[k_x_i])*(k_points_all[:,1] == ky_rangey[k_y_i]))>0);
        indmbz = np.where(((k_points[:,0] == kx_rangex[k_x_i])*(k_points[:,1] == ky_rangey[k_y_i]))>0);

        if(np.size(ind)>0)& (np.size(indmbz)>0):
            energy_cut[k_x_i,k_y_i] =Edisp[ind,1];
"""


energy_cut2 = np.zeros([kn_pointsx,kn_pointsy]);

for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        if hexagon((kx_rangex[k_x_i],ky_rangey[k_y_i])):
            ind = np.where(((k_points_all[:,0] == kx_rangex[k_x_i])*(k_points_all[:,1] == ky_rangey[k_y_i]))>0);
            energy_cut2[k_x_i,k_y_i] =Edisp[ind,1];


print(np.max(energy_cut2), np.min(energy_cut2))
plt.imshow(energy_cut2,cmap='jet')
plt.colorbar()
plt.show()

energy_cut = np.zeros([kn_pointsx,kn_pointsy]);

"""
#umklapps_V=[GM1,GM2,GM1+GM2,GM1-GM2,-GM1,-GM2,-(GM1+GM2),-(GM1-GM2)]
umklapps_V=[GM1,GM2,GM1+GM2,-GM1,-GM2,-(GM1+GM2)]
for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        if hexagon((kx_rangex[k_x_i],ky_rangey[k_y_i])):
            ind = np.where(((k_points_all[:,0] == kx_rangex[k_x_i])*(k_points_all[:,1] == ky_rangey[k_y_i]))>0);
            energy_cut[k_x_i,k_y_i] =Edisp[ind,1];
        else:
            for u in umklapps_V:
                if hexagon((kx_rangex[k_x_i]-u[0],ky_rangey[k_y_i]-u[1])):
                    ind = np.where(((k_points_all[:,0] == (kx_rangex[k_x_i]))*(k_points_all[:,1] == (ky_rangey[k_y_i]) ))>0);
                    energy_cut[k_x_i,k_y_i] =Edisp[ind,1];

"""

for k_x_i in range(kn_pointsx):
    for k_y_i in range(kn_pointsy):
        ind = np.where(((k_points_all[:,0] == (kx_rangex[k_x_i]))*(k_points_all[:,1] == (ky_rangey[k_y_i]) ))>0);
        energy_cut[k_x_i,k_y_i] =Edisp[ind,1];

print(np.max(energy_cut), np.min(energy_cut))
plt.imshow(energy_cut,cmap='jet')
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
kx_rangexp = np.linspace(-k_window_sizex,k_window_sizex,5*kn_pointsx)
ky_rangeyp = np.linspace(-k_window_sizey,k_window_sizey,5*kn_pointsy)
k1p,k2p= np.meshgrid(kx_rangexp,ky_rangeyp) #grid to calculate wavefunct




f = interpolate.interp2d(k1,k2, energy_cut, kind='cubic')

f_x = interpolate.bisplev(kx_rangex,ky_rangey, f.tck, dx=1, dy=0)
#f_x = interpolate.interp2d(k1,k2, energy_cut, kind='cubic',dx=1, dy=0)
#plt.imshow(f(kx_rangexp,ky_rangeyp),cmap='jet',vmax=np.max(energy_cut), vmin=np.min(energy_cut))
#plt.colorbar()
#plt.show()


"""
np.savetxt('disp.out', f(kx_rangexp,ky_rangeyp), delimiter=',')
np.savetxt('vx.out', f_x, delimiter=',')
"""


plt.contour(k1p, k2p, f(kx_rangexp,ky_rangeyp),25);
plt.show()

e0=-0.001
c = plt.contour(k1p, k2p, f(kx_rangexp,ky_rangeyp), [e0])
v = c.collections[0].get_paths()[0].vertices
xs = v[:,0]
ys = v[:,1]
#plt.scatter(x,y)
plt.show()
plt.scatter(xs,ys)
plt.show()


sizev=np.size(xs)
thet=np.linspace(-np.pi,np.pi,sizev)
th=np.arctan2(ys,xs)

#plt.scatter(th,absv)
#plt.scatter(th,absv*np.cos(th))
#plt.show()
##########################################
#Phonon dispersion
##########################################
'''
n_points = 40;
KX = k_window_sizex
KY = 2.*pi./3.*2.*sin(th./2); % 4pi/3*sin(th/2)
MX = KX;
MY = 0;

path_q_x = linspace(0,10*MX,n_points);
path_q_y = linspace(0,10*MY,n_points);
paths = sqrt(path_q_x.^2+path_q_y.^2);
paths(2)-paths(1)
sqrt(step_x.^2+step_y.^2)
c_s = 12000.*1.05.*10^(-34)/(1.6*10^(-19)).*1./(0.246.*10^(-9));
m_umklapps = 0:1:2;
ph_branches = 0:1:2;
'''
"next step is to organize the eigenvectors to have the same dimesnsionality as the number of points in the original BZ and use this to calculate bubble and vertices for for scattering with critical nematic boson. Also have to check the issue with nematoelastic coupling that develops hot spots in FS"
