import numpy as np
import Hamiltonian
import MoireLattice
import matplotlib.pyplot as plt
import sys
import numpy.linalg as la


fillings = np.array([0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
mu_values = np.array([0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])

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


filling_index=int(sys.argv[1]) #0-25
##########################################
#parameters energy calculation
##########################################
a_graphene=2.46*(1e-10) #in meters
hbvf=0.003404*0.1973269804*1e-6 /a_graphene #ev*m
# hbvf = 2.1354; # eV
theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
nbands=4 #Number of bands 
Nsamp=int(sys.argv[2])
kappa_p=0.0797/0.0975;
kappa=kappa_p;
up = 0.0975; # eV
u = kappa*up; # eV

l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
ln=MoireLattice.MoireTriangLattice(Nsamp,theta,1)
lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one
[KX,KY]=lq.Generate_lattice()
# plt.scatter(KX,KY)
# plt.show()
Npoi=np.size(KX)
[q1,q1,q3]=l.q
q=la.norm(q1)

hvkd=hbvf*q
Kvec=(2*lq.b[0,:]+lq.b[1,:])/3 
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
h=Hamiltonian.Ham(hvkd, alpha, xi, 0, 0,n1,n2, l, nbands)
print(h)

# h=Hamiltonian.Ham(1, alpha_andrei, -1, 0,0,n1,n2, lq,nbands)
# h.umklapp_lattice()

h=Hamiltonian.Ham(hvkd, alpha_andrei, 1, KX[5],KY[5],n1,n2, lq,nbands)


Ene_valley_plus_a=np.empty((0))
Ene_valley_min_a=np.empty((0))
psi_plus_a=[]
psi_min_a=[]
print( KX[0],KY[0]  )
E1,wave1=h.eigens()

# print("starting dispersion ..........")
# # for l in range(Nsamp*Nsamp):
for l in range(Npoi):

    h=Hamiltonian.Ham(hvkd, alpha, 1, KX[l],KY[l],n1,n2, lq,nbands)
    E1,wave1=h.eigens()
    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
    psi_plus_a.append(wave1)

    h=Hamiltonian.Ham(hvkd, alpha, -1, KX[l],KY[l],n1,n2, lq,nbands)
    E1,wave1=h.eigens()
    Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
    psi_min_a.append(wave1)

    
    printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

##relevant wavefunctions and energies for the + valley
psi_plus=np.array(psi_plus_a)
psi_plus_conj=np.conj(np.array(psi_plus_a))
Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])

psi_min=np.array(psi_min_a)
psi_min_conj=np.conj(np.array(psi_min_a))
Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

bds1=[]
for i in range(nbands):
    plt.scatter(KX,KY, s=30, c=Ene_valley_plus[:,i])
    bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
    print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("2plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()
print("minimum bw_plus was:",np.min(np.array(bds1)))
bds1=[]
for i in range(nbands):
    plt.scatter(KX,KY, s=30, c=Ene_valley_min[:,i])
    bds1.append(np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]))
    print("bandwidth plus,",int(i),np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("2minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
    plt.close()
print("minimum bw_min was:",np.min(np.array(bds1)))

Ene_valley_plus_a=np.empty((0))
Ene_valley_min_a=np.empty((0))
psi_plus_a=[]
psi_min_a=[]
nbands=14 #Number of bands 
kpath=lq.High_symmetry_path()
# plt.scatter(kpath[:,0],kpath[:,1])
# VV=lq.boundary()
# plt.plot(VV[:,0], VV[:,1])
# plt.show()
Npoi=np.shape(kpath)[0]

h=Hamiltonian.Ham(1, 10*alpha, 1, 0,0,n1,n2, lq,nbands)
E1,wave1=h.eigens()
for l in range(Npoi):
    

    h=Hamiltonian.Ham(1, alpha_andrei, 1, kpath[l,0],kpath[l,1],n1,n2, lq,nbands)
    # h.umklapp_lattice()
    # break
    E1,wave1=h.eigens()
    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)

    psi_plus_a.append(wave1)
    printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
print(np.shape(Ene_valley_plus_a))
qa=np.linspace(0,1,Npoi)
for i in range(nbands):
    plt.plot(qa,Ene_valley_plus[:,i] , c='b')

# plt.ylim([-0.08,0.08])
plt.show()

bds1=[]
for i in range(nbands):
    bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
    print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))


print("minimum bw_plus was:",np.min(np.array(bds1)))
