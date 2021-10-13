import numpy as np
import Hamiltonian
import Hamiltonian_min
import Hamiltonian_2v
import MoireLattice
import matplotlib.pyplot as plt
import sys
import numpy.linalg as la
import time


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
mu=mu_values[filling_index]/1000
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


#Various alpha values
hvfK_andrei=19.81
#andreis params
w=0.110 #in ev
hvfkd_andrei=hvfK_andrei*np.sin(theta/2) #wrong missing 1/sqrt3
alpha_andrei=w/hvfkd_andrei
alpha=w/hvkd
alpha_andrei_corrected=(np.sqrt(3)/2)*w/hvfkd_andrei
#magic angles
amag1=0.5695
amag2=0.605
amag3=0.65
#angle with flat band in the chiral limit
alph2= 0.5856



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

#JY params 
# hbvf = 2.7; # eV
# hvkd=hbvf*q
# kappa=0.75
# up = 0.105; # eV
# u = kappa*up; # eV
# alpha=up/hvkd
# alph=alpha


PH=False
print("kappa is..", kappa)
print("alpha is..", alpha)


#################
##############



#################################
#################################
#################################
# e_k
#################################
#################################
#################################





########TTEST FOR THE DISPERSION FOR ALL K AND THE BANDSTRUCTURE ALONG HIGH SYMMETRY POINTS
# for l in range(Npoi):

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]


# # print("starting dispersion ..........")
# # # for l in range(Nsamp*Nsamp):
# s=time.time()
# hpl=Hamiltonian.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
# for l in range(Npoi):
#     E1,wave1=hpl.eigens(KX[l],KY[l],nbands)
#     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
#     psi_plus_a.append(wave1)


#     E1,wave1=hmin.eigens(KX[l],KY[l],nbands)
#     Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
#     psi_min_a.append(wave1)

#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# e=time.time()
# print("time to diag over MBZ", e-s)
# ##relevant wavefunctions and energies for the + valley
# psi_plus=np.array(psi_plus_a)
# psi_plus_conj=np.conj(np.array(psi_plus_a))
# Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
# e0=(np.min(Ene_valley_plus[:,int(nbands/2)])+np.max(Ene_valley_plus[:,int(nbands/2)-1]))/2.0
# Ene_valley_plus=Ene_valley_plus-e0
# print(e0)

# psi_min=np.array(psi_min_a)
# psi_min_conj=np.conj(np.array(psi_min_a))
# Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])-e0


# bds1=[]
# for i in range(nbands):
#     plt.scatter(KX,KY, s=30, c=Ene_valley_plus[:,i])
#     bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
#     print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("2plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()
# print("minimum bw_plus was:",np.min(np.array(bds1)))

# bds1=[]
# for i in range(nbands):
#     plt.scatter(KX,KY, s=30, c=Ene_valley_min[:,i])
#     bds1.append(np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]))
#     print("bandwidth plus,",int(i),np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]))
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("2minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()
# print("minimum bw_min was:",np.min(np.array(bds1)))

# #################################
# #################################
# #################################
# # High symmetry points
# #################################
# #################################
# #################################

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]


# nbands=14 #Number of bands 
# kpath=lq.High_symmetry_path()
# # plt.scatter(kpath[:,0],kpath[:,1])
# # VV=lq.boundary()
# # plt.plot(VV[:,0], VV[:,1])
# # plt.show()
# Npoi=np.shape(kpath)[0]
# hpl=Hamiltonian.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
# for l in range(Npoi):
#     # h.umklapp_lattice()
#     # break
#     E1,wave1=hpl.eigens(kpath[l,0],kpath[l,1],nbands)
#     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
#     psi_plus_a.append(wave1)


#     E1,wave1=hmin.eigens(kpath[l,0],kpath[l,1],nbands)
#     Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
# Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

# print(np.shape(Ene_valley_plus_a))
# qa=np.linspace(0,1,Npoi)
# for i in range(nbands):
#     plt.plot(qa,Ene_valley_plus[:,i] , c='b')
#     plt.plot(qa,Ene_valley_min[:,i] , c='r', ls="--")
# plt.xlim([0,1])
# plt.ylim([-0.08,0.08])
# plt.show()

# bds1=[]
# for i in range(nbands):
#     bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
#     print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))


# print("minimum bw_plus was:",np.min(np.array(bds1)))

# #################################
# #################################
# #################################
# #  Testing FS contour method and square grid interpolation
# #################################
# #################################
# #################################

# hpl=Hamiltonian.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM(hvkd, alph, -1, lq,kappa,PH)

# [f_interp,k_window_sizex,k_window_sizey]=hpl.FSinterp(False,False, mu)

# [xFS_dense,yFS_dense]=hpl.FS_contour( 200, mu)
# plt.scatter(xFS_dense,yFS_dense)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

# #################################
# #################################
# #################################
# # Wavefunctions
# #################################
# #################################
# #################################


Ene_valley_plus_a=np.empty((0))
Ene_valley_min_a=np.empty((0))
psi_plus_a=[]
psi_min_a=[]
rot_C2z=lq.C2z
rot_C3z=lq.C3z

nbands=14 #Number of bands 
kpath=lq.High_symmetry_path()
# plt.scatter(kpath[:,0],kpath[:,1])
# VV=lq.boundary()
# plt.plot(VV[:,0], VV[:,1])
# plt.show()
Npoi=np.shape(kpath)[0]
hpl=Hamiltonian.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
hmin=Hamiltonian_min.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
Edif=[]
overlaps=[]
for l in range(Npoi):
    # h.umklapp_lattice()
    # break
    E1p,wave1p=hpl.eigens(kpath[l,0],kpath[l,1],nbands)
    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
    psi_plus_a.append(wave1p)


    E1m,wave1m=hmin.eigens(-kpath[l,0],-kpath[l,1],nbands)
    Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
    psi_min_a.append(wave1m)

    Edif.append(E1p-E1m)

    psi1=np.conj(wave1p[:,6])
    psi2=hmin.Op_rot_psi( wave1m[:,6] , rot_C2z)
    ov=np.array(np.abs(np.conj(psi1.T)@psi2 )).flatten() [0]
    overlaps.append(ov)
    printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

plt.plot(Edif)
plt.show()

plt.plot(np.array(overlaps))
plt.ylim([0,2])
plt.show()

print(np.shape(Ene_valley_plus_a))
qa=np.linspace(0,1,Npoi)
for i in range(nbands):
    plt.plot(qa,Ene_valley_plus[:,i] , c='b')
    plt.plot(qa,Ene_valley_min[:,i] , c='r', ls="--")
plt.xlim([0,1])
plt.ylim([-0.08,0.08])
plt.show()

bds1=[]
for i in range(nbands):
    bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
    print("bandwidth plus,",int(i),np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))


print("minimum bw_plus was:",np.min(np.array(bds1)))



# e=time.time()


# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]

# Ene_valley_plus_a2=np.empty((0))
# Ene_valley_min_a2=np.empty((0))
# psi_plus_a2=[]
# psi_min_a2=[]

# rot_C2z=lq.C2z
# rot_C3z=lq.C3z
# nbands=14 #Number of bands 
# kpath=lq.High_symmetry_path()
# # plt.scatter(kpath[:,0],kpath[:,1])
# # VV=lq.boundary()
# # plt.plot(VV[:,0], VV[:,1])
# # plt.show()
# Npoi=np.shape(kpath)[0]
# hpl=Hamiltonian.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
# print("p1")
# # hpl2=Hamiltonian_2v.Ham_BM(hvkd, alph, 1, lq,kappa,PH)
# # print("p2")
# # hmin=Hamiltonian_min.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
# # print("m1")
# hmin=Hamiltonian.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
# print("m1")
# # hmin2=Hamiltonian_2v.Ham_BM(hvkd, alph, -1, lq,kappa,PH)
# # print("m2")


# overlaps=[]

# randind=10
# E1_p,wave1_p=hpl.eigens(kpath[randind,0],kpath[randind,1],nbands)
# Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1_p)
# psi_plus_a.append(wave1_p)

# # E1_p2,wave1_p2=hpl2.eigens(kpath[randind,0],kpath[randind,1],nbands)
# # Ene_valley_plus_a2=np.append(Ene_valley_plus_a2,E1_p2)
# # psi_plus_a2.append(wave1_p2)

# E1_m,wave1_m=hmin.eigens(-kpath[randind,0],-kpath[randind,1],nbands)
# Ene_valley_min_a=np.append(Ene_valley_min_a,E1_m)
# psi_min_a.append(wave1_m)

# # E1_m2,wave1_m2=hmin2.eigens(-kpath[randind,0],-kpath[randind,1],nbands)
# # Ene_valley_min_a2=np.append(Ene_valley_min_a2,E1_m2)
# # psi_min_a2.append(wave1_m2)

# print(kpath[randind,0],kpath[randind,1])
# # print(E1_m,E1_p)
# for nbn in range(nbands):
#     psi1=np.conj(wave1_p[:,nbn])
#     # psi1p=np.conj(wave1_p2[:,nbn])
#     # psi2=(wave1_m[:,nbn])
#     psi2=hmin.Op_rot_psi( wave1_m[:,nbn] , rot_C2z)
#     # psi2p=hmin2.Op_rot_psi( wave1_m2[:,nbn] , rot_C2z)

#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     # plt.plot(np.abs(psi2p))
#     plt.show()


#     # psi1=(wave1_p[:,nbn])
#     # psi2=wave1_m[:,nbn] 
#     # psi2=wave1_m[:,nbn] 
#     # print("nomr plus",np.dot(np.conj(wave1_p[:,nbn]).T,wave1_p[:,nbn]))
#     # print("C2z... initial ks",KX[l],KY[l], "...Final ks...", KXc2z[l],KYc2z[l] )
#     ov=np.array(np.abs(np.conj(psi1.T)@psi2 )).flatten() [0]
#     print("...overlap...",  ov)
#     overlaps.append(ov)
#     print(E1_m[nbn],E1_p[nbn])

# plt.plot(np.array(overlaps))
# plt.show()

        
