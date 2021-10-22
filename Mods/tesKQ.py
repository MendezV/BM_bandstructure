import numpy as np
import Hamiltonian
import MoireLattice
import matplotlib.pyplot as plt
import sys
import numpy.linalg as la
import time
 
#TODO: implement the new test for the KQ lattice see whats failing
fillings = np.array([0,0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
mu_values = np.array([0,0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])

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

# # #################################
# # #################################
# # #################################
# # # Wavefunctions
# # #################################
# # #################################
# # #################################


# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]
# rot_C2z=lq.C2z
# rot_C3z=lq.C3z

# nbands=14 #Number of bands 
# kpath=lq.High_symmetry_path()
# # plt.scatter(kpath[:,0],kpath[:,1])
# # VV=lq.boundary()
# # plt.plot(VV[:,0], VV[:,1])
# # plt.show()
# Npoi=np.shape(kpath)[0]
# hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
# Edif=[]
# overlaps=[]
# for l in range(Npoi):
#     # h.umklapp_lattice()
#     # break
#     E1p,wave1p=hpl.eigens(kpath[l,0],kpath[l,1],nbands)
#     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
#     psi_plus_a.append(wave1p)


#     E1m,wave1m=hmin.eigens(kpath[l,0],kpath[l,1],nbands)
#     Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
#     psi_min_a.append(wave1m)

#     Edif.append(E1p-E1m)

#     psi1=np.conj(wave1p[:,6])
#     psi2=hmin.Op_rot_psi( wave1m[:,6] , rot_C2z)
#     ov=np.array(np.abs(np.conj(psi1.T)@psi2 )).flatten() [0]
#     overlaps.append(ov)
#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
# Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

# # plt.plot(Edif)
# # plt.show()

# # plt.plot(np.array(overlaps))
# # plt.ylim([0,2])
# # plt.show()

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


# # #################################
# # #################################
# # #################################
# # # Disp after all changes above
# # #################################
# # #################################
# # #################################

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]
# rot_C2z=lq.C2z
# rot_C3z=lq.C3z

# nbands=14 #Number of bands 
# kpath=lq.High_symmetry_path()
# # plt.scatter(kpath[:,0],kpath[:,1])
# # VV=lq.boundary()
# # plt.plot(VV[:,0], VV[:,1])
# # plt.show()
# Npoi=np.shape(kpath)[0]
# hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
# Edif=[]
# overlaps=[]
# for l in range(Npoi):
#     # h.umklapp_lattice()
#     # break
#     E1p,wave1p=hpl.eigens(kpath[l,0],kpath[l,1],nbands)
#     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
#     psi_plus_a.append(wave1p)


#     E1m,wave1m=hmin.eigens(kpath[l,0],kpath[l,1],nbands)
#     Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
#     psi_min_a.append(wave1m)

#     Edif.append(E1p-E1m)

#     psi1=np.conj(wave1p[:,6])
#     psi2=hmin.Op_rot_psi( wave1m[:,6] , rot_C2z)
#     ov=np.array(np.abs(np.conj(psi1.T)@psi2 )).flatten() [0]
#     overlaps.append(ov)
#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
# Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

# # plt.plot(Edif)
# # plt.show()

# # plt.plot(np.array(overlaps))
# # plt.ylim([0,2])
# # plt.show()

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



# # #################################
# # #################################
# # #################################
# # # Form factors
# # #################################
# # #################################
# # #################################

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]

# rot_C2z=lq.C2z
# rot_C3z=lq.C3z
# # print("starting dispersion ..........")
# # # for l in range(Nsamp*Nsamp):
# s=time.time()
# hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
# hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
# overlaps=[]
# for l in range(Npoi):
#     E1p,wave1p=hpl.eigens(KX[l],KY[l],nbands)
#     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
#     psi_plus_a.append(wave1p)


#     E1m,wave1m=hmin.eigens(-KX[l],-KY[l],nbands)
#     Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
#     psi_min_a.append(wave1m)

#     psi1=np.conj(wave1p[:,3])
#     psi2=hmin.Op_rot_psi( wave1m[:,3] , rot_C2z)
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     # ov=np.array(np.abs(np.conj(psi1.T)@psi2 )).flatten() [0]
#     overlaps.append(ov)

#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# print((overlaps))
# print(np.abs(overlaps))

# plt.plot(np.abs(overlaps))
# plt.plot(np.real(overlaps))
# plt.show()
# e=time.time()
# print("time to diag over MBZ", e-s)
# ##relevant wavefunctions and energies for the + valley
# psi_plus=np.array(psi_plus_a)
# Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])

# psi_min=np.array(psi_min_a)
# Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

# FFp=Hamiltonian.FormFactors(psi_plus, 1, lq)
# L00p=FFp.denFF_s()
# FFm=Hamiltonian.FormFactors(psi_min, -1, lq)
# L00m=FFm.denFF_s()
# print(np.shape(L00p),np.shape(L00m) )

######transpose complex conj plus
# for i in range(4):
#     for j in range(4):
#         abs1=[]
#         abs2=[]
#         abs3=[]
#         for k in range(Npoi):
#             for kp in range(Npoi):
#                 abs1.append(np.abs(np.conj(L00p[k,i,kp,j])-L00p[kp,j,k,i] ))
#                 abs2.append(np.abs((L00p[k,i,kp,j])))
#                 abs3.append(np.abs(L00p[kp,j,k,i] ))
#         plt.plot(abs1, c='r')
#         plt.plot(abs2, c='b')
#         plt.plot(abs3,c='k')
# plt.show()


# # ######transpose complex conj minus
# for i in range(4):
#     for j in range(4):
#         abs1=[]
#         abs2=[]
#         abs3=[]
#         for k in range(Npoi):
#             for kp in range(Npoi):
#                 abs1.append(np.abs(np.conj(L00m[k,i,kp,j])-L00m[kp,j,k,i] ))
#                 abs2.append(np.abs((L00m[k,i,kp,j])))
#                 abs3.append(np.abs(L00m[kp,j,k,i] ))
#         plt.plot(abs1, c='b')
#         # plt.plot(abs2, c='b')
#         # plt.plot(abs3,c='k')
# plt.show()


###### complex conj valley flip
# [KXc2z,KYc2z, Indc2z]=lq.C2zLatt(KX,KY)
# # for i in range(4):
# for i in range(1,3):
#     # for j in range(4):
#     for j in range(1,3):
#         abs1=[]
#         abs2=[]
#         abs3=[]
#         for k in range(Npoi):
#             for kp in range(Npoi):
#                 # print(kp,Indc2z[kp])
#                 abs1.append(  np.abs(np.conj(L00p[k,i,kp,j])) - np.abs(L00m[k,i,kp,j] ) )
#                 abs2.append(np.abs((L00p[k,i,kp,j])))
#                 abs3.append(np.abs(L00m[int(Indc2z[k]),i,int(Indc2z[kp]),j] ))

#                 # print("1:  ",np.conj(L00p[k,i,kp,j]))
#                 # print("1lat:  ", KX[k], KY[k], " ; ",KX[kp], KY[kp],)
#                 # print("2:  ",L00m[int(Indc2z[k]),i,int(Indc2z[kp]),j])
#                 # print("2lat:  ", KX[int(Indc2z[k])], KY[int(Indc2z[k])], " ; ",KX[int(Indc2z[kp])], KY[int(Indc2z[kp])],)
#         plt.plot(abs1, label=str(i)+"  "+str(j))
#         # plt.plot(abs2, c='b')
#         # plt.plot(abs3,c='k')

# plt.legend()
# plt.show()

##plots dispersion
# bds1=[]
# for i in range(nbands):
#     plt.scatter(KX,KY, s=30, c=Ene_valley_plus[:,i])
#     bds1.append(np.max(Ene_valley_plus[:,i])-np.min(Ene_valley_plus[:,i]))
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("2plusvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()


# bds1=[]
# for i in range(nbands):
#     plt.scatter(KX,KY, s=30, c=Ene_valley_min[:,i])
#     bds1.append(np.max(Ene_valley_min[:,i])-np.min(Ene_valley_min[:,i]))
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.colorbar()
#     plt.savefig("2minvalley_E"+str(i)+"_size_"+str(Nsamp)+".png")
#     plt.close()


# # #################################
# # #################################
# # ##C3 symmetry vmin
# # #################################
# # #################################
# # #################################

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]

# rot_C2z=lq.C2z
# rot_C3z=lq.C3z

# [KXc3z,KYc3z, Indc3z]=lq.C3zLatt(KX,KY)
# [KXc3z2,KYc3z2, Indc3z2]=lq.C3zLatt(KXc3z,KYc3z)
# # print("starting dispersion ..........")
# # # for l in range(Nsamp*Nsamp):
# s=time.time()
# hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
# overlaps1=[]
# overlaps2=[]
# for l in range(Npoi):
    
#     E1m_unitary,wave1m_unitary=hmin.eigens(KX[l],KY[l],nbands)

#     E1m_rot0,wave1m_rot0=hmin.eigens(KXc3z[l],KYc3z[l],nbands)
    
#     E1m_rot1,wave1m_rot1=hmin.eigens(KXc3z2[l],KYc3z2[l],nbands)


#     psi1=wave1m_rot0[:,2]
#     psi2=hmin.C3unitary(hmin.Op_rot_psi( wave1m_unitary[:,2] , rot_C3z),1)
#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     plt.show()
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     print("0 sing rot + sing rot")
#     print(ov,np.abs(ov))
#     print(np.sum(np.abs(psi1)-np.abs(psi2)))

#     psi1=wave1m_rot0[:,2]
#     psi2=hmin.C3unitary(hmin.Op_rot_psi( wave1m_unitary[:,2] , rot_C3z),-1)
#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     plt.show()
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     print("1 sing rot - sing rot")
#     print(ov,np.abs(ov))
#     print(np.sum(np.abs(psi1)-np.abs(psi2)))

#     psi1=wave1m_rot0[:,2]
#     psi2=hmin.C3unitary(hmin.Op_rot_psi(hmin.Op_rot_psi( wave1m_unitary[:,2] , rot_C3z), rot_C3z),1)
#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     plt.show()
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     print("2 sing rot + double rot")
#     print(ov,np.abs(ov))
#     print(np.sum(np.abs(psi1)-np.abs(psi2)))

#     psi1=wave1m_rot0[:,2]
#     psi2=hmin.C3unitary(hmin.Op_rot_psi(hmin.Op_rot_psi( wave1m_unitary[:,2] , rot_C3z), rot_C3z),-1)
#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     plt.show()
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     print("3 sing rot - double rot")
#     print(ov,np.abs(ov))
#     print(np.sum(np.abs(psi1)-np.abs(psi2)))


#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# print((overlaps1))
# print(np.abs(overlaps1))

# plt.plot(np.abs(overlaps1))
# plt.plot(np.abs(overlaps2))
# plt.show()
# e=time.time()
# print("time to diag over MBZ", e-s)




# # #################################
# # #################################
# # ##C3 symmetry vmin
# # #################################
# # #################################
# # #################################

# Ene_valley_plus_a=np.empty((0))
# Ene_valley_min_a=np.empty((0))
# psi_plus_a=[]
# psi_min_a=[]

# rot_C2z=lq.C2z
# rot_C3z=lq.C3z

# [KXc3z,KYc3z, Indc3z]=lq.C3zLatt(KX,KY)
# [KXc3z2,KYc3z2, Indc3z2]=lq.C3zLatt(KXc3z,KYc3z)
# # print("starting dispersion ..........")
# # # for l in range(Nsamp*Nsamp):
# s=time.time()
# hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
# overlaps1=[]
# overlaps2=[]
# for l in range(Npoi):
    
#     E1m_unitary,wave1m_unitary=hpl.eigens(KX[l],KY[l],nbands)

#     E1m_rot0,wave1m_rot0=hpl.eigens(KXc3z[l],KYc3z[l],nbands)
    
#     E1m_rot1,wave1m_rot1=hpl.eigens(KXc3z2[l],KYc3z2[l],nbands)


#     psi1=wave1m_rot0[:,2]
#     psi2=hpl.C3unitary(hpl.Op_rot_psi( wave1m_unitary[:,2] , rot_C3z),1)
#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     plt.plot(np.abs(np.abs(psi1)-np.abs(psi2)))
#     plt.axhline(np.mean(np.abs(np.abs(psi1)-np.abs(psi2))))
#     plt.show()
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     print("0 sing rot + sing rot")
#     print(ov,np.abs(ov))
#     print(np.mean(np.abs(np.abs(psi1)-np.abs(psi2))))

#     psi1=wave1m_rot0[:,2]
#     psi2=hpl.C3unitary(hpl.Op_rot_psi( wave1m_unitary[:,2] , rot_C3z),-1)
#     plt.plot(np.abs(psi1))
#     plt.plot(np.abs(psi2))
#     plt.plot(np.abs(np.abs(psi1)-np.abs(psi2)))
#     plt.axhline(np.mean(np.abs(np.abs(psi1)-np.abs(psi2))))
#     plt.show()
#     ov=np.array((np.conj(psi1.T)@psi2 )).flatten()[0]
#     print("1 sing rot - sing rot")
#     print(ov,np.abs(ov))
#     print(np.mean(np.abs(np.abs(psi1)-np.abs(psi2))))


#     printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

# print((overlaps1))
# print(np.abs(overlaps1))

# plt.plot(np.abs(overlaps1))
# plt.plot(np.abs(overlaps2))
# plt.show()
# e=time.time()
# print("time to diag over MBZ", e-s)




# # #################################
# # #################################
# # #################################
# # # Form factors C3
# # #################################
# # #################################
# # #################################

Ene_valley_plus_a=np.empty((0))
Ene_valley_plus_ac3=np.empty((0))
psi_plus_a=[]
psi_plus_ac3=[]

rot_C2z=lq.C2z
rot_C3z=lq.C3z
[KXc3z,KYc3z, Indc3z]=lq.C3zLatt(KX,KY)
# print("starting dispersion ..........")
# # for l in range(Nsamp*Nsamp):
s=time.time()
hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
overlaps=[]
nbands=2
for l in range(Npoi):
    E1p,wave1p=hpl.eigens(KX[l],KY[l],nbands)
    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
    psi_plus_a.append(wave1p)


    E1p_c3z,wave1p_c3z=hpl.eigens(KXc3z[l],KYc3z[l],nbands)
    wave1p_c3z_rot=hpl.Op_rot_psi( wave1p_c3z , rot_C3z)
    Ene_valley_plus_ac3=np.append(Ene_valley_plus_ac3,E1p_c3z)
    psi_plus_ac3.append(wave1p_c3z)


    printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

e=time.time()
print("time to diag over MBZ", e-s)
##relevant wavefunctions and energies for the + valley
psi_plus=np.array(psi_plus_a)
Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])

psi_plusc3=np.array(psi_plus_ac3)
Ene_valley_plusc3= np.reshape(Ene_valley_plus_ac3,[Npoi,nbands])

print(np.shape(psi_plus),np.shape(psi_plusc3))

FFp=Hamiltonian.FormFactors(psi_plus, 1, lq)
L00p=FFp.NemFFL_a_plus()
FFc3=Hamiltonian.FormFactors(psi_plusc3, 1, lq)
L00m=FFc3.NemFFL_a_plus()
print(np.shape(L00p),np.shape(L00m) )

ind0=np
#####transpose complex conj plus


diffar=[]
K=[]
KP=[]
cos1=[]
cos2=[]
# for k in range(Npoi):
#     for kp in range(Npoi):
#         K.append(KX[k]-KX[kp])
#         KP.append(KY[k]-KY[kp])
#         undet=np.abs(np.linalg.det(L00p[k,:,kp,:]))
#         dosdet=np.abs(np.linalg.det(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
#         diffar.append( undet - dosdet )
#         cos1.append(undet)
#         cos2.append(dosdet)
kp=np.argmin(KX**2+KY**2)
for k in range(Npoi):
    undet=np.abs(np.linalg.det(L00p[k,:,kp,:]))
    dosdet=np.abs(np.linalg.det(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
    diffar.append( undet - dosdet )
    cos1.append(undet)
    cos2.append(dosdet)

plt.plot(diffar)
plt.show()

plt.scatter(KX,KY,c=cos1)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.scatter(KX,KY,c=cos2)
plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax = plt.axes(projection='3d')

ax.scatter3D(KX,KY,cos1, c=cos1);
plt.show()

# # #################################
# # #################################
# # #################################
# # # Form factors C3 minus valley
# # #################################
# # #################################
# # #################################

Ene_valley_plus_a=np.empty((0))
Ene_valley_plus_ac3=np.empty((0))
psi_plus_a=[]
psi_plus_ac3=[]

rot_C2z=lq.C2z
rot_C3z=lq.C3z
[KXc3z,KYc3z, Indc3z]=lq.C3zLatt(KX,KY)
# print("starting dispersion ..........")
# # for l in range(Nsamp*Nsamp):
s=time.time()
hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
overlaps=[]
nbands=2
for l in range(Npoi):
    E1p,wave1p=hmin.eigens(KX[l],KY[l],nbands)
    Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
    psi_plus_a.append(wave1p)


    printProgressBar(l + 1, Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

e=time.time()
print("time to diag over MBZ", e-s)
##relevant wavefunctions and energies for the + valley
psi_plus=np.array(psi_plus_a)
Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])



print(np.shape(psi_plus),np.shape(psi_plusc3))

FFp=Hamiltonian.FormFactors(psi_plus, -1, lq)
L00p=FFp.NemFFL_a()

ind0=np
#####transpose complex conj plus
diffar=[]
for i in range(nbands):
    for j in range(nbands):
        abs1=[]
        abs2=[]
        abs3=[]
        for k in range(Npoi):
            for kp in range(Npoi):
                # print(KX[k]-KX[kp],KY[k]-KY[kp], KXc3z[k]-KXc3z[kp],KYc3z[k]-KYc3z[kp])
                # print(np.abs(np.linalg.det(L00p[k,:,kp,:]))-np.abs(np.linalg.det(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:])))
                diffar.append(   np.abs(np.linalg.det(L00p[k,:,kp,:]))-np.abs(np.linalg.det(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:]))   )
                # abs1.append(np.abs(L00p[k,i,kp,j]-L00m[kp,j,k,i] ))
#         plt.plot(abs1, c='r')
# plt.show()
plt.plot(diffar)
plt.show()

