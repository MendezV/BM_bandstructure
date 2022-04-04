import numpy as np
import MoireLattice 
import matplotlib.pyplot as plt
from scipy import linalg as la
 
paulix=np.array([[0,1],[1,0]])
pauliy=np.array([[0,-1j],[1j,0]])
pauliz=np.array([[1,0],[0,-1]])
modulation_thet=1.05
Nsamp=6


#Lattice parameters 
#lattices with different normalizations 
theta=modulation_thet*np.pi/180  # magic angle 
l=MoireLattice.MoireTriangLattice(Nsamp,theta,0) 
lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one 

[KX1bz,KY1bz]=lq.Generate_lattice()
[KX,KY]=lq.Generate_Umklapp_lattice2(KX1bz, KY1bz,1) #for the momentum transfer lattice
[KQX,KQY]=lq.Generate_Umklapp_lattice2(KX1bz, KY1bz,2) #for the momentum transfer lattice
Npoi1bz=np.size(KX1bz); print(Npoi1bz, "1bz numer of sampling lattice points")
Npoi=np.size(KX); print(Npoi, "X numer of sampling lattice points")
NpoiQ=np.size(KQX); print(NpoiQ, "Q numer of sampling lattice points")
[q1,q2,q3]=l.q
q=la.norm(q1)
umkl=0
print(f"taking {umkl} umklapps")
VV=lq.boundary()

plt.scatter(KX,KY)
plt.savefig("tt.png")
plt.close()

def anaeig_p(kx,ky):
    kv=np.array([kx,ky])
    knorm=np.sqrt(kv.T@kv)
    k=kx+1j*ky
    kbar=kx-1j*ky        
    hk_n=knorm
        
    psi2=np.array([+1*np.exp(1j*np.angle(kbar)), 1])/np.sqrt(2)
    psi1=np.array([-1*np.exp(1j*np.angle(kbar)), 1])/np.sqrt(2)
    # Eigvals
    (Eigvals,Eigvect)=(np.array([-knorm, knorm]),np.array([psi1,psi2]).T)
    
    return Eigvals,Eigvect
    
def anaeig_m(kx,ky):
    kv=np.array([kx,ky])
    knorm=np.sqrt(kv.T@kv)
    mk=-kx-1j*ky
    mkbar=-kx+1j*ky
    
        
    psi2=np.array([+1*np.exp(1j*np.angle(mk)), 1])/np.sqrt(2)
    psi1=np.array([-1*np.exp(1j*np.angle(mk)), 1])/np.sqrt(2)
    # Eigvals
    (Eigvals,Eigvect)=(np.array([-knorm, knorm]),np.array([psi1,psi2]).T)
    
    return Eigvals,Eigvect

def calceig_p(kx,ky):
    mat=kx*paulix+ky*pauliy#+(1e-1)*pauliz
    (Eigvals,Eigvect)= la.eigh(mat)  #returns sorted eigenvalues
    nbands=2
    
    for nband in range(nbands):
        psi_p=Eigvect[:,nband]
        maxisind = np.unravel_index(np.argmax(np.abs(np.imag(psi_p)), axis=None), psi_p.shape)[0]
        phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
        Eigvect[:,nband]=Eigvect[:,nband]*np.exp(-1j*phas)
    
    if Eigvals[0]==Eigvals[1]:
        psi2=np.array([1, 1])/np.sqrt(2)
        psi1=np.array([-1, 1])/np.sqrt(2)
        Eigvect[:,0]=psi1
        Eigvect[:,1]=psi2
    return Eigvals,Eigvect

def calceig_m(kx,ky):
    mat=-kx*paulix+ky*pauliy#+(1e-1)*pauliz
    (Eigvals,Eigvect)= la.eigh(mat)  #returns sorted eigenvalues
    nbands=2

    for nband in range(nbands):
        psi_p=Eigvect[:,nband]
        maxisind = np.unravel_index(np.argmax(np.abs(np.imag(psi_p)), axis=None), psi_p.shape)[0]
        phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
        Eigvect[:,nband]=Eigvect[:,nband]*np.exp(-1j*phas)
    
    if Eigvals[0]==Eigvals[1]:
        psi2=np.array([1, 1])/np.sqrt(2)
        psi1=np.array([-1, 1])/np.sqrt(2)
        Eigvect[:,0]=psi1
        Eigvect[:,1]=psi2
        
        
    return Eigvals,Eigvect
Eup=np.zeros(Npoi)
Edown=np.zeros(Npoi)
psi=[]

ana_Eup=np.zeros(Npoi)
ana_Edown=np.zeros(Npoi)
ana_psi=[]
kxoff=[]
kyoff=[]
for i in range(Npoi):
    kx=KX[i]
    ky=KY[i]
    Eigvals_ana, Eigvect_ana=anaeig_m(kx,ky)
    Eigvals, Eigvect=calceig_m(kx,ky)
    avang=np.abs(np.mean(np.angle(Eigvect_ana/Eigvect)))
    if avang>1e-10:
        print(np.angle(Eigvect_ana/Eigvect), "here", kx,ky)
        kxoff.append(kx);kyoff.append(ky)
    
    Eup[i]=Eigvals[1]
    Edown[i]=Eigvals[0]
    psi.append(Eigvect)
    
plt.scatter(KX,KY,c=Eup)
plt.savefig("tt2.png")
plt.close()

kx_ind=np.where(KY==0)[0]
kx=KX[kx_ind]
Eup_p=Eup[kx_ind]
Edown_p=Edown[kx_ind]
plt.scatter(kx,Eup_p)
plt.scatter(kx, Edown_p)
plt.scatter(kx, np.abs(kx),marker='x')

plt.scatter(kx, -np.abs(kx),marker='x')
plt.savefig("tt3.png")
plt.close()

psiarr=np.array(psi)
Lambda_Tens=np.tensordot(np.conj(psiarr),psiarr, axes=([1],[1]))
print(np.shape(Lambda_Tens))
k_ind=np.argmin( np.array(KX)**2 +np.array(KY)**2)

tensdet=[]
for i in range(Npoi):
    dd=np.abs(la.det(np.abs(Lambda_Tens[k_ind,:,i,:])**2))
    tensdet.append(dd)
    
    
plt.scatter(KX,KY, c=tensdet)
plt.scatter(kxoff,kyoff, c='r')
plt.colorbar()
plt.savefig("tt4.png")
plt.close()


def Fdirac(kx,ky):
    mk=-kx-1j*ky
    mkbar=-kx+1j*ky
    phi=np.angle(mk)
    F=np.zeros([np.size(phi), 2, np.size(phi), 2]) +0*1j
    phi1, phi2=np.meshgrid(phi,phi)
    for n in range(2):
        for n2 in range(2):
            s1=(-1)**n
            s2=(-1)**n2
            F[:,n,:,n2]= ( 1+s1*s2*np.exp(1j*(phi1-phi2)) )/2  
            
    return F

Lambda_Tens=Fdirac(KX,KY)
print(np.shape(Lambda_Tens))
k_ind=np.argmin( np.array(KX)**2 +np.array(KY)**2)

tensdet=[]
for i in range(Npoi):
    dd=np.abs(la.det(np.abs(Lambda_Tens[k_ind,:,i,:])**2))
    tensdet.append(dd)
    
    
plt.scatter(KX,KY, c=tensdet)
plt.colorbar()

plt.savefig("tt5.png")
plt.close()



# Lambda_Tens=Fdirac(KQX,KQY)
# K=[]
# KP=[]
# cos1=[]
# cos2=[]
# for s in range(NpoiQ):
#     kp=np.argmin( (KQX-KQX[s])**2 +(KQY-KQY[s])**2)
#     undet=0
#     K.append(KQX[kp])
#     KP.append(KQY[kp])
#     for k in range(Npoi):
#         ks=np.argmin( (KQX-KX[k])**2 +(KQY-KY[k])**2)
#         undet=undet+np.abs(np.linalg.det(np.abs(Lambda_Tens[ks,:,kp,:])**2))  
#     cos1.append(undet/Npoi)
    
# plt.scatter(K,KP, c=cos1, vmin=0.630,vmax=0.640, s=3)
# plt.colorbar()
# plt.savefig("testdet.png")
# plt.close()



# Eup=np.zeros(NpoiQ)
# Edown=np.zeros(NpoiQ)
# psi=[]


# kxoff=[]
# kyoff=[]
# for i in range(NpoiQ):
#     kx=KQX[i]
#     ky=KQY[i]
#     Eigvals, Eigvect=calceig_p(kx,ky)

    
#     Eup[i]=Eigvals[1]
#     Edown[i]=Eigvals[0]
#     psi.append(Eigvect)

# psiarr=np.array(psi)
# Lambda_Tens=np.tensordot(np.conj(psiarr),psiarr, axes=([1],[1]))
# K=[]
# KP=[]
# cos1=[]
# cos2=[]
# for s in range(NpoiQ):
#     kp=np.argmin( (KQX-KQX[s])**2 +(KQY-KQY[s])**2)
#     undet=0
#     K.append(KQX[kp])
#     KP.append(KQY[kp])
#     for k in range(Npoi):
#         ks=np.argmin( (KQX-KX[k])**2 +(KQY-KY[k])**2)
#         undet=undet+np.abs(np.linalg.det(np.abs(Lambda_Tens[ks,:,kp,:])**2))  
#     cos1.append(undet/Npoi)
    
# kp=np.argmin( np.array(K)**2 +np.array(KP)**2)
# del K[kp]
# del KP[kp]
# del cos1[kp]

# plt.scatter(K,KP, c=cos1, vmin=0.630,vmax=0.640, s=3)
# plt.colorbar()
# plt.savefig("testdet2.png")
# plt.close()


Lambda_Tens=Fdirac(KQX,KQY)
K=[]
KP=[]
cos1=[]
cos2=[]
for s in range(Npoi):
    undet=0
    K.append(KX[s])
    KP.append(KY[s])
    for k in range(Npoi1bz):
        k1=np.argmin( (KQX-(KX1bz[k]+KX[s]))**2 +(KQY-(KY1bz[k]+KY[s]))**2)
        k2=np.argmin( (KQX-KX1bz[k])**2 +(KQY-KY1bz[k])**2)
        undet=undet+np.abs(np.linalg.det(np.abs(Lambda_Tens[k1,:,k2,:])**2))  
    cos1.append(undet/Npoi1bz)
    
kp=np.argmin( np.array(K)**2 +np.array(KP)**2)
del K[kp]
del KP[kp]
del cos1[kp]

a=np.where(np.array(KP)==0)[0]
print(np.where(np.array(KP)==0),np.where(np.array(K)==0))

plt.scatter(np.array(K)[a],np.array(cos1)[a])
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_cutx.png")
plt.close()



plt.scatter(K,KP, c=cos1)
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_.png")
plt.close()

plt.scatter(K,cos1)
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_cut.png")
plt.close()

plt.scatter(KP,cos1)
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_cut2.png")
plt.close()



########################

Eup=np.zeros(NpoiQ)
Edown=np.zeros(NpoiQ)
psi=[]


kxoff=[]
kyoff=[]
for i in range(NpoiQ):
    kx=KQX[i]
    ky=KQY[i]
    Eigvals, Eigvect=calceig_p(kx,ky)

    
    Eup[i]=Eigvals[1]
    Edown[i]=Eigvals[0]
    psi.append(Eigvect)

psiarr=np.array(psi)
Lambda_Tens=np.tensordot(np.conj(psiarr),psiarr, axes=([1],[1]))

K=[]
KP=[]
cos1=[]
cos2=[]
for s in range(Npoi):
    undet=0
    K.append(KX[s])
    KP.append(KY[s])
    for k in range(Npoi1bz):
        k1=np.argmin( (KQX-(KX1bz[k]+KX[s]))**2 +(KQY-(KY1bz[k]+KY[s]))**2)
        k2=np.argmin( (KQX-KX1bz[k])**2 +(KQY-KY1bz[k])**2)
        undet=undet+np.abs(np.linalg.det(np.abs(Lambda_Tens[k1,:,k2,:])**2))  
    cos1.append(undet/Npoi1bz)
    
kp=np.argmin( np.array(K)**2 +np.array(KP)**2)
del K[kp]
del KP[kp]
del cos1[kp]

plt.scatter(K,KP, c=cos1)
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_2.png")
plt.close()

plt.scatter(K,cos1)
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_2cut.png")
plt.close()

plt.scatter(KP,cos1)
plt.colorbar()
plt.savefig("TestC3_symm_det_0p_dirac_2cut2.png")
plt.close()