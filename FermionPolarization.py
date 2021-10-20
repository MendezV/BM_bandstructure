import numpy as np
import scipy
from scipy import linalg as la
import time
import sys
import Hamiltonian
import MoireLattice
import matplotlib.pyplot as plt

#TODO: method for electron-phonon coupling Omega
#TODO: paralelize when calculating the eigenvalues
#TODO: executable for jobs an tidy workspace
#TODO: way of calculating that involves the derivative of the Lindhart
#TODO: cyprians calculation along momentum cut (in ee bubble method)

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
    print(f"\r{prefix} |{bar}| {percent}% {suffix} ", end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


class ee_Bubble:

    def __init__(self, latt, nbands, hpl, hmin, KX, KY):
        self.lq=latt
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        self.KX=KX
        self.KY=KY
        self.Npoi=np.size(KX)
        [self.KQX, self.KQY, self.Ik]=latt.Generate_momentum_transfer_lattice( KX, KY)
        self.Npoi_Q=np.size(self.KQX)
        [self.psi_plus,self.Ene_valley_plus,self.psi_min,self.Ene_valley_min]=self.precompute_E_psi()
        self.eta=np.mean( np.abs( np.diff( self.Ene_valley_plus[:,int(nbands/2)].flatten() )  ) )/2
        FFp=Hamiltonian.FormFactors(self.psi_plus, 1, latt)
        FFm=Hamiltonian.FormFactors(self.psi_min, 1, latt)
        self.L00p=FFp.denFF_s()
        self.L00m=FFm.denFF_s()
        self.dS_in=latt.VolMBZ/self.Npoi
        

    def nf(self, e, T):
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(1+np.exp( e/T ))
        else:
            return np.heaviside(-e,0.5)


    def nb(self, e, T):
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(np.exp( e/T )-1)
        else:
            return -np.heaviside(-e,0.5)


    def precompute_E_psi_noQ(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi):
            E1,wave1=self.hpl.eigens(self.KX[l],self.KY[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(self.KX[l],self.KY[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            psi_min_a.append(wave1)

            printProgressBar(l + 1, self.Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi,self.nbands])


        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]

    def precompute_E_psi(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi_Q):
            E1,wave1=self.hpl.eigens(self.KQX[l],self.KQY[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(self.KQX[l],self.KQY[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            psi_min_a.append(wave1)

            printProgressBar(l + 1, self.Npoi_Q, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi_Q,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi_Q,self.nbands])


        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]

    def integrand_ZT(self,nkq,nk,ekn,ekm,w,mu):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,1.0) # at zero its 1
        nfkq=np.heaviside(-edkq,1.0) #at zero is 1
        eps=self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)

    def integrand_T(self,nkq,nk,ekn,ekm,w,mu,T):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #finite temp
        nfk= self.nf(edk,T)
        nfkq= self.nf(edkq,T)

        eps=self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)

    def Compute(self, mu, omegas, kpath):

        integ=[]
        s=time.time()

        print("starting bubble.......")

        path=np.arange(0,self.Npoi)
        for omegas_m_i in omegas:
            sd=[]
            for l in path:  #for calculating only along path in FBZ
                bub=0
                
                qx=kpath[int(l), 0]
                qy=kpath[int(l), 1]
                Ikq=[]
                for s in range(self.Npoi):
                    kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
                    indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
                    Ikq.append(indmin)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.L00p[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])
                Lambda_Tens_min_kq_k=np.array([self.L00m[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])


                integrand_var=0
                #####all bands for the + and - valley
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        ek_n=self.Ene_valley_plus[:,nband]
                        ek_m=self.Ene_valley_plus[:,mband]
                        Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        # integrand_var=integrand_var+(Lambda_Tens_plus_k_kq_mn)*(Lambda_Tens_plus_kq_k_nm)*integrand(Ikq,Ik,ek_n,ek_m,omegas_m_i,mu,T)
                        

                        ek_n=self.Ene_valley_min[:,nband]
                        ek_m=self.Ene_valley_min[:,mband]
                        Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        

                e=time.time()
            
                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
            
        integ_arr_no_reshape=np.array(integ)#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",e-s)
        return integ_arr_no_reshape


        
class ep_Bubble:

    def __init__(self, latt, nbands, hpl, hmin, KX, KY, symmetric):
        self.lq=latt
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        self.KX=KX
        self.KY=KY
        self.Npoi=np.size(KX)
        [self.KQX, self.KQY, self.Ik]=latt.Generate_momentum_transfer_lattice( KX, KY)
        self.Npoi_Q=np.size(self.KQX)
        [self.psi_plus,self.Ene_valley_plus,self.psi_min,self.Ene_valley_min]=self.precompute_E_psi()
        self.eta=np.mean( np.abs( np.diff( self.Ene_valley_plus[:,int(nbands/2)].flatten() )  ) )/2
        FFp=Hamiltonian.FormFactors(self.psi_plus, 1, latt)
        FFm=Hamiltonian.FormFactors(self.psi_min, 1, latt)

        if symmetric:
            self.L00p=FFp.denFFL_s()
            self.L00m=FFm.denFFL_s()
            self.LnemLp=FFp.NemFFL_s()
            self.LnemLm=FFm.NemFFL_s()
            self.LnemTp=FFp.NemFFT_s()
            self.LnemTm=FFm.NemFFT_s()
        else:
            self.L00p=FFp.denFFL_a()
            self.L00m=FFm.denFFL_a()
            self.LnemLp=FFp.NemFFL_a()
            self.LnemLm=FFm.NemFFL_a()
            self.LnemTp=FFp.NemFFT_a()
            self.LnemTm=FFm.NemFFT_a()
        self.dS_in=latt.VolMBZ/self.Npoi
        

    def nf(self, e, T):
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(1+np.exp( e/T ))
        else:
            return np.heaviside(-e,0.5)


    def nb(self, e, T):
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(np.exp( e/T )-1)
        else:
            return -np.heaviside(-e,0.5)


    def precompute_E_psi_noQ(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi):
            E1,wave1=self.hpl.eigens(self.KX[l],self.KY[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(self.KX[l],self.KY[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            psi_min_a.append(wave1)

            printProgressBar(l + 1, self.Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi,self.nbands])


        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]

    def precompute_E_psi(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi_Q):
            E1,wave1=self.hpl.eigens(self.KQX[l],self.KQY[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(self.KQX[l],self.KQY[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            psi_min_a.append(wave1)

            printProgressBar(l + 1, self.Npoi_Q, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi_Q,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi_Q,self.nbands])


        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]

    def integrand_ZT(self,nkq,nk,ekn,ekm,w,mu):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,1.0) # at zero its 1
        nfkq=np.heaviside(-edkq,1.0) #at zero is 1
        eps=self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)

    def integrand_T(self,nkq,nk,ekn,ekm,w,mu,T):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #finite temp
        nfk= self.nf(edk,T)
        nfkq= self.nf(edkq,T)

        eps=self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)

    def Compute(self, mu, omegas, kpath):

        integ=[]
        s=time.time()

        print("starting bubble.......")

        path=np.arange(0,self.Npoi)
        for omegas_m_i in omegas:
            sd=[]
            for l in path:  #for calculating only along path in FBZ
                bub=0
                
                qx=kpath[int(l), 0]
                qy=kpath[int(l), 1]
                Ikq=[]
                for s in range(self.Npoi):
                    kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
                    indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
                    Ikq.append(indmin)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.L00p[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])
                Lambda_Tens_min_kq_k=np.array([self.L00m[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])


                integrand_var=0
                #####all bands for the + and - valley
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        ek_n=self.Ene_valley_plus[:,nband]
                        ek_m=self.Ene_valley_plus[:,mband]
                        Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        # integrand_var=integrand_var+(Lambda_Tens_plus_k_kq_mn)*(Lambda_Tens_plus_kq_k_nm)*integrand(Ikq,Ik,ek_n,ek_m,omegas_m_i,mu,T)
                        

                        ek_n=self.Ene_valley_min[:,nband]
                        ek_m=self.Ene_valley_min[:,mband]
                        Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        

                e=time.time()
            
                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
            
        integ_arr_no_reshape=np.array(integ)#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",e-s)
        return integ_arr_no_reshape


    
        
def main() -> int:

    #TODO: input the ratio of the interlayer hoppings and the twist angle by user IO


    #parameters for the calculation
    theta=1.05*np.pi/180  # magic angle
    fillings = np.array([0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
    mu_values = np.array([0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])

    
    try:
        filling_index=int(sys.argv[1]) #0-25

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose chemical potential for desired filling")

    try:
        N_SFs=25 #number of SF's currently implemented
        a=np.arange(N_SFs)
        a[filling_index]

    except (IndexError):
        raise Exception(f"Index has to be between 0 and {N_SFs-1}")


    try:
        Nsamp=int(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")

    #lattices with different normalizations
    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
    ln=MoireLattice.MoireTriangLattice(Nsamp,theta,1)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one
    [KX,KY]=lq.Generate_lattice()
    Npoi=np.size(KX)
    [q1,q1,q3]=l.q
    q=la.norm(q1)
    VV=lq.boundary()

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

    #JY params 
    # hbvf = 2.7; # eV
    # hvkd=hbvf*q
    # kappa=0.75
    # up = 0.105; # eV
    # u = kappa*up; # eV
    # alpha=up/hvkd
    # alph=alpha

    #other electronic params
    filling_index=int(sys.argv[1]) #0-25
    mu=mu_values[filling_index]/1000
    filling=fillings[filling_index]
    nbands=4
    hbarc=0.1973269804*1e-6 #ev*m
    alpha=137.0359895 #fine structure constant
    a_graphene=2.46*(1e-10) #in meters
    ee2=(hbarc/a_graphene)/alpha
    kappa_di=3.03

    #phonon parameters
    c_light=299792458 #m/s
    M=1.99264687992e-26 * 5.6095861672249e+38/1000 # [in units of eV]
    hhbar=6.582119569e-13 /1000 #(in eV s)
    sqrt_hbar_M=hhbar*np.sqrt(hhbar/M)*c_light
    alpha_ep=2 # in ev
    beta_ep=4 #in ev
    c_phonon=21400 #m/s
    omegacoef=hhbar*c_phonon/a_graphene #proportionality bw q and omega
    symmetric=True #whether we are looking at the symmetric or the antisymmetric mode

    print("kappa is..", kappa)
    print("alpha is..", alpha)


    [path,kpath,HSP_index]=lq.embedded_High_symmetry_path(KX,KY)
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(kpath[:,0],kpath[:,1], s=30, c='g' )
    plt.gca().set_aspect('equal')
    plt.show()

    hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
    hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)

    B1=ee_Bubble(lq, nbands, hpl, hmin, KX, KY)
    omega=[1e-14]
    kpath=np.array([KX,KY]).T
    integ=B1.Compute(mu, omega, kpath)
    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX,KY, s=20, c=np.real(integ))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("pKQenergycuttestreal_"+str(Nsamp)+"_nu_"+str(filling)+".png")
    plt.close()
    print("the minimum real part is ...", np.min(np.real(integ)))

    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX,KY, s=20, c=np.imag(integ))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("pKQenergycuttestimag_"+str(Nsamp)+"_nu_"+str(filling)+".png")
    plt.close()
    print("the maximum imaginary part is ...", np.max(np.imag(integ)))

    plt.plot(VV[:,0],VV[:,1])
    plt.scatter(KX,KY, s=20, c=np.abs(integ))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("pKQenergycuttestabs_"+str(Nsamp)+"_nu_"+str(filling)+".png")
    plt.close()




    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
