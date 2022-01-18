import numpy as np
import scipy
from scipy import linalg as la
import time
import sys
import Hamiltonian
import MoireLattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import concurrent.futures
import functools

#TODO: plot dets see if this controls width -- cannnot be if there is filling dependence 
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



        
class ep_Bubble:
    """
    A class used to represent electron phonon bubbles

    ...

    Attributes
    ----------
    
    number of bands
    hpl hmin hamiltonians for minus and plus valleys
    
    lattice
    umkl number of umklap processes to sum over
    KX1bz, KY1bz kpoints in the first bz
    KX KY K points including umklapps
    KQX KQY kpoints with an umkl+1 umklapps to account for momentum transfers
    NpoiXX number of points in each of the three latices above
    
    Methods
    -------
    
    
    """

    ################################
    """
    CONSTRUCTOR for the ep_Bubble class 
    """
    ################################
    
    def __init__(self, latt, nbands, hpl, hmin, symmetric, mode, cons , test, umkl):

        
        ################################
        #lattice attributes
        ################################
        self.latt=latt
        self.umkl=umkl
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice()
        [self.KX,self.KY]=latt.Generate_Umklapp_lattice2(self.KX1bz, self.KY1bz,self.umkl) #for the integration grid 
        [self.KQX,self.KQY]=latt.Generate_Umklapp_lattice2(self.KX1bz, self.KY1bz,self.umkl+1) #for the momentum transfer lattice
        self.Npoi1bz=np.size(self.KX1bz)
        self.Npoi=np.size(self.KX)
        self.NpoiQ=np.size(self.KQX)
        
        self.Ik=latt.insertion_index( self.KX,self.KY, self.KQX, self.KQY)
        [q1,q2,q3]=latt.qvect()
        self.qscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        self.dS_in=1/self.Npoi1bz
        
        
        ################################
        #dispersion attributes
        ################################
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        [self.psi_plus,self.Ene_valley_plus_1bz,self.psi_min,self.Ene_valley_min_1bz]=self.precompute_E_psi()
        
        self.Ene_valley_plus=self.hpl.ExtendE(self.Ene_valley_plus_1bz , self.umkl+1)
        self.Ene_valley_min=self.hmin.ExtendE(self.Ene_valley_min_1bz , self.umkl+1)
        
        
        ################################
        ###selecting eta
        ################################
        eps_l=[]
        for i in range(nbands):
            eps_l.append(np.mean( np.abs( np.diff( self.Ene_valley_plus_1bz[:,i].flatten() )  ) )/2)
            eps_l.append(np.mean( np.abs( np.diff( self.Ene_valley_min_1bz[:,i].flatten() )  ) )/2)
        eps_a=np.array(eps_l)
        eps=np.min(eps_a)
        
        
        self.eta=eps
        self.eta_dirac_delta=eps/4
        self.eta_cutoff=eps
        self.eta_small_imag=0.1*eps
        
        
        ################################
        #generating form factors
        ################################
        self.FFp=Hamiltonian.FormFactors_umklapp(self.psi_plus, 1, latt, self.umkl+1,self.hpl)
        self.FFm=Hamiltonian.FormFactors_umklapp(self.psi_min, -1, latt, self.umkl+1,self.hmin)
        
        
        [self.alpha_ep, self.beta_ep,  self.Wupsilon, self.agraph, self.mass ]=cons
        self.mode=mode
        self.symmetric=symmetric
        self.name="_mode_"+self.mode+"_symmetry_"+self.symmetric+"_alpha_"+str(self.alpha_ep)+"_beta_"+str(self.beta_ep)+"_umklp_"+str(umkl)+"_kappa_"+str(self.hpl.kappa)+"_theta_"+str(self.latt.theta)

        if symmetric=="s":
            if mode=="L":
                self.L00p=self.FFp.denqFFL_s()
                self.L00m=self.FFm.denqFFL_s()
                self.Lnemp=self.FFp.NemqFFL_s()
                self.Lnemm=self.FFm.NemqFFL_s()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()
            else: #Tmode
                self.Lnemp=self.FFp.NemqFFT_s()
                self.Lnemm=self.FFm.NemqFFT_s()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()
        else: # a- mode
            if mode=="L":
                self.L00p=self.FFp.denqFFL_a()
                self.L00m=self.FFm.denqFFL_a()
                self.Lnemp=self.FFp.NemqFFL_a()
                self.Lnemm=self.FFm.NemqFFL_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()
            else: #Tmode
                self.Lnemp=self.FFp.NemqFFT_a()
                self.Lnemm=self.FFm.NemqFFT_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()

        


        
        ################################
        #testing form factors for symmetry
        ################################
        if test:
            print("testing symmetry of the form factors...")
            [KXc3z,KYc3z, Indc3z]=self.latt.C3zLatt(self.KQX,self.KQY)
            diffarp=[]
            diffarm=[]
            K=[]
            KP=[]
            cos1=[]
            cos2=[]
            kp=np.argmin(self.KQX**2 +self.KQY**2)
            for k in range(self.NpoiQ):
                K.append(self.KQX[k]-self.KQX[kp])
                KP.append(self.KQY[k]-self.KQY[kp])
                #Regular FF
                # Plus Valley FF Omega
                undet=np.abs(np.linalg.det(self.Omega_FFp[k,:,kp,:]))
                dosdet=np.abs(np.linalg.det(self.Omega_FFp[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                cos1.append(undet)
                diffarp.append( undet   - dosdet   )
                # Minus Valley FF Omega
                undet=np.abs(np.linalg.det(self.Omega_FFm[k,:,kp,:]))
                dosdet=np.abs(np.linalg.det(self.Omega_FFm[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                cos2.append(undet)
                diffarm.append( undet   - dosdet   )
            
            #saving the data for the test
            np.save("TestC3_symm_diffp"+self.name+".npy",diffarp)
            np.save("TestC3_symm_diffm"+self.name+".npy",diffarm)
            np.save("TestC3_symm_K"+self.name+".npy",K)
            np.save("TestC3_symm_KP"+self.name+".npy",KP)
            np.save("TestC3_symm_det"+self.name+".npy",cos1)

            if np.mean(np.abs(diffarp))<1e-7:
                print("plus valley form factor passed the C3 symmetry test with average difference... ",np.mean(np.abs(diffarp)))
            else:
                print("failed C3 symmetry test, plus valley...",np.mean(np.abs(diffarp)))
                np.save("TestC3_symm_diffp"+self.name+".npy",diffarp)
                np.save("TestC3_symm_K"+self.name+".npy",K)
                np.save("TestC3_symm_KP"+self.name+".npy",KP)
                np.save("TestC3_symm_det"+self.name+".npy",cos1)
            if np.mean(np.abs(diffarm))<1e-7:
                print("minus valley form factor passed the C3 symmetry test with average difference... ",np.mean(np.abs(diffarm)))
            else:
                print("failed C3 symmetry test, minus valley...",np.mean(np.abs(diffarp)))
                np.save("TestC3_symm_diffp"+self.name+".npy",diffarm)
                np.save("TestC3_symm_K"+self.name+".npy",K)
                np.save("TestC3_symm_KP"+self.name+".npy",KP)
                np.save("TestC3_symm_det"+self.name+".npy",cos2)

            print("finished testing symmetry of the form factors...")



    ################################
    """
    METHODS for the ep_Bubble class 
    """
    ################################
    
    def nf(self, e, T):
        
        """[summary]
        fermi occupation function with a truncation if the argument exceeds 
        double precission

        Args:
            e ([double]): [description] energy
            T ([double]): [description] temperature

        Returns:
            [double]: [description] value of the fermi function 
        """
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(1+np.exp( e/T ))
        else:
            return np.heaviside(-e,0.5)


    def nb(self, e, T):
        """[summary]
        bose occupation function with a truncation if the argument exceeds 
        double precission

        Args:
            e ([double]): [description] energy
            T ([double]): [description] temperature

        Returns:
            [double]: [description] value of the fermi function 
        """
        rat=np.abs(np.max(e/T))
        if rat<700:
            return 1/(np.exp( e/T )-1)
        else:
            return -np.heaviside(-e,0.5)


    
    def precompute_E_psi(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz):
            E1,wave1=self.hpl.eigens(self.KX1bz[l],self.KY1bz[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(self.KX1bz[l],self.KY1bz[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            psi_min_a.append(wave1)

            # printProgressBar(l + 1, self.Npoi_Q, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi1bz,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi1bz,self.nbands])

        
        

        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]
    

    def OmegaL(self):
        

        
        
        Omega_FFp=(self.alpha_ep*self.L00p+self.beta_ep*self.Lnemp)#/np.sqrt(self.Npoi)
        Omega_FFm=(self.alpha_ep*self.L00m+self.beta_ep*self.Lnemm)#/np.sqrt(self.Npoi)
                
        return [Omega_FFp,Omega_FFm]

    def OmegaT(self):

        Omega_FFp=(self.beta_ep*self.Lnemp)
        Omega_FFm=(self.beta_ep*self.Lnemm)

        return [Omega_FFp,Omega_FFm]

    def integrand_ZT(self,nkq,nk,ekn,ekm,w,mu):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1
        eps=self.eta_small_imag

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        # fac_p=(nfkq-nfk)/(-(edkq-edk))
        return (fac_p)

    def deltad(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    def integrand_ZT_lh(self,nkq,nk,ekn,ekm,w,mu):
        
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1
        deltad_cut=1e-17*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-self.eta_cutoff, 0.0)/(deltad_cut-(edkq-edk))
        fac_p2=(self.deltad( edk, self.eta_dirac_delta))*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        
        return (fac_p+fac_p2)


    def integrand_T(self,nkq,nk,ekn,ekm,w,mu,T):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #finite temp
        nfk= self.nf(edk,T)
        nfkq= self.nf(edkq,T)

        eps=0.5*self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)

    def Compute_lh(self, mu, omegas, kpath):

        integ=[]
        sb=time.time()

        print("starting bubble.......",np.shape(kpath)[0])

        path=np.arange(0,np.shape(kpath)[0])
        for omegas_m_i in omegas:
            sd=[]
            for l in path:  #for calculating only along path in FBZ
                bub=0
                
                qx=kpath[int(l), 0]
                qy=kpath[int(l), 1]
                Ikq=[]
                
                Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.Omega_FFp[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])
                Lambda_Tens_min_kq_k=np.array([self.Omega_FFm[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])


                integrand_var=0
                #####all bands for the + and - valley
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        ek_n=self.Ene_valley_plus[:,nband]
                        ek_m=self.Ene_valley_plus[:,mband]
                        Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT_lh(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)


                        ek_n=self.Ene_valley_min[:,nband]
                        ek_m=self.Ene_valley_min[:,mband]
                        Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT_lh(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        

                eb=time.time()
            
                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
            
        integ_arr_no_reshape=np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        return integ_arr_no_reshape
    
    def Compute(self, mu, omegas, kpath):

        integ=[]
        sb=time.time()

        print("starting bubble.......",np.shape(kpath)[0])

        path=np.arange(0,np.shape(kpath)[0])
        
        for omegas_m_i in omegas:
            sd=[]
            for l in path:  #for calculating only along path in FBZ
                bub=0
                
                qx=kpath[int(l), 0]
                qy=kpath[int(l), 1]
                
                Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.Omega_FFp[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])
                Lambda_Tens_min_kq_k=np.array([self.Omega_FFm[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])


                integrand_var=0
                #####all bands for the + and - valley
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        ek_n=self.Ene_valley_plus[:,nband]
                        ek_m=self.Ene_valley_plus[:,mband]
                        Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)


                        ek_n=self.Ene_valley_min[:,nband]
                        ek_m=self.Ene_valley_min[:,mband]
                        Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        

                eb=time.time()
            
                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
            
        integ_arr_no_reshape=np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        return integ_arr_no_reshape
    
    def parCompute(self, theta, omegas, kpath, VV, Nsamp,  muv, fil, prop_BZ,ind):
        mu=muv[ind]
        integ=[]
        sb=time.time()

        print("starting bubble.......",np.shape(kpath)[0])

        path=np.arange(0,np.shape(kpath)[0])
        
        for omegas_m_i in omegas:
            sd=[]
            for l in path:  #for calculating only along path in FBZ
                bub=0
                
                qx=kpath[int(l), 0]
                qy=kpath[int(l), 1]
                
                Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.Omega_FFp[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])
                Lambda_Tens_min_kq_k=np.array([self.Omega_FFm[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])


                integrand_var=0
                #####all bands for the + and - valley
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        ek_n=self.Ene_valley_plus[:,nband]
                        ek_m=self.Ene_valley_plus[:,mband]
                        Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                        # Lambda_Tens_plus_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)


                        ek_n=self.Ene_valley_min[:,nband]
                        ek_m=self.Ene_valley_min[:,mband]
                        Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                        # Lambda_Tens_min_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        

                eb=time.time()
            
                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
        
        integ_arr_no_reshape=np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        
        
        print("the filling is .. " , fil[ind])
        integ=integ_arr_no_reshape.flatten()  #in ev
        popt, res, c, resc=self.extract_cs( integ, prop_BZ)
        integ= self.Wupsilon*integ
        print("parameters of the fit...", c)
        print("residual of the fit...", res)

        self.plot_res( integ, self.KX1bz,self.KY1bz, VV, fil[ind], Nsamp,c, res, str(theta)+"_ieps_")
        
        return [integ,res, c]

    def parCompute_lh(self, theta, omegas, kpath, VV, Nsamp,  muv, fil, prop_BZ,ind):
        mu=muv[ind]
        integ=[]
        sb=time.time()

        print("starting bubble.......",np.shape(kpath)[0])

        path=np.arange(0,np.shape(kpath)[0])
        
        for omegas_m_i in omegas:
            sd=[]
            for l in path:  #for calculating only along path in FBZ
                bub=0
                
                qx=kpath[int(l), 0]
                qy=kpath[int(l), 1]
                
                Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.Omega_FFp[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])
                Lambda_Tens_min_kq_k=np.array([self.Omega_FFm[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi)])


                integrand_var=0
                #####all bands for the + and - valley
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        ek_n=self.Ene_valley_plus[:,nband]
                        ek_m=self.Ene_valley_plus[:,mband]
                        Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                        # Lambda_Tens_plus_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT_lh(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)


                        ek_n=self.Ene_valley_min[:,nband]
                        ek_m=self.Ene_valley_min[:,mband]
                        Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                        # Lambda_Tens_min_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                        integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT_lh(Ikq,self.Ik,ek_n,ek_m,omegas_m_i,mu)
                        

                eb=time.time()
            
                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
        
        integ_arr_no_reshape=np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        
        
        print("the filling is .. " ,  fil[ind])
        integ=integ_arr_no_reshape.flatten() #in ev
        popt, res, c, resc=self.extract_cs( integ, prop_BZ)
        integ= self.Wupsilon*integ
        print("parameters of the fit...", c)
        print("residual of the fit...", res)

        self.plot_res( integ, self.KX1bz,self.KY1bz, VV, fil[ind], Nsamp,c, res, str(theta)+"_lh_")
        
        return [integ, res, c]

    def extract_cs(self, integ, prop_BZ):
        qq=self.qscale/self.agraph
        scaling_fac= self.Wupsilon/(qq*qq*self.mass)
        # scaling_fac= self.Wupsilon*(self.agraph**2) /(self.mass*(self.qscale**2))
        print("scalings...",scaling_fac,  self.Wupsilon)
        [KX_m, KY_m, ind]=self.latt.mask_KPs( self.KX1bz,self.KY1bz, prop_BZ)
        # plt.scatter(self.KX1bz,self.KY1bz, c=np.real(integ))
        # plt.show()
        id=np.ones(np.shape(KX_m))
        Gmat=np.array([ KX_m**2,KY_m**2]).T
        GTG=Gmat.T@Gmat
        d=np.real(integ[ind])
        plt.scatter(KX_m, KY_m, c=d)
        plt.colorbar()
        plt.savefig("fit.png")
        plt.close()
        b=Gmat.T@d
        popt=la.pinv(GTG)@b
        res=np.sqrt(np.sum((Gmat@popt-d)**2)) #residual of the least squares procedure
        nn=np.sqrt(np.sum((d)**2)) #residual of the least squares procedure
        print(popt, res, np.sqrt(np.mean(popt)*scaling_fac),np.sqrt(res*scaling_fac/nn))
        return popt, res/nn, np.sqrt(np.mean(popt)*scaling_fac),np.sqrt(res*scaling_fac/nn)

    
    def extract_cs_path(self, integ, kpath):
        qq=self.qscale/self.agraph
        scaling_fac= self.Wupsilon/(qq*qq*self.mass)

        # scaling_fac= self.Wupsilon*(self.agraph**2) /(self.mass*(self.qscale**2))
        qx=kpath[:, 0]
        qy=kpath[:, 1]
        # plt.scatter(self.KX1bz,self.KY1bz, c=np.real(integ))
        # plt.show()
        Gmat=np.array([qx**2,qy**2]).T
        GTG=Gmat.T@Gmat
        d=np.real(integ)
        plt.scatter(qx,qy, c=d)
        plt.colorbar()
        plt.savefig("fit2.png")
        plt.close()
        b=Gmat.T@d
        popt=la.pinv(GTG)@b
        plt.scatter(qx,qy, c=(popt[0]*qx*qx+popt[1]*qy*qy))
        plt.colorbar()
        plt.savefig("fit3.png")
        plt.close()
        res=np.sqrt(np.sum((Gmat@popt-d)**2)) #residual of the least squares procedure
        nn=np.sqrt(np.sum((d)**2)) #residual of the least squares procedure
        print(popt, res, np.sqrt(np.mean(popt)*scaling_fac),np.sqrt(res*scaling_fac/nn))
        return popt, res/nn, np.sqrt(np.mean(popt)*scaling_fac),np.sqrt(res*scaling_fac/nn)


    def quad_pi(self, x, y, popt):
        return popt[0]+popt[1]*x+ popt[2]*y+ popt[3]*x**2+popt[4]*x*y+ popt[5]*y**2
    
    def plot_res(self, integ, KX,KY, VV, filling, Nsamp, c , res, add_tag):
        identifier=add_tag+str(Nsamp)+"_nu_"+str(filling)+self.name
        
        plt.plot(VV[:,0],VV[:,1])
        plt.scatter(KX,KY, s=20, c=np.real(integ))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.savefig("Pi_ep_energy_cut_real_"+identifier+".png")
        plt.close()
        print("the minimum real part is ...", np.min(np.real(integ)))

        plt.plot(VV[:,0],VV[:,1])
        plt.scatter(KX,KY, s=20, c=np.abs(integ))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.savefig("Pi_ep_energy_cut_abs_"+identifier+".png")
        plt.close()

        print("saving data from the run ...")

        with open("bubble_data_"+identifier+".npy", 'wb') as f:
            np.save(f, integ)
        with open("bubble_data_kx_"+identifier+".npy", 'wb') as f:
            np.save(f, KX)
        with open("bubble_data_ky_"+identifier+".npy", 'wb') as f:
            np.save(f, KY)
        with open("cparams_data_"+identifier+".npy", 'wb') as f:
            np.save(f, c)
        with open("cparams_res_data_"+identifier+".npy", 'wb') as f:
            np.save(f, res)


    def Fill_sweep(self,fillings, mu_values,VV, Nsamp, c_phonon,theta):
        prop_BZ=0.15
        cs=[]
        cs_lh=[]
        rs=[]
        rs_lh=[]
        selfE=[]
        selfE_lh=[]
        
        
        qp=np.arange(np.size(fillings))
        s=time.time()
        omega=[1e-14]
        kpath=np.array([self.KX1bz,self.KY1bz]).T
        calc = functools.partial(self.parCompute, theta, omega, kpath, VV, Nsamp, mu_values,fillings,prop_BZ)
        MAX_WORKERS=25
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(calc, qpval): qpval for qpval in qp
            }

            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()  # read the future object for result
                selfE.append(result[0])
                cs.append(result[2])
                rs.append(result[1])
                qpval = future_to_file[future]  
        

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     results_pl = executor.map(calc, qp, chunksize=1)

        #     for result in results_pl:
        #         selfE.append(result[0])
        #         cs.append(result[2])
        #         rs.append(result[1])


        e=time.time()
        print("time for sweep ieps", e-s)
        
        s=time.time()
        omega=[1e-14]
        kpath=np.array([self.KX1bz,self.KY1bz]).T
        calc = functools.partial(self.parCompute_lh, theta, omega, kpath, VV, Nsamp, mu_values,fillings,prop_BZ)
        MAX_WORKERS=25
        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_file = {
                executor.submit(self.parCompute_lh, theta, omega, kpath, VV, Nsamp, mu_values,fillings,prop_BZ, qpval): qpval for qpval in qp
            }

            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()  # read the future object for result
                selfE_lh.append(result[0])
                cs_lh.append(result[2])
                rs_lh.append(result[1])
                qpval_lh = future_to_file[future] 

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #     results_pl = executor.map(calc, qp, chunksize=1)

        #     for result in results_pl:
        #         selfE_lh.append(result[0])
        #         cs_lh.append(result[2])
        #         rs_lh.append(result[1])


        e=time.time()
        print("time for sweep delta", e-s)

        # for ide in range(np.size(fillings)):
        #     mu= mu_values[ide]
        #     filling=fillings[ide]
        #     omega=[0]
        #     kpath=np.array([self.KX,self.KY]).T
        #     integ=self.Compute(mu, omega, kpath)
        #     # integ_lh=self.Compute_lh(mu, omega, kpath)

        #     # plt.plot((abs(integ-integ_lh).flatten())/np.mean(abs(integ_lh).flatten()))
        #     # plt.savefig("comparison_of_integrands_"+self.name+"_filling_"+str(filling)+".png")
        #     # plt.close()

        #     print("the filling is .. " , filling)
        #     integ=integ.flatten()
        #     popt, res, c, resc=self.extract_cs( integ, prop_BZ)
        #     print("parameters of the fit...", c)
        #     print("residual of the fit...", res)
        #     print("original coeff...", c_phonon)
        #     self.plot_res( integ, self.KX,self.KY, VV, filling, Nsamp,c, res, "eps")
        #     cs.append(c)
        #     rs.append(resc)

        #     # integ_lh=integ_lh.flatten()
        #     # popt, res, c, resc=self.extract_cs( integ_lh, prop_BZ)
        #     # print("parameters of the fit _lh...", c)
        #     # print("residual of the fit..._lh", res)
        #     # print("original coeff..._lh",  c_phonon)    
        #     # # self.plot_res( integ_lh, self.KX,self.KY, VV, filling, Nsamp,c, res, "lh")
        #     # cs_lh.append(c)
        #     # rs_lh.append(resc)

        
        cep=np.array(cs)/c_phonon
        clh=np.array(cs_lh)/c_phonon
        plt.scatter(fillings, cep, c='b', label='eps')
        plt.plot(fillings, cep, c='k', ls='--')
        plt.scatter(fillings, clh, c='r', label='lh')
        plt.plot(fillings, clh, c='k', ls='--')
        plt.legend()
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$\alpha/ c$"+self.mode)
        plt.savefig("velocities_V_filling_"+self.name+"_"+str(Nsamp)+"_theta_"+str(theta)+".png")
        plt.close()
        # plt.show()

        cep=np.array(cs)/c_phonon
        clh=np.array(cs_lh)/c_phonon
        plt.scatter(fillings, 1-cep**2, c='b', label='eps')
        plt.plot(fillings, 1-cep**2, c='k', ls='--')
        plt.scatter(fillings, 1-clh**2, c='r', label='lh')
        plt.plot(fillings, 1-clh**2, c='k', ls='--')
        plt.legend()
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"$1-(\alpha/ c)^{2}$, "+self.mode+"-mode")
        plt.savefig("velocities_V_renvsq_"+self.name+"_"+str(Nsamp)+"_theta_"+str(theta)+".png")
        plt.close()
        # plt.show()

        rep=np.array(rs)/ c_phonon
        rlh=np.array(rs_lh)/ c_phonon
        plt.scatter(fillings, rep, c='b', label='eps')
        plt.plot(fillings, rep, c='k', ls='--')
        plt.scatter(fillings, rlh, c='r', label='lh')
        plt.plot(fillings, rlh, c='k', ls='--')
        plt.legend()
        plt.xlabel(r"$\nu$")
        plt.ylabel(r"res$ /c$ "+self.mode)
        plt.yscale('log')
        plt.savefig("velocities_res_V_filling_"+self.name+"_"+str(Nsamp)+"_theta_"+str(theta)+".png")
        # plt.close()
        plt.show()

        with open("velocities_V_filling_"+self.name+".npy", 'wb') as f:
                np.save(f, cep)
        with open("velocities_res_V_filling_"+self.name+".npy", 'wb') as f:
                np.save(f, rep)
        with open("velocities_V_lh_filling_"+self.name+".npy", 'wb') as f:
                np.save(f, clh)
        with open("velocities_res_lh_V_filling_"+self.name+".npy", 'wb') as f:
                np.save(f, rlh)

        
def main() -> int:

    """[summary]
    Calculates the polarization bubble for the electron phonon interaction vertex in TBG
    
    In:
        integer that picks the chemical potential for the calculation
        integer linear number of samples to be used 
        str L or T calculate the bubble for the longitudinal or transverse mode
        double additional scale to the Hamiltonian
        
    Out: 

    
    
    Raises:
        Exception: ValueError, IndexError Input integer in the firs argument to choose chemical potential for desired filling
        Exception: ValueError, IndexError Input int for the number of k-point samples total kpoints =(arg[2])**2
        Exception: ValueError, IndexError third arg has to be the mode that one wants to simulate either L or T
        Exception: ValueError, IndexError Fourth argument is a modulation factor from 0 to 1 to change the interaction strength
    """
    
    try:
        filling_index=int(sys.argv[1]) #0-25

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose chemical potential for desired filling")

    try:
        Nsamp=int(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")


    try:
        mode=(sys.argv[3])

    except (ValueError, IndexError):
        raise Exception("third arg has to be the mode that one wants to simulate either L or T")

    try:
        modulation=float(sys.argv[4])

    except (ValueError, IndexError):
        raise Exception("Fourth arguments is a modulation factor from 0 to 1 to change the interaction strength")

    #Lattice parameters 
    #lattices with different normalizations
    theta=modulation*1.05*np.pi/180  # magic angle
    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one
    [KX,KY]=lq.Generate_lattice()
    Npoi=np.size(KX); print(Npoi, "numer of sampling lattice points")
    [q1,q2,q3]=l.q
    q=la.norm(q1)
    umkl=0
    print(f"taking {umkl} umklapps")
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
    PH=True
    

    #JY params 
    # hbvf = 2.7; # eV
    # hvkd=hbvf*q
    # kappa=0.75
    # up = 0.105; # eV
    # u = kappa*up; # eV
    # alpha=up/hvkd
    # alph=alpha
    
    print("hbvf is ..",hbvf )
    print("q is...", q)
    print("hvkd is...", hvkd)
    print("kappa is..", kappa)
    print("alpha is..", alph)
    print("the twist angle is ..", theta)

    #electron parameters
    nbands=2
    hbarc=0.1973269804*1e-6 #ev*m
    alpha=137.0359895 #fine structure constant
    a_graphene=2.458*(1e-10) #in meters this is the lattice constant NOT the carbon-carbon distance
    e_el=1.6021766*(10**(-19))  #in joule/ev
    ee2=(hbarc/a_graphene)/alpha
    kappa_di=3.03

    #phonon parameters
    c_light=299792458 #m/s
    M=1.99264687992e-26 * (c_light*c_light/e_el) # [in units of eV]
    mass=M/(c_light**2) # in ev *s^2/m^2
    hhbar=6.582119569e-16 #(in eV s)
    alpha_ep=0*2# in ev
    beta_ep=4 #in ev
    c_phonon=21400 #m/s
    
    #calculating effective coupling
    A1mbz=lq.VolMBZ*((q**2)/(a_graphene**2))
    AWZ_graphene=np.sqrt(3)*a_graphene*a_graphene/2
    A1bz=(2*np.pi)**2 / AWZ_graphene
    alpha_ep_effective=np.sqrt(A1mbz/A1bz)*alpha_ep
    beta_ep_effective=np.sqrt(A1mbz/A1bz)*beta_ep
    
    #testing the orders of magnitude for the dimensionless velocity squared
    qq=q/a_graphene
    Wupsilon=(beta_ep_effective**2)*qq*qq
    W=0.008
    ctilde=W*(qq**2)*(mass)*(c_phonon**2)/Wupsilon
    print("phonon params", Wupsilon,ctilde)
    
    #parameters to be passed to the Bubble class
    mode_layer_symmetry="a" #whether we are looking at the symmetric or the antisymmetric mode
    cons=[alpha_ep_effective,beta_ep_effective, Wupsilon, a_graphene, mass] #constants used in the bubble calculation and data anlysis


    hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq, kappa, PH)
    hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq, kappa, PH)
    
    #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
    Ndos=100
    ldos=MoireLattice.MoireTriangLattice(Ndos,theta,2)
    [ Kxp, Kyp]=ldos.Generate_lattice()
    disp=Hamiltonian.Dispersion( ldos, nbands, hpl, hmin)
    Nfils=7
    [fillings,mu_values]=disp.mu_filling_array(Nfils, True, False, False)
    filling_index=int(sys.argv[1]) 
    mu=mu_values[filling_index]
    filling=fillings[filling_index]
    print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
    
    
    #BUBBLE CALCULATION
    test_symmetry=True
    B1=ep_Bubble(lq, nbands, hpl, hmin,  mode_layer_symmetry, mode, cons, test_symmetry, umkl)
    omega=[1e-14]
    kpath=np.array([KX,KY]).T
    integ=B1.Compute(mu, omega, kpath)
    popt, res, c, resc=B1.extract_cs( integ, 0.2)
    B1.plot_res(integ, KX, KY, VV, filling, Nsamp, c , res, "")
    print(np.mean(popt),np.mean(c), resc, c_phonon)
    # B1.Fill_sweep(fillings, mu_values, VV, Nsamp, c_phonon,theta)
    
    # NnarrowSamp=100
    # lnarrowSamp=MoireLattice.MoireTriangLattice(NnarrowSamp,theta,2)
    # [ Kxp, Kyp]=lnarrowSamp.Generate_lattice()
    
    # [KX_m, KY_m, ind]=lnarrowSamp.mask_KPs( Kxp, Kyp, 0.2)
    # kpath=np.array([KX_m, KY_m]).T
    # integ=B1.Compute_lh(mu, omega, kpath)
    # popt, res, c, resc=B1.extract_cs_path( integ, kpath)
    # print(np.mean(popt),np.mean(c), resc, c_phonon)
    # B1.plot_res(integ, KX_m, KY_m, VV, filling, Nsamp, c , res, "small")
    

    
    # return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
