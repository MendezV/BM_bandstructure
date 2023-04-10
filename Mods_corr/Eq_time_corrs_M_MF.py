import numpy as np
import scipy
from scipy import linalg as la
import time
import sys
import MoireLattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import concurrent.futures
import os
import pandas as pd
import Dispersion




class Eq_time_corrs:
    """
    A class used to represent electron phonon bubbles

    ...

    Attributes
    ----------
    
    lattice
    umkl number of umklap processes to sum over
    KX1bz, KY1bz kpoints in the first bz
    KX KY K points including umklapps
    KQX KQY kpoints with an umkl+1 umklapps to account for momentum transfers
    NpoiXX number of points in each of the three latices above
    
        
    dispersion
    number of bands
    hpl hmin hamiltonians for minus and plus valleys
    psi_plus(min) wavefunctions, unitless, enter the calculation of the form factors
    Ene_vallyey_plus(min) dispersion as a function of k in ev
    
    form factors
    FFp(m) are form factor objects from the Hamiltonian module
    they represent the form factors for the plus and minus valley
    L00p(m) are density form factor arrays
    Lnemp(m) are the nematic form factors which can be chosen depending on the value of "symmetric" in the constructor
    Omega_FFp(m) are the form factors that enter the interaction with the normalized coupling constants so that beta_ep=1
    
    constants
    Wupsilon scale of the bubble with units of ev^2 /m^2
    mass carbon atom in ev *s^2/m^2
    a_graphene graphene lattice constant (not the cc distance)
    alpha_ep deformation potential coupling normalized by the gauge coupling
    beta_ep=1 with the current implementation (sorry for the bad code)
    
    
    
    Methods
    -------
    nf,nb fermi and bose occupation functions
    OmegaL(T) form factors for the interaction
    integrand
    
    """

    ################################
    """
    CONSTRUCTOR for the ep_Bubble class 
    """
    ################################
    
    def __init__(self, Mean_field_M, symmetric, mode, dir ):

        
        ################################
        #lattice attributes
        ################################
        self.latt=Mean_field_M.latt
        # print(latt.Npoints, "points in the lattice")
        [q1,q2,q3]=self.latt.qvect()
        self.qscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        print("qq",self.qscale)
        self.dS_in=1/self.latt.Npoi1bz
        
        
        ################################
        # dispersion attributes
        ################################
        self.nbands=Mean_field_M.nbands
        self.HB=Mean_field_M.HB
        self.Ene_valley_plus_1bz=Mean_field_M.Ene_valley_plus_1bz
        self.Ene_valley_min_1bz=Mean_field_M.Ene_valley_min_1bz
        self.Ene_valley_plus_K=Mean_field_M.Ene_valley_plus_K
        self.Ene_valley_min_K=Mean_field_M.Ene_valley_min_K
        self.Ene_valley_plus=Mean_field_M.Ene_valley_plus
        self.Ene_valley_min=Mean_field_M.Ene_valley_min

        
        
        ################################
        ### selecting eta
        ################################
        eps_l=[]
        for i in range(self.nbands):
            eps_l.append(np.mean( np.abs( np.diff( self.Ene_valley_plus_1bz[:,i].flatten() )  ) )/2)
            eps_l.append(np.mean( np.abs( np.diff( self.Ene_valley_min_1bz[:,i].flatten() )  ) )/2)
        eps_a=np.array(eps_l)
        eps=np.min(eps_a)
        
        
        self.eta=eps
        self.eta_dirac_delta=eps/4
        self.eta_cutoff=eps
        self.eta_small_imag=0.1*eps
        
        
        
        
        
        ################################
        #attributes for the name of the object and the extraction of Ctilde
        #note that the couplings are normalized so that beta_ep is 1 and only the relative value bw them matters
        #the dimensions of the bubble are carried by Wupsilon
        #agraph and the mass are used to get the conversion from the normalized bubble to the unitful bubble when doing the fit
        #to extract the sound velocity
        ################################
        self.mode=mode
        self.symmetric=symmetric
        self.name="_mode_"+self.mode+"_symmetry_"+self.symmetric+"_alpha_"+str(Mean_field_M.alpha_ep)+"_beta_"+str(Mean_field_M.beta_ep)+"_umklp_"+str(self.latt.umkl)+"_kappa_"+str(self.HB.hpl.kappa)+"_theta_"+str(self.latt.theta)+"_modeHF_"+str(self.HB.mode)
        self.dir=dir #directory where the correlation functions will be stored
        
        ################################
        #generating form factors
        ################################

        if symmetric=="s":
            if mode=="L":
                self.L00p=self.HB.FFp.denqFFL_s()
                self.L00m=self.HB.FFm.denqFFL_s()
                self.Lnemp=self.HB.FFp.NemqFFL_s()
                self.Lnemm=self.HB.FFm.NemqFFL_s()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()
            elif mode=="T": 
                self.Lnemp=self.HB.FFp.NemqFFT_s()
                self.Lnemm=self.HB.FFm.NemqFFT_s()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()
            elif mode=="long":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.NemqFFL_s(), self.HB.FFm.NemqFFL_s())
            elif mode=="trans":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.NemqFFT_s(), self.HB.FFm.NemqFFT_s())
            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_s(), self.HB.FFm.denqFF_s())
            elif mode=="subl":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.sublFF_s(), self.HB.FFm.sublFF_s())
            elif mode=="nemx":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.nemxFF_s(), self.HB.FFm.nemxFF_s())
            elif mode=="nemy":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.nemyFF_s(), self.HB.FFm.nemyFF_s())
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_s(), self.HB.FFm.denqFF_s())
            
        else: # a- mode
            if mode=="L":
                self.L00p=self.HB.FFp.denqFFL_a()
                self.L00m=self.HB.FFm.denqFFL_a()
                self.Lnemp=self.HB.FFp.NemqFFL_a()
                self.Lnemm=self.HB.FFm.NemqFFL_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()                
            elif mode=="T":
                self.Lnemp=self.HB.FFp.NemqFFT_a()
                self.Lnemm=self.HB.FFm.NemqFFT_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()
            elif mode=="long":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.NemqFFL_a(), self.HB.FFm.NemqFFL_a())
            elif mode=="trans":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.NemqFFT_a(), self.HB.FFm.NemqFFT_a())
            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_s(), self.HB.FFm.denqFF_s())
            elif mode=="subl":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.sublFF_a(), self.HB.FFm.sublFF_a())
            elif mode=="nemx":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.nemxFF_a(), self.HB.FFm.nemxFF_a())
            elif mode=="nemy":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.nemyFF_a(), self.HB.FFm.nemyFF_a())
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_a(), self.HB.FFm.denqFF_a())

        
        
        ################################
        #lattice attributes
        ################################
        IkpMpq=np.zeros([self.latt.Npoi1bz,self.latt.Npoi])
        for q in range(self.latt.Npoi):
            IkpMpq[:,q]=np.array(self.latt.insertion_index( self.latt.KX1bz+self.latt.KX[q]+self.latt.M1[0],self.latt.KY1bz+self.latt.KY[q]+self.latt.M1[1], self.latt.KQX, self.latt.KQY))
        self.IkpMpq=IkpMpq.astype(int)
        print( 'the shape of the index q array',np.shape(self.IkpMpq))
        self.Ik1bz_c=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
        

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
        Tp=T+1e-17 #To capture zero temperature
        rat=np.abs(e/Tp)
        
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
        Tp=T+1e-17 #To capture zero temperature
        rat=np.abs(np.max(e/Tp))
        
        if rat<700:
            return 1/(np.exp( e/T )-1)
        else:
            return -np.heaviside(-e,0.5) # at zero its 1
    

    def OmegaL(self):
        
        Omega_FFp_pre=np.sqrt(self.Wupsilon)*(self.alpha_ep*self.L00p+self.beta_ep*self.Lnemp)
        Omega_FFm_pre=np.sqrt(self.Wupsilon)*(self.alpha_ep*self.L00m+self.beta_ep*self.Lnemm)
        
        [Omega_FFp,Omega_FFm]=self.HB.Form_factor_unitary( Omega_FFp_pre, Omega_FFm_pre)
                
        return [Omega_FFp,Omega_FFm]

    def OmegaT(self):

        Omega_FFp_pre=np.sqrt(self.Wupsilon)*self.beta_ep*self.Lnemp
        Omega_FFm_pre=np.sqrt(self.Wupsilon)*self.beta_ep*self.Lnemm

        
        [Omega_FFp,Omega_FFm]=self.HB.Form_factor_unitary( Omega_FFp_pre, Omega_FFm_pre)
        
        return [Omega_FFp,Omega_FFm]


    def deltad(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    
    def GRw0s(self,ekn,ekm,mu,T):
        edkq=ekn-mu
        edk=ekm-mu

        #finite temp
        nfk= self.nf(edk,T)
        nfkq= self.nf(edkq,T)
        
        ###iepsilon
        eps=self.eta_small_imag ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(-(edkq-edk)+1j*eps)
        return np.real(fac_p)
    
    def GReqs(self,ekn,ekm,mu,T):
        edkq=ekn-mu
        edk=ekm-mu

        #finite temp
        nfkq= self.nf(edkq,T)
        nfk= self.nf(edk,T)
        
        return (1-nfk)*nfkq


    def corr(self,args):
        ( mu, T)=args

        
        sb=time.time()

        print("starting bubble.......")

        integ=np.zeros(self.latt.Npoi)

        for Nq in range(self.latt.Npoi):  #for calculating only along path in FBZ
            
            bub=0
            
            for Nk in range(self.latt.Npoi1bz):
                ikq=self.latt.Ikpq[Nk,Nq]
                ik=self.latt.Ik1bz[Nk]
                
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        Lp=self.Omega_FFp[ikq, ik,nband,mband]
                        ekq_p_n=self.Ene_valley_plus[ikq,nband]
                        ek_p_m=self.Ene_valley_plus[ik,mband]
                        GRs_p=self.GRw0s(ekq_p_n,ek_p_m,mu,T)
                        integrand_var=np.abs(Lp*np.conj(Lp))*GRs_p
                        bub=bub+integrand_var
                        
                        
                        Lm=self.Omega_FFm[ikq, ik,nband,mband]
                        ekq_m_n=self.Ene_valley_min[ikq,nband]
                        ek_m_m=self.Ene_valley_min[ik,mband]
                        GRs_m=self.GRw0s(ekq_m_n,ek_m_m,mu,T)
                        integrand_var=np.abs(Lm*np.conj(Lm))*GRs_m
                        bub=bub+integrand_var

            integ[Nq]=bub

        eb=time.time()
        

        print("time for bubble...",eb-sb)
        
        res= integ*self.dS_in 
        self.savedata(res, mu, T, '')
        return res
    
    
    def corr_eq(self,args):
        ( mu, T)=args

        
        sb=time.time()

        print("starting bubble.......")

        integ=np.zeros(self.latt.Npoi)

        for Nq in range(self.latt.Npoi):  #for calculating only along path in FBZ
            
            bub=0
            
            for Nk in range(self.latt.Npoi1bz):
                ikq=self.latt.Ikpq[Nk,Nq]
                ik=self.latt.Ik1bz[Nk]
                
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        Lp=self.Omega_FFp[ikq, ik,nband,mband]
                        ekq_p_n=self.Ene_valley_plus[ikq,nband]
                        ek_p_m=self.Ene_valley_plus[ik,mband]
                        GRs_p=self.GReqs(ekq_p_n,ek_p_m,mu,T)
                        integrand_var=np.abs(Lp*np.conj(Lp))*GRs_p
                        bub=bub+integrand_var
                        
                        
                        Lm=self.Omega_FFm[ikq, ik,nband,mband]
                        ekq_m_n=self.Ene_valley_min[ikq,nband]
                        ek_m_m=self.Ene_valley_min[ik,mband]
                        GRs_m=self.GReqs(ekq_m_n,ek_m_m,mu,T) 
                        integrand_var=np.abs(Lm*np.conj(Lm))*GRs_m
                        bub=bub+integrand_var

            integ[Nq]=bub

        eb=time.time()
        

        print("time for bubble...",eb-sb)
        
        res= integ*self.dS_in 
        self.savedata(res, mu, T, '')
        return res
    
    def savedata(self, integ, mu_value, T, add_tag):
        
        Nsamp=self.latt.Npoints
        index_f=np.argmin((mu_value-self.mu_values)**2)
        filling=self.fillings[index_f]
        identifier=add_tag+str(Nsamp)+self.name
        Nss=np.size(self.latt.KX)

        #products of the run
        Pibub=np.hstack(integ)

        filling_list=[]
        mu_list=[]
        filling_list=[filling]*Nss
        mu_list=[mu_value]*Nss
            
        KXall=self.latt.KX
        KYall=self.latt.KY
        fillingarr=np.array(filling_list)
        muarr=np.array(mu_list)
        disp_m1=self.Ene_valley_min_K[:,0].flatten()
        disp_m2=self.Ene_valley_min_K[:,1].flatten()
        disp_p1=self.Ene_valley_plus_K[:,0].flatten()
        disp_p2=self.Ene_valley_plus_K[:,1].flatten()

        
        #constants
        thetas_arr=np.array([self.latt.theta]*(Nss))
        kappa_arr=np.array([self.HB.hpl.kappa]*(Nss))
        
        print('checking sizes of the arrays for hdf5 storage')
        print(Nss, Nss,np.size(Pibub),np.size(KXall), np.size(KYall), np.size(fillingarr), np.size(muarr), np.size(thetas_arr), np.size(kappa_arr), np.size(disp_m1), np.size(disp_m2), np.size(disp_p1), np.size(disp_p2), np.size(T))

            
        df = pd.DataFrame({'bub': Pibub, 'kx': KXall, 'ky': KYall,'nu': fillingarr,'mu':muarr, 'theta': thetas_arr, 'kappa': kappa_arr, 'Em1':disp_m1, 'Em2':disp_m2,'Ep1':disp_p1,'Ep2':disp_p2 , 'T':T})
        df.to_hdf(self.dir+'/corr_eq'+identifier+'_nu_'+str(filling)+'_T_'+str(T)+'.h5', key='df', mode='w')


        return None
    
    def Fill_sweep(self, Nfils, T, parallel=False):
        
        if Nfils>1:
            [fillings,mu_values]=self.HB.disp.mu_filling_array(Nfils,self.Ene_valley_min_1bz, self.Ene_valley_plus_1bz)
        else:
            fillings=np.array([0])
            mu_values=np.array([0])
        print('fillings and mu for the sweep....\n',fillings, mu_values)
        integ=[]
        self.fillings=fillings
        self.mu_values=mu_values

        qp=np.arange(np.size(fillings))
        s=time.time()

        arglist=[]
        for i in qp:
            arglist.append( ( mu_values[i],T) )

        if parallel==True:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future_to_file = {
                    
                    executor.submit(self.corr, arglist[qpval]): qpval for qpval in qp
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()  # read the future object for result
                    integ.append(result[0])
                    qpval = future_to_file[future]  
        else:
            print('no parallelization on filling')
            for qpval in qp:
                self.corr(arglist[qpval])

        
        
        e=time.time()
        t=e-s
        print("time for sweep delta", t)
        
                
        return t
    
    def Fill_sweep_eq(self, Nfils, T, parallel=False):
        
        if Nfils>1:
            [fillings,mu_values]=self.HB.disp.mu_filling_array(Nfils,self.Ene_valley_min_1bz, self.Ene_valley_plus_1bz)
        else:
            fillings=np.array([0])
            mu_values=np.array([0])
        print('fillings and mu for the sweep....\n',fillings, mu_values)
        integ=[]
        self.fillings=fillings
        self.mu_values=mu_values

        qp=np.arange(np.size(fillings))
        s=time.time()

        arglist=[]
        for i in qp:
            arglist.append( ( mu_values[i],T) )

        if parallel==True:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future_to_file = {
                    
                    executor.submit(self.corr_eq, arglist[qpval]): qpval for qpval in qp
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()  # read the future object for result
                    integ.append(result[0])
                    qpval = future_to_file[future]  
        else:
            print('no parallelization on filling')
            for qpval in qp:
                self.corr_eq(arglist[qpval])

        
        
        e=time.time()
        t=e-s
        print("time for sweep delta", t)
        
                
        return t
    
    def Corr_eq(self, mu_values, fillings, T, parallel=False):

        print('fillings and mu for the sweep....\n',fillings, mu_values)
        integ=[]
        self.fillings=fillings
        self.mu_values=mu_values

        qp=np.arange(np.size(fillings))
        s=time.time()

        arglist=[]
        for i in qp:
            arglist.append( ( mu_values[i],T) )

        if parallel==True:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future_to_file = {
                    
                    executor.submit(self.corr, arglist[qpval]): qpval for qpval in qp
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()  # read the future object for result
                    integ.append(result[0])
                    qpval = future_to_file[future]  
        else:
            print('no parallelization on filling')
            for qpval in qp:
                self.corr(arglist[qpval])

        
        
        e=time.time()
        t=e-s
        print("time for sweep delta", t)
        
                
        return t
    
    
    def Corr(self, mu_values, fillings, T, parallel=False):

        print('fillings and mu for the sweep....\n',fillings, mu_values)
        integ=[]
        self.fillings=fillings
        self.mu_values=mu_values

        qp=np.arange(np.size(fillings))
        s=time.time()

        arglist=[]
        for i in qp:
            arglist.append( ( mu_values[i],T) )

        if parallel==True:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future_to_file = {
                    
                    executor.submit(self.corr, arglist[qpval]): qpval for qpval in qp
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()  # read the future object for result
                    integ.append(result[0])
                    qpval = future_to_file[future]  
        else:
            print('no parallelization on filling')
            for qpval in qp:
                self.corr(arglist[qpval])

        
        
        e=time.time()
        t=e-s
        print("time for sweep delta", t)
        
                
        return t
    
    def MF_corr_eq(self, args):
        
        ( mu, T, phi_T)=args
        Form_fact_p = self.Omega_FFp
        
        sb=time.time()

        print("starting bubble.......")

        integ=np.zeros(self.latt.Npoi)

        
        N_Mp=2 # number of symmetry breaking momenta +1
        
        Lp_pre=np.zeros([ N_Mp*self.nbands, N_Mp*self.nbands], dtype=np.cdouble)
        Lm_pre=np.zeros([ N_Mp*self.nbands, N_Mp*self.nbands], dtype=np.cdouble)
        
        Lp_u=np.zeros([ N_Mp*self.nbands, N_Mp*self.nbands], dtype=np.cdouble)
        Lm_u=np.zeros([ N_Mp*self.nbands, N_Mp*self.nbands], dtype=np.cdouble)
        
        Hqp_kq=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=np.cdouble)
        Hqm_kq=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=np.cdouble)
        
        Hqp_k=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=np.cdouble)
        Hqm_k=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=np.cdouble)
        
        Eval_plus_kq = np.zeros([N_Mp*self.nbands])
        Eval_min_kq = np.zeros([N_Mp*self.nbands])
        
        Vval_plus_kq=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=complex)
        Vval_min_kq=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=complex)
        
        Eval_plus_k = np.zeros([N_Mp*self.nbands])
        Eval_min_k = np.zeros([N_Mp*self.nbands])
        
        Vval_plus_k=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=complex)
        Vval_min_k=np.zeros([N_Mp*self.nbands,N_Mp*self.nbands], dtype=complex)

        # Mean_field_M
        for Nq in range(self.latt.Npoi):  #for calculating only along path in FBZ
            
            Hqp_kq=self.Mean_field_M.Heq_p_MBZ[Nq,:,:]+phi_T*self.Mean_field_M.HT_p_MBZ[Nq,:,:]
            Hqm_kq=self.Mean_field_M.Heq_m_MBZ[Nq,:,:]+phi_T*self.Mean_field_M.HT_m_MBZ[Nq,:,:]
            
            Eval_plus_kq, Vval_plus_kq = np.linalg.eigh(Hqp_kq)
            Eval_min_kq, Vval_min_kq = np.linalg.eigh(Hqm_kq)

            
            for Nk in range(self.latt.Npoi1bz):  #for calculating only along path in FBZ
                Nk_d = self.Ik1bz_c[Nk]
                
                Hqp_k=self.Mean_field_M.Heq_p_MBZ[Nk_d,:,:]+phi_T*self.Mean_field_M.HT_p_MBZ[Nk_d,:,:]
                Hqm_k=self.Mean_field_M.Heq_m_MBZ[Nk_d,:,:]+phi_T*self.Mean_field_M.HT_m_MBZ[Nk_d,:,:]
                
                Eval_plus_k, Vval_plus_k = np.linalg.eigh(Hqp_k)
                Eval_min_k, Vval_min_k = np.linalg.eigh(Hqm_k)
                
                ik=self.latt.Ik1bz[Nk]
                ikq=self.latt.Ikpq[Nk,Nq]
                ikMq=self.IkpMpq[Nk,Nq]

                Lp1=  Form_fact_p[ikq, ik, :, :] 
                Lm1 = - (self.sx@Lp1@self.sx) #using chiral
                
                Lp2 =  Form_fact_p[ikMq, ik, :, :]
                Lm2 = - (self.sx@Lp2@self.sx) #using chiral
                
                for nband in range(self.nbands):
                    for mband in range(self.nbands):
                        
                        Lp_pre[nband,mband]=Lp_pre[nband,mband]+Lp1[nband,mband]
                        Lm_pre[Nq, Nk,nband,mband]=Lm_pre[Nk,nband,mband]+np.conj(Lm1.T)[nband,mband]
                        
                        Lp_pre[self.nbands+nband,self.nbands+mband]=Lp_pre[self.nbands+nband,self.nbands+mband]+Lp2[nband,mband]
                        Lm_pre[self.nbands+nband,self.nbands+mband]=Lm_pre[self.nbands+nband,self.nbands+mband]+np.conj(Lm2.T)[nband,mband]
                        
                Lp_u = np.conj(np.transpose(Vval_plus_kq))@(Lp_pre@Vval_plus_k)
                Lm_u = np.conj(np.transpose(Vval_min_kq))@(Lm_pre@Vval_min_k)
                
                for nband in range(N_Mp*self.nbands):
                    for mband in range(N_Mp*self.nbands):
                        
                        Lp=Lp_u[nband,mband]
                        ekq_p_n=Eval_plus_kq[nband]
                        ek_p_m=Eval_plus_k[mband]
                        GRs_p=self.GReqs(ekq_p_n,ek_p_m,mu,T)
                        integrand_var=np.abs(Lp*np.conj(Lp))*GRs_p
                        bub=bub+integrand_var
                        
                        
                        Lm=Lm_u[nband,mband]
                        ekq_m_n=Eval_min_kq[nband]
                        ek_m_m=Eval_min_k[mband]
                        GRs_m=self.GReqs(ekq_m_n,ek_m_m,mu,T) 
                        integrand_var=np.abs(Lm*np.conj(Lm))*GRs_m
                        bub=bub+integrand_var

            integ[Nq]=bub

                        
        eb=time.time()
        

        print("time for bubble...",eb-sb)
        
        res= integ*self.dS_in 
        self.savedata(res, mu, T, 'MF_M')
        return res
    
    
    def MF_Corr_eq(self, mu_values, fillings, T , phiT, parallel=False):

        print('fillings and mu for the sweep....\n',fillings, mu_values)
        integ=[]
        self.fillings=fillings
        self.mu_values=mu_values

        qp=np.arange(np.size(fillings))
        s=time.time()

        arglist=[]
        for i in qp:
            arglist.append( ( mu_values[i], T, phiT) )

        if parallel==True:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                future_to_file = {
                    
                    executor.submit(self.corr, arglist[qpval]): qpval for qpval in qp
                }

                for future in concurrent.futures.as_completed(future_to_file):
                    result = future.result()  # read the future object for result
                    integ.append(result[0])
                    qpval = future_to_file[future]  
        else:
            print('no parallelization on filling')
            for qpval in qp:
                self.corr(arglist[qpval])

        
        
        e=time.time()
        t=e-s
        print("time for sweep delta", t)
        
                
        return t


     
        
def main() -> int:

    """[summary]
    Calculates the polarization bubble for the electron phonon interaction vertex in TBG
    
    In:
        integer that picks the chemical potential for the calculation
        integer linear number of samples to be used 
        str L or T calculate the bubble for the longitudinal or transverse mode
        double additional scale to modulate twist angle
        double additional scale to modulate kappa
        
    Out: 
        hdf5 dataset with the filling sweep 
    
    
    Raises:
        Exception: ValueError, IndexError Input integer in the firs argument to choose chemical potential for desired filling
        Exception: ValueError, IndexError Input int for the number of k-point samples total kpoints =(arg[2])**2
        Exception: ValueError, IndexError third arg has to be the mode that one wants to simulate either L or T
        Exception: ValueError, IndexError Fourth argument is a modulation factor from 0 to 1 to change the twist angle
        Exception: ValueError, IndexError Fifth argument is a modulation factor from 0 to 1 to change the interlayer hopping ratio

    """
    
    #####
    # Parameters Diag: samples
    ####
    try:
        Nsamp=int(sys.argv[1])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")

    #####
    # Electron parameters: filling angle kappa HF
    ####
    
    try:
        filling_index=int(sys.argv[2]) 

    except (ValueError, IndexError):
        raise Exception("Input integer in the first argument to choose chemical potential for desired filling")

    try:
        modulation_theta=float(sys.argv[3])

    except (ValueError, IndexError):
        raise Exception("Third argument is the twist angle")

    try:
        modulation_kappa=float(sys.argv[4])

    except (ValueError, IndexError):
        raise Exception("Fourth argument is the value of kappa")
    
    try:
        mode_HF=int(sys.argv[5])

    except (ValueError, IndexError):
        raise Exception("Fifth argument determines whether HF renormalized bands are used, 1 for HF, 0 for bare")
    #####
    # Phonon parameters: polarization L or T
    ####
    
    try:
        mode=(sys.argv[6])

    except (ValueError, IndexError):
        raise Exception("sixth arg has to be the mode that one wants to simulate either L or T")

    
    print("\n \n")
    print("lattice sampling...")
    #Lattice parameters 
    #lattices with different normalizations
    theta=modulation_theta*np.pi/180  # magic angle
    c6sym=True
    umkl=1 #the number of umklaps where we calculate an observable ie Pi(q), for momentum transfers we need umkl+1 umklapps when scattering from the 1bz
    l=MoireLattice.MoireTriangLattice(Nsamp, theta, 0, c6sym, umkl)
    lq=MoireLattice.MoireTriangLattice(Nsamp, theta, 2, c6sym, umkl) #this one is normalized
    [q1,q2,q3]=l.q
    q=np.sqrt(q1@q1)
    print(f"taking {umkl} umklapps")
    VV=lq.boundary()


    #kosh params realistic  -- this is the closest to the actual Band Struct used in the paper
    # hbvf = 2.1354; # eV
    # hvkd=hbvf*q
    # kappa_p=0.0797/0.0975
    # kappa=kappa_p
    # up = 0.0975; # eV
    # u = kappa*up; # eV
    # alpha=up/hvkd
    # alph=alpha
    PH=True
    

    #JY params 
    hbvf = (3/(2*np.sqrt(3)))*2.7; # eV
    hvkd=hbvf*q
    kappa=modulation_kappa
    up = 0.105; # eV
    u = kappa*up; # eV
    alpha=up/hvkd
    alph=alpha

    #Andrei params 
    # hbvf = 19.81/(8*np.pi/3); # eV
    # hvkd=hbvf*q
    # kappa=1
    # up = 0.110; # eV
    # u = kappa*up; # eV
    # alpha=up/hvkd
    # alph=alpha
    print("\n \n")
    print("parameters of the hamiltonian...")
    print("hbvf is ..",hbvf )
    print("q is...", q)
    print("hvkd is...", hvkd)
    print("kappa is..", kappa)
    print("alpha is..", alph)
    print("the twist angle is ..", theta)
    print("\n \n")

    #electron parameters
    nbands=2
    nremote_bands=0
    hbarc=0.1973269804*1e-6 #ev*m
    alpha=137.0359895 #fine structure constant
    a_graphene=2.458*(1e-10) #in meters this is the lattice constant NOT the carbon-carbon distance
    e_el=1.6021766*(10**(-19))  #in joule/ev
    ee2=(hbarc/a_graphene)/alpha
    eps_inv = 1/10
    d_screening=20*(1e-9)/a_graphene
    d_screening_norm=d_screening*lq.qnor()
    epsilon_0 = 8.85*1e-12
    ev_conv = e_el
    Vcoul=( e_el*e_el*eps_inv*d_screening/(2*epsilon_0*a_graphene) )
    V0= (  Vcoul/lq.Vol_WZ() )/ev_conv
    print(V0, 'la energia de coulomb en ev')
    print("\n \n")

    #phonon parameters
    c_light=299792458 #m/s
    M=1.99264687992e-26 * (c_light*c_light/e_el) # [in units of eV]
    mass=M/(c_light**2) # in ev *s^2/m^2
    alpha_ep=0 # in ev
    beta_ep=4 #in ev SHOULD ALWAYS BE GREATER THAN ZERO
    if mode=="L":
        c_phonon=21400 #m/s
    if mode=="T":
        c_phonon=13600 #m/s
    else:
        c_phonon=21400 #m/s
    
    #calculating effective coupling
    A1mbz=lq.VolMBZ*((q**2)/(a_graphene**2))
    AWZ_graphene=np.sqrt(3)*a_graphene*a_graphene/2
    A1bz=(2*np.pi)**2 / AWZ_graphene
    alpha_ep_effective=np.sqrt(1/2)*np.sqrt(A1mbz/A1bz)*alpha_ep #sqrt 1/2 from 2 atoms per unit cell in graphene
    beta_ep_effective=np.sqrt(1/2)*np.sqrt(A1mbz/A1bz)*beta_ep #sqrt 1/2 from 2 atoms per unit cell in graphene
    alpha_ep_effective_tilde=alpha_ep_effective/beta_ep_effective
    beta_ep_effective_tilde=beta_ep_effective/beta_ep_effective
    
    #testing the orders of magnitude for the dimensionless velocity squared
    qq=q/a_graphene
    Wupsilon=(beta_ep_effective**2)*qq*qq
    W=0.008
    #ctilde=W*(qq**2)*(mass)*(c_phonon**2)/Wupsilon
    print("phonon params", Wupsilon )
    print("phonon params upsilon", Wupsilon/W )
    print("area ratio", A1mbz/A1bz, (2*np.sin(theta/2))**2   )
    print("correct factor by which the interaction is reduced",np.sqrt(2)/(2*np.sin(theta/2)))
    print("c tilde",np.sqrt((Wupsilon/W)*(1/(qq**2))*(1/mass) ))
    print("\n \n")
    
    #parameters to be passed to the Bubble class
    mode_layer_symmetry="a" #whether we are looking at the symmetric or the antisymmetric mode
    cons=[alpha_ep_effective_tilde,beta_ep_effective_tilde, Wupsilon, a_graphene, mass] #constants used in the bubble calculation and data anlysis

    
    #Hartree fock correction to the bandstructure
    hpl=Dispersion.Ham_BM(hvkd, alph, 1, lq, kappa, PH, 1) #last argument is whether or not we have interlayer hopping
    hmin=Dispersion.Ham_BM(hvkd, alph, -1, lq, kappa, PH, 1 ) #last argument is whether or not we have interlayer hopping
    
    
    substract=0 #0 for decoupled layers
    HB=Dispersion.HF_BandStruc( lq, hpl, hmin, hpl, hmin, nremote_bands, nbands, substract,  [V0, d_screening_norm], mode_HF)
    
    
    #BUBBLE CALCULATION        
    test_symmetry=True
    B1=Eq_time_corrs(lq, nbands, HB,  mode_layer_symmetry, mode, cons, test_symmetry, umkl)
    # a=B1.corr( args=(0.0,0.0))


    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
