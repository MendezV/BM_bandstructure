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
    
    def __init__(self, latt, nbands, HB, symmetric, mode, cons , test, umkl):

        
        ################################
        #lattice attributes
        ################################
        self.latt=latt
        # print(latt.Npoints, "points in the lattice")
        self.umkl=umkl
        
        self.Ik=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
        [q1,q2,q3]=latt.qvect()
        self.qscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        print("qq",self.qscale)
        self.dS_in=1/self.latt.Npoi1bz
        
        
        ################################
        #dispersion attributes
        ################################
        self.nbands=nbands
        self.HB=HB
        self.Ene_valley_plus_1bz=HB.E_HFp
        self.Ene_valley_min_1bz=HB.E_HFm
        self.Ene_valley_plus_K=HB.E_HFp_K
        self.Ene_valley_min_K=HB.E_HFm_K
        self.Ene_valley_plus=HB.E_HFp_ex
        self.Ene_valley_min=HB.E_HFm_ex

        
        
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
        #attributes for the name of the object and the extraction of Ctilde
        #note that the couplings are normalized so that beta_ep is 1 and only the relative value bw them matters
        #the dimensions of the bubble are carried by Wupsilon
        #agraph and the mass are used to get the conversion from the normalized bubble to the unitful bubble when doing the fit
        #to extract the sound velocity
        ################################
        [self.alpha_ep, self.beta_ep,  self.Wupsilon, self.agraph, self.mass ]=cons #constants for the exraction of the effective velocity
        self.mode=mode
        self.symmetric=symmetric
        self.name="_mode_"+self.mode+"_symmetry_"+self.symmetric+"_alpha_"+str(self.alpha_ep)+"_beta_"+str(self.beta_ep)+"_umklp_"+str(umkl)+"_kappa_"+str(self.HB.hpl.kappa)+"_theta_"+str(self.latt.theta)+"_modeHF_"+str(HB.mode)

        
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
            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_s(), self.HB.FFm.denqFF_s())
            elif mode=="subl":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.sublFF_s(), self.HB.FFm.sublFF_s())
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
            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_s(), self.HB.FFm.denqFF_s())
            elif mode=="subl":
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.sublFF_a(), self.HB.FFm.sublFF_a())
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.HB.Form_factor_unitary(self.HB.FFp.denqFF_a(), self.HB.FFm.denqFF_a())

        

    ################################
    """
    METHODS for the ep_Bubble class 
    """
    ################################
    
    # def nf(self, e, T):
        
    #     """[summary]
    #     fermi occupation function with a truncation if the argument exceeds 
    #     double precission

    #     Args:
    #         e ([double]): [description] energy
    #         T ([double]): [description] temperature

    #     Returns:
    #         [double]: [description] value of the fermi function 
    #     """
    #     Tp=T+1e-17 #To capture zero temperature
    #     rat=np.abs(np.max(e/Tp))
        
    #     if rat<700:
    #         return 1/(1+np.exp( e/T ))
    #     else:
    #         return np.heaviside(-e,0.5)
        
        
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
        
        Omega_FFp_pre=(self.alpha_ep*self.L00p+self.beta_ep*self.Lnemp)
        Omega_FFm_pre=(self.alpha_ep*self.L00m+self.beta_ep*self.Lnemm)
        
        [Omega_FFp,Omega_FFm]=self.HB.Form_factor_unitary( Omega_FFp_pre, Omega_FFm_pre)
                
        return [Omega_FFp,Omega_FFm]

    def OmegaT(self):

        Omega_FFp_pre=(self.beta_ep*self.Lnemp)
        Omega_FFm_pre=(self.beta_ep*self.Lnemm)

        
        [Omega_FFp,Omega_FFm]=self.HB.Form_factor_unitary( Omega_FFp_pre, Omega_FFm_pre)
        
        return [Omega_FFp,Omega_FFm]


    def deltad(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)


    # def integrand_T(self,nkq,nk,ekn,ekm,w,mu,T):
    #     edkq=ekn[nkq]-mu
    #     edk=ekm[nk]-mu

    #     #finite temp
    #     nfk= self.nf(edk,T)
    #     nfkq= self.nf(edkq,T)

    #     ####delta
    #     # deltad_cut=1e-17*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
    #     # fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-self.eta_cutoff, 0.0)/(deltad_cut-(edkq-edk))
    #     # fac_p2=(self.deltad( edk, self.eta_dirac_delta))*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        
    #     # return (fac_p+fac_p2)
        
    #     ###iepsilon
    #     eps=self.eta_small_imag ##SENSITIVE TO DISPERSION

    #     fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
    #     return (fac_p)
    
    def GReqs(self,ekn,ekm,mu,T):
        edkq=ekn-mu
        edk=ekm-mu

        #finite temp
        nfk= self.nf(edk,T)
        nfkq= self.nf(edkq,T)

        ####delta
        # deltad_cut=1e-17*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        # fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-self.eta_cutoff, 0.0)/(deltad_cut-(edkq-edk))
        # fac_p2=(self.deltad( edk, self.eta_dirac_delta))*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        
        # return (fac_p+fac_p2)
        
        ###iepsilon
        eps=self.eta_small_imag ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(-(edkq-edk)+1j*eps)
        return np.real(fac_p)


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
        
        res= self.Wupsilon*integ*self.dS_in
        return res

     
        
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
    umkl=3 #the number of umklaps where we calculate an observable ie Pi(q), for momentum transfers we need umkl+1 umklapps when scattering from the 1bz
    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0,c6sym,umkl)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2,c6sym,umkl) #this one is normalized
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
    

    
    
        
    mu=0
    filling=0
    print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
    
    substract=0 #0 for decoupled layers
    mu=0
    filling=0
    HB=Dispersion.HF_BandStruc( lq, hpl, hmin, hpl, hmin, nremote_bands, nbands, substract,  [V0, d_screening_norm], mode_HF)
    
    
    #BUBBLE CALCULATION
    bs=lq.boundary()
    print(np.shape(bs))
            
    test_symmetry=True
    B1=ep_Bubble(lq, nbands, HB,  mode_layer_symmetry, mode, cons, test_symmetry, umkl)
    a=B1.corr( args=(0.0,0.0))
    plt.scatter(lq.KX,lq.KY,c=a, cmap='Blues_r')
    plt.colorbar()
    plt.plot(bs[:,0], bs[:,1],c='r')
    plt.savefig("bubmensh.png")
    plt.close()

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
