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
import Mean_Field_Latt



class FormFactors_kq():
    def __init__(self, psi_kq, psi_k, ks,  xi, Bands=None):
        
        #lattice and kpoint attributes
        self.xi=xi
        self.Nu=int(np.shape(psi_k)[0]/4) #4, 2 for sublattice and 2 for layer
        
        [kq,k]=ks
        self.kx=k[0]
        self.ky=k[0]
        self.kqx=kq[0]
        self.kqy=kq[0]
        
        self.qx=self.kqx-self.kx
        self.qy=self.kqy-self.ky
        self.q=np.sqrt(self.qx**2+self.qy**2)
        
        
        #delimiting bands to be used 
        if Bands is None:
            self.tot_nbands = np.shape(psi_k)[1]
            inindex=0
            finindex=self.tot_nbands
            # print("calculating form factors with all bands is input wavefunction")
        else:
            self.tot_nbands= Bands
            initBands=np.shape(psi_k)[1]
            inindex=int(initBands/2)-int(Bands/2)
            finindex=int(initBands/2)+int(Bands/2)
            # print(f"truncating wavefunction and calculating form factors with only {Bands} bands")
            # print(f"originally we had {initBands} bands, we sliced from {inindex} to {finindex} for Form Facts (upper bound is excluded)")
            
        self.psi_k = psi_k[:,inindex:finindex] #has dimension # 4*N, nbands
        self.psi_kq = psi_kq[:,inindex:finindex] #has dimension # 4*N, nbands
        
        self.cpsi_k=np.conj(self.psi_k)
        self.cpsi_kq=np.conj(self.psi_kq)

    def __repr__(self):
        return "Form factors for valley {xi}".format( xi=self.xi)

    def matmult(self, layer, sublattice):
        if layer==0 and sublattice==0:

            return self.psi_k
            
        else:
            pauli0=np.array([[1,0],[0,1]])
            paulix=np.array([[0,1],[1,0]])
            pauliy=np.array([[0,-1j],[1j,0]])
            pauliz=np.array([[1,0],[0,-1]])

            pau=[pauli0,paulix,pauliy,pauliz]
            Qmat=np.eye(self.Nu)
        
            mat=np.kron(pau[layer],np.kron(Qmat, pau[sublattice]))

            return  mat@self.psi_k
            

    def calcFormFactor(self, layer, sublattice):
        mult_psi=self.matmult(layer,sublattice)
        Lambda_Tens=(self.cpsi_kq.T)@mult_psi
        
        return Lambda_Tens
    


    def fq(self ):
                        
        return (self.qx**2-self.qy**2)/self.q

    def gq(self):
          
        return 2*(self.qx*self.qy)/self.q


    def hq(self):
                        
        return self.q


        

    ########### Anti-symmetric displacement of the layers
    def denqFF_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return L30

    def denqFFL_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return self.hq()*L30


    def NemqFFL_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFL=self.fq() *L31-self.xi*self.gq()*L32
        return Nem_FFL

    def NemqFFT_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFT=-self.gq() *L31- self.xi*self.fq()*L32
        return Nem_FFT

    ########### Symmetric displacement of the layers
    def denFF_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return L00

    def denqFFL_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return self.hq()*L00

    def NemqFFL_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFL=self.fq() *L01-self.xi*self.gq()*L02
        return Nem_FFL

    def NemqFFT_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFT=-self.gq()*L01 - self.xi*self.fq()*L02
        return Nem_FFT

    ################################
    # form factor tests
    ################################
    
    def test_C2T_dens(self, LamP):
        pauliz=np.array([[1,0],[0,-1]])
        if self.tot_nbands==2:
            transf_mat=pauliz@(np.conj(LamP)@pauliz)
            mean_dif=np.mean( np.abs( LamP - transf_mat  ) )
            if mean_dif>1e-9:
                print(self.xi,'form factor failed C2T by', mean_dif )
                print("1\n",LamP)
                print("2\n",transf_mat )
    
    def test_Cstar_dens(self, LamP,LamM):
        #this test is very lax a more correct approach would be to hard code the symmtery for serious calculations
        # the main issue is the loss of precission in the extension if the wave functions and the ammound of matrix operations 
        # to get to the form factors
        paulix=np.array([[0,1],[1,0]])
        if self.tot_nbands==2:
            transf_mat=paulix@(LamM@paulix) 
            mean_dif=np.mean( np.abs( LamP - transf_mat  ) )
            if mean_dif>1e-3:
                print(self.xi,'form factor failed Cstar by', mean_dif )
                print("1\n",LamP)
                print("2\n",transf_mat )
                
    def test_Csub_dens(self, LamP):
        paulix=np.array([[0,1],[1,0]])
        if self.tot_nbands==2:
            transf_mat=paulix@(LamP@paulix) 
            mean_dif=np.mean( np.abs( LamP - transf_mat  ) )
            if mean_dif>1e-9:
                print(self.xi,'form factor failed csub by', mean_dif )
                print("1\n",LamP )
                print("2\n",transf_mat )

    def test_C2T_nemc3_FFs(self, LamP):
        pauliz=np.array([[1,0],[0,-1]])
        if self.tot_nbands==2:
            transf_mat=pauliz@(np.conj(LamP)@pauliz)
            mean_dif=np.mean( np.abs( LamP - transf_mat  ) )
            if mean_dif>1e-9:
                print(self.xi,'form factor failed C2T by', mean_dif )
                print("1\n",LamP)
                print("2\n",transf_mat )


    def test_Cstar_nemc3_FFs(self, LamP,LamM):
        #this test is very lax a more correct approach would be to hard code the symmtery for serious calculations
        # the main issue is the loss of precission in the extension if the wave functions and the ammound of matrix operations 
        # to get to the form factors
        paulix=np.array([[0,1],[1,0]])
        if self.tot_nbands==2:
            transf_mat=-paulix@(LamM@paulix) 
            mean_dif=np.mean( np.abs( LamP - transf_mat  ) )
            if mean_dif>1e-3:
                print(self.xi,'form factor failed Cstar by', mean_dif )
                print("1\n",LamP)
                print("2\n",transf_mat )
                    
                    
    def test_Csub_nemc3_FFs(self, LamP):
        paulix=np.array([[0,1],[1,0]])
        if self.tot_nbands==2:
            transf_mat=-paulix@(LamP@paulix) 
            mean_dif=np.mean( np.abs( LamP - transf_mat  ) )
            if mean_dif>1e-9:
                print(self.xi,'form factor failed csub by', mean_dif )
                print("1\n",LamP )
                print("2\n",transf_mat )
    
      
   
class Phon_bare_BandStruc:
    
    def __init__(self, latt, hpl, hmin, nbands, cons, field, qins_X, qins_Y, sym, mode, layersym ):

        
        self.latt=latt
        
        
        self.nbands_init=2 
        self.nbands=nbands

        
        self.ini_band=int(self.nbands_init/2)-int(self.nbands/2)
        self.fini_band=int(self.nbands_init/2)+int(self.nbands/2)
        
        
        self.hpl=hpl
        self.hmin=hmin
        
        self.field=field
        self.sym=sym     #whether we take all c3 invariant momenta
        if self.sym:
            ang=2*np.pi/3
            qins_X_2=np.cos(ang)*qins_X-np.sin(ang)*qins_Y
            qins_Y_2=np.sin(ang)*qins_X+np.cos(ang)*qins_Y
            ang=4*np.pi/3
            qins_X_3=np.cos(ang)*qins_X-np.sin(ang)*qins_Y
            qins_Y_3=np.sin(ang)*qins_X+np.cos(ang)*qins_Y
            self.qins_X=[qins_X,-qins_X,qins_X_2,-qins_X_2,qins_X_3,-qins_X_3]
            self.qins_Y=[qins_Y,-qins_Y,qins_Y_2,-qins_Y_2,qins_Y_3,-qins_Y_3]
            self.nqs=6
            self.lmf = None
            print(' "RBVS" lattice not implemented')
            exit()
        else:
            self.qins_X=[qins_X,-qins_X]
            self.qins_Y=[qins_Y,-qins_Y]
            self.nqs=2
            self.lmf =Mean_Field_Latt.MF_Lattice_M1_phonon(self.latt)
            

        [self.alpha_ep, self.beta_ep,  self.Wupsilon, self.agraph, self.mass ]=cons #constants for the exraction of the effective velocity
        self.disp=Dispersion.Dispersion( latt, self.nbands, hpl, hmin)
        
        #for converting phonon displacements into the right units
        [q1,q2,q3]=latt.qvect()
        self.qscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        
        
        ev_meter=1e+3 *1e-10
        phiM=1e-13
        print(self.beta_ep,ev_meter*self.beta_ep*(self.qscale/self.agraph),1e+3*phiM*self.beta_ep*(self.qscale/self.agraph))
        phiM=1e-12
        print(self.beta_ep,ev_meter*self.beta_ep*(self.qscale/self.agraph),1e+3*phiM*self.beta_ep*(self.qscale/self.agraph))
        phiM=5e-12
        print(self.beta_ep,ev_meter*self.beta_ep*(self.qscale/self.agraph),1e+3*phiM*self.beta_ep*(self.qscale/self.agraph))
        phiM=1e-11
        print(self.beta_ep,ev_meter*self.beta_ep*(self.qscale/self.agraph),1e+3*phiM*self.beta_ep*(self.qscale/self.agraph))
        
        
        self.High_symmetry(0.0)
        self.High_symmetry(0.001*self.agraph)
        self.High_symmetry(0.005*self.agraph)
        self.High_symmetry(0.01*self.agraph)
        self.High_symmetry(0.05*self.agraph)
        self.High_symmetry(0.1*self.agraph)
        
    def eig(self, kx,ky, phi_M ):
        
        [wave1p_k,E1p_k,wave1m_k,E1m_k]=self.disp.E_gauge_psi( kx , ky )
        [wave1p_kq,E1p_kq,wave1m_kq,E1m_kq]=self.disp.E_gauge_psi( kx + self.latt.M1[0] , ky+self.latt.M1[1] )

        k=[ kx , ky]
        kq=[kx+self.latt.M1[0] , ky+self.latt.M1[1]]
        FFp=FormFactors_kq( wave1p_kq[:,self.ini_band:self.fini_band],wave1p_k[:,self.ini_band:self.fini_band], [kq,k],  1)
        Lnemp=FFp.NemqFFT_a()
        Trans=self.beta_ep*Lnemp
       
        off_D_k_kq=Trans*phi_M*(self.qscale/self.agraph)
        Ham_P=np.bmat([[np.diag(E1p_k),off_D_k_kq],[np.conj(off_D_k_kq.T),np.diag(E1p_kq)]])
        
        (Eigvals,Eigvect)= np.linalg.eigh(Ham_P)  #returns sorted eigenvalues
        return [Eigvals,Eigvect]
    
    def eig_M(self, kx,ky, phi_M ):
        
        [wave1p_k,E1p_k,wave1m_k,E1m_k]=self.disp.E_gauge_psi( kx , ky )
        [wave1p_kq,E1p_kq,wave1m_kq,E1m_kq]=self.disp.E_gauge_psi( kx + self.latt.M1[0] , ky+self.latt.M1[1] )

        k=[ kx , ky]
        kq=[kx+self.latt.M1[0] , ky+self.latt.M1[1]]
        FFp=FormFactors_kq( wave1m_kq[:,self.ini_band:self.fini_band],wave1m_k[:,self.ini_band:self.fini_band], [kq,k],  -1)
        Lnemp=FFp.NemqFFT_a()
        Trans=self.beta_ep*Lnemp
       
        off_D_k_kq=Trans*phi_M*(self.qscale/self.agraph)
        Ham_M=np.bmat([[np.diag(E1m_k),off_D_k_kq],[np.conj(off_D_k_kq.T),np.diag(E1m_kq)]])
        
        (Eigvals,Eigvect)= np.linalg.eigh(Ham_M)  #returns sorted eigenvalues
        return [Eigvals,Eigvect]
        
    
    def High_symmetry(self, phiM):
        print("\n")
        print("band structure across high symmetry directions")
       
        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]

        nbands=4 #Number of bands 
        kpath=self.latt.High_symmetry_path()
        
        kx=kpath[:,0]
        ky=kpath[:,1]
        VV=self.latt.boundary()
        plt.scatter(kx,ky, c='r')
        plt.plot(VV[:,0], VV[:,1], c='b')
        plt.savefig("highsym_path.png")
        plt.close()

        Npoi=np.shape(kpath)[0]
        
        for l in range(Npoi):
            # h.umklapp_lattice()
            # break
            E1p,wave1p=self.eig(kpath[l,0],kpath[l,1],phiM)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
            psi_plus_a.append(wave1p)


            E1m,wave1m=self.eig_M(kpath[l,0],kpath[l,1],phiM)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
            psi_min_a.append(wave1m)

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

    

        print("shape of the energies..",np.shape(Ene_valley_plus_a))
        qa=np.linspace(0,1,Npoi)
        for i in range(nbands):
            plt.plot(qa,Ene_valley_plus[:,i] , c='b')
            plt.plot(qa,Ene_valley_min[:,i] , c='r', ls="--")
            if i==int(nbands/2-1):
                min_min  = np.min(Ene_valley_min[:,i])
                min_plus = np.min(Ene_valley_plus[:,i])
            
            elif i==int(nbands/2):
                max_min  = np.max(Ene_valley_min[:,i])
                max_plus = np.max(Ene_valley_plus[:,i])
        maxV=np.max([max_min,max_plus])
        minC=np.max([min_min,min_plus])
        BW=maxV-minC
        print("the bandwidth is ..." ,BW)
        plt.xlim([0,1])
        # plt.ylim([-0.008,0.008])
        plt.savefig("highsym_phi_"+str(phiM)+".png")
        plt.close()
        print("finished band structure along high symmetry directions")
        print("\n \n")
        return [Ene_valley_plus, Ene_valley_min]

        

     
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
    umkl=1 #the number of umklaps where we calculate an observable ie Pi(q), for momentum transfers we need umkl+1 umklapps when scattering from the 1bz #fock corrections converge at large umklapp of 4
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
    eps_inv = 1.0/5.0
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
    cons=[alpha_ep_effective,beta_ep_effective, Wupsilon, a_graphene, mass] #constants used in the bubble calculation and data anlysis

    
    #Hartree fock correction to the bandstructure
    hpl=Dispersion.Ham_BM(hvkd, alph, 1, lq, kappa, PH, 1) #last argument is whether or not we have interlayer hopping
    hmin=Dispersion.Ham_BM(hvkd, alph, -1, lq, kappa, PH, 1 ) #last argument is whether or not we have interlayer hopping
    
    
    
    substract=0 #0 for decoupled layers
    mu=0
    filling=0
    field=1
    layersym="a"
    qins_X=lq.M1[0]
    qins_Y=lq.M1[1]
    sym=False
    HB=Phon_bare_BandStruc( lq,  hpl, hmin, nbands,  cons, field, qins_X, qins_Y, sym, mode, layersym )

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
