import numpy as np
import scipy
from scipy import linalg as la
import time
import sys
import Hamiltonian
import Lattice
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import concurrent.futures
import functools
import pandas as pd

#TODO: plot dets see if this controls width -- cannnot be if there is filling dependence 

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

    def __init__(self, latt, nbands, hpl, hmin, test, umkl):

         ################################
        #lattice attributes
        ################################
        self.latt=latt
        # print(latt.Npoints, "points in the lattice")
        self.umkl=umkl
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice() #for the integration grid, we integrate over these
        [self.KX,self.KY]=latt.Generate_Umklapp_lattice(self.KX1bz, self.KY1bz,self.umkl) #for the q external momenta
        [self.KQX,self.KQY]=latt.Generate_Umklapp_lattice(self.KX1bz, self.KY1bz,self.umkl+1) #for the momentum transfer lattice
        
        self.Npoi1bz=np.size(self.KX1bz)
        self.Npoi=np.size(self.KX)
        self.NpoiQ=np.size(self.KQX)
        
        self.Ik=latt.insertion_index( self.KX1bz,self.KY1bz, self.KQX, self.KQY)
        self.dS_in=1/self.Npoi1bz
        
        
        ################################
        #dispersion attributes
        ################################
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        disp=Hamiltonian.Dispersion( latt, nbands, hpl, hmin)
        [self.psi_plus_1bz,self.Ene_valley_plus_1bz,self.psi_min_1bz,self.Ene_valley_min_1bz]=disp.precompute_E_psi()
        [self.psi_plus,self.Ene_valley_plus,self.psi_min,self.Ene_valley_min]=disp.precompute_E_psi_u(self.KQX,self.KQY)
        
        
        
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
        
        
        #generating form factors
        self.FFp=Hamiltonian.FormFactors(self.psi_plus, 1, latt, self.umkl+1)
        self.FFm=Hamiltonian.FormFactors(self.psi_min, -1, latt, self.umkl+1)

        self.name="diree"
        # self.Omega_FFp=self.FFp.denqFF()
        # self.Omega_FFm=self.FFm.denqFF()
        self.Omega_FFp=self.FFp.Fdirac()
        self.Omega_FFm=self.FFm.Fdirac()
        

        self.dS_in=latt.VolMBZ/self.Npoi1bz


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
                undet=np.abs(np.linalg.det(np.abs(self.Omega_FFp[k,:,kp,:])**2))  
                dosdet=np.abs(np.linalg.det(np.abs(self.Omega_FFp[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2))
                # undet=np.abs(np.linalg.det(self.Omega_FFp[k,:,kp,:]))
                # dosdet=np.abs(np.linalg.det(self.Omega_FFp[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                cos1.append(undet)
                diffarp.append( undet   - dosdet   )
                # Minus Valley FF Omega
                undet=np.abs(np.linalg.det(np.abs(self.Omega_FFm[k,:,kp,:])**2))
                dosdet=np.abs(np.linalg.det(np.abs(self.Omega_FFm[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2))
                # undet=np.abs(np.linalg.det(self.Omega_FFm[k,:,kp,:]))
                # dosdet=np.abs(np.linalg.det(self.Omega_FFm[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                cos2.append(undet)
                diffarm.append( undet   - dosdet   )
            

            plt.scatter(K,KP, c=cos1, s=4)
            plt.colorbar()
            plt.savefig("TestC3_symm_det_p"+self.name+".png")
            plt.close()
            
            plt.scatter(K,KP, c=cos2,s=4)
            plt.colorbar()
            plt.savefig("TestC3_symm_det_m"+self.name+".png")
            plt.close()

            # if np.mean(np.abs(diffarp))<1e-7:
            #     print("plus valley form factor passed the C3 symmetry test with average difference... ",np.mean(np.abs(diffarp)))
            # else:
            #     print("failed C3 symmetry test, plus valley...",np.mean(np.abs(diffarp)))
            #     #saving data for revision
            #     np.save("TestC3_symm_diff_p"+self.name+".npy",diffarp)
            #     np.save("TestC3_symm_K_p"+self.name+".npy",K)
            #     np.save("TestC3_symm_KP_p"+self.name+".npy",KP)
            #     np.save("TestC3_symm_det_p"+self.name+".npy",cos1)
            # if np.mean(np.abs(diffarm))<1e-7:
            #     print("minus valley form factor passed the C3 symmetry test with average difference... ",np.mean(np.abs(diffarm)))
            # else:
            #     print("failed C3 symmetry test, minus valley...",np.mean(np.abs(diffarp)))
            #     #saving data for revision
            #     np.save("TestC3_symm_diff_m"+self.name+".npy",diffarm)
            #     np.save("TestC3_symm_K_m"+self.name+".npy",K)
            #     np.save("TestC3_symm_KP_m"+self.name+".npy",KP)
            #     np.save("TestC3_symm_det_m"+self.name+".npy",cos2)

            print("finished testing symmetry of the form factors...")
            
            # ###DOS
            # Ndos=latt.Npoints
            # ldos=Lattice.TriangLattice(Ndos, 0) #this one
            # [self.KXdos, self.KYdos]=ldos.Generate_lattice()
            # self.Npoidos=np.size(self.KXdos)
            # [self.Ene_valley_plus_dos,self.Ene_valley_min_dos]=self.precompute_E_psi_dos()
            # # with open('dispersions/Edisp_'+str(Ndos)+'.npy', 'wb') as f:
            # #     np.save(f, self.Ene_valley_plus_dos)
            # # with open('dispersions/Edism_'+str(Ndos)+'.npy', 'wb') as f:
            # #     np.save(f, self.Ene_valley_min_dos)
            # # print("Loading  ..........")
            
            # # with open('dispersions/Edisp_'+str(Ndos)+'.npy', 'rb') as f:
            # #     self.Ene_valley_plus_dos=np.load(f)
            # # with open('dispersions/Edism_'+str(Ndos)+'.npy', 'rb') as f:
            # #     self.Ene_valley_min_dos=np.load(f)
                
            # plt.scatter(self.KQX, self.KQY, c=self.Ene_valley_plus[:,0])
            # plt.colorbar()
            # plt.savefig("energydisp0.png")
            # plt.close()
            # plt.scatter(self.KQX, self.KQY, c=self.Ene_valley_plus[:,1])
            # plt.colorbar()
            # plt.savefig("energydisp1.png")
            # plt.close()
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(self.KQX, self.KQY,self.Ene_valley_plus[:,1], c=self.Ene_valley_plus[:,1]);
            # ax.scatter3D(self.KQX, self.KQY,self.Ene_valley_plus[:,0], c=self.Ene_valley_plus[:,1]);

            # plt.savefig("ene.png")
            # plt.close()
            # ind0=np.where(self.KQY==0)
            # plt.plot(self.KQX[ind0], self.Ene_valley_plus[ind0,1].flatten())
            # plt.plot(self.KQX[ind0], self.Ene_valley_plus[ind0,0].flatten())
            # plt.savefig("eneri.png")
            # plt.close()

            # [self.bins,self.valt, self.f2 ]=hpl.DOS(self.Ene_valley_plus_dos,self.Ene_valley_min_dos)
            # [self.earr,self.dosarr,self.f2 ]=hpl.DOS2(self.Ene_valley_plus_dos,self.Ene_valley_min_dos,1/np.size(self.KXdos))

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


    def deltad(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)
    
    def test_deltad(self):
        dels=[0.001,0.01,0.1,1,2,5,10]
        ees=np.linspace(-10,10,int(20/self.eta))
       
        for de in dels:
            plt.plot(ees,self.deltad(ees,self.eta*de) )
            plt.savefig(str(de)+".png")
            print("testing delta...", de,np.trapz(self.deltad(ees,self.eta*de))*self.eta)


    def integrand_T(self,nkq,nk,ekn,ekm,w,mu,T):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

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

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)
    
    def integrand_ZT(self,nkq,nk,ekn,ekm,w,mu):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1

        ####delta
        # deltad_cut=1e-17*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        # fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-self.eta_cutoff, 0.0)/(deltad_cut-(edkq-edk))
        # fac_p2=(self.deltad( edk, self.eta_dirac_delta))*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        
        # return (fac_p+fac_p2)
        
        ####iepsilon
        eps=self.eta_small_imag ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)


    
    def parCompute(self, args):
        
        (omegas, kpath,  muv, ind)=args
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
                
                Ikq=self.latt.insertion_index( self.KX1bz+qx,self.KY1bz+qy, self.KQX, self.KQY)

            
                #first index is momentum, second is band third and fourth are the second momentum arg and the fifth is another band index
                Lambda_Tens_plus_kq_k=np.array([self.Omega_FFp[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi1bz)])
                Lambda_Tens_min_kq_k=np.array([self.Omega_FFm[Ikq[ss],:,self.Ik[ss],:] for ss in range(self.Npoi1bz)])


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
                        

                bub=bub+np.sum(integrand_var)*self.dS_in

                sd.append( bub )

            integ.append(sd)
        eb=time.time()
        
        
        integ_arr_no_reshape=np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        

        integ=integ_arr_no_reshape.flatten()  #in ev
        
        return [integ]
    

    def plot_res(self, integ, KX,KY, VV, filling, Nsamp, add_tag):
        identifier=add_tag+str(Nsamp)+"_nu_"+str(filling)+self.name
        plt.plot(VV[:,0],VV[:,1])
        plt.scatter(KX,KY, s=20, c=np.real(integ))
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.clim(0,8000)
        plt.colorbar()
        plt.savefig("Pi_ep_energy_cut_real_"+identifier+".png")
        plt.close()
        print("the minimum real part is ...", np.min(np.real(integ)))
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(KX,KY,np.real(integ), c=np.real(integ));
        plt.savefig("3drealinteg.png")
        plt.close()

        plt.plot(VV[:,0],VV[:,1])
        plt.scatter(KX,KY, s=20, c=np.imag(integ))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.savefig("Pi_ep_energy_cut_imag_"+identifier+".png")
        plt.close()
        print("the maximum imaginary part is ...", np.max(np.imag(integ)))

        plt.plot(VV[:,0],VV[:,1])
        plt.scatter(KX,KY, s=20, c=np.abs(integ))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.colorbar()
        plt.savefig("Pi_ep_energy_cut_abs_"+identifier+".png")
        plt.close()


        print("testing symmetry of the result...")
        [KXc3z,KYc3z, Indc3z]=self.latt.C3zLatt(KX,KY)
        intc3=[integ[int(ii)] for ii in Indc3z]
        plt.plot(np.abs(integ- intc3)/np.abs(integ), 'ro')
        plt.plot(np.abs(integ- intc3)/np.abs(integ), c='k', ls='--')
        plt.savefig("Test_C3_symm_bubble_"+identifier+".png")
        plt.close()
        print("finished testing symmetry of the result...")

        print("saving data from the run ...")

        with open("bubble_data_"+identifier+".npy", 'wb') as f:
            np.save(f, integ)
        with open("bubble_data_kx_"+identifier+".npy", 'wb') as f:
            np.save(f, KX)
        with open("bubble_data_ky_"+identifier+".npy", 'wb') as f:
            np.save(f, KY)



    def epsilon_sweep(self,fillings, mu_values,earr, dos):
        
        
        scalings=[1e-8,0.001,0.01,0.1,1,2,5, 100]
        for e in scalings:
            cs=[]
            cs_lh=[]
            eddy=self.eta_small_imag
            self.eta_small_imag=e*eddy
            
            #have to divide by the vlume of the MBZ to get the same normalization
            Vol=self.lq.VolMBZ
            
            mmin=np.min([np.min(self.Ene_valley_plus),np.min(self.Ene_valley_min)])
            mmax=np.max([np.max(self.Ene_valley_plus),np.max(self.Ene_valley_min)])
            NN=int((mmax-mmin)/self.eta)+int((int((mmax-mmin)/self.eta)+1)%2) #making sure there is a bin at zero energy

            bins2=np.linspace(earr[0],earr[-1],NN)
            for ide in range(np.size(bins2)):
                # mu= mu_values[ide]
                # filling=fillings[ide]
                omega=[0]
                kpath=np.array([0,0]).T
                integ=self.Compute_noff(bins2[ide], omega, kpath)
                integ_lh=self.Compute_lh_noff(bins2[ide], omega, kpath)
                cs.append(np.real(integ)/Vol)
                cs_lh.append(integ_lh/Vol)
            self.eta_small_imag=eddy
            # print(e)

            # print(cs)
            # print(cs_lh)
            plt.title("scaling fact cutoff "+str(e))
            plt.plot(earr, dos, label="hist")
            plt.plot(earr, dos, label="integ")
            plt.legend()
            plt.savefig("comparedos"+str(e)+".png")
            plt.close()
            plt.title("scaling fact cutoff"+str(e))
            plt.plot(bins2,cs, label="ieps")
            plt.plot(bins2,cs_lh, label="derivative")
            plt.plot(earr, dos, label="hist")
            plt.legend()
            plt.title("scaling fact cutoff "+str(e))
            print("normalizations ", e , np.sum(cs)*(bins2[1]-bins2[0]),np.sum(cs_lh)*(bins2[1]-bins2[0]))
            plt.savefig("scaling"+str(e)+".png")
            plt.close()
            
            plt.plot(bins2,cs, label="ieps")
            plt.plot(bins2,cs_lh, label="derivative")
            plt.plot(earr, dos, label="integ")
            plt.title("scaling fact "+str(e))
            plt.legend()
            plt.savefig("scaling_2_"+str(e)+".png")
            plt.close()
            
    def savedata(self, integ, filling, Nsamp, add_tag):
        
        identifier=add_tag+str(Nsamp)+self.name
        Nfills=np.size(filling)
        Nss=np.size(self.KX)

        #products of the run
        Pibub=np.hstack(integ)
        filling_list=[]
        for i,ff in enumerate(filling):
            filling_list=filling_list+[filling[i]]*Nss
            
        KXall=np.hstack([self.KX]*Nfills)
        KYall=np.hstack([self.KY]*Nfills)
        fillingarr=np.array(filling_list)
        
            
        df = pd.DataFrame({'bub': Pibub, 'kx': KXall, 'ky': KYall,'nu': fillingarr})
        df.to_hdf('data'+identifier+'.h5', key='df', mode='w')


        return None
            
    def Fill_sweep(self,fillings, mu_values,VV, Nsamp):
        
        selfE=[]

        qp=np.arange(np.size(fillings))
        s=time.time()
        omega=[1e-14]
        kpath=np.array([self.KX,self.KY]).T
        
        arglist=[]
        for i, qpval in enumerate(qp):
            arglist.append( (omega, kpath, mu_values, qpval) )

        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_file = {
                
                executor.submit(self.parCompute, arglist[qpval]): qpval for qpval in qp
            }

            for future in concurrent.futures.as_completed(future_to_file):
                result = future.result()  # read the future object for result
                selfE.append(result[0])
                qpval = future_to_file[future]  

        
        
        e=time.time()
        t=e-s
        print("time for sweep delta", t)
        
        self.savedata( selfE, fillings, Nsamp, "")
                
        return t
        
        

        
def main() -> int:
    
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

    #lattices with different normalizations

    latt=Lattice.TriangLattice(Nsamp,0)
    [KX,KY]=latt.Generate_lattice()
    Npoi=np.size(KX); print(Npoi, "numer of sampling lattice points")
    umkl=0
    print(f"taking {umkl} umklapps")
    VV=latt.boundary()
    dS_in=latt.VolMBZ/Npoi
    print(dS_in)


    hbvf = 2.1354; # eV
    print("hbvf is ..",hbvf )


    nbands=2
    hbarc=0.1973269804*1e-6 #ev*m
    alpha=137.0359895 #fine structure constant
    a_graphene=2.458*(1e-10) #in meters
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
    gamma=np.sqrt(hhbar/(a_graphene*mass*c_phonon))
    gammap=(gamma**2)/((a_graphene**2)*((2*np.pi)**2)) 
    scaling_fac=( a_graphene**2) /(mass)
    print("phonon params...", gammap/1e+11 , gamma, np.sqrt(gammap*scaling_fac),gammap*scaling_fac)
    mode_layer_symmetry="a" #whether we are looking at the symmetric or the antisymmetric mode
    cons=[alpha_ep, beta_ep, gammap, a_graphene, mass] #constants used in the bubble calculation and data anlysis


    hpl=Hamiltonian.Ham_p(hbvf, latt)
    hm=Hamiltonian.Ham_m(hbvf, latt)


    
    #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
    Ndos=100
    ldos=Lattice.TriangLattice(Ndos,0)
    disp=Hamiltonian.Dispersion( ldos, nbands, hpl, hm)
    Nfils=20
    [fillings,mu_values]=disp.mu_filling_array(Nfils, False, True, True)
    filling_index=int(sys.argv[1]) 
    mu=mu_values[filling_index]
    filling=fillings[filling_index]
    print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
    [earr, dos]=disp.dos_filling_array(Nfils, False, True, True)
    [earr2, dos2]=disp.dos2_filling_array(Nfils, False, True, True, dS_in)
    
    plt.plot(earr, dos)
    plt.plot(earr2, dos2)
    plt.savefig("comparingdos.png")
    plt.close()
    
    #BUBBLE CALCULATION
    test_symmetry=True
    B1=ee_Bubble(latt, nbands, hpl, hm,test_symmetry, umkl)
    omega=[0]
    [KX,KY]=latt.Generate_Umklapp_lattice(KX, KY,0) #for the integration grid 

    # kpath=np.array([KX,KY]).T
    # integ=B1.Compute(mu, omega, kpath)
    # B1.plot_res(integ, KX,KY, VV, filling, Nsamp, "")
    # B1.epsilon_sweep(fillings, mu_values,earr, dos)
    
    # #BUBBLE CALCULATION
    # test_symmetry=True
    # B1=ep_Bubble(lq, nbands, hpl, hmin,  mode_layer_symmetry, mode, cons, test_symmetry, umkl)
    # omega=[1e-14]
    # kpath=np.array([KX,KY]).T
    # integ=B1.Compute(mu, omega, kpath)
    # popt, res, c, resc=B1.extract_cs( integ, 0.2)
    # B1.plot_res(integ, KX,KY, VV, filling, Nsamp, c , res, "")
    # print(np.mean(popt),np.mean(c), resc, c_phonon)
    # # B1.Fill_sweep(fillings, mu_values, VV, Nsamp, c_phonon,theta)
    
    # NnarrowSamp=100
    # lnarrowSamp=Lattice.TriangLattice(NnarrowSamp,theta,2)
    # [ Kxp, Kyp]=lnarrowSamp.Generate_lattice()
    
    # [KX_m, KY_m, ind]=lnarrowSamp.mask_KPs( Kxp, Kyp, 0.2)
    # kpath=np.array([KX_m, KY_m]).T
    # integ=B1.Compute_lh(mu, omega, kpath)
    # popt, res, c, resc=B1.extract_cs_path( integ, kpath)
    # print(np.mean(popt),np.mean(c), resc, c_phonon)
    # B1.plot_res(integ, KX_m, KY_m, VV, filling, Nsamp, c , res, "small")
    
     #BUBBLE CALCULATION
    B1.Fill_sweep([0], [0], VV, Nsamp)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
