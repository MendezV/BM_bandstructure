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

 
#TODO: parameter file that contains Nsamp, Numklaps, kappa, theta, mode, default_filling, alpha, beta, alphamod, betamod

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

    def __init__(self, latt, nbands, hpl, hmin, symmetric, mode, cons , test, umkl):
        self.lq=latt
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice()
        self.Npoi1bz=np.size(self.KX1bz)
        [self.KX,self.KY]=latt.Generate_Umklapp_lattice(self.KX1bz, self.KY1bz,umkl)
        self.Npoi=np.size(self.KX)
        self.latt=latt
        self.umkl=umkl
        [q1,q2,q3]=latt.qvect()
        self.Gscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice()
        [self.KQX, self.KQY, self.Ik]=latt.Generate_momentum_transfer_umklapp_lattice( self.KX1bz, self.KY1bz,  self.KX,self.KY)
        self.Npoi_Q=np.size(self.KQX)
        # ##########
        # [self.psi_plus,self.Ene_valley_plus,self.psi_min,self.Ene_valley_min]=self.precompute_E_psi_noQ()
        # hpl.DOS(self.Ene_valley_plus,self.Ene_valley_min)

        # #########
        [self.psi_plus,self.Ene_valley_plus,self.psi_min,self.Ene_valley_min]=self.precompute_E_psi()
        self.eta=np.mean( np.abs( np.diff( self.Ene_valley_plus[:,int(nbands/2)].flatten() )  ) )/2
        self.FFp=Hamiltonian.FormFactors(self.psi_plus, 1, latt, self.umkl)
        self.FFm=Hamiltonian.FormFactors(self.psi_min, -1, latt, self.umkl)
        [self.alpha_ep, self.beta_ep, self.gamma, self.agraph, self.mass ]=cons
        self.mode=mode
        self.symmetric=symmetric
        self.name="_mode_"+self.mode+"_symmetry_"+self.symmetric+"_alpha_"+str(self.alpha_ep)+"_beta_"+str(self.beta_ep)

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

        self.dS_in=latt.VolMBZ/self.Npoi


        
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
            for k in range(self.Npoi_Q):
                K.append(self.KQX[k]-self.KQX[kp])
                KP.append(self.KQY[k]-self.KQY[kp])
                #Regular FF
                # Plus Valley FF Omega
                # undet=np.abs(np.linalg.det(self.Lnemp[k,:,kp,:]))
                # dosdet=np.abs(np.linalg.det(self.Lnemp[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                undet=np.abs(np.linalg.det(self.Omega_FFp[k,:,kp,:]))
                dosdet=np.abs(np.linalg.det(self.Omega_FFp[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                # cos1.append(undet)
                # cos2.append(dosdet)
                diffarp.append( undet   - dosdet   )
                # Minus Valley FF Omega
                # undet=np.abs(np.linalg.det(self.Lnemm[k,:,kp,:]))
                # dosdet=np.abs(np.linalg.det(self.Lnemm[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                undet=np.abs(np.linalg.det(self.Omega_FFm[k,:,kp,:]))
                dosdet=np.abs(np.linalg.det(self.Omega_FFm[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                diffarm.append( undet   - dosdet   )
                

            plt.plot(diffarp, label="plus valley")
            plt.plot(diffarm, label="minsu valley")
            plt.title("FF- C3 FF")
            plt.savefig("TestC3_symm"+self.name+".png")
            # plt.show()
            plt.close()

            # plt.scatter(K,KP,c=cos1)
            # plt.colorbar()
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.show()

            # plt.scatter(K,KP,c=cos2)
            # plt.colorbar()
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.show()

            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax = plt.axes(projection='3d')
            # ax.scatter3D(K,KP,cos1, c=cos1);
            # plt.show()

            print("finished testing symmetry of the form factors...")


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

        plt.scatter(self.KX, self.KY, c=Ene_valley_min[:,-1])
        plt.show()


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

    def parallel_precompute_E_psi(self):
        Mac_maxthreads=6
        Desk_maxthreads=12

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        qp=np.array([self.KQX, self.KQY]).T
        s=time.time()
        eigplus = functools.partial(self.hpl.parallel_eigens, self.nbands)
        eigmin = functools.partial(self.hmin.parallel_eigens, self.nbands)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_pl = executor.map(eigplus, qp, chunksize=int(np.size(qp)/Mac_maxthreads))
            results_min = executor.map(eigmin, qp, chunksize=int(np.size(qp)/Mac_maxthreads))

            for result in results_pl:
                Ene_valley_plus_a=np.append(Ene_valley_plus_a,result[0])
                psi_plus_a.append(result[1])
            for result in results_min:
                Ene_valley_min_a=np.append(Ene_valley_min_a,result[0])
                psi_min_a.append(result[1])

        e=time.time()


        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoi_Q,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoi_Q,self.nbands])


        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]
    
    def w_ph_L(self):
        # return self.omegacoef*(self.FFp.h_denominator(self.Lnemp)) #h corresponds to the norm, later will have to code different dispersions
        # plt.plot(1/np.sort(self.FFp.h_denominator(self.Lnemp).flatten()))
        # plt.show()
        return self.FFp.h_denominator(self.Lnemp) #h corresponds to the norm, later will have to code different dispersions
        
    def OmegaL(self):
        
        # overall_coef=self.sqrt_hbar_M/np.sqrt(self.w_ph_L())
        overall_coef=1/np.sqrt(self.w_ph_L())
        
        
        Omega_FFp=overall_coef*(self.alpha_ep*self.L00p+self.beta_ep*self.Lnemp)#/np.sqrt(self.Npoi)
        Omega_FFm=overall_coef*(self.alpha_ep*self.L00m+self.beta_ep*self.Lnemm)#/np.sqrt(self.Npoi)
                
        return [Omega_FFp,Omega_FFm]

    def w_ph_T(self):
        return self.FFp.h_denominator(self.Lnemp) #h corresponds to the norm, later will have to code different dispersions
        
    def OmegaT(self):
        overall_coef=1/np.sqrt(self.w_ph_T())
        Omega_FFp=overall_coef*(self.beta_ep*self.Lnemp)
        Omega_FFm=overall_coef*(self.beta_ep*self.Lnemm)

        return [Omega_FFp,Omega_FFm]

    def integrand_ZT(self,nkq,nk,ekn,ekm,w,mu):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1
        eps=0.5*self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        return (fac_p)

    def deltad(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    def integrand_ZT_lh(self,nkq,nk,ekn,ekm,w,mu):
        eps=0.5*self.eta ##SENSITIVE TO DISPERSION
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1

        deltad_cut=1e-17*np.heaviside(eps-np.abs(edkq-edk), 1.0)
        fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-eps, 0.0)/(deltad_cut-(edkq-edk))
        fac_p2=0.0

        # fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-eps, 0.0)/(deltad_cut-(edkq-edk))
        # fac_p2=(self.deltad( edk, 50*eps))*np.heaviside(eps-np.abs(edkq-edk), 1.0)
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
                for s in range(self.Npoi):
                    kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
                    indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
                    Ikq.append(indmin)

            
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
        return self.gamma*integ_arr_no_reshape
    
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
                Ikq=[]
                for s in range(self.Npoi):
                    kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
                    indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
                    Ikq.append(indmin)

            
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
        return self.gamma*integ_arr_no_reshape
    
    def parCompute(self, theta, omegas, kpath, VV, Nsamp,  muv, fil,ind):
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
                Ikq=[]
                for s in range(self.Npoi):
                    kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
                    indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
                    Ikq.append(indmin)

            
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
        
        prop_BZ=0.2
        integ_arr_no_reshape=self.gamma*np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        
        
        print("the filling is .. " , mu)
        integ=integ_arr_no_reshape.flatten()*1000 #convertion to mev
        popt, res, c, resc=self.extract_cs( integ, prop_BZ)
        print("parameters of the fit...", c)
        print("residual of the fit...", res)

        self.plot_res( integ, self.KX1bz,self.KY1bz, VV, fil[ind], Nsamp,c, res, str(theta))
        
        return [integ,res, c]

    def parCompute_lh(self, theta, omegas, kpath, VV, Nsamp,  muv, fil,ind):
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
                Ikq=[]
                for s in range(self.Npoi):
                    kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
                    indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
                    Ikq.append(indmin)

            
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
        
        prop_BZ=0.2
        integ_arr_no_reshape=self.gamma*np.array(integ).flatten()#/(8*Vol_rec) #8= 4bands x 2valleys
        print("time for bubble...",eb-sb)
        
        
        print("the filling is .. " , mu)
        integ=integ_arr_no_reshape.flatten()*1000 #convertion to mev
        popt, res, c, resc=self.extract_cs( integ, prop_BZ)
        print("parameters of the fit...", c)
        print("residual of the fit...", res)

        self.plot_res( integ, self.KX1bz,self.KY1bz, VV, fil[ind], Nsamp,c, res, str(theta))
        
        return [integ,res, c]

    def extract_cs(self, integ, thres):
        scaling_fac=self.agraph**2 /(self.mass*(self.Gscale**2))
        print(scaling_fac)
        [KX_m, KY_m, ind]=self.latt.mask_KPs( self.KX1bz,self.KY1bz, thres)
        # plt.scatter(self.KX1bz,self.KY1bz, c=np.real(integ))
        # plt.show()
        id=np.ones(np.shape(KX_m))
        Gmat=np.array([id, KX_m,KY_m,KX_m**2,KY_m*KX_m,KY_m**2]).T
        GTG=Gmat.T@Gmat
        d=np.real(integ[ind])
        # plt.scatter(KX_m, KY_m, c=d)
        # plt.show()
        b=Gmat.T@d
        popt=la.pinv(GTG)@b
        res=np.sqrt(np.sum((Gmat@popt-d)**2)) #residual of the least squares procedure
        nn=np.sqrt(np.sum((d)**2)) #residual of the least squares procedure
        print(popt, res/nn, np.sqrt(np.mean(popt)*scaling_fac),np.sqrt(res*scaling_fac/nn))
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
        with open("cparams_data_"+identifier+".npy", 'wb') as f:
            np.save(f, c)
        with open("cparams_res_data_"+identifier+".npy", 'wb') as f:
            np.save(f, res)


    def Fill_sweep(self,fillings, mu_values,VV, Nsamp, c_phonon,theta):
        prop_BZ=0.2
        cs=[]
        cs_lh=[]
        rs=[]
        rs_lh=[]
        selfE=[]
        selfE_lh=[]
        
        
        print("starting dispersion ..........")
        qp=np.arange(np.size(fillings))
        s=time.time()
        omega=[1e-14]
        kpath=np.array([self.KX1bz,self.KY1bz]).T
        calc = functools.partial(self.parCompute, theta, omega, kpath, VV, Nsamp, mu_values/1000,fillings)
        maxthreads=12

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_pl = executor.map(calc, qp, chunksize=int(np.size(qp)/maxthreads))

            for result in results_pl:
                selfE.append(result[0])
                cs.append(result[2])
                rs.append(result[1])


        e=time.time()
        
        s=time.time()
        omega=[1e-14]
        kpath=np.array([self.KX1bz,self.KY1bz]).T
        calc = functools.partial(self.parCompute_lh, theta, omega, kpath, VV, Nsamp, mu_values/1000,fillings)
        maxthreads=12

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results_pl = executor.map(calc, qp, chunksize=int(np.size(qp)/maxthreads))

            for result in results_pl:
                selfE_lh.append(result[0])
                cs_lh.append(result[2])
                rs_lh.append(result[1])


        e=time.time()

        # for ide in range(np.size(fillings)):
        #     mu= mu_values[ide]/1000
        #     filling=fillings[ide]
        #     omega=[0]
        #     kpath=np.array([self.KX,self.KY]).T
        #     integ=self.Compute(mu, omega, kpath)
        #     # integ_lh=self.Compute_lh(mu, omega, kpath)

        #     # plt.plot((abs(integ-integ_lh).flatten())/np.mean(abs(integ_lh).flatten()))
        #     # plt.savefig("comparison_of_integrands_"+self.name+"_filling_"+str(filling)+".png")
        #     # plt.close()

        #     print("the filling is .. " , filling)
        #     integ=integ.flatten()*1000 #convertion to mev
        #     popt, res, c, resc=self.extract_cs( integ, prop_BZ)
        #     print("parameters of the fit...", c)
        #     print("residual of the fit...", res)
        #     print("original coeff...", c_phonon)
        #     self.plot_res( integ, self.KX,self.KY, VV, filling, Nsamp,c, res, "eps")
        #     cs.append(c)
        #     rs.append(resc)

        #     # integ_lh=integ_lh.flatten()*1000 #convertion to mev
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
        # plt.close()
        plt.show()

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
        # plt.close()
        plt.show()

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
        plt.close()
        # plt.show()

        with open("velocities_V_filling_"+self.name+"_theta_"+str(theta)+".npy", 'wb') as f:
                np.save(f, cep)
        with open("velocities_res_V_filling_"+self.name+"_theta_"+str(theta)+".npy", 'wb') as f:
                np.save(f, rep)

        
class ee_Bubble:

    def __init__(self, latt, nbands, hpl, hmin, test, umkl, theta):

        #changes to eliominate arg KX KY put KX1bz first, then call umklapp lattice and put slef in KX, KY
        #change to implement dos 
        self.lq=latt
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        self.umkl=umkl
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice()
        self.Npoi1bz=np.size(self.KX1bz)
        [self.KX,self.KY]=latt.Generate_Umklapp_lattice2(self.KX1bz, self.KY1bz,self.umkl) #for the integration grid 
        [self.KQX,self.KQY]=latt.Generate_Umklapp_lattice2(self.KX1bz, self.KY1bz,self.umkl+1) #for the momentum transfer lattice
        self.Npoi=np.size(self.KX)
        self.NpoiQ=np.size(self.KQX)
        self.latt=latt
        [q1,q2,q3]=latt.qvect()
        self.Gscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        [self.psi_plus,self.Ene_valley_plus,self.psi_min,self.Ene_valley_min]=self.precompute_E_psi()
        
        EpK=self.hpl.ExtendE(self.Ene_valley_plus , self.umkl)
        EmK=self.hmin.ExtendE(self.Ene_valley_min , self.umkl)
        # plt.scatter(self.KX1bz,self.KY1bz, c=self.Ene_valley_plus[:,0])
        # plt.colorbar()
        # plt.savefig("1bzplusdisp.png")
        # plt.close()
        # plt.scatter(self.KX,self.KY, c=EpK[:,0], s=1)
        # plt.colorbar()
        # plt.savefig("plusdisp.png")
        # plt.close()
        # plt.scatter(self.KX,self.KY, c=EmK[:,0], s=1)
        # plt.colorbar()
        # plt.savefig("mindisp.png")
        # plt.close()
        
        self.eta=np.mean( np.abs( np.diff( self.Ene_valley_plus[:,int(nbands/2)].flatten() )  ) )/2
        self.FFp=Hamiltonian.FormFactors_umklapp(self.psi_plus, 1, latt, self.umkl+1,self.hpl)
        self.FFm=Hamiltonian.FormFactors_umklapp(self.psi_min, -1, latt, self.umkl+1,self.hmin)
        self.name="ee"
        self.L00p=self.FFp.denqFF_s()
        self.L00m=self.FFm.denqFF_s()
        
        

        self.dS_in=latt.VolMBZ/self.Npoi


        
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
                # undet=np.abs(np.linalg.det(self.Lnemp[k,:,kp,:]))
                # dosdet=np.abs(np.linalg.det(self.Lnemp[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                undet=np.abs(np.linalg.det(self.L00p[k,:,kp,:]))
                dosdet=np.abs(np.linalg.det(self.L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                cos1.append(undet)
                cos2.append(dosdet)
                diffarp.append( undet   - dosdet   )
                # Minus Valley FF Omega
                # undet=np.abs(np.linalg.det(self.Lnemm[k,:,kp,:]))
                # dosdet=np.abs(np.linalg.det(self.Lnemm[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                undet=np.abs(np.linalg.det(self.L00m[k,:,kp,:]))
                dosdet=np.abs(np.linalg.det(self.L00m[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
                diffarm.append( undet   - dosdet   )
                

            plt.plot(diffarp, label="plus valley")
            plt.plot(diffarm, label="minsu valley")
            plt.title("FF- C3 FF")
            plt.legend()
            plt.savefig("TestC3_symm"+self.name+".png")
            # plt.show()
            plt.close()

            plt.scatter(K,KP,c=cos1)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("cos1scat.png")
            plt.close()

            plt.scatter(K,KP,c=cos2)
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("cos2scat.png")
            plt.close()

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(K,KP,cos1, c=cos1);
            plt.savefig("cos1.png")
            plt.close()

            print("finished testing symmetry of the form factors...")


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
    
    

   

    def integrand_ZT(self,nkq,nk,ekn,ekm,w,mu):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1
        eps=0.1*self.eta ##SENSITIVE TO DISPERSION

        fac_p=(nfkq-nfk)/(w-(edkq-edk)+1j*eps)
        # fac_p=(nfkq-nfk)/(-(edkq-edk))
        return (fac_p)

    def deltad(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)

    def integrand_ZT_lh(self,nkq,nk,ekn,ekm,w,mu):
        eps=0.1*self.eta ##SENSITIVE TO DISPERSION
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1
        deltad_cut=1e-17*np.heaviside(eps-np.abs(edkq-edk), 1.0)
        fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-eps, 0.0)/(deltad_cut-(edkq-edk))
        fac_p2=0.0

        # fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-eps, 0.0)/(deltad_cut-(edkq-edk))
        fac_p2=(self.deltad( edk, 10*self.eta))*np.heaviside(eps-np.abs(edkq-edk), 0.0)

        return (fac_p+fac_p2)


    def integrand_T(self,nkq,nk,ekn,ekm,w,mu,T):
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #finite temp
        nfk= self.nf(edk,T)
        nfkq= self.nf(edkq,T)

        eps=self.eta ##SENSITIVE TO DISPERSION

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
    ###########No FORM FACOTR
    def Compute_lh_noff(self, mu, omegas, kpath):

        integ=[]
        sb=time.time()

        print("starting bubble.......",np.shape(kpath)[0])

        
        bub=0
            
        qx=0.2#kpath[int(l), 0]
        qy=0.2#kpath[int(l), 1]
        Ikq=[]
        for s in range(self.Npoi):
            kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
            indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
            Ikq.append(indmin)
    
        integrand_var=0
        #####all bands for the + and - valley
        for nband in range(self.nbands):
            for mband in range(self.nbands):
                
                ek_n=self.Ene_valley_plus[:,nband]
                ek_m=self.Ene_valley_plus[:,mband]
                # Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                Lambda_Tens_plus_kq_k_nm=int(nband==mband)  #TO SdWITCH OFF THE FORM FACTORS
                integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT_lh(Ikq,self.Ik,ek_n,ek_m,0,mu)
                ek_n=self.Ene_valley_min[:,nband]
                ek_m=self.Ene_valley_min[:,mband]
                # Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                Lambda_Tens_min_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT_lh(Ikq,self.Ik,ek_n,ek_m,0,mu)
                
        eb=time.time()
    
        bub=bub+np.sum(integrand_var)*self.dS_in
        print("time for bubble...",eb-sb)
        return bub
    
    def Compute_noff(self, mu, omegas, kpath):

        integ=[]
        sb=time.time()

        print("starting bubble.......",np.shape(kpath)[0])

        
        bub=0
            
        qx=0.2#kpath[int(l), 0]
        qy=0.2#kpath[int(l), 1]
        Ikq=[]
        for s in range(self.Npoi):
            kxq,kyq=self.KX[s]+qx,self.KY[s]+qy
            indmin=np.argmin(np.sqrt((self.KQX-kxq)**2+(self.KQY-kyq)**2))
            Ikq.append(indmin)
    
        integrand_var=0
        #####all bands for the + and - valley
        for nband in range(self.nbands):
            for mband in range(self.nbands):
                
                ek_n=self.Ene_valley_plus[:,nband]
                ek_m=self.Ene_valley_plus[:,mband]
                # Lambda_Tens_plus_kq_k_nm=Lambda_Tens_plus_kq_k[:,nband,mband]
                Lambda_Tens_plus_kq_k_nm=int(nband==mband)  #TO SdWITCH OFF THE FORM FACTORS
                integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_plus_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,0,mu)
                ek_n=self.Ene_valley_min[:,nband]
                ek_m=self.Ene_valley_min[:,mband]
                # Lambda_Tens_min_kq_k_nm=Lambda_Tens_min_kq_k[:,nband,mband]
                Lambda_Tens_min_kq_k_nm=int(nband==mband)  #TO SWITCH OFF THE FORM FACTORS
                integrand_var=integrand_var+np.abs(np.abs( Lambda_Tens_min_kq_k_nm )**2)*self.integrand_ZT(Ikq,self.Ik,ek_n,ek_m,0,mu)
                
        eb=time.time()
    
        bub=bub+np.sum(integrand_var)*self.dS_in
        print("time for bubble...",eb-sb)
        return bub

    
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



    def epsilon_sweep(self,fillings, mu_values):
        
        
        scalings=[0.2,0.3,0.4]
        for e in scalings:
            cs=[]
            cs_lh=[]
            eddy=self.eta
            self.eta=e*eddy
            bins2=np.linspace(self.bins[0],self.bins[-1],500)
            for ide in range(np.size(bins2)):
                # mu= mu_values[ide]/1000
                # filling=fillings[ide]
                omega=[0]
                kpath=np.array([0,0]).T
                integ=self.Compute_noff(bins2[ide], omega, kpath)
                integ_lh=self.Compute_lh_noff(bins2[ide], omega, kpath)
                cs.append(integ/2)
                cs_lh.append(integ_lh/2)
            self.eta=eddy
            print(e)

            print(cs)
            print(cs_lh)

            plt.plot(bins2,cs)
            plt.plot(bins2,cs_lh)
            plt.plot(self.bins, self.valt)
            plt.show()
        

            


    
        
def main() -> int:

    #parameters for the calculation
    theta= 1.04*1.05*np.pi/180  # magic angle
    fillings = np.array([0.0,0.1341,0.2682,0.4201,0.5720,0.6808,0.7897,0.8994,1.0092,1.1217,1.2341,1.3616,1.4890,1.7107,1.9324,2.0786,2.2248,2.4558,2.6868,2.8436,3.0004,3.1202,3.2400,3.3720,3.5039,3.6269,3.7498])
    mu_values = np.array([0.0,0.0625,0.1000,0.1266,0.1429,0.1508,0.1587,0.1666,0.1746,0.1843,0.1945,0.2075,0.2222,0.2524,0.2890,0.3171,0.3492,0.4089,0.4830,0.5454,0.6190,0.6860,0.7619,0.8664,1.0000,1.1642,1.4127])

    
    try:
        filling_index=int(sys.argv[1]) #0-25

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose chemical potential for desired filling")

    try:
        N_SFs=26 #number of SF's currently implemented
        a=np.arange(N_SFs)
        a[filling_index]

    except (IndexError):
        raise Exception(f"Index has to be between 0 and {N_SFs-1}")


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
    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
    ln=MoireLattice.MoireTriangLattice(Nsamp,theta,1)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one
    [KX,KY]=lq.Generate_lattice()
    Npoi=np.size(KX); print(Npoi, "numer of sampling lattice points")
    [q1,q2,q3]=l.q
    q=la.norm(q1)
    umkl=3
    print(f"taking ${umkl} umklapps")
    VV=lq.boundary()
    # [KXu,KYu]=lq.Generate_Umklapp_lattice(KX,KY,umkl)
    # [KQX, KQY, Ik]=lq.Generate_momentum_transfer_umklapp_lattice( KX, KY,  KXu, KYu)
    

    #kosh params realistic  -- this is the closest to the actual Band Struct used in the paper
    hbvf = 2.1354; # eV
    hvkd=hbvf*q
    kappa_p=0.0797/0.0975
    kappa=kappa_p*modulation
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

    #other electronic params
    filling_index=int(sys.argv[1]) 
    mu=mu_values[filling_index]/1000
    filling=fillings[filling_index]
    nbands=2
    hbarc=0.1973269804*1e-6 #ev*m
    alpha=137.0359895 #fine structure constant
    a_graphene=2.46*(1e-10) #in meters
    ee2=(hbarc/a_graphene)/alpha
    kappa_di=3.03

    #phonon parameters
    c_light=299792458 #m/s
    M=1.99264687992e-26 * 5.6095861672249e+38/1000 # [in units of eV]
    mass=M/(c_light**2) # in ev *s^2/m^2
    hhbar=6.582119569e-13 /1000 #(in eV s)
    alpha_ep=0*2*1# in ev
    beta_ep=4 #in ev
    c_phonon=21400 #m/s
    gamma=np.sqrt(hhbar*q/(a_graphene*mass*c_phonon))
    gammap=(q*q*gamma**2/a_graphene**2)/(4*np.pi*np.pi)
    print("phonon params...", gammap )
    mode_layer_symmetry="a" #whether we are looking at the symmetric or the antisymmetric mode
    cons=[alpha_ep, beta_ep, gammap, a_graphene, mass] #constants used in the bubble calculation and data anlysis

    print("kappa is..", kappa)
    print("alpha is..", alph)


    # [path,kpath,HSP_index]=lq.embedded_High_symmetry_path(KX,KY)
    # plt.plot(VV[:,0],VV[:,1])
    # plt.scatter(kpath[:,0],kpath[:,1], s=30, c='g' )
    # plt.gca().set_aspect('equal')
    # plt.show()
    
    hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq, kappa, PH)
    hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq, kappa, PH)
    
    # test_symmetry=True
    # B1=ee_Bubble_2(lq, nbands, hpl, hmin, test_symmetry, umkl, theta)
    # omega=[1e-14]
    # kpath=np.array([KX,KY]).T
    # integ=B1.Compute(mu, omega, kpath)
    # B1.plot_res( integ, KX,KY, VV, filling, Nsamp, "FG")
    # integ=B1.Compute_lh(mu, omega, kpath)
    # B1.plot_res( integ, KX,KY, VV, filling, Nsamp, "FGlh")
    # B1.epsilon_sweep(fillings, mu_values)
    
    #testing umklapp
    
    test_symmetry=True
    B1=ee_Bubble(lq, nbands, hpl, hmin, test_symmetry, umkl, theta)
    # omega=[1e-14]
    # kpath=np.array([KX,KY]).T
    # integ=B1.Compute(mu, omega, kpath)
    # B1.plot_res( integ, KX,KY, VV, filling, Nsamp, "FG")
    # integ=B1.Compute_lh(mu, omega, kpath)
    # B1.plot_res( integ, KX,KY, VV, filling, Nsamp, "FGlh")
    # B1.epsilon_sweep(fillings, mu_values)
    

    # test_symmetry=True
    # B1=ep_Bubble(lq, nbands, hpl, hmin,  mode_layer_symmetry, mode, cons, test_symmetry, umkl)
    # omega=[1e-14]
    # kpath=np.array([KX,KY]).T
    # integ=B1.Compute_lh(mu, omega, kpath)
    # popt, res, c, resc=B1.extract_cs( integ, 1)
    # B1.plot_res(integ, KX,KY, VV, filling, Nsamp, c , res, "")
    # print(np.mean(popt),np.mean(c), resc, c_phonon)
    # B1.Fill_sweep(fillings, mu_values, VV, Nsamp, c_phonon,theta)
    

    
    # return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
