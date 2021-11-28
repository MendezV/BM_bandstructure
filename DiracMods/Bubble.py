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
        self.Ik=latt.insertion_index( self.KX,self.KY, self.KQX, self.KQY)
        self.Npoi=np.size(self.KX)
        self.NpoiQ=np.size(self.KQX)
        self.latt=latt
        [q1,q2,q3]=latt.qvect()
        self.qscale=la.norm(q1) #necessary for rescaling since we where working with a normalized lattice 
        [self.psi_plus,self.Ene_valley_plus_1bz,self.psi_min,self.Ene_valley_min_1bz]=self.precompute_E_psi()
        
        self.Ene_valley_plus=self.hpl.ExtendE(self.Ene_valley_plus_1bz , self.umkl+1)
        self.Ene_valley_min=self.hmin.ExtendE(self.Ene_valley_min_1bz , self.umkl+1)
        
        ###selecting eta
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
        self.FFp=Hamiltonian.FormFactors_umklapp(self.psi_plus, 1, latt, self.umkl+1,self.hpl)
        self.FFm=Hamiltonian.FormFactors_umklapp(self.psi_min, -1, latt, self.umkl+1,self.hmin)
        
        self.name="ee"
        self.L00p=self.FFp.denqFF_s()
        self.L00m=self.FFm.denqFF_s()
        
        

        self.dS_in=latt.VolMBZ/self.Npoi1bz


        
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
            
            ###DOS
            Ndos=latt.Npoints
            ldos=Lattice.TriangLattice(Ndos,theta, 2) #this one
            [self.KXdos, self.KYdos]=ldos.Generate_lattice()
            self.Npoidos=np.size(self.KXdos)
            [self.Ene_valley_plus_dos,self.Ene_valley_min_dos]=self.precompute_E_psi_dos()
            # with open('dispersions/Edisp_'+str(Ndos)+'.npy', 'wb') as f:
            #     np.save(f, self.Ene_valley_plus_dos)
            # with open('dispersions/Edism_'+str(Ndos)+'.npy', 'wb') as f:
            #     np.save(f, self.Ene_valley_min_dos)
            # print("Loading  ..........")
            
            # with open('dispersions/Edisp_'+str(Ndos)+'.npy', 'rb') as f:
            #     self.Ene_valley_plus_dos=np.load(f)
            # with open('dispersions/Edism_'+str(Ndos)+'.npy', 'rb') as f:
            #     self.Ene_valley_min_dos=np.load(f)
                
            plt.scatter(self.KXdos, self.KYdos, c=self.Ene_valley_plus_dos[:,0])
            plt.colorbar()
            plt.savefig("energydisp0.png")
            plt.close()
            plt.scatter(self.KXdos, self.KYdos, c=self.Ene_valley_plus_dos[:,1])
            plt.colorbar()
            plt.savefig("energydisp1.png")
            plt.close()
            plt.scatter(self.KXdos, self.KYdos, c=self.Ene_valley_min_dos[:,0])
            plt.colorbar()
            plt.savefig("energydism0.png")
            plt.close()
            plt.scatter(self.KXdos, self.KYdos, c=self.Ene_valley_min_dos[:,1])
            plt.colorbar()
            plt.savefig("energydism1.png")
            plt.close()

            [self.bins,self.valt, self.f2 ]=hpl.DOS(self.Ene_valley_plus_dos,self.Ene_valley_min_dos)
            [self.earr,self.dosarr,self.f2 ]=hpl.DOS2(self.Ene_valley_plus_dos,self.Ene_valley_min_dos,1/np.size(self.KXdos))

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
    
    def precompute_E_psi_dos(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))


        print("starting dispersion for DOS..........")
        
        s=time.time()
        
        for l in range(self.Npoidos):
            E1,wave1=self.hpl.eigens(self.KXdos[l],self.KYdos[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)


            E1,wave1=self.hmin.eigens(self.KXdos[l],self.KYdos[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)

            # printProgressBar(l + 1, self.Npoi, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.Npoidos,self.nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.Npoidos,self.nbands])
        
        

        return [Ene_valley_plus,Ene_valley_min]
    
    

   

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
    
    def test_deltad(self):
        dels=[0.001,0.01,0.1,1,2,5,10]
        ees=np.linspace(-10,10,int(20/self.eta))
       
        for de in dels:
            plt.plot(ees,self.deltad(ees,self.eta*de) )
            plt.savefig(str(de)+".png")
            print("testing delta...", de,np.trapz(self.deltad(ees,self.eta*de))*self.eta)

    def integrand_ZT_lh(self,nkq,nk,ekn,ekm,w,mu):
        
        edkq=ekn[nkq]-mu
        edk=ekm[nk]-mu

        #zero temp
        nfk=np.heaviside(-edk,0.5) # at zero its 1
        nfkq=np.heaviside(-edkq,0.5) #at zero is 1
        deltad_cut=1e-17*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-self.eta_cutoff, 0.0)/(deltad_cut-(edkq-edk))
        fac_p2=(self.deltad( edk, self.eta_dirac_delta))*np.heaviside(self.eta_cutoff-np.abs(edkq-edk), 1.0)
        
        # for i in range(np.size(edkq)):
        #     if np.abs((edkq-edk)[i])<self.eta_cutoff:
        #         print("1",self.KQX[nkq[i]],self.KQY[nkq[i]], ekn[nkq[i]]-mu)
        #         print("2",self.KQX[nk[i]],self.KQY[nk[i]],ekm[nk[i]]-mu)
        #         print("3",deltad_cut[i],  (edkq-edk)[i],fac_p[i],fac_p2[i] )
            
        # # fac_p=(nfkq-nfk)*np.heaviside(np.abs(edkq-edk)-eps, 0.0)/(deltad_cut-(edkq-edk))
        # print(eps)

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
        
        self.test_deltad()

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
                # print(qx,qy)

                Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)

            
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
                
                Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)

            
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

        # print("starting bubble.......",np.shape(kpath)[0])

        
        bub=0
            
        qx=0.15#kpath[int(l), 0]
        qy=0.15#kpath[int(l), 1]

        Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)
    
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
        # print("time for bubble...",eb-sb)
        return bub
    
    def Compute_noff(self, mu, omegas, kpath):

        integ=[]
        sb=time.time()

        # print("starting bubble.......",np.shape(kpath)[0])

        
        bub=0
            
        qx=0.15#kpath[int(l), 0]
        qy=0.15#kpath[int(l), 1]
        Ikq=self.latt.insertion_index( self.KX+qx,self.KY+qy, self.KQX, self.KQY)
    
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
        # print("time for bubble...",eb-sb)
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

            bins2=np.linspace(self.bins[0],self.bins[-1],NN)
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
            plt.plot(self.bins, self.valt, label="hist")
            plt.plot(self.earr,self.dosarr, label="integ")
            plt.legend()
            plt.savefig("comparedos"+str(e)+".png")
            plt.close()
            plt.title("scaling fact cutoff"+str(e))
            plt.plot(bins2,cs, label="ieps")
            plt.plot(bins2,cs_lh, label="derivative")
            plt.plot(self.bins, self.valt, label="hist")
            plt.legend()
            plt.title("scaling fact cutoff "+str(e))
            print("normalizations ", e , np.sum(cs)*(bins2[1]-bins2[0]),np.sum(cs_lh)*(bins2[1]-bins2[0]))
            plt.savefig("scaling"+str(e)+".png")
            plt.close()
            
            plt.plot(bins2,cs, label="ieps")
            plt.plot(bins2,cs_lh, label="derivative")
            plt.plot(self.earr,self.dosarr, label="integ")
            plt.title("scaling fact "+str(e))
            plt.legend()
            plt.savefig("scaling_2_"+str(e)+".png")
            plt.close()
        

            


    
        
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

    theta=modulation*1.05*np.pi/180  # magic angle
    l=Lattice.TriangLattice(Nsamp,theta,0)
    ln=Lattice.TriangLattice(Nsamp,theta,1)
    lq=Lattice.TriangLattice(Nsamp,theta,2) #this one
    [KX,KY]=lq.Generate_lattice()
    Npoi=np.size(KX); print(Npoi, "numer of sampling lattice points")
    [q1,q1,q3]=l.q
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
    gamma=np.sqrt(hhbar*q/(a_graphene*mass*c_phonon))
    gammap=(q**2)*(gamma**2)/((a_graphene**2)*((2*np.pi)**2)) 
    scaling_fac=( a_graphene**2) /(mass*(q**2))
    print("phonon params...", gammap/1e+11 , gamma, np.sqrt(gammap*scaling_fac),gammap*scaling_fac)
    mode_layer_symmetry="a" #whether we are looking at the symmetric or the antisymmetric mode
    cons=[alpha_ep, beta_ep, gammap, a_graphene, mass] #constants used in the bubble calculation and data anlysis


    hpl=Hamiltonian.Ham_BM_p(hvkd, alph, 1, lq, kappa, PH)
    hmin=Hamiltonian.Ham_BM_m(hvkd, alph, -1, lq, kappa, PH)
    
    #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
    Ndos=100
    ldos=Lattice.TriangLattice(Ndos,theta,2)
    [ Kxp, Kyp]=ldos.Generate_lattice()
    disp=Hamiltonian.Dispersion( ldos, nbands, hpl, hmin)
    Nfils=3
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
    B1.plot_res(integ, KX,KY, VV, filling, Nsamp, c , res, "")
    print(np.mean(popt),np.mean(c), resc, c_phonon)
    # B1.Fill_sweep(fillings, mu_values, VV, Nsamp, c_phonon,theta)
    
    NnarrowSamp=100
    lnarrowSamp=Lattice.TriangLattice(NnarrowSamp,theta,2)
    [ Kxp, Kyp]=lnarrowSamp.Generate_lattice()
    
    [KX_m, KY_m, ind]=lnarrowSamp.mask_KPs( Kxp, Kyp, 0.2)
    kpath=np.array([KX_m, KY_m]).T
    integ=B1.Compute_lh(mu, omega, kpath)
    popt, res, c, resc=B1.extract_cs_path( integ, kpath)
    print(np.mean(popt),np.mean(c), resc, c_phonon)
    B1.plot_res(integ, KX_m, KY_m, VV, filling, Nsamp, c , res, "small")
    

    
    # return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
