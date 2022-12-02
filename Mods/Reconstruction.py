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
     
class Phon_bare_BandStruc:
    
    
    def __init__(self, latt,  hpl, hmin, nbands,  cons, field, qins_X, qins_Y, sym, mode, layersym ):

        
        self.latt=latt
        
        
        self.nbands_init=4*hpl.Dim
        self.nbands=nbands

        
        self.ini_band=int(self.nbands_init/2)-int(self.nbands/2)
        self.fini_band=int(self.nbands_init/2)+int(self.nbands/2)
        
        
        self.hpl=hpl
        self.hmin=hmin
        
        self.field=field
        self.sym=sym     #whether we take all c3 invariant mome
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
        else:
            self.qins_X=[qins_X,-qins_X]
            self.qins_Y=[qins_Y,-qins_Y]
            self.nqs=2

        [self.alpha_ep, self.beta_ep,  self.Wupsilon, self.agraph, self.mass ]=cons #constants for the exraction of the effective velocity

        
        #TODO: check that the form factors are working properly when I increase the number of bands
    
        ################################
        #dispersion attributes
        ################################

        disp=Dispersion( latt, self.nbands_init, hpl, hmin)
        [self.psi_plus_1bz,self.Ene_valley_plus_1bz,self.psi_min_1bz,self.Ene_valley_min_1bz]=disp.precompute_E_psi()

        self.Ene_valley_plus=self.hpl.ExtendE(self.Ene_valley_plus_1bz[:,self.ini_band:self.fini_band] , self.latt.umkl_Q)
        self.Ene_valley_min=self.hmin.ExtendE(self.Ene_valley_min_1bz[:,self.ini_band:self.fini_band] , self.latt.umkl_Q)

        self.psi_plus=self.hpl.ExtendPsi(self.psi_plus_1bz, self.latt.umkl_Q)
        self.psi_min=self.hmin.ExtendPsi(self.psi_min_1bz, self.latt.umkl_Q)
        
        
        ################################
        #generating form factors
        ################################
        self.FFp=FormFactors(self.psi_plus[:,:,self.ini_band:self.fini_band], 1, latt, -1,self.hpl)
        self.FFm=FormFactors(self.psi_min[:,:,self.ini_band:self.fini_band], -1, latt, -1,self.hmin)
        
        
        
        # s=time.time()
        # EHFp=[]
        # U_transf=[]
        # EHFm=[]
        # U_transfm=[]
        # [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denFF_s(), self.FFm.denFF_s())
        # Ik_pre=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
        # for k in Ik_pre:
        #     hpl=0+0*1j
        #     hmin=0+0*1j
        #     for q in range(self.nqs):
        #         k1=np.argmin( np.sqrt( (self.latt.KQX-(self.latt.KQX[k]+self.qins_X[q]))**2 + (self.latt.KQY-(self.latt.KQY[k]+self.qins_Y[q]))**2 ) )
        #         first_check=np.sqrt( (self.latt.KQX[k1]-(self.latt.KQX[k]+self.qins_X[q]))**2 +(self.latt.KQY[k1]-(self.latt.KQY[k]+self.qins_Y[q]))**2  )
                
        #         k2=np.argmin( np.sqrt( (self.latt.KQX-(-self.latt.KQX[k]))**2 + (self.latt.KQY-(-self.latt.KQY[k]))**2  ) )
        #         second_check=np.sqrt( (self.latt.KQX[k2]-(-self.latt.KQX[k] ))**2 +(self.latt.KQY[k2]-(-self.latt.KQY[k] ))**2)
                
        #         k3=np.argmin( np.sqrt( (self.latt.KQX-(-self.latt.KQX[k]-self.qins_X[q]))**2 + (self.latt.KQY-(-self.latt.KQY[k]-self.qins_Y[q]))**2 ) )
        #         third_check=np.sqrt( (self.latt.KQX[k1]-(-self.latt.KQX[k]-self.qins_X[q]))**2 +(self.latt.KQY[k1]-(-self.latt.KQY[k]-self.qins_Y[q]))**2  )
                
                
        #         if first_check<0.5/self.latt.NpoiQ:
                    
        #             hpl=hpl+self.Omega_FFp[k1,:,k,:]
        #             hmin=hmin+self.Omega_FFm[k1,:,k,:]
                    
        #             print('\n')
        #             print(self.qins_X[q],self.qins_Y[q],self.latt.KQX[k],self.latt.KQY[k])
        #             print('check sum',self.latt.KQX[k1],self.latt.KQY[k1],self.latt.KQX[k]+self.qins_X[q],self.latt.KQY[k]+self.qins_Y[q])
        #             print('minus sum', self.latt.KQX[k3],self.latt.KQY[k3])
        #             print('minus vec og',self.latt.KQX[k2],self.latt.KQY[k2])
                    
        #             print('p: ',self.Omega_FFp[k1,:,k,:])
        #             print('\n')
        #             print('pm: ',self.Omega_FFp[k3,:,k2,:])
        #             (Eigvals,Eigvect)= np.linalg.eig(self.Omega_FFp[k1,:,k,:])  #returns sorted eigenvalues
        #             (Eigvals2,Eigvect2)= np.linalg.eig(self.Omega_FFp[k3,:,k2,:])  #returns sorted eigenvalues
        #             print('eigs.....',Eigvals,Eigvals2)
        #             print('\n')
        #             print('\n')
        #             print('m: ',self.Omega_FFm[k1,:,k,:])
        #             print('\n')
        #             print('mm: ',self.Omega_FFm[k3,:,k2,:])
        #             print('\n')
        #             (Eigvals,Eigvect)= np.linalg.eig(self.Omega_FFm[k1,:,k,:])  #returns sorted eigenvalues
        #             (Eigvals2,Eigvect2)= np.linalg.eig(self.Omega_FFm[k3,:,k2,:])  #returns sorted eigenvalues
        #             print('eigs.....',Eigvals,Eigvals2)
            
        #     print('\n')
        #     print('\n')
                
        #     (Eigvals,Eigvect)= np.linalg.eig(hpl)  #returns sorted eigenvalues
        #     print('total eigs p:',Eigvals)  
        #     EHFp.append(Eigvals)
        #     U_transf.append(Eigvect)
            
        #     (Eigvals,Eigvect)= np.linalg.eig(hmin)  #returns sorted eigenvalues
        #     EHFm.append(Eigvals)
        #     U_transfm.append(Eigvect)
        #     print('total eigs m:',Eigvals) 
        
        ################################
        #generating form factors
        ################################
       
        if layersym=="s":
            if mode=="L":
                self.L00p=self.FFp.denqFFL_s()
                self.L00m=self.FFm.denqFFL_s()
                self.Lnemp=self.FFp.NemqFFL_s()
                self.Lnemm=self.FFm.NemqFFL_s()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()
            elif mode=="T": 
                self.Lnemp=self.FFp.NemqFFT_s()
                self.Lnemm=self.FFm.NemqFFT_s()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()
            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denFF_s(), self.FFm.denFF_s())
                
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denFF_s(), self.FFm.denFF_s())
            
        else: # a- mode
            if mode=="L":
                self.L00p=self.FFp.denqFFL_a()
                self.L00m=self.FFm.denqFFL_a()
                self.Lnemp=self.FFp.NemqFFL_a()
                self.Lnemm=self.FFm.NemqFFL_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()
                
                ##Plotting form factors
                self.FFm.plotFF(self.Omega_FFm, mode+'_Om')
                self.FFp.plotFF(self.Omega_FFp, mode+'_Op')
                
                   
            elif mode=="T":
                self.Lnemp=self.FFp.NemqFFT_a()
                self.Lnemm=self.FFm.NemqFFT_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()
                
                ##Plotting form factors
                self.FFm.plotFF(self.Omega_FFm, mode+'_Om')
                self.FFp.plotFF(self.Omega_FFp, mode+'_Op')
                

            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denFF_s(), self.FFm.denFF_s())
                
                ##Plotting form factors
                self.FFm.plotFF(self.Omega_FFm, mode+'_Om')
                self.FFp.plotFF(self.Omega_FFp, mode+'_Op')
                
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denqFF_a(), self.FFm.denqFF_a())

        
        
        e=self.FFp.plotMatEig(self.Omega_FFp)
        # e=self.FFp.plotMatEig(self.Omega_FFm)
        
        # [mat_s,matp, matm]=self.make_mat_phon()
        # [mat_s_eps,matp2_eps, matm_eps]=self.make_mat_phon()
        # Hp=matp2_eps+matp+np.conj(matp.T)
        # Hm=matm_eps+matm+np.conj(matm.T)
        
        # (self.Eigvals_p,self.Eigvect_p)= np.linalg.eigh(Hp) 
        # (self.Eigvals_m,self.Eigvect_m)= np.linalg.eigh(Hm) 
        

        # print(f"starting dispersion with {self.latt.Npoi} points..........")
        
        # s=time.time()
        # EHFp=[]
        # U_transf=[]
        # EHFm=[]
        # U_transfm=[]
        # paulix=np.array([[0,1],[1,0]])
        
        # #if we use TRS to reconstruct minus valley
        # diagI = np.zeros([int(self.nbands/2),int(self.nbands/2)]);
        # for ib in range(int(self.nbands/2)):
        #     diagI[ib,int(self.nbands/2-1-ib)]=1;
        # sx=np.kron(paulix,diagI)
        
        # ##only works for instabilities that occur at repetitions of Gamma
        # Ik_pre=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
        # plt.scatter(self.latt.KQX,self.latt.KQY)
        # plt.scatter(self.latt.KX1bz,self.latt.KY1bz)
        # for k in Ik_pre:
            
        #     H0p=np.diag(self.Ene_valley_plus[k,:])
        #     Delt_p=np.zeros([self.nbands,self.nbands])+1j*1e-20
        #     Delt_m=np.zeros([self.nbands,self.nbands])+1j*1e-20
        #     c=0
        #     # plt.scatter(self.latt.KQX,self.latt.KQY)
        #     for q in range(self.nqs):
        #         k1=np.argmin( np.sqrt( (self.latt.KQX-(self.latt.KQX[k]+self.qins_X[q]))**2 + (self.latt.KQY-(self.latt.KQY[k]+self.qins_Y[q]))**2 ) )
        #         first_check=np.sqrt( (self.latt.KQX[k1]-(self.latt.KQX[k]+self.qins_X[q]))**2 +(self.latt.KQY[k1]-(self.latt.KQY[k]+self.qins_Y[q]))**2)
            
                
        #         # plt.scatter( self.latt.KQX[k] + self.qins_X[q]  ,self.latt.KQY[k]+self.qins_Y[q], c='b')
        #         # plt.scatter( self.latt.KQX[k1]  ,self.latt.KQY[k1], c='r')  
                
                
                
                
        #         k2=np.argmin( np.sqrt( (self.latt.KQX-(-self.latt.KQX[k]))**2 + (self.latt.KQY-(-self.latt.KQY[k]))**2  ) )
        #         second_check=np.sqrt( (self.latt.KQX[k2]-(-self.latt.KQX[k] ))**2 +(self.latt.KQY[k2]-(-self.latt.KQY[k] ))**2)
                
        #         k3=np.argmin( np.sqrt( (self.latt.KQX-(-self.latt.KQX[k]-self.qins_X[q]))**2 + (self.latt.KQY-(-self.latt.KQY[k]-self.qins_Y[q]))**2 ) )
        #         third_check=np.sqrt( (self.latt.KQX[k1]-(-self.latt.KQX[k]-self.qins_X[q]))**2 +(self.latt.KQY[k1]-(-self.latt.KQY[k]-self.qins_Y[q]))**2  )
                
                
        #         print('p: ',self.Omega_FFp[k1,:,k,:])
        #         print('\n')
        #         print('pm: ',self.Omega_FFp[k3,:,k2,:])
        #         (Eigvals,Eigvect)= np.linalg.eig(self.Omega_FFp[k1,:,k,:])  #returns sorted eigenvalues
        #         (Eigvals2,Eigvect2)= np.linalg.eig(self.Omega_FFp[k3,:,k2,:])  #returns sorted eigenvalues
        #         print('eigs.....',Eigvals,Eigvals2)
        #         print('\n')
        #         print('\n')
        #         print('m: ',self.Omega_FFm[k1,:,k,:])
        #         print('\n')
        #         print('mm: ',self.Omega_FFm[k3,:,k2,:])
        #         print('\n')            
        #         print('m: ',self.Omega_FFm[k1,:,k,:])
        #         print('\n')
        #         (Eigvals,Eigvect)= np.linalg.eig(self.Omega_FFm[k1,:,k,:])  #returns sorted eigenvalues
        #         (Eigvals2,Eigvect2)= np.linalg.eig(self.Omega_FFm[k3,:,k2,:])  #returns sorted eigenvalues
        #         print('eigs.....',Eigvals,Eigvals2)
                
        #         if first_check<0.5/self.latt.NpoiQ:
        #             c=c+1
        #             Delt_p=Delt_p+self.Omega_FFp[k1,:,k,:]*self.field
        #             Delt_m=Delt_m+self.Omega_FFm[k1,:,k,:]*self.field
                    
        #     # plt.scatter( self.latt.KQX[k]  ,self.latt.KQY[k]) 
        #     # plt.show()
        #     if c<self.nqs:
        #         print('not all points plus')
                    
            
        #     Hpl=H0p*0+Delt_p
        #     (Eigvals,Eigvect)= np.linalg.eigh(Hpl)  #returns sorted eigenvalues

        #     EHFp.append(Eigvals)
        #     U_transf.append(Eigvect)
            
        #     H0min=np.diag(self.Ene_valley_min[k,:])
            
        #     Hmin=H0min*0+Delt_m
        #     # print('\n')
        #     # print(self.latt.KQX[k],self.latt.KQY[k])
        #     # print('\n')
        #     # print('rel_1',np.abs(Delt_p+sx@(Delt_m@sx)))
        #     # print('\n')
        #     # print('rel_2',np.abs(Delt_p-sx@(Delt_m@sx)))
        #     # print('\n')
            
        #     if np.mean(np.abs(Hpl+sx@(Hmin@sx)))>1e-8:
        #         plt.scatter(self.latt.KQX[k],self.latt.KQY[k], c='r')
        #         print('\n')
        #         print('rel_1',np.mean(np.abs(Delt_p+sx@(Delt_m@sx))))
        #         print('rel_2',np.mean(np.abs(Delt_p-sx@(Delt_m@sx))))
        #         print('\n')
            
        #     (Eigvals,Eigvect)= np.linalg.eigh(Hmin)  #returns sorted eigenvalues
        #     EHFm.append(Eigvals)
        #     U_transfm.append(Eigvect)
         
        # plt.show()
      
                
        # Ik=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX[Ik_pre],self.latt.KQY[Ik_pre])
        # self.E_HFp=np.array(EHFp)[Ik, :]
        # self.E_HFp_K=self.hpl.ExtendE(self.E_HFp, self.latt.umkl)
        # self.E_HFp_ex=self.hpl.ExtendE(self.E_HFp, self.latt.umkl_Q)
        # self.Up=np.array(U_transf)
        
        # self.E_HFm=np.array(EHFm)[Ik, :]
        # self.E_HFm_K=self.hmin.ExtendE(self.E_HFm, self.latt.umkl)
        # self.E_HFm_ex=self.hmin.ExtendE(self.E_HFm, self.latt.umkl_Q)
        # self.Um=np.array(U_transfm)
        
        # #plots of the Bandstructre if needed
        # print(np.size(Ik),np.shape(self.E_HFp),np.shape(self.E_HFm), 'sizes of the energy arrays in HF module')
        # self.plots_bands()
        
        # e=time.time()
        # print(f'time for Diag {e-s}')

        # #plots of the Bandstructre if needed
        # # self.plots_bands()
        
    
    def plots_bands(self):
        
        for nsb in range(self.nbands):
            plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFp[:,nsb], s=40)
            plt.colorbar()
            plt.savefig("EHF"+str(nsb)+"p_kappa"+str(self.hpl.kappa)+".png")
            plt.close()
            
            
            
            plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFm[:,nsb], s=40)
            plt.colorbar()
            plt.savefig("EHF"+str(nsb)+"m_kappa"+str(self.hpl.kappa)+".png")
            plt.close()
            
        
        
        #############################
        # high symmetry path
        #############################
        [path,kpath,HSP_index]=self.latt.embedded_High_symmetry_path(self.latt.KQX,self.latt.KQY)
        pth=np.arange(np.size(path))
        
        for nsb in range(self.nbands):
            plt.plot(self.E_HFp_ex[path,nsb], c='b')
            plt.scatter(pth,self.E_HFp_ex[path,nsb], c='b', s=9)
            
        for nsb in range(self.nbands):
            plt.plot(self.E_HFm_ex[path,nsb], ls='--', c='r')
            plt.scatter(pth,self.E_HFm_ex[path,nsb], c='r', s=9)
            
        plt.savefig("disp_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        plt.plot(self.latt.KQX[path], self.latt.KQY[path])
        VV=self.latt.boundary()
        plt.scatter(self.latt.KQX[path], self.latt.KQY[path], c='r')
        plt.plot(VV[:,0], VV[:,1], c='b')
        plt.savefig("HSPp_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        print("Bandwith,", np.max(self.E_HFp_ex[:,1])-np.min(self.E_HFp_ex[:,0]))
        
        return None
    

    def OmegaL(self):
        
        Omega_FFp_pre=(self.alpha_ep*self.L00p+self.beta_ep*self.Lnemp)
        Omega_FFm_pre=(self.alpha_ep*self.L00m+self.beta_ep*self.Lnemm)
        
        [Omega_FFp,Omega_FFm]=self.Form_factor_unitary( Omega_FFp_pre, Omega_FFm_pre)
                
        return [Omega_FFp,Omega_FFm]

    def OmegaT(self):

        Omega_FFp_pre=(self.beta_ep*self.Lnemp)
        Omega_FFm_pre=(self.beta_ep*self.Lnemm)

        
        [Omega_FFp,Omega_FFm]=self.Form_factor_unitary( Omega_FFp_pre, Omega_FFm_pre)
        
        return [Omega_FFp,Omega_FFm]
    
    def Form_factor_unitary(self, FormFactor_p, FormFactor_m):

        FormFactor_new_p=FormFactor_p 
        FormFactor_new_m=FormFactor_m
        
        return [FormFactor_new_p,FormFactor_new_m]
    
    
    def make_mat_phon(self):
        
        mat_s=np.zeros([self.latt.NpoiQ,self.latt.NpoiQ])
        matp=np.zeros([self.latt.NpoiQ*self.nbands,self.latt.NpoiQ*self.nbands])
        matm=np.zeros([self.latt.NpoiQ*self.nbands,self.latt.NpoiQ*self.nbands])
        
        #could be problematic at edges of BZ
        for k in range(self.latt.NpoiQ):
            mat=np.zeros([self.latt.NpoiQ,self.latt.NpoiQ])
            k1=np.argmin( (self.latt.KQX-(self.latt.KQX[k]+self.qins_X[0]))**2 +(self.latt.KQY-(self.latt.KQY[k]+self.qins_Y[0]))**2)
            k2=np.argmin( (self.latt.KQX-self.latt.KQX[k])**2 +(self.latt.KQY-self.latt.KQY[k])**2)
            first_check=np.sqrt( (self.latt.KQX[k1]-(self.latt.KQX[k]+self.qins_X[0]))**2 +(self.latt.KQY[k1]-(self.latt.KQY[k]+self.qins_Y[0]))**2)
            
            if first_check<0.5/self.latt.NpoiQ:
                mat[k1,k2]=1
                mat_s[k1,k2]=1
                matp=matp+np.kron(mat,self.Omega_FFp[k1,:,k2,:])*self.field
                matm=matm+np.kron(mat,self.Omega_FFm[k1,:,k2,:])*self.field
            
        return [mat_s,matp, matm]

    def make_mat_eps(self):
        
        mat_s=np.zeros([self.latt.NpoiQ,self.latt.NpoiQ])
        matp=np.zeros([self.latt.NpoiQ*self.nbands,self.latt.NpoiQ*self.nbands])
        matm=np.zeros([self.latt.NpoiQ*self.nbands,self.latt.NpoiQ*self.nbands])
        
        #could be problematic at edges of BZ
        for k in range(self.latt.NpoiQ):
            mat=np.zeros([self.latt.NpoiQ,self.latt.NpoiQ])
            mat[k,k]=1
            matp=matp+np.kron(mat,np.diag(self.Ene_valley_plus[k,:]))
            matm=matm+np.kron(mat,np.diag(self.Ene_valley_min[k,:]))
            
        return [mat_s,matp, matm]
    

        

     
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
    
    
    hpl_decoupled=Dispersion.Ham_BM(hvkd, alph, 1, lq, kappa, PH,0)
    hmin_decoupled=Dispersion.Ham_BM(hvkd, alph, -1, lq, kappa, PH,0)
    
    
    if mode_HF==1:
        
        substract=0 #0 for decoupled layers
        mu=0
        filling=0
        HB=Dispersion.HF_BandStruc( lq, hpl, hmin, hpl_decoupled, hmin_decoupled, nremote_bands, nbands, substract,  [V0, d_screening_norm], mode_HF)
        
        
    else:
        
        substract=0 #0 for decoupled layers
        mu=0
        filling=0
        HB=Dispersion.HF_BandStruc( lq, hpl, hmin, hpl_decoupled, hmin_decoupled, nremote_bands, nbands, substract,  [V0, d_screening_norm], mode_HF)
        
    substract=0 #0 for decoupled layers
    mu=0
    filling=0
    field=1
    layersym="a"
    qins_X=lq.GMvec[0][0]
    qins_Y=lq.GMvec[0][1]
    sym=True
    HB=Dispersion.Phon_bare_BandStruc( lq,  hpl, hmin, nbands,  cons, field, qins_X, qins_Y, sym, mode, layersym )

    return 0

if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit
