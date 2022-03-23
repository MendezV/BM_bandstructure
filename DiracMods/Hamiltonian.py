import numpy as np
import Lattice
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import Lattice
from scipy.interpolate import interp1d
from scipy.linalg import circulant
import sys  

class Ham():
    def __init__(self, hbvf, latt, rescale=None):
        if rescale is None:
            self.rescale = 1
        else:
            self.rescale = rescale
        
        self.hvkd = self.rescale*hbvf
        self.latt=latt
        
        
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION
    
    def eigens(self, kx,ky):
        [GM1,GM2]=self.latt.LMvec

        e1=(GM1+GM2)/3
        e2=(-2*GM1+GM2)/3
        e3=(-2*GM2+GM1)/3
        
        
        W3=self.hvkd #0.00375/3  #in ev
        k=np.array([kx,ky])
        
        hk=W3*(np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3))
        hk_n=np.abs(hk)
        # print(k@e1,k@e2,k@e3,hk_n, 2*np.pi*1/3)
        psi2=np.array([+1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)
        psi1=np.array([-1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)
        
        # mat=np.array([[0,hk],[np.conj(hk),0]])
        # print((mat@psi1)/psi1, -hk_n )
        # print((mat@psi2)/psi2, hk_n )
        
        psi=np.zeros([np.size(psi1), 2])+0*1j
        psi[:,0]=psi1
        psi[:,1]=psi2

        
        return np.array([-hk_n,hk_n ]), psi
    
    def ExtendE(self,E_k , umklapp):
        Gu=self.latt.Umklapp_List(umklapp)
        
        Elist=[]
        for GG in Gu:
            Elist=Elist+[E_k]
            
        return np.vstack(Elist)

    
    

class Ham_p():
    def __init__(self, hbvf, latt, rescale=None):
        if rescale is None:
            self.rescale = 1
        else:
            self.rescale = rescale
        
        self.hvkd = self.rescale*hbvf
        self.latt=latt
        
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION
    
    def eigens(self, kx,ky):
        # W3=0.00375/3  #in ev
        kv=np.array([kx,ky])
        knorm=np.sqrt(kv.T@kv)
        k=kx+1j*ky
        kbar=kx-1j*ky
        
        hk_n=self.hvkd*knorm
        
        psi2=np.array([+1*np.exp(1j*np.angle(kbar)), 1])/np.sqrt(2)
        psi1=np.array([-1*np.exp(1j*np.angle(kbar)), 1])/np.sqrt(2)
    
        # mat=hbvf*np.array([[0,kbar],[k,0]])
        # print((mat@psi1)/psi1, -hk_n )
        # print((mat@psi2)/psi2, hk_n )
        
        psi=np.zeros([np.size(psi1), 2])+0*1j
        psi[:,0]=psi1
        psi[:,1]=psi2

        
        return np.array([-hk_n,hk_n ]), psi
    
    
class Ham_m():
    def __init__(self, hbvf, latt, rescale=None):
        if rescale is None:
            self.rescale = 1
        else:
            self.rescale = rescale
        
        self.hvkd = self.rescale*hbvf
        self.latt=latt
        
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION

    def eigens(self, kx,ky):
        # W3=0.00375/3  #in ev
        kv=np.array([kx,ky])
        knorm=np.sqrt(kv.T@kv)
        mk=-kx-1j*ky
        mkbar=-kx+1j*ky
        
        hk_n=self.hvkd*knorm
        
        psi2=np.array([+1*np.exp(1j*np.angle(mk)), 1])/np.sqrt(2)
        psi1=np.array([-1*np.exp(1j*np.angle(mk)), 1])/np.sqrt(2)
    
        # mat=hbvf*np.array([[0,mk],[mkbar,0]])
        # print((mat@psi1)/psi1, -hk_n )
        # print((mat@psi2)/psi2, hk_n )
        # print(np.conj(psi1.T)@psi1)
        
        psi=np.zeros([np.size(psi1), 2])+0*1j
        psi[:,0]=psi1
        psi[:,1]=psi2
        
        return np.array([-hk_n,hk_n ]), psi
    
    


class Dispersion():
    
    def __init__(self, latt, nbands, hpl, hmin):

        self.lat=latt
        
        #changes to eliominate arg KX KY put KX1bz first, then call umklapp lattice and put slef in KX, KY
        #change to implement dos 
        self.lq=latt
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        [self.KX1bz, self.KY1bz]=latt.Generate_lattice()
        self.Npoi1bz=np.size(self.KX1bz)
        self.latt=latt
        # [self.psi_plus,self.Ene_valley_plus_1bz,self.psi_min,self.Ene_valley_min_1bz]=self.precompute_E_psi()
    
    def precompute_E_psi(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(self.Npoi1bz):
            E1,wave1=self.hpl.eigens(self.KX1bz[l],self.KY1bz[l])
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(self.KX1bz[l],self.KY1bz[l])
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
    
    
    def precompute_E_psi_u(self, KXu,KYu):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]
        
        Npoiu=np.size(KXu)


        print("starting dispersion ..........")
        
        s=time.time()
        
        for l in range(Npoiu):
            E1,wave1=self.hpl.eigens(KXu[l],KYu[l])
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            psi_plus_a.append(wave1)


            E1,wave1=self.hmin.eigens(KXu[l],KYu[l])
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            psi_min_a.append(wave1)

            # printProgressBar(l + 1, self.Npoi_Q, prefix = 'Progress Diag2:', suffix = 'Complete', length = 50)

        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        psi_plus=np.array(psi_plus_a)
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoiu,self.nbands])

        psi_min=np.array(psi_min_a)
        Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoiu,self.nbands])

        
        

        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]
    
    ###########DOS FOR DEBUGGING
    
    def DOS(self,Ene_valley_plus_pre,Ene_valley_min_pre):
        [Ene_valley_plus,Ene_valley_min]=[Ene_valley_plus_pre,Ene_valley_min_pre]
        nbands=np.shape(Ene_valley_plus)[1]
        print("number of bands in density of states calculation," ,nbands)
        eps_l=[]
        for i in range(nbands):
            eps_l.append(np.mean( np.abs( np.diff( Ene_valley_plus[:,i].flatten() )  ) )/2)
            eps_l.append(np.mean( np.abs( np.diff( Ene_valley_min[:,i].flatten() )  ) )/2)
        eps_a=np.array(eps_l)
        eps=np.min(eps_a)*10
        print("and epsilon is ...", eps)
        
        mmin=np.min([np.min(Ene_valley_plus),np.min(Ene_valley_min)])
        mmax=np.max([np.max(Ene_valley_plus),np.max(Ene_valley_min)])
        NN=int((mmax-mmin)/eps)+int((int((mmax-mmin)/eps)+1)%2) #making sure there is a bin at zero energy
        binn=np.linspace(mmin,mmax,NN+1)
        valt=np.zeros(NN)

        val_p,bins_p=np.histogram(Ene_valley_plus[:,0].flatten(), bins=binn,density=True)
        valt=valt+val_p
        val_p,bins_p=np.histogram(Ene_valley_plus[:,1].flatten(), bins=binn,density=True)
        valt=valt+val_p

        val_m,bins_m=np.histogram(Ene_valley_min[:,0].flatten(), bins=binn,density=True)
        valt=valt+val_m
        val_m,bins_m=np.histogram(Ene_valley_min[:,1].flatten(), bins=binn,density=True)
        valt=valt+val_m
        
        bins=(binn[:-1]+binn[1:])/2
        
        
        
        
        valt=2*valt
        f2 = interp1d(binn[:-1],valt, kind='cubic')
        de=(bins[1]-bins[0])
        print("sum of the hist, normed?", np.sum(valt)*de)
        
        plt.plot(bins,valt)
        plt.scatter(bins,valt, s=1)
        plt.savefig("dos1.png")
        plt.close()
        

        return [bins,valt,f2 ]
    def deltados(self, x, epsil):
        return (1/(np.pi*epsil))/(1+(x/epsil)**2)
    
    def DOS2(self,Ene_valley_plus_pre,Ene_valley_min_pre,dS_in):
        
        [Ene_valley_plus,Ene_valley_min]=[Ene_valley_plus_pre,Ene_valley_min_pre]
        nbands=np.shape(Ene_valley_plus)[1]
        print(nbands)
        eps_l=[]
        for i in range(nbands):
            eps_l.append(np.mean( np.abs( np.diff( Ene_valley_plus[:,i].flatten() )  ) )/2)
            eps_l.append(np.mean( np.abs( np.diff( Ene_valley_min[:,i].flatten() )  ) )/2)
        eps_a=np.array(eps_l)
        eps=np.min(eps_a)*10
        
        
        mmin=np.min([np.min(Ene_valley_plus),np.min(Ene_valley_min)])
        mmax=np.max([np.max(Ene_valley_plus),np.max(Ene_valley_min)])
        NN=int((mmax-mmin)/eps)+int((int((mmax-mmin)/eps)+1)%2) #making sure there is a bin at zero energy
        earr=np.linspace(mmin,mmax,NN+1)
        epsil=eps/10
        de=earr[1]-earr[0]
        dosl=[]
        dos1band=[]
        print("the volume element is ",dS_in, "and epsilon is ...", epsil)
        
        for i in range(np.size(earr)):
            predos=0
            for j in range(nbands):
                
                predos_plus=self.deltados(Ene_valley_plus[:,j]-earr[i], epsil)
                predos_min=self.deltados(Ene_valley_min[:,j]-earr[i], epsil)
                predos=predos+predos_plus+predos_min
                
                # print(np.sum(predos_plus  )*dS_in)
                # print(np.sum(predos_min  )*dS_in)
                # print("sum of the hist, normed?", np.sum(predos)*dS_in)
            # print("4 real sum of the hist, normed?", np.sum(predos)*dS_in)
            dos1band.append(np.sum(predos_plus  )*dS_in )
            dosl.append( np.sum(predos  )*dS_in )
            # print(np.sum(self.deltados(earr, epsil)*de))
        dosarr1band=2*np.array(dos1band) 
        dosarr=2*np.array(dosl) #extra 2 for spin
        f2 = interp1d(earr,dosarr, kind='cubic')
        print("sum of the hist, normed?", np.sum(dosarr)*de)
        print("sum of the hist, normed?", np.sum(dosarr1band)*de)
        
        
    
        return [earr,dosarr,f2 ]
    
    def bisection(self,f,a,b,N):
        '''Approximate solution of f(x)=0 on interval [a,b] by bisection method.

        Parameters
        ----------
        f : function
            The function for which we are trying to approximate a solution f(x)=0.
        a,b : numbers
            The interval in which to search for a solution. The function returns
            None if f(a)*f(b) >= 0 since a solution is not guaranteed.
        N : (positive) integer
            The number of iterations to implement.

        Returns
        -------
        x_N : number
            The midpoint of the Nth interval computed by the bisection method. The
            initial interval [a_0,b_0] is given by [a,b]. If f(m_n) == 0 for some
            midpoint m_n = (a_n + b_n)/2, then the function returns this solution.
            If all signs of values f(a_n), f(b_n) and f(m_n) are the same at any
            iteration, the bisection method fails and return None.

        Examples
        --------
        >>> f = lambda x: x**2 - x - 1
        >>> bisection(f,1,2,25)
        1.618033990263939
        >>> f = lambda x: (2*x - 1)*(x - 3)
        >>> bisection(f,0,1,10)
        0.5
        '''
        if f(a)*f(b) >= 0:
            print("Bisection method fails.")
            return None
        a_n = a
        b_n = b
        for n in range(1,N+1):
            m_n = (a_n + b_n)/2
            f_m_n = f(m_n)
            if f(a_n)*f_m_n < 0:
                a_n = a_n
                b_n = m_n
            elif f(b_n)*f_m_n < 0:
                a_n = m_n
                b_n = b_n
            elif f_m_n == 0:
                print("Found exact solution.")
                return m_n
            else:
                print("Bisection method fails.")
                return None
        return (a_n + b_n)/2

    def chem_for_filling(self, fill, f2, earr):
        
        NN=10000
        mine=earr[1]
        maxe=earr[-2]
        mus=np.linspace(mine,maxe, NN)
        dosarr=f2(mus)
        de=mus[1]-mus[0]
        
        
        
        #FILLING FOR EACH CHEMICAL POTENTIAL
        ndens=[]
        for mu_ind in range(NN):
            N=np.trapz(dosarr[0:mu_ind])*de
            ndens.append(N)
            
        nn=np.array(ndens)
        nn=8*(nn/nn[-1])  - 4



        
        fn = interp1d(mus,nn-fill, kind='cubic')
        fn2 = interp1d(mus,nn, kind='cubic')
        
        mu=self.bisection(fn,mine,maxe,50)
        nfil=fn2(mu)
        if fill==0.0:
            mu=0.0
            nfil=0.0
         
        if fill>0:
            errfil=abs((nfil-fill)/fill)
            if errfil>0.1:
                print("TOO MUCH ERROR IN THE FILLING CALCULATION") 
            
        return [mu, nfil, mus,nn]
    
    def mu_filling_array(self, Nfil, read, write, calculate):
        
        fillings=np.linspace(0,3.9,Nfil)
        
        if calculate:
            [psi_plus,Ene_valley_plus_dos,psi_min,Ene_valley_min_dos]=self.precompute_E_psi()
        if read:
            print("Loading  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_min_dos=np.load(f)
    
        if write:
            print("saving  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)
            with open('dispersions/dirEdism_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_min_dos)
                
        

        [earr, dos, f2 ]=self.DOS(Ene_valley_plus_dos,Ene_valley_min_dos)

        mu_values=[]        
        for fill in fillings:
            [mu, nfil, es,nn]=self.chem_for_filling( fill, f2, earr)
            mu_values.append(mu)


        
        return [fillings,np.array(mu_values)]
    
    def dos_filling_array(self, Nfil, read, write, calculate):
        
        fillings=np.linspace(0,3.9,Nfil)
        
        if calculate:
            [psi_plus,Ene_valley_plus_dos,psi_min,Ene_valley_min_dos]=self.precompute_E_psi()
        if read:
            print("Loading  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_min_dos=np.load(f)
    
        if write:
            print("saving  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)
            with open('dispersions/dirEdism_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_min_dos)
                
        

        [earr, dos, f2 ]=self.DOS(Ene_valley_plus_dos,Ene_valley_min_dos)


        
        return [earr, dos]
    def dos2_filling_array(self, Nfil, read, write, calculate,dS_in):
        
        fillings=np.linspace(0,3.9,Nfil)
        
        if calculate:
            [psi_plus,Ene_valley_plus_dos,psi_min,Ene_valley_min_dos]=self.precompute_E_psi()
        if read:
            print("Loading  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'rb') as f:
                Ene_valley_min_dos=np.load(f)
    
        if write:
            print("saving  ..........")
            with open('dispersions/dirEdisp_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)
            with open('dispersions/dirEdism_'+str(self.lq.Npoints)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_min_dos)
                
        

        [earr, dos, f2 ]=self.DOS2(Ene_valley_plus_dos,Ene_valley_min_dos,dS_in)

        
        return [earr, dos]
    
    ### FERMI SURFACE ANALYSIS

    #creates a square grid, interpolates 
    def FSinterp(self, save_d, read_d, ham):

        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        GM=self.latt.GMs

        Nsamp = 40
        nbands= 4

        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(GM1,GM2)
        VV=np.array(Vertices_list+[Vertices_list[0]])
        k_window_sizex = K[1][0]*1.1 #4*(np.pi/np.sqrt(3))*np.sin(theta/2) (half size of  MBZ from edge to edge)
        k_window_sizey = K[2][1]
        Radius_inscribed_hex=1.0000005*k_window_sizey
        kx_rangex = np.linspace(-k_window_sizex,k_window_sizex,Nsamp) #normalization q
        ky_rangey = np.linspace(-k_window_sizey,k_window_sizey,Nsamp) #normalization q
        bz = np.zeros([Nsamp,Nsamp])
        ##########################################
        #Setting up kpoint arrays
        ##########################################
        #Number of relevant kpoints
        for x in range(Nsamp ):
            for y in range(Nsamp ):
                if self.latt.hexagon1((kx_rangex[x],ky_rangey[y]),Radius_inscribed_hex):
                    bz[x,y]=1
        num_kpoints=int(np.sum(bz))
        tot_kpoints=Nsamp*Nsamp


        #kpoint arrays
        k_points = np.zeros([num_kpoints,2]) #matrix of brillion zone points (inside hexagon)
        k_points_all = np.zeros([tot_kpoints,2]) #positions of all  the kpoints

        #filling kpoint arrays
        count1=0 #counting kpoints in the Hexagon
        count2=0 #counting kpoints in the original grid
        for x in range(Nsamp):
            for y in range(Nsamp):
                pos=[kx_rangex[x],ky_rangey[y]] #position of the kpoint
                k_points_all[count2,:]=pos #saving the position to the larger grid
                if self.latt.hexagon1((kx_rangex[x],ky_rangey[y]),Radius_inscribed_hex):
                    k_points[count1,:]=pos #saving the kpoint in the hexagon only
                    count1=count1+1
                count2=count2+1
        
        spect=[]

        for kkk in k_points_all:
            E1,wave1=ham.eigens(kkk[0], kkk[1],nbands)
            cois=[E1,wave1]
            spect.append(np.real(cois[0]))

        Edisp=np.array(spect)
        if save_d:
            with open('dispersions/sqEdisp_'+str(Nsamp)+'.npy', 'wb') as f:
                np.save(f, Edisp)

        if read_d:
            print("Loading  ..........")
            with open('dispersions/sqEdisp_'+str(Nsamp)+'.npy', 'rb') as f:
                Edisp=np.load(f)


        energy_cut = np.zeros([Nsamp, Nsamp]);
        for k_x_i in range(Nsamp):
            for k_y_i in range(Nsamp):
                ind = np.where(((k_points_all[:,0] == (kx_rangex[k_x_i]))*(k_points_all[:,1] == (ky_rangey[k_y_i]) ))>0);
                energy_cut[k_x_i,k_y_i] =Edisp[ind,2];

        # print(np.max(energy_cut), np.min(energy_cut))
        # plt.imshow(energy_cut.T) #transpose since x and y coordinates dont match the i j indices displayed in imshow
        # plt.colorbar()
        # plt.show()


        k1,k2= np.meshgrid(kx_rangex,ky_rangey) #grid to calculate wavefunct
        kx_rangexp = np.linspace(-k_window_sizex,k_window_sizex,Nsamp)
        ky_rangeyp = np.linspace(-k_window_sizey,k_window_sizey,Nsamp)
        k1p,k2p= np.meshgrid(kx_rangexp,ky_rangeyp) #grid to calculate wavefunct


        f_interp = interpolate.interp2d(k1,k2, energy_cut.T, kind='linear')

        # plt.plot(VV[:,0],VV[:,1])
        # plt.contour(k1p, k2p, f_interp(kx_rangexp,ky_rangeyp),[mu],cmap='RdYlBu')
        # plt.show()
        return [f_interp,k_window_sizex,k_window_sizey]



    #if used in the middle of plotting will close the plot
    def FS_contour(self, Np, mu, ham):
        #option for saving the square grid dispersion
        save_d=False
        read_d=False
        [f_interp,k_window_sizex,k_window_sizey]=self.FSinterp( save_d, read_d, ham)
        y = np.linspace(-k_window_sizex,k_window_sizex, 4603)
        x = np.linspace(-k_window_sizey,k_window_sizey, 4603)
        X, Y = np.meshgrid(x, y)
        Z = f_interp(x,y)  #choose dispersion
        c= plt.contour(X, Y, Z, levels=[mu],linewidths=3, cmap='summer');
        plt.close()
        #plt.show()
        numcont=np.shape(c.collections[0].get_paths())[0]
        
        if numcont==1:
            v = c.collections[0].get_paths()[0].vertices
        else:
            contourchoose=0
            v = c.collections[0].get_paths()[0].vertices
            sizecontour_prev=np.prod(np.shape(v))
            for ind in range(1,numcont):
                v = c.collections[0].get_paths()[ind].vertices
                sizecontour=np.prod(np.shape(v))
                if sizecontour>sizecontour_prev:
                    contourchoose=ind
            v = c.collections[0].get_paths()[contourchoose].vertices
        NFSpoints=Np
        xFS_dense = v[::int(np.size(v[:,1])/NFSpoints),0]
        yFS_dense = v[::int(np.size(v[:,1])/NFSpoints),1]
        
        return [xFS_dense,yFS_dense]
    
    def High_symmetry(self):
        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]

        nbands=self.nbands
        kpath=self.latt.High_symmetry_path()

        Npoi=np.shape(kpath)[0]
        for l in range(Npoi):
            # h.umklapp_lattice()
            # break
            E1p,wave1p=self.hpl.eigens(kpath[l,0],kpath[l,1])
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
            psi_plus_a.append(wave1p)


            E1m,wave1m=self.hmin.eigens(kpath[l,0],kpath[l,1])
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
            psi_min_a.append(wave1m)

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

    

        print("the shape of the energy array is",np.shape(Ene_valley_plus))
        qa=np.linspace(0,1,Npoi)
        for i in range(nbands):
            plt.plot(qa,Ene_valley_plus[:,i] , c='b')
            plt.plot(qa,Ene_valley_min[:,i] , c='r', ls="--")
        plt.xlim([0,1])
        # plt.ylim([-0.009,0.009])
        plt.savefig("highsym.png")
        plt.close()
        return [Ene_valley_plus, Ene_valley_min]
    


class FormFactors():
    def __init__(self, psi, xi, lat, umklapp):
        self.psi = psi #has dimension #kpoints, 4*N, nbands
        self.lat=lat
        self.cpsi=np.conj(psi)
        self.xi=xi

        Gu=lat.Umklapp_List(umklapp)
        [KX,KY]=lat.Generate_lattice()
        [KXu,KYu]=lat.Generate_Umklapp_lattice(KX,KY,umklapp)
        
        self.kx=KXu
        self.ky=KYu
       
        #momentum transfer lattice
        kqx1, kqx2=np.meshgrid(self.kx,self.kx)
        kqy1, kqy2=np.meshgrid(self.ky,self.ky)
        self.qx=kqx1-kqx2
        self.qy=kqy1-kqy2
        self.q=np.sqrt(self.qx**2+self.qy**2)+1e-17
        
        # not the easiest here
        # psilist=[]
        # for GG in Gu:
        #     shi1=int(GG[0])
        #     shi2=int(GG[1])
        #     [psishift,Ene,psishift,Ene]=disp.precompute_E_psi_u(KQX,KQY)
        #     # psishift=ham.trans_psi2(psi_p, shi1, shi2)
        #     psilist=psilist+[psi]
        # self.psi=np.vstack(psilist)
        # self.cpsi=np.conj(self.psi)
            

    def __repr__(self):
        return "Form factors for valley {xi}".format( xi=self.xi)

    def matmult(self, sublattice):
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        pau=[pauli0,paulix,pauliy,pauliz]
        

        mat=pau[sublattice]
        
        # print("shapes in all the matrices being mult", np.shape(mat), np.shape(self.psi), np.shape(mat@self.psi))
        psimult=[]
        for i in range(np.shape(self.psi)[0]):
            psimult=psimult+[mat@self.psi[i,:,:]]
        mult_psi=np.array(psimult)

        return  mult_psi#mat@self.psi

    def calcFormFactor(self,  sublattice):
        s=time.time()
        print("calculating tensor that stores the overlaps........")
        mult_psi=self.matmult(sublattice)
        Lambda_Tens=np.tensordot(self.cpsi,mult_psi, axes=([1],[1]))
        e=time.time()
        print("finsihed the overlaps..........", e-s)
        return(Lambda_Tens)
    

    #######fourth round
    def fq(self, FF ):

        farr= np.ones(np.shape(FF))
        for i in range(np.shape(FF)[1]):
            for j in range(np.shape(FF)[1]):
                farr[:, i, :, j]=(self.qx**2-self.qy**2)/self.q
                        
        return farr

    def gq(self,FF):
        garr= np.ones(np.shape(FF))
        
        for i in range(np.shape(FF)[1]):
            for j in range(np.shape(FF)[1]):
                garr[:, i, :, j]=2*(self.qx*self.qy)/self.q
                        
        return garr 


    def hq(self,FF):
        harr= np.ones(np.shape(FF))
        
        for i in range(np.shape(FF)[1]):
            for j in range(np.shape(FF)[1]):
                harr[:, i, :, j]=self.q
                        
        return harr 

    def h_denominator(self,FF):
        qx=self.kx[1]-self.kx[0]
        qy=self.ky[1]-self.ky[0]
        qmin=np.sqrt(qx**2+qy**2)
        harr= np.ones(np.shape(FF))
        qcut=np.array(self.q)
        qanom=qcut[np.where(qcut<0.01*qmin)]
        qcut[np.where(qcut<0.01*qmin)]=np.ones(np.shape(qanom))*qmin
        
        for i in range(np.shape(FF)[1]):
            for j in range(np.shape(FF)[1]):
                harr[:, i, :, j]=qcut
                        
        return harr

    

    ########### Anti-symmetric displacement of the layers
    def denqFF(self):
        L30=self.calcFormFactor(sublattice=0)
        return L30

    def denqFFL(self):
        L30=self.calcFormFactor( sublattice=0)
        return self.hq(L30)*L30


    def NemqFFL(self):
        L31=self.calcFormFactor( sublattice=1)
        L32=self.calcFormFactor(  sublattice=2)
        Nem_FFL=self.fq(L31) *L31-self.xi*self.gq(L32)*L32
        return Nem_FFL
    

    def NemqFFT(self):
        L31=self.calcFormFactor(sublattice=1)
        L32=self.calcFormFactor(sublattice=2)
        Nem_FFT=-self.gq(L31) *L31- self.xi*self.fq(L32)*L32
        return Nem_FFT


    def Fdirac(self):

        phi=np.angle(self.xi*self.kx-1j*self.ky)
        F=np.zeros([np.size(phi), 2, np.size(phi), 2]) +0*1j
        phi1, phi2=np.meshgrid(phi,phi)
        for n in range(2):
            for n2 in range(2):
                s1=(-1)**n
                s2=(-1)**n2
                F[:,n,:,n2]= ( 1+s1*s2*np.exp(1j*(phi1-phi2)) )/2  
                
        return F
    
    def F_tb(self):
        [GM1,GM2]=self.lat.LMvec

        e1=(GM1+GM2)/3
        e2=(-2*GM1+GM2)/3
        e3=(-2*GM2+GM1)/3

        k=np.array([self.kx,self.ky]).T
        
        hk=(np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3))

        phi=np.angle(hk)
        F=np.zeros([np.size(phi), 2, np.size(phi), 2])+0*1j
        phi1, phi2=np.meshgrid(phi,phi)
        for n in range(2):
            for n2 in range(2):
                s1=(-1)**n
                s2=(-1)**n2
                F[:,n,:,n2]= ( 1+s1*s2*np.exp(1j*(phi1-phi2)) )/2  
                
        return F
                
    

def main() -> int:
    ##when we use this main, we are exclusively testing the moire hamiltonian symmetries and methods
    from scipy import linalg as la
    

    
    try:
        filling_index=int(sys.argv[1]) 

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose chemical potential for desired filling")


    try:
        Nsamp=int(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")


    ##########################################
    #parameters energy calculation
    ##########################################
    graphene=False
    if graphene==True:
        a_graphene=2.46*(1e-10) #in meters
        nbands=2 
        Nsamp=int(sys.argv[2])

        #Lattice generation
        latt=Lattice.TriangLattice(Nsamp,0)
        
        #parameters for the bandstructure
        hbvf = 2.1354; # eV
        print("hbvf is ..",hbvf )
        
        #generating the dispersion 
        resc=1 #0.001
        h=Ham(hbvf, latt, resc)
        disp=Dispersion( latt, nbands, h,h)
        disp.High_symmetry()
        
        #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
        Ndos=100
        ldos=Lattice.TriangLattice(Ndos,0)
        disp=Dispersion( ldos, nbands, h, h)
        Nfils=20
        [fillings,mu_values]=disp.mu_filling_array(Nfils, False, False, True) #read write calculate
        filling_index=int(sys.argv[1]) 
        mu=mu_values[filling_index]
        filling=fillings[filling_index]
        print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
        
        #testing the form factors
        
        umkl=2
        [KX1bz, KY1bz]=latt.Generate_lattice()
        Npoi1bz=np.size(KX1bz)
        [KX,KY]=latt.Generate_Umklapp_lattice(KX1bz, KY1bz,umkl) #for the integration grid 
        [KQX,KQY]=latt.Generate_Umklapp_lattice(KX1bz, KY1bz,umkl+1) #for the momentum transfer lattice
        Npoi=np.size(KX)
        NpoiQ=np.size(KQX)
        [KX,KY]=latt.Generate_lattice()
        [KXc3z,KYc3z, Indc3z]=latt.C3zLatt(KQX,KQY)
        disp=Dispersion( latt, nbands, h, h)
        [psi,Ene,psi,Ene]=disp.precompute_E_psi_u(KQX,KQY)
        FF=FormFactors(psi, 1, latt, umkl+1)

        # L00p=FF.F_tb()
        # L00m=FF.F_tb()
        L00p=FF.denqFF()
        L00m=FF.denqFF()
        # L00p=FF.NemqFFL()
        # L00m=FF.NemqFFL()
        test= True
        name="ee"
        if test:
            print("testing symmetry of the form factors...")
            [KXc3z,KYc3z, Indc3z]=latt.C3zLatt(KQX,KQY)
            diffarp=[]
            diffarm=[]
            K=[]
            KP=[]
            cos1=[]
            cos2=[]
            kp=np.argmin(KQX**2 +KQY**2)
            for k in range(NpoiQ):
                K.append(KQX[k]-KQX[kp])
                KP.append(KQY[k]-KQY[kp])
                #Regular FF
                undet=np.abs(np.linalg.det(np.abs(L00p[k,:,kp,:])**2 ))
                dosdet=np.abs(np.linalg.det(np.abs(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2 ))
                cos1.append(undet)
                cos2.append(dosdet)
                diffarp.append( undet   - dosdet   )
                # Minus Valley FF Omega
                undet=np.abs(np.linalg.det(np.abs(L00m[k,:,kp,:])**2 ))
                dosdet=np.abs(np.linalg.det(np.abs(L00m[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2 ))
                diffarm.append( undet   - dosdet   )

            
            VV=latt.boundary()
            plt.plot(diffarp, label="plus valley")
            plt.plot(diffarm, label="minsu valley")
            plt.title("FF- C3 FF")
            plt.legend()
            plt.savefig("TestC3_symm"+name+".png")
            # plt.show()
            plt.close()  
            plt.scatter(K,KP,c=cos1, s=3)
            plt.plot(VV[:,0],VV[:,1], c='r')
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("cos1scat.png")
            plt.close( ) 
            plt.scatter(K,KP,c=cos2, s=3)
            plt.plot(VV[:,0],VV[:,1], c='r')
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("cos2scat.png")
            plt.close(  )
            print("finished testing symmetry of the form factors...")
    else:
        a_graphene=2.46*(1e-10) #in meters
        nbands=2 
        Nsamp=int(sys.argv[2])

        #Lattice generation
        latt=Lattice.TriangLattice(Nsamp,0)
        
        #parameters for the bandstructure
        hbvf = 2.1354; # eV
        print("hbvf is ..",hbvf )
        
        #generating the dispersion 
        resc=1 #0.001
        hpl=Ham_p(hbvf, latt, resc)
        hm=Ham_m(hbvf, latt, resc)
        disp=Dispersion( latt, nbands, hpl,hm)
        disp.High_symmetry()
        
        #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
        Ndos=100
        ldos=Lattice.TriangLattice(Ndos,0)
        disp=Dispersion( ldos, nbands, hpl, hm)
        Nfils=20
        [fillings,mu_values]=disp.mu_filling_array(Nfils, False, False, True) #read write calculate
        filling_index=int(sys.argv[1]) 
        mu=mu_values[filling_index]
        filling=fillings[filling_index]
        print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
        
        #testing the form factors
        
        umkl=2
        [KX1bz, KY1bz]=latt.Generate_lattice()
        Npoi1bz=np.size(KX1bz)
        [KX,KY]=latt.Generate_Umklapp_lattice(KX1bz, KY1bz,umkl) #for the integration grid 
        [KQX,KQY]=latt.Generate_Umklapp_lattice(KX1bz, KY1bz,umkl+1) #for the momentum transfer lattice
        Npoi=np.size(KX)
        NpoiQ=np.size(KQX)
        [KX,KY]=latt.Generate_lattice()
        [KXc3z,KYc3z, Indc3z]=latt.C3zLatt(KQX,KQY)
        disp=Dispersion( latt, nbands, hpl, hm)
        [psi,Ene,psi,Ene]=disp.precompute_E_psi_u(KQX,KQY)
        FF=FormFactors(psi, 1, latt, umkl+1)

        # L00p=FF.F_tb()
        # L00m=FF.F_tb()
        L00p=FF.denqFF()
        L00m=FF.denqFF()
        # L00p=FF.NemqFFL()
        # L00m=FF.NemqFFL()
        test= True
        name="ee"
        if test:
            print("testing symmetry of the form factors...")
            [KXc3z,KYc3z, Indc3z]=latt.C3zLatt(KQX,KQY)
            diffarp=[]
            diffarm=[]
            K=[]
            KP=[]
            cos1=[]
            cos2=[]
            kp=np.argmin(KQX**2 +KQY**2)
            for k in range(NpoiQ):
                K.append(KQX[k]-KQX[kp])
                KP.append(KQY[k]-KQY[kp])
                #Regular FF
                undet=np.abs(np.linalg.det(np.abs(L00p[k,:,kp,:])**2 ))
                dosdet=np.abs(np.linalg.det(np.abs(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2 ))
                cos1.append(undet)
                cos2.append(dosdet)
                diffarp.append( undet   - dosdet   )
                # Minus Valley FF Omega
                undet=np.abs(np.linalg.det(np.abs(L00m[k,:,kp,:])**2 ))
                dosdet=np.abs(np.linalg.det(np.abs(L00m[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2 ))
                diffarm.append( undet   - dosdet   )

            
            VV=latt.boundary()
            plt.plot(diffarp, label="plus valley")
            plt.plot(diffarm, label="minsu valley")
            plt.title("FF- C3 FF")
            plt.legend()
            plt.savefig("TestC3_symm"+name+".png")
            # plt.show()
            plt.close()  
            plt.scatter(K,KP,c=cos1, s=3)
            plt.plot(VV[:,0],VV[:,1], c='r')
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("cos1scat.png")
            plt.close( ) 
            plt.scatter(K,KP,c=cos2, s=3)
            plt.plot(VV[:,0],VV[:,1], c='r')
            plt.colorbar()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.savefig("cos2scat.png")
            plt.close(  )
            print("finished testing symmetry of the form factors...")

def helper():
    [KX,KY]=latt.Generate_lattice()
    plt.scatter(KX,KY)
    plt.savefig("trave.png")
    plt.close()

    
    umkl=1
    [KXu,KYu]=latt.Generate_Umklapp_lattice(KX,KY,umkl)
    Npoi_u=np.size(KXu)
    plt.scatter(KXu, KYu)
    plt.scatter(KX, KY)
    [KXc3z,KYc3z, Indc3z]=latt.C3zLatt(KXu,KYu)
    
    
    [psi,Ene,psi,Ene]=disp.precompute_E_psi_u(KXu, KYu)

    plt.scatter(KXu,KYu, c=Ene[:,1])
    plt.savefig("dispersion_pre.png")
    plt.close()
    
    hpl=Ham_p(hbvf, latt)
    hmin=Ham_m(hbvf, latt)
    # [GM1, GM2]=l.GMvec
    # n,m=1,0
    # K=n*GM1+GM2*m
    # kx=3
    # ky=0.
    # kxu=kx+K[0]
    # kyu=ky+K[1]
    # plt.scatter([kx],[ky],s=30)
    # plt.scatter([kxu],[kyu],s=30)
    # plt.savefig("fig0.png")
    # plt.close()
    
    
    
    # [E1,psi1]=hpl.eigens(kx,ky)
    # [E2,psi2]=hpl.eigens(kxu,kyu)
    # print("energy", E1,E2)
    # print("overlap",np.conj(psi1[:,0]).T@psi1[:,0],np.conj(psi2[:,0]).T@psi2[:,0],np.abs(np.conj(psi1[:,0]).T@psi2[:,0]))
    # print("overlap",np.conj(psi1[:,0]).T@psi1[:,1],np.abs(np.conj(psi1[:,1]).T@psi2[:,1]))

    disp=Dispersion( latt, nbands, hpl, hmin)
    [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]=disp.precompute_E_psi_u(KXu, KYu)

    plt.scatter(KXu,KYu, c=Ene_valley_min[:,1])
    plt.savefig("dispersion.png")
    plt.close()
    
    # disp.High_symmetry()
    
    #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
    Ndos=100
    ldos=Lattice.TriangLattice(Ndos,0)
    disp=Dispersion( ldos, nbands, hpl, hpl)
    Nfils=20
    [fillings,mu_values]=disp.mu_filling_array(Nfils, False, True, True)
    filling_index=int(sys.argv[1]) 
    mu=mu_values[filling_index]
    filling=fillings[filling_index]
    print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
    
    umkl=0
    [KX1bz, KY1bz]=latt.Generate_lattice()
    Npoi1bz=np.size(KX1bz)
    [KX,KY]=latt.Generate_Umklapp_lattice(KX1bz, KY1bz,umkl) #for the integration grid 
    [KQX,KQY]=latt.Generate_Umklapp_lattice(KX1bz, KY1bz,umkl+1) #for the momentum transfer lattice
    Npoi=np.size(KX)
    NpoiQ=np.size(KQX)
    
    print(np.shape(psi_plus))
    
    #generating form factors
    FFp=FormFactors(psi_plus, 1, latt, umkl+1)
    FFm=FormFactors(psi_min, -1, latt, umkl+1)
    FF=FormFactors(psi, 1, latt, umkl+1)

    name="ee"
    # L00=FFp.F_tb()
    L00p=FF.F_tb()
    L00m=FF.F_tb()


    test= True
    if test:
        print("testing symmetry of the form factors...")
        [KXc3z,KYc3z, Indc3z]=latt.C3zLatt(KQX,KQY)
        diffarp=[]
        diffarm=[]
        K=[]
        KP=[]
        cos1=[]
        cos2=[]
        kp=np.argmin(KQX**2 +KQY**2)
        for k in range(NpoiQ):
            K.append(KQX[k]-KQX[kp])
            KP.append(KQY[k]-KQY[kp])
            #Regular FF
            # Plus Valley FF Omega
            # undet=np.abs(np.linalg.det(Lnemp[k,:,kp,:]))
            # dosdet=np.abs(np.linalg.det(Lnemp[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
            undet=np.linalg.det(np.abs(L00p[k,:,kp,:])**2 )
            dosdet=np.linalg.det(np.abs(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2 )
            cos1.append(undet)
            cos2.append(dosdet)
            diffarp.append( undet   - dosdet   )
            # Minus Valley FF Omega
            # undet=np.abs(np.linalg.det(Lnemm[k,:,kp,:]))
            # dosdet=np.abs(np.linalg.det(Lnemm[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
            undet=np.linalg.det(np.abs(L00m[k,:,kp,:])**2 )
            dosdet=np.linalg.det(np.abs(L00m[int(Indc3z[k]),:,int(Indc3z[kp]),:])**2 )
            diffarm.append( undet   - dosdet   )

        plt.plot(diffarp, label="plus valley")
        plt.plot(diffarm, label="minsu valley")
        plt.title("FF- C3 FF")
        plt.legend()
        plt.savefig("TestC3_symm"+name+".png")
        # plt.show()
        plt.close()  
        plt.scatter(K,KP,c=cos1)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("cos1scat.png")
        plt.close( ) 
        plt.scatter(K,KP,c=cos2)
        plt.colorbar()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig("cos2scat.png")
        plt.close(  )
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(K,KP,cos1, c=cos1);
        plt.savefig("cos1.png")
        plt.close(  )
        print("finished testing symmetry of the form factors...")


if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
