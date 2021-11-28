import numpy as np
import Lattice
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import Lattice
from scipy.interpolate import interp1d
from scipy.linalg import circulant
  

class Ham():
    def __init__(self, hbvf, latt):

        self.hvkd = hbvf
        self.latt=latt
        
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION
    
    def eigens(self, kx,ky):
        [GM1,GM2]=self.latt.AMvec

        e1=(GM1+GM2)/3
        e2=(-2*GM1+GM2)/3
        e3=(-2*GM2+GM1)/3
        
        
        W3=0.00375/3  #in ev
        k=np.array([kx,ky])
        
        hk=np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3)
        hk_n=np.abs(W3*hk)
        # print(k@e1,k@e2,k@e3,hk_n, 2*np.pi*1/3)
        psi1=np.array([+1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)
        psi2=np.array([-1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)

        return np.array([-hk_n,hk_n]), np.array([psi2,psi1])
    
    def eigens2(self, kx,ky):
        hbvf = 0.001*2.1354; # eV
        # W3=0.00375/3  #in ev
        kv=np.array([kx,ky])
        knorm=np.sqrt(kv.T@kv)
        k=kx+1j*ky
        kbar=kx-1j*ky
        
        psi1=np.array([+np.exp(1j*np.angle(kbar)/2), np.exp(1j*np.angle(k)/2)])/np.sqrt(2)
        psi2=np.array([-np.exp(1j*np.angle(kbar)/2), np.exp(1j*np.angle(k)/2)])/np.sqrt(2)

        return np.array([-hbvf*knorm,hbvf*knorm ]), np.array([psi2,psi1])





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
        eps=np.min(eps_a)*1.5
        
        mmin=np.min([np.min(Ene_valley_plus),np.min(Ene_valley_min)])
        mmax=np.max([np.max(Ene_valley_plus),np.max(Ene_valley_min)])
        NN=int((mmax-mmin)/eps)+int((int((mmax-mmin)/eps)+1)%2) #making sure there is a bin at zero energy
        binn=np.linspace(mmin,mmax,NN+1)
        valt=np.zeros(NN)

        val_p,bins_p=np.histogram(Ene_valley_plus.flatten(), bins=binn,density=True)
        valt=valt+val_p

        val_m,bins_m=np.histogram(Ene_valley_min.flatten(), bins=binn,density=True)
        valt=valt+val_m
        
        bins=(binn[:-1]+binn[1:])/2
        
        
        
        
        valt=2*2*valt
        f2 = interp1d(binn[:-1],valt, kind='cubic')
        de=(bins[1]-bins[0])
        print("sum of the hist, normed?", np.sum(valt)*de)
        
        plt.plot(bins,valt)
        plt.scatter(bins,valt, s=1)
        plt.savefig("dos.png")
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
        eps=np.min(eps_a)
        
        mmin=np.min([np.min(Ene_valley_plus),np.min(Ene_valley_min)])
        mmax=np.max([np.max(Ene_valley_plus),np.max(Ene_valley_min)])
        NN=int((mmax-mmin)/eps)+int((int((mmax-mmin)/eps)+1)%2) #making sure there is a bin at zero energy
        earr=np.linspace(mmin,mmax,NN+1)
        epsil=eps/4
        de=earr[1]-earr[0]
        dosl=[]
        print("the volume element is ",dS_in)
        
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
            dosl.append( np.sum(predos  )*dS_in )
            # print(np.sum(self.deltados(earr, epsil)*de))

        dosarr=2*np.array(dosl) #extra 2 for spin
        f2 = interp1d(earr,dosarr, kind='cubic')
        print("sum of the hist, normed?", np.sum(dosarr)*de)
        
        
    
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
            [Ene_valley_plus_dos,Ene_valley_min_dos]=self.precompute_E_psi()
        if read:
            print("Loading  ..........")
            with open('dispersions/Edisp_'+str(self.lq.Npoints)+'_theta_'+str(self.lq.theta)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)
            with open('dispersions/Edisp_'+str(self.lq.Npoints)+'_theta_'+str(self.lq.theta)+'.npy', 'rb') as f:
                Ene_valley_min_dos=np.load(f)
    
        if write:
            print("saving  ..........")
            with open('dispersions/Edisp_'+str(self.lq.Npoints)+'_theta_'+str(self.lq.theta)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)
            with open('dispersions/Edism_'+str(self.lq.Npoints)+'_theta_'+str(self.lq.theta)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_min_dos)
                
        

        [earr, dos, f2 ]=self.DOS(Ene_valley_plus_dos,Ene_valley_min_dos)

        mu_values=[]        
        for fill in fillings:
            [mu, nfil, es,nn]=self.chem_for_filling( fill, f2, earr)
            mu_values.append(mu)


        
        return [fillings,np.array(mu_values)]
    
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
        plt.ylim([-0.009,0.009])
        plt.savefig("highsym.png")
        plt.show()
        return [Ene_valley_plus, Ene_valley_min]
    

class FormFactors_umklapp():
    def __init__(self, psi_p, xi, lat, umklapp, ham):
        self.psi_p = psi_p #has dimension #kpoints, 4*N, nbands
        self.lat=lat
        self.cpsi_p=np.conj(psi_p)
        self.xi=xi
        self.Nu=int(np.shape(self.psi_p)[1]/4) #4, 2 for sublattice and 2 for layer

        
        [KX,KY]=lat.Generate_lattice()
        
        Gu=lat.Umklapp_List(umklapp)
        [KXu,KYu]=lat.Generate_Umklapp_lattice2( KX, KY,umklapp)

        self.kx=KXu
        self.ky=KYu

        #momentum transfer lattice
        kqx1, kqx2=np.meshgrid(self.kx,self.kx)
        kqy1, kqy2=np.meshgrid(self.ky,self.ky)
        self.qx=kqx1-kqx2
        self.qy=kqy1-kqy2
        self.q=np.sqrt(self.qx**2+self.qy**2)+1e-17
        
        self.qmin_x=KXu[1]-KXu[0]
        self.qmin_y=KYu[1]-KYu[0]
        self.qmin=np.sqrt(self.qmin_x**2+self.qmin_y**2)
        psilist=[]
        for GG in Gu:
            shi1=int(GG[0])
            shi2=int(GG[1])
            psishift=ham.trans_psi2(psi_p, shi1, shi2)
            psilist=psilist+[psishift]
        self.psi=np.vstack(psilist)
        self.cpsi=np.conj(self.psi)
        print(np.shape(self.psi), np.shape(psi_p), np.shape(self.kx))
            

    def __repr__(self):
        return "Form factors for valley {xi}".format( xi=self.xi)

    def matmult(self, layer, sublattice):
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        pau=[pauli0,paulix,pauliy,pauliz]
        Qmat=np.eye(self.Nu)
        

        mat=np.kron(pau[layer],np.kron(Qmat, pau[sublattice]))
        
        psimult=[]
        for i in range(np.shape(self.psi)[0]):
            psimult=psimult+[mat@self.psi[i,:,:]]
        mult_psi=np.array(psimult)

        return  mult_psi#mat@self.psi

    def calcFormFactor(self, layer, sublattice):
        s=time.time()
        print("calculating tensor that stores the overlaps........")
        mult_psi=self.matmult(layer,sublattice)
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
        qmin=self.qmin
        harr= np.ones(np.shape(FF))
        qcut=np.array(self.q)
        qanom=qcut[np.where(qcut<0.01*qmin)]
        qcut[np.where(qcut<0.01*qmin)]=np.ones(np.shape(qanom))*qmin
        
        for i in range(np.shape(FF)[1]):
            for j in range(np.shape(FF)[1]):
                harr[:, i, :, j]=qcut
                        
        return harr


    

    ########### Anti-symmetric displacement of the layers
    def denqFF_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return L30

    def denqFFL_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return self.hq(L30)*L30


    def NemqFFL_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFL=self.fq(L31) *L31-self.xi*self.gq(L32)*L32
        return Nem_FFL

    def NemqFFT_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFT=-self.gq(L31) *L31- self.xi*self.fq(L32)*L32
        return Nem_FFT

    ########### Symmetric displacement of the layers
    def denqFF_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return L00

    def denqFFL_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return self.hq(L00)*L00

    def NemqFFL_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFL=self.fq(L01) *L01-self.xi*self.gq(L02)*L02
        return Nem_FFL

    def NemqFFT_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFT=-self.gq(L01)*L01 - self.xi*self.fq(L02)*L02
        return Nem_FFT
    
    

def main() -> int:
    ##when we use this main, we are exclusively testing the moire hamiltonian symmetries and methods
    from scipy import linalg as la
    
    
    
    #parameters for the calculation
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



    filling_index=int(sys.argv[1]) #0-25
    mu=mu_values[filling_index]/1000
    ##########################################
    #parameters energy calculation
    ##########################################
    a_graphene=2.46*(1e-10) #in meters
    #hbvf=0.003404*0.1973269804*1e-6 /a_graphene #ev*m

    Nsamp=int(sys.argv[2])




    l=Lattice.TriangLattice(Nsamp,0)
    [KX,KY]=l.Generate_lattice()
    plt.scatter(KX,KY)
    plt.savefig("trave.png")
    
    xi=1
    #kosh params realistic  -- this is the closest to the actual Band Struct used in the paper
    hbvf = 2.1354; # eV
    print("hbvf is ..",hbvf )



    Ene_valley_plus_a=np.empty((0))
    Ene_valley_plus_ac3=np.empty((0))
    psi_plus_a=[]
    psi_plus_ac3=[]

    umkl=1
    [KXu,KYu]=l.Generate_Umklapp_lattice(KX,KY,umkl)
    Npoi_u=np.size(KXu)
    plt.scatter(KXu, KYu)
    plt.scatter(KX, KY)
    plt.savefig("fig0.png")
    plt.close()

    [KXc3z,KYc3z, Indc3z]=l.C3zLatt(KXu,KYu)

    hpl=Ham(hbvf, l)
    nbands=2 
    disp=Dispersion( l, nbands, hpl, hpl)
    disp.High_symmetry()


if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
