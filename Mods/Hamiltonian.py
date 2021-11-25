import numpy as np
import MoireLattice
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import MoireLattice
from scipy.interpolate import interp1d
from scipy.linalg import circulant
  

class Ham_BM_p():
    def __init__(self, hvkd, alpha, xi, latt, kappa, PH):

        self.hvkd = hvkd
        self.alpha= alpha
        self.xi = xi
        
        
        self.latt=latt
        self.kappa=kappa
        self.PH=PH #particle hole symmetry
        self.gap=0#1e-8#artificial gap
        
        #precomputed momentum lattice and interlayer coupling
       
        self.cuttoff_momentum_lat=self.umklapp_lattice()


        self.U=np.matrix(self.InterlayerU())

        #constant shift for the dispersion
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(GM1,GM2)
        self.e0=0
        E,psi=self.eigens( K[2][0],K[2][1], 2)
        self.e0=(E[0]+E[1])/2  #offset to set dirac points at zero energy. If there is a gap the dispersion is just centered at zero
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION 

    def umklapp_lattice(self):

        #large lattice that will be cuttoff
        Numklpx=30
        Numklpy=30
        gridpx=np.arange(-int(Numklpx/2),int(Numklpx/2),1) #grid to calculate wavefunct
        gridpy=np.arange(-int(Numklpy/2),int(Numklpy/2),1) #grid to calculate wavefunct
        n1,n2=np.meshgrid(gridpx,gridpy) #grid to calculate wavefunct

        #q vectors defined as the difference of the K point positions in the two layers
        # and 2pi/3 rotations of that.
        [q1,q2,q3]=self.latt.q
        #getting the momentum lattice to be diagonalized
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        GM=self.latt.GMs
        qx_difb = + GM1[0]*n1 + GM2[0]*n2 + 2*self.xi*q1[0]
        qy_difb = + GM1[1]*n1 + GM2[1]*n2 + 2*self.xi*q1[1]
        valsb = np.sqrt(qx_difb**2+qy_difb**2)
        #cutoff=7.*GM*0.7
        cutoff=9.*GM*0.7
        ind_to_sum_b = np.where(valsb <cutoff) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array

        #cutoff lattice
        n1_val_b = n1[ind_to_sum_b] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
        n2_val_b = n2[ind_to_sum_b] #
        Nb = np.shape(ind_to_sum_b)[1] ##number of indices for which the condition above is satisfied
        G0xb= GM1[0]*n1_val_b+GM2[0]*n2_val_b #umklapp vectors within the cutoff
        G0yb= GM1[1]*n1_val_b+GM2[1]*n2_val_b #umklapp vectors within the cutoff

        #reciprocal lattices for both layers
        #flipping the order so that same points occur in the same index for plus and minus valleys
        qx_t = -qx_difb[ind_to_sum_b]
        qy_t = -qy_difb[ind_to_sum_b]
        qx_b = qx_difb[ind_to_sum_b]#[::int(self.xi)]
        qy_b = qy_difb[ind_to_sum_b]#[::int(self.xi)]


        # plt.scatter(qx_t ,qy_t,c='k', s=np.arange(Nb)+1 )
        # plt.scatter(qx_b ,qy_b , c='r', s=np.arange(Nb)+1 )
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.savefig("momelat.png")
        # plt.close()
        return [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t,qx_b, qy_b] 

    def umklapp_lattice_rot(self, rot):
        [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        G0xb_p= rot[0,0]*G0xb + rot[0,1]*G0yb
        G0yb_p= rot[1,0]*G0xb + rot[1,1]*G0yb
        qx_t_p= rot[0,0]*qx_t + rot[0,1]*qy_t
        qy_t_p= rot[1,0]*qx_t + rot[1,1]*qy_t
        qx_b_p= rot[0,0]*qx_b + rot[0,1]*qy_b
        qy_b_p= rot[1,0]*qx_b + rot[1,1]*qy_b
        return [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]
    def umklapp_lattice_trans(self,trans):
        [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        G0xb_p= G0xb + trans[0]
        G0yb_p= G0yb + trans[1]
        qx_t_p= qx_t + trans[0]
        qy_t_p= qy_t +trans[1]
        qx_b_p= qx_b + trans[0]
        qy_b_p= qy_b +trans[1]
        return [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]
    


    def diracH(self, kx, ky):

        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        tau=self.xi
        hvkd=self.hvkd

        Qplusx = qx_t
        Qplusy = qy_t
        Qminx = qx_b
        Qminy = qy_b

        ###checking if particle hole symmetric model is chosen
        if(self.PH):
            # #top layer
            qx_1 = kx - Qplusx
            qy_1 = ky - Qplusy

            # #bottom layer
            qx_2 = kx -Qminx
            qy_2 = ky -Qminy
        else:
            # #top layer
            ROTtop=self.latt.rot_min
            kkx_1 = kx - Qplusx
            kky_1 = ky - Qplusy
            qx_1 = ROTtop[0,0]*kkx_1+ROTtop[0,1]*kky_1
            qy_1 = ROTtop[1,0]*kkx_1+ROTtop[1,1]*kky_1
            # #bottom layer
            ROTbot=self.latt.rot_plus
            kkx_2 = kx -Qminx
            kky_2 = ky -Qminy
            qx_2 = ROTbot[0,0]*kkx_2+ROTbot[0,1]*kky_2
            qy_2 = ROTbot[1,0]*kkx_2+ROTbot[1,1]*kky_2

        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        H1=hvkd*(np.kron(np.diag(qx_1),tau*paulix)+np.kron(np.diag(qy_1),pauliy)) +np.kron(self.gap*np.eye(Nb),pauliz) # ARITCFICIAL GAP ADDED
        H2=hvkd*(np.kron(np.diag(qx_2),tau*paulix)+np.kron(np.diag(qy_2),pauliy)) +np.kron(self.gap*np.eye(Nb),pauliz) # ARITCFICIAL GAP ADDED
        return [H1,H2]
    
    def diracH2(self, kx, ky):

        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        tau=self.xi
        hvkd=self.hvkd



        ###checking if particle hole symmetric model is chosen
        
        # #top layer
        qx_1 = kx 
        qy_1 = ky 
        # #bottom layer
        qx_2 = kx 
        qy_2 = ky 
    
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        H1=hvkd*(qx_1*tau*paulix+qy_1*pauliy) +self.gap*pauliz # ARITCFICIAL GAP ADDED
        H2=hvkd*(qx_1*tau*paulix+qy_1*pauliy) +self.gap*pauliz # ARITCFICIAL GAP ADDED
        return [H1,H2]



    def InterlayerU(self):
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        tau=self.xi
        [q1,q2,q3]=self.latt.q
        

        Qplusx = qx_t
        Qplusy = qy_t
        Qminx = qx_b
        Qminy = qy_b

        matGq1=np.zeros([Nb,Nb])
        matGq2=np.zeros([Nb,Nb])
        matGq3=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(Nb):

            indi1=np.where(np.sqrt(  (Qplusx[i]-Qminx + tau*q1[0])**2+(Qplusy[i]-Qminy + tau*q1[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq1[i,indi1]=1

            indi1=np.where(np.sqrt(  (Qplusx[i]-Qminx + tau*q2[0])**2+(Qplusy[i]-Qminy+ tau*q2[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq2[i,indi1]=1 #indi1+1=i
    
            indi1=np.where(np.sqrt(  (Qplusx[i]-Qminx + tau*q3[0])**2+(Qplusy[i]-Qminy + tau*q3[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq3[i,indi1]=1


        #Matrices that  appeared as coefficients of the real space ops
        #full product is the kronecker product of both matrices

        Mdelt1=matGq1
        Mdelt2=matGq2
        Mdelt3=matGq3
       

        w0=self.kappa
        w1=1
        phi = 2*np.pi/3    
        z = np.exp(-1j*phi*tau)
        zs = np.exp(1j*phi*tau)
        
        T1 = np.array([[w0,w1],[w1,w0]])
        T2 = zs*np.array([[w0,w1*zs],[w1*z,w0]])
        T3 = z*np.array([[w0,w1*z],[w1*zs,w0]])


        #different basis
        # T1 = np.array([[w0,w1],[w1,w0]])
        # T2 = np.array([[w0,w1*zs],[w1*z,w0]])
        # T3 = np.array([[w0,w1*z],[w1*zs,w0]])

        U=self.hvkd*self.alpha*( np.kron(Mdelt1,T1) + np.kron(Mdelt2,T2)+ np.kron(Mdelt3,T3)) #interlayer coupling

        return U
        
    def eigens(self, kx,ky, nbands):
        
        U=self.U
        Udag=U.H
        [H1,H2]=self.diracH( kx, ky)
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]

        for nband in range(nbands):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)[0]
            # print("wave1p;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)

        return Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi
    
    def eigens_dirac(self, kx,ky, nbands):
        
        U=self.U*0
        Udag=U.H
        P=self.PH
        self.PH=True
        [H1,H2]=self.diracH( kx, ky)
        self.PH=P
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]

        for nband in range(nbands):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1p;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)

        return Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi
    
    def eigens_dirac2(self, kx,ky, nbands):
        
        [H1,H2]=self.diracH2( kx, ky)
        
        Hxi=np.bmat([[H2, H2*0 ], [H2*0, H1]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,:]
        nbands_D=2

        for nband in range(nbands_D):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1m;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)
            

        return Eigvals[1:3]-self.e0, psi[:,1:3]
    
    def eigens_dirac3(self, kx,ky, nbands):
        e1=np.array([np.cos(2*np.pi*1/3), np.sin(2*np.pi*1/3)])*2*2*np.pi*1/3/np.sqrt(3)
        e2=np.array([np.cos(2*np.pi*2/3), np.sin(2*np.pi*2/3)])*2*2*np.pi*1/3/np.sqrt(3)
        e3=np.array([np.cos(2*np.pi*0/3), np.sin(2*np.pi*0/3)])*2*2*np.pi*1/3/np.sqrt(3)
        
        W3=0.00375/3  #in ev
        k=np.array([kx,ky])
        
        hk=np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3)
        hk_n=np.abs(W3*hk)
        # print(k@e1,k@e2,k@e3,hk_n, 2*np.pi*1/3)
        psi1=np.array([+1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)
        psi2=np.array([-1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)

        return np.array([-hk_n,hk_n]), np.array([psi2,psi1])


    def parallel_eigens(self,  nbands,q):
        kx,ky=q[0], q[1]
        U=self.U
        Udag=U.H
        [H1,H2]=self.diracH( kx, ky)
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]

        for nband in range(nbands):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1p;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)

        return [Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi]



    ### FERMI SURFACE ANALYSIS

    #creates a square grid, interpolates and finds contours at fixed mu
    def FSinterp(self, save_d, read_d, mu):

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
            E1,wave1=self.eigens(kkk[0], kkk[1],nbands)
            cois=[E1,wave1]
            spect.append(np.real(cois[0]))

        Edisp=np.array(spect)
        if save_d:
            with open('Edisp_'+str(Nsamp)+'.npy', 'wb') as f:
                np.save(f, Edisp)

        if read_d:
            print("Loading  ..........")
            with open('Edisp_'+str(Nsamp)+'.npy', 'rb') as f:
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
    def FS_contour(self, Np, mu):
        #option for saving the square grid dispersion
        save_d=False
        read_d=False
        [f_interp,k_window_sizex,k_window_sizey]=self.FSinterp( save_d, read_d, mu)
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


    #METHODS FOR MANIPULATING WAVEFUNCTIONS AND FORM FACTORS
    #for reference the pattern of kronecker prod is 
    # layer x umklp x sublattice
    def rot_WF(self, rot):
        [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]=self.umklapp_lattice_rot( rot)
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        
        [q1,q2,q3]=self.latt.q

        

        matGGp1=np.zeros([Nb,Nb])
        matGGp2=np.zeros([Nb,Nb])
        matGGp3=np.zeros([Nb,Nb])
        matGGp4=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(Nb):

            indi1=np.where(np.sqrt(  (qx_t[i]-qx_t_p)**2+(qy_t[i]-qy_t_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp1[i,indi1]=1
                # print(i, indi1, "a")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_b_p)**2+(qy_b[i]-qy_b_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp2[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "b")
    
            indi1=np.where(np.sqrt(  (qx_t[i]-qx_b_p)**2+(qy_t[i]-qy_b_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp3[i,indi1]=1
                # print(i, indi1, "c")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_t_p)**2+(qy_b[i]-qy_t_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp4[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "d")
        
        
        block_tt=matGGp1
        block_tb=matGGp3
        block_bt=matGGp4
        block_bb=matGGp2
        return [block_tt,block_tb,block_bt, block_bb]
        # return np.bmat([[matGGp1,matGGp3], [matGGp4, matGGp2]])
    
    def trans_WF(self, trans):
        [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]=self.umklapp_lattice_trans( trans)
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        
        [q1,q2,q3]=self.latt.q

        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        matGGp1=np.zeros([Nb,Nb])
        matGGp2=np.zeros([Nb,Nb])
        matGGp3=np.zeros([Nb,Nb])
        matGGp4=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(Nb):

            indi1=np.where(np.sqrt(  (qx_t[i]-qx_t_p)**2+(qy_t[i]-qy_t_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp1[i,indi1]=1
                # print(i, indi1, "a")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_b_p)**2+(qy_b[i]-qy_b_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp2[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "b")
    
            indi1=np.where(np.sqrt(  (qx_t[i]-qx_b_p)**2+(qy_t[i]-qy_b_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp3[i,indi1]=1
                # print(i, indi1, "c")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_t_p)**2+(qy_b[i]-qy_t_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp4[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "d")
        sig0=np.eye(2)
        block_tt=np.kron(matGGp1, sig0)
        block_tb=np.kron(matGGp3, sig0)
        block_bt=np.kron(matGGp4, sig0)
        block_bb=np.kron(matGGp2, sig0)
        return np.bmat([[block_tt,block_tb], [block_bt, block_bb]])
        # return np.bmat([[matGGp1,matGGp3], [matGGp4, matGGp2]])
    
    #untested
    def c2x_psi(self, psi):
        rot=self.latt.C2x
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=paulix
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi
        # return rot2@psi
        
    
    def c2z_psi(self, psi):
        rot=self.latt.C2z
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=paulix
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi
        
    def c3z_psi(self, psi):
        rot=self.latt.C3z
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=pauli0*np.cos(2*np.pi/3)+1j*self.xi*pauliz*np.sin(2*np.pi/3)
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi

        
    #enduntested
        
    def trans_psi(self, psi, dirGM1,dirGM2):
        [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        veccirc1=np.roll(np.eye(Nb,1).flatten(), -dirGM1)
        veccirc2=np.roll(np.eye(Nb,1).flatten(), dirGM1)
        sig0=np.eye(2)
        matt=np.kron(circulant(veccirc1), sig0)
        matb=np.kron(circulant(veccirc2), sig0)
        mat=np.bmat([[matt,matt*0], [matt*0, matb]])



        return mat@psi
    
    def trans_psi2(self, psi, dirGM1,dirGM2):
        
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        Trans=dirGM1*GM1+dirGM2*GM2
        mat = self.trans_WF(Trans)
        ###This is very error prone, I'm assuming two options either a pure wvefunction
        #or a wavefunction where the first index is k , the second is 4N and the third is band
        nind=len(np.shape(psi))
        if nind==2:
            matmul=mat@psi
        if nind==3:
            psimult=[]
            for i in range(np.shape(psi)[0]):
                psimult=psimult+[mat@psi[i,:,:]]
            matmul=np.array(psimult)
        else:
            print("not going to work, check umklapp shift")
            matmul=mat@psi
                
        

        return matmul
        return mat@psi


    def Op_mu_N_sig_psi(self, psi, layer, sublattice, umkpl):
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        pau=[pauli0,paulix,pauliy,pauliz]
        N=self.cuttoff_momentum_lat[3]
        Um=np.eye(N, k=umkpl)

        mat=np.kron(pau[layer],np.kron(Um, pau[sublattice]))
        return  mat@psi
    
    
    def C3unitary(self, psi, sign):
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        pau=[pauli0,paulix,pauliy,pauliz]
        N=self.cuttoff_momentum_lat[3]
        Um=np.eye(N)
        w1=np.exp(sign*1j*(2*np.pi/3)*self.xi)
        w2=np.exp(-sign*1j*(2*np.pi/3)*self.xi)
        # Omega=np.array([[w1,0],[0,w2]])
        Omega=np.array([[0,w2],[w1,0]])


        mat=np.kron(pauli0,np.kron(Um, Omega))
        return  mat@psi

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
        
        plt.plot(bins,valt)
        plt.scatter(bins,valt, s=1)
        
        # plt.ylim([0,8])
        plt.savefig("dos.png")
        plt.close()
        
        
        valt=2*valt
        f2 = interp1d(binn[:-1],valt, kind='cubic')
        de=(bins[1]-bins[0])
        print("sum of the hist, normed?", np.sum(valt)*de)
        

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

        dosarr=np.array(dosl)
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
            
        return [mu, nfil, mus,nn]
        
    
    def ExtendE(self,E_k , umklapp):
        Gu=self.latt.Umklapp_List(umklapp)
        
        Elist=[]
        for GG in Gu:
            Elist=Elist+[E_k]
            
        return np.vstack(Elist)


class Ham_BM_m():
    def __init__(self, hvkd, alpha, xi, latt, kappa, PH):

        self.hvkd = hvkd
        self.alpha= alpha
        self.xi = xi
        
        
        self.latt=latt
        self.kappa=kappa
        self.PH=PH #particle hole symmetry
        self.gap=0#1e-8#artificial gap
        
        #precomputed momentum lattice and interlayer coupling
       
        self.cuttoff_momentum_lat=self.umklapp_lattice()


        self.U=np.matrix(self.InterlayerU())

        #constant shift for the dispersion
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        Vertices_list, Gamma, K, Kp, M, Mp=self.latt.FBZ_points(GM1,GM2)
        self.e0=0
        E,psi=self.eigens( K[2][0],K[2][1], 2)
        self.e0=(E[0]+E[1])/2  #offset to set dirac points at zero energy. If there is a gap the dispersion is just centered at zero
        
    def __repr__(self):
        return "Hamiltonian with alpha parameter {alpha} and scale {hvkd}".format( alpha =self.alpha,hvkd=self.hvkd)


    # METHODS FOR CALCULATING THE DISPERSION 

    def umklapp_lattice(self):

        #large lattice that will be cuttoff
        Numklpx=30
        Numklpy=30
        gridpx=np.arange(-int(Numklpx/2),int(Numklpx/2),1) #grid to calculate wavefunct
        gridpy=np.arange(-int(Numklpy/2),int(Numklpy/2),1) #grid to calculate wavefunct
        n1,n2=np.meshgrid(gridpx,gridpy) #grid to calculate wavefunct

        #q vectors defined as the difference of the K point positions in the two layers
        # and 2pi/3 rotations of that.
        [q1,q2,q3]=self.latt.q
        #getting the momentum lattice to be diagonalized
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        GM=self.latt.GMs
        qx_difb = + GM1[0]*n1 + GM2[0]*n2 + 2*self.xi*q1[0]
        qy_difb = + GM1[1]*n1 + GM2[1]*n2 + 2*self.xi*q1[1]
        valsb = np.sqrt(qx_difb**2+qy_difb**2)
        cutoff=9.*GM*0.7
        ind_to_sum_b = np.where(valsb <cutoff) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array

        #cutoff lattice
        n1_val_b = n1[ind_to_sum_b] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
        n2_val_b = n2[ind_to_sum_b] #
        Nb = np.shape(ind_to_sum_b)[1] ##number of indices for which the condition above is satisfied
        G0xb= GM1[0]*n1_val_b+GM2[0]*n2_val_b #umklapp vectors within the cutoff
        G0yb= GM1[1]*n1_val_b+GM2[1]*n2_val_b #umklapp vectors within the cutoff

        #reciprocal lattices for both layers
        #flipping the order so that same points occur in the same index for plus and minus valleys
        qx_t = -qx_difb[ind_to_sum_b][::int(self.xi)]
        qy_t = -qy_difb[ind_to_sum_b][::int(self.xi)]
        qx_b = qx_difb[ind_to_sum_b][::int(self.xi)]
        qy_b = qy_difb[ind_to_sum_b][::int(self.xi)]


        # plt.scatter(qx_t ,qy_t,c='k', s=np.arange(Nb)+1 )
        # plt.scatter(qx_b ,qy_b , c='r', s=np.arange(Nb)+1 )
        # plt.show()
        return [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t,qx_b, qy_b] 

    def umklapp_lattice_rot(self, rot):
        [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        G0xb_p= rot[0,0]*G0xb + rot[0,1]*G0yb
        G0yb_p= rot[1,0]*G0xb + rot[1,1]*G0yb
        qx_t_p= rot[0,0]*qx_t + rot[0,1]*qy_t
        qy_t_p= rot[1,0]*qx_t + rot[1,1]*qy_t
        qx_b_p= rot[0,0]*qx_b + rot[0,1]*qy_b
        qy_b_p= rot[1,0]*qx_b + rot[1,1]*qy_b
        return [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]

    def umklapp_lattice_trans(self,trans):
        [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        G0xb_p= G0xb + trans[0]
        G0yb_p= G0yb + trans[1]
        qx_t_p= qx_t + trans[0]
        qy_t_p= qy_t +trans[1]
        qx_b_p= qx_b + trans[0]
        qy_b_p= qy_b +trans[1]
        return [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]
    


    def diracH(self, kx, ky):

        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        tau=self.xi
        hvkd=self.hvkd

        Qplusx = qx_t
        Qplusy = qy_t
        Qminx = qx_b
        Qminy = qy_b

        ###checking if particle hole symmetric model is chosen
        if(self.PH):
            # #top layer
            qx_1 = kx - Qplusx
            qy_1 = ky - Qplusy

            # #bottom layer
            qx_2 = kx -Qminx
            qy_2 = ky -Qminy
        else:
            # #top layer
            ROTtop=self.latt.rot_min
            kkx_1 = kx - Qplusx
            kky_1 = ky - Qplusy
            qx_1 = ROTtop[0,0]*kkx_1+ROTtop[0,1]*kky_1
            qy_1 = ROTtop[1,0]*kkx_1+ROTtop[1,1]*kky_1
            # #bottom layer
            ROTbot=self.latt.rot_plus
            kkx_2 = kx -Qminx
            kky_2 = ky -Qminy
            qx_2 = ROTbot[0,0]*kkx_2+ROTbot[0,1]*kky_2
            qy_2 = ROTbot[1,0]*kkx_2+ROTbot[1,1]*kky_2

        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        H1=hvkd*(np.kron(np.diag(qx_1),tau*paulix)+np.kron(np.diag(qy_1),pauliy)) +np.kron(self.gap*np.eye(Nb),pauliz) # ARITCFICIAL GAP ADDED
        H2=hvkd*(np.kron(np.diag(qx_2),tau*paulix)+np.kron(np.diag(qy_2),pauliy)) +np.kron(self.gap*np.eye(Nb),pauliz) # ARITCFICIAL GAP ADDED
        return [H1,H2]
    
    def diracH2(self, kx, ky):

        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        tau=self.xi
        hvkd=self.hvkd



        ###checking if particle hole symmetric model is chosen
        
        # #top layer
        qx_1 = kx 
        qy_1 = ky 
        # #bottom layer
        qx_2 = kx 
        qy_2 = ky 
    
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        H1=hvkd*(qx_1*tau*paulix+qy_1*pauliy) +self.gap*pauliz # ARITCFICIAL GAP ADDED
        H2=hvkd*(qx_1*tau*paulix+qy_1*pauliy) +self.gap*pauliz # ARITCFICIAL GAP ADDED
        return [H1,H2]


    def InterlayerU(self):
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        tau=self.xi
        [q1,q2,q3]=self.latt.q
        

        Qplusx = qx_t
        Qplusy = qy_t
        Qminx = qx_b
        Qminy = qy_b

        matGq1=np.zeros([Nb,Nb])
        matGq2=np.zeros([Nb,Nb])
        matGq3=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(Nb):

            indi1=np.where(np.sqrt(  (Qplusx[i]-Qminx + tau*q1[0])**2+(Qplusy[i]-Qminy + tau*q1[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq1[i,indi1]=1

            indi1=np.where(np.sqrt(  (Qplusx[i]-Qminx + tau*q2[0])**2+(Qplusy[i]-Qminy+ tau*q2[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq2[i,indi1]=1 #indi1+1=i
    
            indi1=np.where(np.sqrt(  (Qplusx[i]-Qminx + tau*q3[0])**2+(Qplusy[i]-Qminy + tau*q3[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq3[i,indi1]=1


        #Matrices that  appeared as coefficients of the real space ops
        #full product is the kronecker product of both matrices

        Mdelt1=matGq1
        Mdelt2=matGq2
        Mdelt3=matGq3
       

        w0=self.kappa
        w1=1
        phi = 2*np.pi/3    
        z = np.exp(-1j*phi*tau)
        zs = np.exp(1j*phi*tau)
        
        T1 = np.array([[w0,w1],[w1,w0]])
        T2 = zs*np.array([[w0,w1*zs],[w1*z,w0]])
        T3 = z*np.array([[w0,w1*z],[w1*zs,w0]])


        #different basis
        # T1 = np.array([[w0,w1],[w1,w0]])
        # T2 = np.array([[w0,w1*zs],[w1*z,w0]])
        # T3 = np.array([[w0,w1*z],[w1*zs,w0]])

        U=self.hvkd*self.alpha*( np.kron(Mdelt1,T1) + np.kron(Mdelt2,T2)+ np.kron(Mdelt3,T3)) #interlayer coupling

        return U
        
    def eigens(self, kx,ky, nbands):
        
        U=self.U
        Udag=U.H
        [H1,H2]=self.diracH( kx, ky)
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H2, U ], [Udag, H1]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]

        for nband in range(nbands):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1m;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)
            

        return Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi
    
    def eigens_dirac(self, kx,ky, nbands):
        
        U=self.U*0
        Udag=U.H
        P=self.PH
        self.PH=True
        [H1,H2]=self.diracH( kx, ky)
        self.PH=P
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H2, U ], [Udag, H1]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]

        for nband in range(nbands):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1m;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)
            

        return Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi
    
    def eigens_dirac2(self, kx,ky, nbands):
        
        [H1,H2]=self.diracH2( kx, ky)
        
        Hxi=np.bmat([[H2, H2*0 ], [H2*0, H1]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,:]
        nbands_D=2

        for nband in range(nbands_D):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1m;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)
            

        return Eigvals[1:3]-self.e0, psi[:,1:3]
    
    def eigens_dirac3(self, kx,ky, nbands):
        e1=np.array([np.cos(2*np.pi*1/3), np.sin(2*np.pi*1/3)])*2*2*np.pi*1/3/np.sqrt(3)
        e2=np.array([np.cos(2*np.pi*2/3), np.sin(2*np.pi*2/3)])*2*2*np.pi*1/3/np.sqrt(3)
        e3=np.array([np.cos(2*np.pi*0/3), np.sin(2*np.pi*0/3)])*2*2*np.pi*1/3/np.sqrt(3)
        
        W3=0.00375/3  #in ev
        k=np.array([kx,ky])
        
        hk=np.exp(1j*k@e1)+np.exp(1j*k@e2)+np.exp(1j*k@e3)
        hk_n=np.abs(W3*hk)
        # print(k@e1,k@e2,k@e3,hk_n, 2*np.pi*1/3)
        psi1=np.array([+1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)
        psi2=np.array([-1*np.exp(1j*np.angle(hk)), 1])/np.sqrt(2)

        return np.array([-hk_n,hk_n]), np.array([psi2,psi1])


    def parallel_eigens(self,  nbands,q):
        kx,ky=q[0], q[1]
        U=self.U
        Udag=U.H
        [H1,H2]=self.diracH( kx, ky)
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H2, U ], [Udag, H1]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]

        for nband in range(nbands):
            psi_p=psi[:,nband]
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1m;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)
            

        return [Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi]


    ### FERMI SURFACE ANALYSIS

    #creates a square grid, interpolates and finds contours at fixed mu
    def FSinterp(self, save_d, read_d, mu):

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
            E1,wave1=self.eigens(kkk[0], kkk[1],nbands)
            cois=[E1,wave1]
            spect.append(np.real(cois[0]))

        Edisp=np.array(spect)
        if save_d:
            with open('Edisp_'+str(Nsamp)+'.npy', 'wb') as f:
                np.save(f, Edisp)

        if read_d:
            print("Loading  ..........")
            with open('Edisp_'+str(Nsamp)+'.npy', 'rb') as f:
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
    def FS_contour(self, Np, mu):
        #option for saving the square grid dispersion
        save_d=False
        read_d=False
        [f_interp,k_window_sizex,k_window_sizey]=self.FSinterp( save_d, read_d, mu)
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


    #METHODS FOR MANIPULATING WAVEFUNCTIONS AND FORM FACTORS
    #for reference the pattern of kronecker prod is 
    # layer x umklp x sublattice
    def rot_WF(self, rot):
        [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]=self.umklapp_lattice_rot( rot)
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        
        [q1,q2,q3]=self.latt.q

        

        matGGp1=np.zeros([Nb,Nb])
        matGGp2=np.zeros([Nb,Nb])
        matGGp3=np.zeros([Nb,Nb])
        matGGp4=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(Nb):

            indi1=np.where(np.sqrt(  (qx_t[i]-qx_t_p)**2+(qy_t[i]-qy_t_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp1[i,indi1]=1
                # print(i, indi1, "a")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_b_p)**2+(qy_b[i]-qy_b_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp2[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "b")
    
            indi1=np.where(np.sqrt(  (qx_t[i]-qx_b_p)**2+(qy_t[i]-qy_b_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp3[i,indi1]=1
                # print(i, indi1, "c")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_t_p)**2+(qy_b[i]-qy_t_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp4[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "d")
        
        
        block_tt=matGGp1
        block_tb=matGGp3
        block_bt=matGGp4
        block_bb=matGGp2
        return [block_tt,block_tb,block_bt, block_bb]
        # return np.bmat([[matGGp1,matGGp3], [matGGp4, matGGp2]])
    
    def trans_WF(self, trans):
        [ G0xb_p, G0yb_p , ind_to_sum_b, Nb, qx_t_p, qy_t_p, qx_b_p, qy_b_p]=self.umklapp_lattice_trans( trans)
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat
        
        [q1,q2,q3]=self.latt.q

        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        matGGp1=np.zeros([Nb,Nb])
        matGGp2=np.zeros([Nb,Nb])
        matGGp3=np.zeros([Nb,Nb])
        matGGp4=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(Nb):

            indi1=np.where(np.sqrt(  (qx_t[i]-qx_t_p)**2+(qy_t[i]-qy_t_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp1[i,indi1]=1
                # print(i, indi1, "a")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_b_p)**2+(qy_b[i]-qy_b_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp2[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "b")
    
            indi1=np.where(np.sqrt(  (qx_t[i]-qx_b_p)**2+(qy_t[i]-qy_b_p)**2  )<tres)
            if np.size(indi1)>0:
                matGGp3[i,indi1]=1
                # print(i, indi1, "c")

            indi1=np.where(np.sqrt(  (qx_b[i]-qx_t_p)**2+(qy_b[i]-qy_t_p)**2   )<tres)
            if np.size(indi1)>0:
                matGGp4[i,indi1]=1 #indi1+1=i
                # print(i, indi1, "d")
        sig0=np.eye(2)
        block_tt=np.kron(matGGp1, sig0)
        block_tb=np.kron(matGGp3, sig0)
        block_bt=np.kron(matGGp4, sig0)
        block_bb=np.kron(matGGp2, sig0)
        return np.bmat([[block_tt,block_tb], [block_bt, block_bb]])
        # return np.bmat([[matGGp1,matGGp3], [matGGp4, matGGp2]])
    
    #untested
    def c2x_psi(self, psi):
        rot=self.latt.C2x
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=paulix
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi
        # return rot2@psi
        
    
    def c2z_psi(self, psi):
        rot=self.latt.C2z
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=paulix
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi
        
    def c3z_psi(self, psi):
        rot=self.latt.C3z
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=pauli0*np.cos(2*np.pi/3)+1j*self.xi*pauliz*np.sin(2*np.pi/3)
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi

        
    #enduntested
        
    def trans_psi(self, psi, dirGM1,dirGM2):
        [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        veccirc1=np.roll(np.eye(Nb,1).flatten(), -dirGM1)
        veccirc2=np.roll(np.eye(Nb,1).flatten(), dirGM1)
        sig0=np.eye(2)
        matt=np.kron(circulant(veccirc1), sig0)
        matb=np.kron(circulant(veccirc2), sig0)
        mat=np.bmat([[matt,matt*0], [matt*0, matb]])



        return mat@psi
    
    def trans_psi2(self, psi, dirGM1,dirGM2):
        
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        Trans=-dirGM1*GM1-dirGM2*GM2
        mat = self.trans_WF(Trans)
        
        nind=len(np.shape(psi))
        ###This is very error prone, I'm assuming two options either a pure wvefunction
        #or a wavefunction where the first index is k , the second is 4N and the third is band
        if nind==2:
            matmul=mat@psi
        if nind==3:
            psimult=[]
            for i in range(np.shape(psi)[0]):
                psimult=psimult+[mat@psi[i,:,:]]
            matmul=np.array(psimult)
        else:
            print("not going to work, check umklapp shift")
            matmul=mat@psi
                
        

        return matmul

    def Op_mu_N_sig_psi(self, psi, layer, sublattice, umkpl):
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        pau=[pauli0,paulix,pauliy,pauliz]
        N=self.cuttoff_momentum_lat[3]
        Um=np.eye(N, k=umkpl)

        mat=np.kron(pau[layer],np.kron(Um, pau[sublattice]))
        return  mat@psi
    
    def C3unitary(self, psi, sign):
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        pau=[pauli0,paulix,pauliy,pauliz]
        N=self.cuttoff_momentum_lat[3]
        Um=np.eye(N)
        w1=np.exp(sign*1j*(2*np.pi/3)*self.xi)
        w2=np.exp(-sign*1j*(2*np.pi/3)*self.xi)
        Omega=np.array([[w1,0],[0,w2]])


        mat=np.kron(pauli0,np.kron(Um, Omega))
        return  mat@psi

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
        
        plt.plot(bins,valt)
        plt.scatter(bins,valt, s=1)
        
        # plt.ylim([0,8])
        plt.savefig("dos.png")
        plt.close()
        
        
        valt=2*valt
        f2 = interp1d(binn[:-1],valt, kind='cubic')
        de=(bins[1]-bins[0])
        print("sum of the hist, normed?", np.sum(valt)*de)
        

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

        dosarr=np.array(dosl)
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
            
        return [mu, nfil, mus,nn]
        
    
    def ExtendE(self,E_k , umklapp):
        Gu=self.latt.Umklapp_List(umklapp)
        
        Elist=[]
        for GG in Gu:
            Elist=Elist+[E_k]
            
        return np.vstack(Elist)
    


class FormFactors():
    def __init__(self, psi, xi, lat, umklapp):
        self.psi = psi #has dimension #kpoints, 4*N, nbands
        self.lat=lat
        self.cpsi=np.conj(psi)
        self.xi=xi
        self.Nu=int(np.shape(self.psi)[1]/4) #4, 2 for sublattice and 2 for layer

        
        [KX,KY]=lat.Generate_lattice()
        [KXu,KYu]=lat.Generate_Umklapp_lattice(KX,KY,umklapp)
        
        #G processes
        self.MGS_1=[[0,1],[1,0],[0,-1],[-1,0],[-1,-1],[1,1]] #1G
        self.MGS1=self.MGS_1+[[-1,-2],[-2,-1],[-1,1],[1,2],[2,1],[1,-1]] #1G and possible corners
        self.MGS_2=self.MGS1+[[-2,-2],[0,-2],[2,0],[2,2],[0,2],[-2,0]] #2G
        self.MGS2=self.MGS_2+[[-2,-3],[-1,-3],[1,-2],[2,-1],[3,1],[3,2],[2,3],[1,3],[-1,2],[-2,1],[-3,-1],[-3,-2]] #2G and possible corners
        self.MGS_3=self.MGS2+[[-3,-3],[0,-3],[3,0],[3,3],[0,3],[-3,0]] #3G
        
        if umklapp==1:
            GSu=self.MGS_1
        elif umklapp==2:
            GSu=self.MGS_2
        elif umklapp==3:
            GSu=self.MGS_3
        else:
            GSu=[]

        

        
        self.kx=KXu
        self.ky=KYu
        [self.KQX, self.KQY, self.Ik]=lat.Generate_momentum_transfer_umklapp_lattice( KX, KY,  KXu, KYu)
    
        #momentum transfer lattice
        kqx1, kqx2=np.meshgrid(self.KQX,self.KQX)
        kqy1, kqy2=np.meshgrid(self.KQY,self.KQY)
        self.qx=kqx1-kqx2
        self.qy=kqy1-kqy2
        self.q=np.sqrt(self.qx**2+self.qy**2)+1e-17
            

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
        
        # print("shapes in all the matrices being mult", np.shape(mat), np.shape(self.psi), np.shape(mat@self.psi))
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
    
    ########### Functions for the nematic form factors
    def f(self, FF ):
        farr= np.ones(np.shape(FF))
        for k_i in range(np.size(self.kx)):
            for k_ip in range(np.size(self.kx)):
                qx=self.kx[k_i]-self.kx[k_ip]
                qy=self.ky[k_i]-self.ky[k_ip]
                q=np.sqrt(qx**2+qy**2)+1e-17
                for i in range(np.shape(FF)[1]):
                    for j in range(np.shape(FF)[1]):
                        farr[k_i, i, k_ip, j]=(qx**2-qy**2)/q
        return farr

    def g(self,FF):
        garr= np.ones(np.shape(FF))
        for k_i in range(np.size(self.kx)):
            for k_ip in range(np.size(self.kx)):
                qx=self.kx[k_i]-self.kx[k_ip]
                qy=self.ky[k_i]-self.ky[k_ip]
                q=np.sqrt(qx**2+qy**2)+1e-17
                for i in range(np.shape(FF)[1]):
                    for j in range(np.shape(FF)[1]):
                        garr[k_i, i, k_ip, j]=2*(qx*qy)/q
        return garr 



    def h(self,FF):

        harr= np.ones(np.shape(FF))
        for k_i in range(np.size(self.kx)):
            for k_ip in range(np.size(self.kx)):
                qx=self.kx[k_i]-self.kx[k_ip]
                qy=self.ky[k_i]-self.ky[k_ip]
                q=np.sqrt(qx**2+qy**2)+1e-17
                for i in range(np.shape(FF)[1]):
                    for j in range(np.shape(FF)[1]):
                        harr[k_i, i, k_ip, j]=q
        return harr


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
        qx=self.KQX[1]-self.KQX[0]
        qy=self.KQY[1]-self.KQY[0]
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
    def denFF_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return L30

    def denFFL_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return self.h(L30)*L30


    def NemFFL_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFL=self.f(L31) *L31-self.xi*self.g(L32)*L32
        return Nem_FFL
    
    def NemFFL_a_plus(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFL=self.f(L31) *L31+self.xi*self.g(L32)*L32
        return Nem_FFL

    def NemFFT_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFT=-self.g(L31) *L31- self.xi*self.f(L32)*L32
        return Nem_FFT

    ########### Symmetric displacement of the layers
    def denFF_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return L00

    def denFFL_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return self.h(L00)*L00

    def NemFFL_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFL=self.f(L01) *L01-self.xi*self.g(L02)*L02
        return Nem_FFL

    def NemFFT_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFT=-self.g(L01)*L01 - self.xi*self.f(L02)*L02
        return Nem_FFT

    

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
    
    def NemqFFL_a_plus(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFL=self.fq(L31) *L31+self.xi*self.gq(L32)*L32
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
    
    
    def Fdirac(self):

        phi=np.angle(self.KQX+1j*self.KQY)
        F=np.zeros([np.size(phi), 2, np.size(phi), 2])
        phi1, phi2=np.meshgrid(phi,phi)
        for n in range(2):
            for n2 in range(2):
                F[:,n,:,n2]=np.sqrt( ( 1+np.cos(phi1-phi2)*(-1)**(n+n2))/2  +1e-17*1j)
                
        return F
                

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
    hbvf=0.003404*0.1973269804*1e-6 /a_graphene #ev*m
    # hbvf = 2.1354; # eV
    theta=1.05*np.pi/180  #1.05*np.pi/180 #twist Angle
    nbands=4 #Number of bands 
    Nsamp=int(sys.argv[2])
    kappa_p=0.0797/0.0975;
    kappa=kappa_p;
    up = 0.0975; # eV
    u = kappa*up; # eV



    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0)
    ln=MoireLattice.MoireTriangLattice(Nsamp,theta,1)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2) #this one
    [KX,KY]=lq.Generate_lattice()
    # plt.scatter(KX,KY)
    # plt.show()
    Npoi=np.size(KX)
    [q1,q1,q3]=l.q
    q=la.norm(q1)
    [GM1,GM2]=lq.GMvec

    hvkd=hbvf*q
    Kvec=(2*lq.b[0,:]+lq.b[1,:])/3 
    K=la.norm(Kvec)
    GG=la.norm(l.b[0,:])
    print(q , 2*K*np.sin(theta/2))


    #Various alpha values
    hvfK_andrei=19.81
    #andreis params
    w=0.110 #in ev
    hvfkd_andrei=hvfK_andrei*np.sin(theta/2) #wrong missing 1/sqrt3
    alpha_andrei=w/hvfkd_andrei
    alpha=w/hvkd
    alpha_andrei_corrected=(np.sqrt(3)/2)*w/hvfkd_andrei
    #magic angles
    amag1=0.5695
    amag2=0.605
    amag3=0.65
    #angle with flat band in the chiral limit
    alph2= 0.5856
    PH=True



    xi=1
    #kosh params realistic  -- this is the closest to the actual Band Struct used in the paper
    hbvf = 2.1354; # eV
    hvkd=hbvf*q
    kappa_p=0.0797/0.0975
    kappa=kappa_p
    up = 0.0975; # eV
    u = kappa*up; # eV
    alpha=up/hvkd
    alph=alpha
    
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
    
        
    # # #################################
    # # #################################
    # # #################################
    # # # Form factors C3
    # # #################################
    # # #################################
    # # #################################

    Ene_valley_plus_a=np.empty((0))
    Ene_valley_plus_ac3=np.empty((0))
    psi_plus_a=[]
    psi_plus_ac3=[]

    rot_C2z=lq.C2z
    rot_C3z=lq.C3z

    [KQX, KQY, Ik]=lq.Generate_momentum_transfer_lattice( KX, KY)
    umkl=1
    [KXu,KYu]=lq.Generate_Umklapp_lattice2(KX,KY,umkl)
    [KQXu, KQYu, Ik]=lq.Generate_momentum_transfer_umklapp_lattice( KX, KY,  KXu, KYu)
    Npoi_u=np.size(KXu)
    plt.scatter(KQX, KQY)
    plt.scatter(KXu, KYu)
    plt.scatter(KX, KY)
    plt.savefig("fig0.png")
    plt.close()

    [KXc3z,KYc3z, Indc3z]=lq.C3zLatt(KXu,KYu)
    # print("starting dispersion ..........")
    # # for l in range(Nsamp*Nsamp):
    s=time.time()
    hpl=Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
    # hpl=Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
    overlaps=[]
    nbands=2 
    '''
    #testing the wavefunction on rotation and translation
    l=35
    shi1=int(0)
    shi2=int(3)
    vecT=shi1*GM1 +shi2*GM2
    
    plt.scatter(KXu,KYu)
    plt.scatter(KX,KY)
    plt.scatter(KX[l],KY[l])
    plt.scatter(KX[l]+vecT[0],KY[l]+vecT[1])
    plt.savefig("latpoint.png")
    plt.close()
    
    print("the first umklapp vector is ",GM1)
    print("the second umklapp vector is ",GM2)
    E1p,wave1p=hpl.eigens(KX[l],KY[l],nbands)
    E1p2,wave1p2=hpl.eigens(KX[l]+vecT[0],KY[l]+vecT[1],nbands)
    wave1p3=hpl.trans_psi2(wave1p, shi1,shi2)
    print(E1p2, E1p, np.shape(wave1p))
    
   
    
    print("gauge fixing working??")
    maxind1=np.unravel_index(np.argmax(np.abs(wave1p[:,1]), axis=None), wave1p[:,1].shape)[0]
    print(wave1p[maxind1,1], maxind1)
    maxind2=np.unravel_index(np.argmax(np.abs(wave1p2[:,1]), axis=None), wave1p2[:,1].shape)[0]
    print(wave1p2[maxind2, 1], maxind2)
    maxind3=np.unravel_index(np.argmax(np.abs(wave1p3[:,1]), axis=None), wave1p3[:,1].shape)[0]
    print(wave1p3[maxind3, 1], maxind3)
    
    
    # plt.plot(np.real(wave1p[:,1])-np.real(wave1p3[:,1]))
    plt.plot(np.real(wave1p2[:,1]))
    plt.plot(np.real(wave1p3[:,1]))
    plt.savefig("wavesre.png")
    plt.close()
    # plt.plot(np.imag(wave1p[:,1])-np.imag(wave1p3[:,1]))
    plt.plot(np.imag(wave1p2[:,1]))
    plt.plot(np.imag(wave1p3[:,1]))
    plt.savefig("wavesim.png")
    plt.close()
    # plt.plot(np.abs(wave1p[:,1])-np.abs(wave1p3[:,1]))
    plt.plot(np.abs(wave1p2[:,1]))
    plt.plot(np.abs(wave1p3[:,1]))
    plt.axvline(maxind2)
    plt.axvline(maxind1)
    plt.savefig("wavesabs.png")
    plt.close()
    
    plt.plot(np.diag(np.abs( np.conj(wave1p2.T)@wave1p3 )))
    # plt.ylim(0.5,1.5)
    # plt.plot(np.abs(wave1p2[:,1]))
    # plt.plot(np.abs(wave1p3[:,1]))
    plt.savefig("waveinner.png")
    plt.close()
    
    print("before shift \n", np.conj(wave1p.T)@wave1p2)
    print("after shift \n",np.conj(wave1p3.T)@wave1p2 ) 
    print("after shift abs \n",np.abs(np.conj(wave1p3.T)@wave1p2 )) 
    
    
    MGS_1=[[0,1],[1,0],[0,-1],[-1,0],[-1,-1],[1,1]] #1G
    MGS1=MGS_1+[[-1,-2],[-2,-1],[-1,1],[1,2],[2,1],[1,-1]] #1G and possible corners
    MGS_2=MGS1+[[-2,-2],[0,-2],[2,0],[2,2],[0,2],[-2,0]] #2G
    MGS2=MGS_2+[[-2,-3],[-1,-3],[1,-2],[2,-1],[3,1],[3,2],[2,3],[1,3],[-1,2],[-2,1],[-3,-1],[-3,-2]] #2G and possible corners
    MGS_3=MGS2+[[-3,-3],[0,-3],[3,0],[3,3],[0,3],[-3,0]] #3G
    inner2=[]
    dist=[]
    for MG in MGS_3:
        l=35
        shi1=int(MG[0])
        shi2=int(MG[1])
        vecT=shi1*GM1 +shi2*GM2
        E1p,wave1p=hpl.eigens(KX[l],KY[l],nbands)
        E1p2,wave1p2=hpl.eigens(KX[l]+vecT[0],KY[l]+vecT[1],nbands)
        wave1p3=hpl.trans_psi2(wave1p, shi1,shi2)
        print(MG)
        # print("before shift \n", np.conj(wave1p.T)@wave1p2)
        # print("after shift \n",np.conj(wave1p3.T)@wave1p2 ) 
        inner=np.abs(np.conj(wave1p3.T)@wave1p2 )
        inner2.append(inner[0,0])
        inner2.append(inner[1,1])
        dist.append(np.sqrt(shi1**2+shi2**2))
        dist.append(np.sqrt(shi1**2+shi2**2))
        print("after shift abs \n",inner) 
        
    plt.scatter(dist,inner2)
    plt.savefig("umklapp_overlaps_"+str(KX[l])+"_"+str(KY[l])+".png")
    plt.close()
    FFp=FormFactors_umklapp(wave1p2, 1, lq,umkl, hpl)
    '''
    
    
    
    '''
    #testing umklapp form factors
    
    umkl=2
    [KXu,KYu]=lq.Generate_Umklapp_lattice2(KX,KY,umkl)
    [KXc3z,KYc3z, Indc3z]=lq.C3zLatt(KXu,KYu)
    Npoi_u=np.size(KXu)
    
    
    
    
    for l in range(Npoi):
        E1p,wave1p=hpl.eigens(KX[l],KY[l],nbands)
        Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
        psi_plus_a.append(wave1p)

    e=time.time()
    print("time to diag over MBZ", e-s)
    ##relevant wavefunctions and energies for the + valley
    psi_plus=np.array(psi_plus_a)
    Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])

    plt.scatter(KX,KY,c=Ene_valley_plus[:,1])
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("fig1.png")
    plt.close()


    FFp=FormFactors_umklapp(psi_plus, 1, lq,umkl, hpl)
    L00p=FFp.NemqFFL_a()

    print(np.shape(L00p) )
    
    
    diffar=[]
    K=[]
    KP=[]
    cos=[]
    cos2=[]

    kp=np.argmin(KXu**2+KYu**2)
    for k in range(Npoi_u):
        undet=np.abs(np.linalg.det(L00p[k,:,kp,:]))
        dosdet=np.abs(np.linalg.det(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
        diffar.append( undet - dosdet )
        cos.append(undet)
        cos2.append(dosdet)
        print(undet, dosdet)

    plt.plot(diffar)
    plt.savefig("fig3.png")
    plt.close()

    plt.scatter(KXu,KYu,c=cos)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("fig4.png")
    plt.close()

    plt.scatter(KXu,KYu,c=cos2)
    plt.colorbar()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("fig5.png")
    plt.close()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax = plt.axes(projection='3d')

    ax.scatter3D(KXu,KYu,cos, c=cos);
    plt.savefig("fig6.png")
    plt.close()
    '''

    
    # for l in range(Npoi_u):
    #     E1p,wave1p=hpl.eigens(KXu[l],KYu[l],nbands)
    #     Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
    #     psi_plus_a.append(wave1p)


    #     E1p_c3z,wave1p_c3z=hpl.eigens(KXc3z[l],KYc3z[l],nbands)
    #     wave1p_c3z_rot=hpl.c3z_psi( wave1p_c3z)
    #     Ene_valley_plus_ac3=np.append(Ene_valley_plus_ac3,E1p_c3z)
    #     psi_plus_ac3.append(wave1p_c3z)


    # e=time.time()
    # print("time to diag over MBZ", e-s)
    # ##relevant wavefunctions and energies for the + valley
    # psi_plus=np.array(psi_plus_a)
    # Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi_u,nbands])

    # psi_plusc3=np.array(psi_plus_ac3)
    # Ene_valley_plusc3= np.reshape(Ene_valley_plus_ac3,[Npoi_u,nbands])


    # plt.scatter(KXu,KYu,c=Ene_valley_plus[:,1])
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("fig1.png")
    # plt.close()

    # plt.scatter(KXu,KYu,c=Ene_valley_plusc3[:,1])
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("fig2.png")
    # plt.close()
    # print(np.shape(psi_plus),np.shape(psi_plusc3))

    # FFp=FormFactors(psi_plus, 1, lq,umkl)
    # L00p=FFp.NemFFL_a()
    # FFc3=FormFactors(psi_plusc3, 1, lq,umkl)
    # L00m=FFc3.NemFFL_a()
    # print(np.shape(L00p),np.shape(L00m) )



    # ind0=np
    # #####transpose complex conj plus


    # diffar=[]
    # K=[]
    # KP=[]
    # cos=[]
    # cos2=[]

    # kp=np.argmin(KXu**2+KYu**2)
    # for k in range(Npoi_u):
    #     # plt.scatter(KX,KY)
    #     # plt.plot(KX[kp],KY[kp], 'o', c='r' )
    #     # plt.plot(KX[k],KY[k], 'o' , c='g')
    #     # plt.plot(KXc3z[k],KYc3z[k], 'o' , c='orange')
    #     # plt.plot(KX[int(Indc3z[k])],KY[int(Indc3z[k])],'o', c='k' )
    #     # plt.show()
    #     undet=np.abs(np.linalg.det(L00p[k,:,kp,:]))
    #     dosdet=np.abs(np.linalg.det(L00p[int(Indc3z[k]),:,int(Indc3z[kp]),:]))
    #     diffar.append( undet - dosdet )
    #     cos.append(undet)
    #     cos2.append(dosdet)
    #     print(undet, dosdet)

    # plt.plot(diffar)
    # plt.savefig("fig3.png")
    # plt.close()

    # plt.scatter(KXu,KYu,c=cos)
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("fig4.png")
    # plt.close()

    # plt.scatter(KXu,KYu,c=cos2)
    # plt.colorbar()
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.savefig("fig5.png")
    # plt.close()

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax = plt.axes(projection='3d')

    # ax.scatter3D(KXu,KYu,cos, c=cos);
    # plt.savefig("fig6.png")
    # plt.close()
    #######################

    ########################
    
    # #################################
    # #################################
    # #################################
    # # Disp cut along high symmetry directions
    # #################################
    # #################################
    # #################################

    Ene_valley_plus_a=np.empty((0))
    Ene_valley_min_a=np.empty((0))
    psi_plus_a=[]
    psi_min_a=[]
    rot_C2z=lq.C2z
    rot_C3z=lq.C3z

    nbands=14 #Number of bands 
    kpath=lq.High_symmetry_path()

    Npoi=np.shape(kpath)[0]
    hpl=Ham_BM_p(hvkd, alph, 1, lq,kappa,PH)
    hmin=Ham_BM_m(hvkd, alph, -1, lq,kappa,PH)
    print("kappa is..", kappa)
    print("alpha is..", alph)
    Edif=[]
    overlaps=[]
    for l in range(Npoi):
        # h.umklapp_lattice()
        # break
        E1p,wave1p=hpl.eigens(kpath[l,0],kpath[l,1],nbands)
        Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
        psi_plus_a.append(wave1p)


        E1m,wave1m=hmin.eigens(kpath[l,0],kpath[l,1],nbands)
        Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
        psi_min_a.append(wave1m)

    Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
    Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

 

    print(np.shape(Ene_valley_plus_a))
    qa=np.linspace(0,1,Npoi)
    for i in range(nbands):
        plt.plot(qa,Ene_valley_plus[:,i] , c='b')
        plt.plot(qa,Ene_valley_min[:,i] , c='r', ls="--")
    plt.xlim([0,1])
    plt.ylim([-0.009,0.009])
    plt.savefig("highsym.png")
    plt.show()


if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
