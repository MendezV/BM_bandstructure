import numpy as np
import MoireLattice
import matplotlib.pyplot as plt
from scipy import interpolate
import time
 

class Ham_BM_p():
    def __init__(self, hvkd, alpha, xi, latt, kappa, PH):

        self.hvkd = hvkd
        self.alpha= alpha
        self.xi = xi
        
        
        self.latt=latt
        self.kappa=kappa
        self.PH=PH #particle hole symmetry
        
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
        cutoff=7.*GM*0.7
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
        
        H1=hvkd*(np.kron(np.diag(qx_1),tau*paulix)+np.kron(np.diag(qy_1),pauliy)) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        H2=hvkd*(np.kron(np.diag(qx_2),tau*paulix)+np.kron(np.diag(qy_2),pauliy)) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
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
            maxisind = np.unravel_index(np.argmax(np.abs(psi_p), axis=None), psi_p.shape)
            # print("wave1p;",psi_p[maxisind])
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)

        return Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi



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
        
        block_tt=matGGp1
        block_tb=matGGp3
        block_bt=matGGp4
        block_bb=matGGp2
        return np.bmat([[block_tt,block_tb], [block_bt, block_bb]])
        # return np.bmat([[matGGp1,matGGp3], [matGGp4, matGGp2]])
    
    def Op_rot_psi(self, psi, rot):

        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        rot_mat = self.rot_WF(rot)
        mat=np.kron(rot_mat,paulix)
        # print("determinant ", np.linalg.det(rot_mat))
        plt.imshow(mat)
        plt.show()
        


        return mat@psi
        # return rot2@psi

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

class Ham_BM_m():
    def __init__(self, hvkd, alpha, xi, latt, kappa, PH):

        self.hvkd = hvkd
        self.alpha= alpha
        self.xi = xi
        
        
        self.latt=latt
        self.kappa=kappa
        self.PH=PH #particle hole symmetry
        
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
        cutoff=7.*GM*0.7
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
        
        H1=hvkd*(np.kron(np.diag(qx_1),tau*paulix)+np.kron(np.diag(qy_1),pauliy)) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        H2=hvkd*(np.kron(np.diag(qx_2),tau*paulix)+np.kron(np.diag(qy_2),pauliy)) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
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
        
        block_tt=matGGp1
        block_tb=matGGp3
        block_bt=matGGp4
        block_bb=matGGp2
        return np.bmat([[block_tt,block_tb], [block_bt, block_bb]])
        # return np.bmat([[matGGp1,matGGp3], [matGGp4, matGGp2]])
    
    def Op_rot_psi(self, psi, rot):

        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])

        rot_mat = self.rot_WF(rot)
        mat=np.kron(rot_mat,pauli0)
        # print("determinant ", np.linalg.det(rot_mat))
        # plt.imshow(mat)
        # plt.show()
        


        return mat@psi
        # return rot2@psi

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


class FormFactors():
    def __init__(self, psi, xi, lat):
        self.psi = psi #has dimension #kpoints, 4*N, nbands
        self.lat=lat
        self.cpsi=np.conj(psi)
        self.xi=xi
        self.Nu=int(np.shape(self.psi)[1]/4) #4, 2 for sublattice and 2 for layer
        [KX,KY]=lat.Generate_lattice()
        self.kx=KX
        self.ky=KY

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
        return  mat@self.psi

    def calcFormFactor(self, layer, sublattice):
        s=time.time()
        print("calculating tensor that stores the overlaps........")
        mult_psi=self.matmult(layer,sublattice)
        Lambda_Tens=np.tensordot(self.cpsi,mult_psi, axes=([1],[1]))
        e=time.time()
        print("finsihed the overlaps..........", e-s)
        return(Lambda_Tens)

    def f(self):
        q=np.sqrt(self.kx**2+self.ky**2)
        return (self.kx**2-self.ky**2)/q

    def g(self):
        q=np.sqrt(self.kx**2+self.ky**2)
        return 2*(self.kx*self.ky)/q

    def h(self):
        q=np.sqrt(self.kx**2+self.ky**2)
        return q

    ########### Anti-symmetric displacement of the layers
    def denFF_a(self):
        L30=self.calcFormFactor( layer=3, sublattice=0)
        return L30

    def NemFFL_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFL=self.f *L31-self.xi*self.g*L32
        return Nem_FFL

    def NemFFT_a(self):
        L31=self.calcFormFactor( layer=3, sublattice=1)
        L32=self.calcFormFactor( layer=3, sublattice=2)
        Nem_FFT=-self.g *L31- self.xi*self.f*L32
        return Nem_FFT
    ########### Symmetric displacement of the layers
    def denFF_s(self):
        L00=self.calcFormFactor( layer=0, sublattice=0)
        return L00

    def NemFFL_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFL=self.f *L01-self.xi*self.g*L02
        return Nem_FFL

    def NemFFT_s(self):
        L01=self.calcFormFactor( layer=0, sublattice=1)
        L02=self.calcFormFactor( layer=0, sublattice=2)
        Nem_FFT=-self.g *L01- self.xi*self.f*L02
        return Nem_FFT

        