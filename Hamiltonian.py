import numpy as np
import MoireLattice
import matplotlib.pyplot as plt


#TODO: make momentum lattice an attribute of the class and make the solver a function of kx and ky

class Ham():
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
        
    def __repr__(self):
        return "Hamiltonian at {kx} {ky} with alpha parameter {alpha} and scale {hvkd}".format(kx=self.kx, ky=self.ky, alpha =self.alpha,hvkd=self.hvkd)

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
        qx_t = -qx_difb[ind_to_sum_b]
        qy_t = -qy_difb[ind_to_sum_b]
        qx_b = qx_difb[ind_to_sum_b]
        qy_b = qy_difb[ind_to_sum_b]
        return [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]

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
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.cuttoff_momentum_lat

        
        U=self.U
        Udag=U.H
        [H1,H2]=self.diracH( kx, ky)
        N =np.shape(U)[0]
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
        (Eigvals,psi)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######HANDLING WITH RESHAPE
        #umklp,umklp, layer, sublattice
        # psi_p=np.zeros([self.Numklpy,self.Numklpx,2,2]) +0*1j
        # psi=np.zeros([self.Numklpy, self.Numklpx, 2,2, self.nbands]) +0*1j

        # for nband in range(self.nbands):
        #     # print(np.shape(ind_to_sum), np.shape(psi_p), np.shape(psi_p[ind_to_sum]))
        #     psi_p[ind_to_sum] = np.reshape(  np.array( np.reshape(Eigvect[:,2*N-int(self.nbands/2)+nband] , [4, N])  ).T, [N, 2, 2] )    

        #     ##GAUGE FIXING by making the 30 30 1 1 component real
        #     # phas=np.angle(psi_p[50-int(xi*25),15,0,0])
        #     # phas=0 ## not fixing the phase
        #     maxisind = np.unravel_index(np.argmax(psi_p, axis=None), psi_p.shape)
        #     phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
        #     psi[:,:,:,:,nband] = psi_p*np.exp(-1j*phas)
        #     # psi[:,nband]=np.reshape(psi_p,[np.shape(n1)[0]*np.shape(n1)[1]*4]).flatten()


        return Eigvals[N-int(nbands/2):N+int(nbands/2)], psi

        