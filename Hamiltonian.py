import numpy as np
import MoireLattice

class Ham():
    def __init__(self, hvkd, alpha, xi, kx, ky, n1, n2, latt, nbands):

        self.hvkd = hvkd
        self.alpha=alpha

        self.kx = kx
        self.ky = ky

        self.xi = xi

        self.n1=n1
        self.n2=n2

        self.Numklpy=np.shape(n1)[0]
        self.Numklpx=np.shape(n1)[1]

        self.latt=latt
        self.nbands=nbands

    def __repr__(self):
        return "Hamiltonian at {kx} {ky} with alpha parameter {alpha} and scale {hvkd}".format(kx=self.kx, ky=self.ky, alpha =self.alpha,hvkd=self.hvkd)
    
    def umklapp_lattice(self):
        #we diagonalize a matrix made up
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        qx_dif = self.kx + GM1[0]*self.n1+GM2[0]*self.n2
        qy_dif = self.ky + GM1[1]*self.n1+GM2[1]*self.n2
        vals = np.sqrt(qx_dif**2+qy_dif**2)
        ind_to_sum = np.where(vals <= 5) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
        n1_val = self.n1[ind_to_sum] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
        n2_val = self.n2[ind_to_sum] #

        N = np.shape(ind_to_sum)[1] ##number of indices for which the condition above is satisfied
    
        G0x= GM1[0]*n1_val+GM2[0]*n2_val
        G0y= GM1[1]*n1_val+GM2[1]*n2_val
        return [G0x, G0y , ind_to_sum, N]


    def diracH(self,G0x, G0y , ind_to_sum, N):
        tau=self.xi
        hvkd=self.hvkd
        [q1,q2,q3]=self.latt.q

        Qplusx = G0x + tau*q1[0]
        Qplusy = G0y + tau*q1[1]
        Qminx = G0x - tau*q1[0]
        Qminy = G0y - tau*q1[1]

        # #top layer
        qx_1 = self.kx -Qplusx
        qy_1 = self.ky -Qplusy

        # #bottom layer
        # q_2=Rotth1@np.array([kx,ky])
        qx_2 = self.kx -Qminx
        qy_2 = self.ky -Qminy

        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        
        H1=hvkd*(np.kron(tau*paulix,np.diag(qx_1))+np.kron(pauliy,np.diag(qy_1))) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        H2=hvkd*(np.kron(tau*paulix,np.diag(qx_2))+np.kron(pauliy,np.diag(qy_2))) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        return [H1,H2]


    def InterlayerU(self,G0x, G0y , ind_to_sum , N):

        tau=self.xi
        hvkd=self.hvkd
        [q1,q2,q3]=self.latt.q
        

        Qplusx = G0x + tau*q1[0]
        Qplusy = G0y + tau*q1[1]
        Qminx = G0x - tau*q1[0]
        Qminy = G0y - tau*q1[1]

        matGq1=np.zeros([N,N])
        matGq2=np.zeros([N,N])
        matGq3=np.zeros([N,N])
        tres=(1e-5)*np.sqrt(q1[0]**2 +q1[1]**2)

        for i in range(N):

            indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q1[0])**2+(Qplusy-Qminy[i] + tau*q1[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq1[indi1,i]=1

            indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q2[0])**2+(Qplusy-Qminy[i] + tau*q2[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq2[indi1,i]=1 #indi1+1=i
 
            indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q3[0])**2+(Qplusy-Qminy[i] + tau*q3[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq3[indi1,i]=1


    #Matrices that  appeared as coefficients of the real space ops
    #full product is the kronecker product of both matrices

        Mdelt1=matGq1
        Mdelt2=matGq2
        Mdelt3=matGq3

        pauli0=np.eye(2,2)
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])

        T11=pauli0+paulix
        T12=pauli0+paulix*np.cos(2*np.pi/3)+pauliy*np.sin(2*np.pi/3)
        T13=pauli0+paulix*np.cos(2*np.pi/3)-pauliy*np.sin(2*np.pi/3)
        U=self.alpha*(np.kron(T11,Mdelt1)+np.kron(T12,Mdelt2)+np.kron(T13,Mdelt3)) #interlayer coupling

        return U*0
    
    def eigens(self):

        [G0x, G0y , ind_to_sum, N]=self.umklapp_lattice()
        U=self.InterlayerU(G0x, G0y , ind_to_sum, N)
        Udag=(U.conj()).T
        [H1,H2]=self.diracH(G0x, G0y , ind_to_sum, N)
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
        #a= np.linalg.eigvalsh(Hxi) - en_shift
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######HANDLING WITH RESHAPE
        #umklp,umklp, layer, sublattice
        psi_p=np.zeros([self.Numklpy,self.Numklpx,2,2]) +0*1j
        # psi=np.zeros([Numklp*Numklp*4,nbands])+0*1j
        psi=np.zeros([self.Numklpy, self.Numklpx, 2,2, self.nbands]) +0*1j

        for nband in range(self.nbands):
            # print(np.shape(ind_to_sum), np.shape(psi_p), np.shape(psi_p[ind_to_sum]))
            psi_p[ind_to_sum] = np.reshape(  np.array( np.reshape(Eigvect[:,2*N-int(self.nbands/2)+nband] , [4, N])  ).T, [N, 2, 2] )    

            ##GAUGE FIXING by making the 30 30 1 1 component real
            # phas=np.angle(psi_p[50-int(xi*25),15,0,0])
            # phas=0 ## not fixing the phase
            maxisind = np.unravel_index(np.argmax(psi_p, axis=None), psi_p.shape)
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,:,:,:,nband] = psi_p*np.exp(-1j*phas)
            # psi[:,nband]=np.reshape(psi_p,[np.shape(n1)[0]*np.shape(n1)[1]*4]).flatten()


        return Eigvals[2*N-int(self.nbands/2):2*N+int(self.nbands/2)], psi

        