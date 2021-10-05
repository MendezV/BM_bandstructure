import numpy as np
import MoireLattice
import matplotlib.pyplot as plt
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
        [q1,q2,q3]=self.latt.q
        #we diagonalize a matrix made up
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        GM=self.latt.GMs
        # qx_dift = -self.kx*0 + GM1[0]*self.n1 + GM2[0]*self.n2 + self.xi*q1[0]-q1[0]
        # qy_dift = -self.ky*0 + GM1[1]*self.n1 + GM2[1]*self.n2 + self.xi*q1[1]-q1[1]
        qx_difb = + GM1[0]*self.n1 + GM2[0]*self.n2 - self.xi*q1[0]-self.xi*q1[0]
        qy_difb = + GM1[1]*self.n1 + GM2[1]*self.n2 - self.xi*q1[1]-self.xi*q1[1]
        # plt.scatter(qx_dift,qy_dift, c='k',s=2)
        # plt.scatter(qx_difb,qy_difb, c='r',s=2)
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()
        # valst = np.sqrt(qx_dift**2+qy_dift**2)
        valsb = np.sqrt(qx_difb**2+qy_difb**2)
        cutoff=7.*GM*0.7
        # ind_to_sum_t = np.where(valst <cutoff) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
        ind_to_sum_b = np.where(valsb <cutoff) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array
        # n1_val_t = self.n1[ind_to_sum_t] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
        # n2_val_t = self.n2[ind_to_sum_t] #
        n1_val_b = self.n1[ind_to_sum_b] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
        n2_val_b = self.n2[ind_to_sum_b] #


        # Nt = np.shape(ind_to_sum_t)[1] ##number of indices for which the condition above is satisfied
        Nb = np.shape(ind_to_sum_b)[1] ##number of indices for which the condition above is satisfied
        # print(Nb, "lattice size")
        # print(Nb*2, "lattice size")
        # if(Nt!=Nb):
        #     print("error with momentum cutoff")
        # G0xt= GM1[0]*n1_val_t+GM2[0]*n2_val_t
        # G0yt= GM1[1]*n1_val_t+GM2[1]*n2_val_t

        G0xb= GM1[0]*n1_val_b+GM2[0]*n2_val_b
        G0yb= GM1[1]*n1_val_b+GM2[1]*n2_val_b

        qx_t = -qx_difb[ind_to_sum_b]#+2*q1[0]
        qy_t = -qy_difb[ind_to_sum_b]#+2*q1[1]
        qx_b = qx_difb[ind_to_sum_b]
        qy_b = qy_difb[ind_to_sum_b]
        # qx_t = qx_dift[ind_to_sum_t]
        # qy_t = qy_dift[ind_to_sum_t]
        # qx_b = qx_difb[ind_to_sum_b]
        # qy_b = qy_difb[ind_to_sum_b]

        # theta=np.linspace(0,2*np.pi, 100)
        # plt.scatter(qx_t,qy_t, c='k',s=2)
        # plt.scatter(qx_b,qy_b, c='r',s=2)
        # plt.scatter(self.kx,self.ky,s=1)
        # plt.scatter(0,0,s=1)
        # # plt.plot(cutoff*np.cos(theta)+self.kx,cutoff*np.sin(theta)+self.ky)
        # # plt.plot(cutoff*np.cos(theta),cutoff*np.sin(theta))
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlim([-1.5*cutoff,1.5*cutoff])
        # plt.ylim([-1.5*cutoff,1.5*cutoff])
        # plt.show()
        # # print(Nb*2)
        # # return [G0xt, G0yt , ind_to_sum_b, Nb,G0xb, G0yb , ind_to_sum_b, Nb,qx_t,qy_t,qx_b,qy_b]
        return [ G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]

    def diracH(self,qx_t,qy_t,qx_b,qy_b):
        tau=self.xi
        hvkd=self.hvkd

        Qplusx = qx_t
        Qplusy = qy_t
        Qminx = qx_b
        Qminy = qy_b

        # #top layer
        # qx_1 = self.kx - Qplusx
        # qy_1 = self.ky - Qplusy
        ROTtop=self.latt.rot_min
        kkx_1 = self.kx - Qplusx
        kky_1 = self.ky - Qplusy
        qx_1 = ROTtop[0,0]*kkx_1+ROTtop[0,1]*kky_1
        qy_1 = ROTtop[1,0]*kkx_1+ROTtop[1,1]*kky_1

        # #bottom layer
        # qx_2 = self.kx -Qminx
        # qy_2 = self.ky -Qminy
        ROTbot=self.latt.rot_plus
        kkx_2 = self.kx -Qminx
        kky_2 = self.ky -Qminy
        qx_2 = ROTbot[0,0]*kkx_2+ROTbot[0,1]*kky_2
        qy_2 = ROTbot[1,0]*kkx_2+ROTbot[1,1]*kky_2

        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        
        # H1=hvkd*(np.kron(tau*paulix,np.diag(qx_1))+np.kron(pauliy,np.diag(qy_1))) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        # H2=hvkd*(np.kron(tau*paulix,np.diag(qx_2))+np.kron(pauliy,np.diag(qy_2))) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        H1=hvkd*(np.kron(np.diag(qx_1),tau*paulix)+np.kron(np.diag(qy_1),pauliy)) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        H2=hvkd*(np.kron(np.diag(qx_2),tau*paulix)+np.kron(np.diag(qy_2),pauliy)) #+np.kron(pauliz,gap*np.eye(N)) # ARITCFICIAL GAP ADDED
        return [H1,H2]


    def InterlayerU(self,qx_t,qy_t,qx_b,qy_b, Nb):

        tau=-self.xi
        [q1,q2,q3]=self.latt.q
        

        Qplusx = qx_t
        Qplusy = qy_t
        Qminx = qx_b
        Qminy = qy_b

        matGq1=np.zeros([Nb,Nb])
        matGq2=np.zeros([Nb,Nb])
        matGq3=np.zeros([Nb,Nb])
        tres=(1e-6)*np.sqrt(q1[0]**2 +q1[1]**2)

        # for i in range(Nb):

        #     indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q1[0])**2+(Qplusy-Qminy[i] + tau*q1[1])**2  )<tres)
        #     if np.size(indi1)>0:
        #         matGq1[i,indi1]=1

        #     indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q2[0])**2+(Qplusy-Qminy[i] + tau*q2[1])**2  )<tres)
        #     if np.size(indi1)>0:
        #         matGq2[i,indi1]=1 #indi1+1=i
 
        #     indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q3[0])**2+(Qplusy-Qminy[i] + tau*q3[1])**2  )<tres)
        #     if np.size(indi1)>0:
        #         matGq3[i,indi1]=1
        # for i in range(Nb):
        #     for j in range(Nb):

        #         if (np.sqrt(  (Qplusx[i]-Qminx[j] + tau*q1[0])**2+(Qplusy[i]-Qminy[j] + tau*q1[1])**2  )<tres):
        #             matGq1[i,j]=1

        #         if (np.sqrt(  (Qplusx[i]-Qminx[j] + tau*q2[0])**2+(Qplusy[i]-Qminy[j] + tau*q2[1])**2  )<tres):
        #             matGq2[i,j]=1 #indi1+1=i
    
        #         if (np.sqrt(  (Qplusx[i]-Qminx[j] + tau*q3[0])**2+(Qplusy[i]-Qminy[j] + tau*q3[1])**2  )<tres):
        #             matGq3[i,j]=1
        
        # matGq12=np.zeros([Nb,Nb])
        # matGq22=np.zeros([Nb,Nb])
        # matGq32=np.zeros([Nb,Nb])
        for i in range(Nb):

            indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q1[0])**2+(Qplusy-Qminy[i] + tau*q1[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq1[i,indi1]=1

            indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q2[0])**2+(Qplusy-Qminy[i] + tau*q2[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq2[i,indi1]=1 #indi1+1=i
    
            indi1=np.where(np.sqrt(  (Qplusx-Qminx[i] + tau*q3[0])**2+(Qplusy-Qminy[i] + tau*q3[1])**2  )<tres)
            if np.size(indi1)>0:
                matGq3[i,indi1]=1


    #Matrices that  appeared as coefficients of the real space ops
    #full product is the kronecker product of both matrices

        Mdelt1=matGq1
        Mdelt2=matGq2
        Mdelt3=matGq3


        pauli0=np.eye(2,2)
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])

       
        # print(Mdelt1+Mdelt2+Mdelt3)
        T11=pauli0+paulix
        T12=pauli0+paulix*np.cos(2*np.pi/3)+tau*pauliy*np.sin(2*np.pi/3)
        T13=pauli0+paulix*np.cos(2*np.pi/3)-tau*pauliy*np.sin(2*np.pi/3)
        # U=self.hvkd*self.alpha*(np.kron(T11,Mdelt1)+np.kron(T12,Mdelt2)+np.kron(T13,Mdelt3)) #interlayer coupling
        U=self.hvkd*self.alpha*(np.kron(Mdelt1,T11)+np.kron(Mdelt2,T12)+np.kron(Mdelt3,T13)) #interlayer coupling

        

        return U
        
    def eigens(self):
        [G0xb, G0yb , ind_to_sum_b, Nb, qx_t, qy_t, qx_b, qy_b]=self.umklapp_lattice()
        # [G0x, G0y , ind_to_sum, N]=self.umklapp_lattice()
        # U=self.InterlayerU(G0x, G0y , ind_to_sum, N)
        U=np.matrix(self.InterlayerU(qx_t,qy_t,qx_b,qy_b, Nb))
        Udag=U.H
        
        # [H1,H2]=self.diracH(G0x, G0y , inds_to_sum, N)
        # [H1,H2]=self.diracH(G0xt, G0yt , ind_to_sum_t, Nt)
        [H1,H2]=self.diracH(qx_t,qy_t,qx_b,qy_b)
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix

        # plt.imshow(np.abs(Hxi))
        # plt.show()
        #a= np.linalg.eigvalsh(Hxi) - en_shift
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues

        #######HANDLING WITH RESHAPE
        #umklp,umklp, layer, sublattice
        psi_p=np.zeros([self.Numklpy,self.Numklpx,2,2]) +0*1j
        # psi=np.zeros([Numklp*Numklp*4,nbands])+0*1j
        psi=np.zeros([self.Numklpy, self.Numklpx, 2,2, self.nbands]) +0*1j

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


        return Eigvals[2*Nb-int(self.nbands/2):2*Nb+int(self.nbands/2)], psi

        