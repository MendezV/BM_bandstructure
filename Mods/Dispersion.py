from decimal import MIN_EMIN
from tkinter.messagebox import NO
import numpy as np
import MoireLattice
import matplotlib.pyplot as plt
from scipy import interpolate
import time
import MoireLattice
from scipy.interpolate import interp1d
from scipy.linalg import circulant
import scipy.linalg as la



# For HF

# Calculate projectors X
# isolate HF active sector X
# form factors X
# filter q points X
# Select projector decoup or simple subs depending on scheme X
# manipulate projectors for dirac points
# Slater components Delta X
# Permute indices FF X
# Make coulomb interaction X
# Fock Term
# Hartree term
# normal ordered
# Background ?
# Build matrix in band space

class Ham_BM():
    def __init__(self, hvkd, alpha, xi, latt, kappa, PH, Interlay=None):

        self.hvkd = hvkd
        self.alpha= alpha
        self.xi = xi #quick fix turns out I flipped all the formulas below
        
        
        self.latt=latt
        self.kappa=kappa
        self.PH=PH #particle hole symmetry
        self.gap=0.0  #artificial gap
        
        #precomputed momentum lattice and interlayer coupling
       
        self.cuttoff_momentum_lat=self.umklapp_lattice()

        if Interlay is None:
            self.Interlay = 1
        else:
            self.Interlay = Interlay
            self.gap=0.0  #artificial gap
            
        self.U=self.Interlay*np.matrix(self.InterlayerU())
        self.Dim=int(np.shape(self.U)[0]/2) #momentum lattice dimension for the matrices

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
        cutoff=5.*GM*0.7
        # cutoff=9.*GM*0.7
        ind_to_sum_b = np.where(valsb <cutoff) #Finding the i,j indices where the difference of  q lies inside threshold, this is a 2 x Nindices array

        #cutoff lattice
        n1_val_b = n1[ind_to_sum_b] # evaluating the indices above, since n1 is a 2d array the result of n1_val is a 1d array of size Nindices
        n2_val_b = n2[ind_to_sum_b] #
        Nb = np.shape(ind_to_sum_b)[1] ##number of indices for which the condition above is satisfied
        G0xb= GM1[0]*n1_val_b+GM2[0]*n2_val_b #umklapp vectors within the cutoff
        G0yb= GM1[1]*n1_val_b+GM2[1]*n2_val_b #umklapp vectors within the cutoff

        #reciprocal lattices for both layers
        #flipping the order so that same points occur in the same index for plus and minus valleys
        if self.xi>0:
            qx_t = -qx_difb[ind_to_sum_b]
            qy_t = -qy_difb[ind_to_sum_b]
            qx_b = qx_difb[ind_to_sum_b]#[::int(self.xi)]
            qy_b = qy_difb[ind_to_sum_b]#[::int(self.xi)]
        else:
            qx_t = -qx_difb[ind_to_sum_b][::int(self.xi)]
            qy_t = -qy_difb[ind_to_sum_b][::int(self.xi)]
            qx_b = qx_difb[ind_to_sum_b][::int(self.xi)]
            qy_b = qy_difb[ind_to_sum_b][::int(self.xi)]


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
        
        # H1=((kx*tau*paulix)+(ky*pauliy)) +self.gap*tau*pauliz
        # H2=((kx*tau*paulix)+(ky*pauliy)) +self.gap*tau*pauliz
        
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
       


        phi = 2*np.pi/3    
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        # T1 = np.array([[w0,w1],[w1,w0]])
        # T2 = zs*np.array([[w0,w1*zs],[w1*z,w0]])
        # T3 = z*np.array([[w0,w1*z],[w1*zs,w0]])
        
        T1=pauli0*self.kappa+paulix
        T2=pauli0*self.kappa+paulix*np.cos(phi)+self.xi*pauliy*np.sin(phi)
        T3=pauli0*self.kappa+paulix*np.cos(phi)-self.xi*pauliy*np.sin(phi)


        U=self.hvkd*self.alpha*( np.kron(Mdelt1,T1) + np.kron(Mdelt2,T2)+ np.kron(Mdelt3,T3)) #interlayer coupling
        # U=0*T1
        return U
        
    def eigens(self, kx,ky, nbands):
        
        if self.xi>0:
            U=self.U
            Udag=U.H
            [H1,H2]=self.diracH( kx, ky)
        else:
            Udag=self.U
            U=Udag.H
            [H2,H1]=self.diracH( kx, ky)
            
        N =int(2*self.Dim)
        
        Hxi=np.bmat([[H1, Udag ], [U, H2]]) #Full matrix
        (Eigvals,Eigvect)= np.linalg.eigh(Hxi)  #returns sorted eigenvalues
        # (Eigvals,Eigvect)= la.eigh(Hxi)  #returns sorted eigenvalues
        
        # Eigvals,Eigvect = np.linalg.eig(Hxi)

        # idx = Eigvals.argsort()   
        # Eigvals = Eigvals[idx]
        # Eigvect = Eigvect[:,idx]
        
        
        #######Gauge Fixing by setting the largest element to be real
        # umklp,umklp, layer, sublattice
        psi=Eigvect[:,N-int(nbands/2):N+int(nbands/2)]
        
        for nband in range(nbands):
            psi_p=psi[:,nband]
            # maxisind = np.unravel_index(np.argmax(np.abs(np.imag(psi_p)), axis=None), psi_p.shape)[0]
            maxisind = np.unravel_index(np.argmax(np.abs( (psi_p) ), axis=None), psi_p.shape)[0]
            phas=np.angle(psi_p[maxisind]) #fixing the phase to the maximum 
            psi[:,nband]=psi[:,nband]*np.exp(-1j*phas)

        return Eigvals[N-int(nbands/2):N+int(nbands/2)]-self.e0, psi
        # return Eigvals-self.e0, psi
    

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
    
    def T_psi(self, psi_p):
        psi=np.conj(psi_p)
        rot=self.latt.C2z
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        submat=pauli0
        [block_ttp,block_tbp,block_btp, block_bbp] = self.rot_WF(rot)
        block_tt=np.kron(block_ttp, submat)
        block_tb=np.kron(block_tbp, submat)
        block_bt=np.kron(block_btp, submat)
        block_bb=np.kron(block_bbp, submat)
        mat=np.bmat([[block_tt,block_tb], [block_bt, block_bb]])


        return mat@psi
    
    def c2zT_psi(self, psi_p):
        psi=np.conj(psi_p)
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        pau=[pauli0,paulix,pauliy,pauliz]
        Qmat=np.eye(self.Dim)
        
        layer=0
        sublattice=1
        # mat=np.kron(pau[layer],np.kron(Qmat, pau[sublattice]))
        mat=np.kron(np.kron(pau[layer],Qmat), pau[sublattice])

        return  mat@psi
    
    def Csub_psi(self, psi):
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        pau=[pauli0,paulix,pauliy,pauliz]
        Qmat=np.eye(self.Dim)
        
        layer=0
        sublattice=3
        # mat=np.kron(pau[layer],np.kron(Qmat, pau[sublattice]))
        mat=np.kron(np.kron(pau[layer],Qmat), pau[sublattice])

        return  mat@psi
    
    #small angle approx antiunitary PH
    def Cstar_psi(self, psi):
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        pau=[pauli0,paulix,pauliy,pauliz]
        Qmat=np.eye(self.Dim)
        
        layer=2
        sublattice=1
        # mat=np.kron(pau[layer],np.kron(Qmat, pau[sublattice]))
        mat=np.kron(np.kron(pau[layer],Qmat), pau[sublattice])

        return  mat@psi

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
        
        [GM1,GM2]=self.latt.GMvec #remove the nor part to make the lattice not normalized
        Trans=self.xi*dirGM1*GM1+self.xi*dirGM2*GM2
        mat = self.trans_WF(Trans)
        ###This is very error prone, I'm assuming two options either a pure wvefunction
        #or a wavefunction where the first index is k , the second is 4N and the third is band
        nind=len(np.shape(psi))
        if nind==2:
            matmul=mat@psi
        elif nind==3:
            psimult=[]
            for i in range(np.shape(psi)[0]):
                psimult=psimult+[mat@psi[i,:,:]]
            matmul=np.array(psimult)
        else:
            print("not going to work, check umklapp shift")
            matmul=mat@psi
                

        return matmul

    
    def ExtendE(self,E_k , umklapp):
        Gu=self.latt.Umklapp_List(umklapp) 
        
        Elist=[]
        for GG in Gu:
            Elist=Elist+[E_k]
            
        return np.vstack(Elist)
    
    def ExtendPsi(self, psi_p, umklapp):
        psilist=[]
        Gu=self.latt.Umklapp_List(umklapp)
        for GG in Gu:
            shi1=int(GG[0])
            shi2=int(GG[1])
            psishift=self.trans_psi(psi_p, shi1, shi2)
            psilist=psilist+[psishift]
        psi=np.vstack(psilist)
        return psi


class Dispersion():
    
    def __init__(self, latt, nbands, hpl, hmin):

        
        #changes to eliominate arg KX KY put KX1bz first, then call umklapp lattice and put slef in KX, KY
        #change to implement dos 
        self.nbands=nbands
        self.hpl=hpl
        self.hmin=hmin
        self.latt=latt
        self.Dim=hpl.Dim
        # [self.psi_plus,self.Ene_valley_plus_1bz,self.psi_min,self.Ene_valley_min_1bz]=self.precompute_E_psi()
    
    def precompute_E_psi(self):

        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]
        psi_min_a_2=[]


        print(f"starting dispersion with {self.latt.Npoi1bz} points..........")
        
        s=time.time()
   
        for l in range(self.latt.Npoi1bz):
            
            E1,wave1=self.hpl.eigens(self.latt.KX1bz[l],self.latt.KY1bz[l],self.nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1)
            # psi_plus_a.append(wave1)
            wave1p=self.gauge_fix( wave1, E1)
            self.check_C2T(wave1p)
            psi_plus_a.append(wave1p)
            
            ########## generate explicitly
            E1,wave1=self.hmin.eigens(self.latt.KX1bz[l],self.latt.KY1bz[l],self.nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            # psi_min_a_2.append(wave1)
            # wave1m=self.gauge_fix( wave1, E1)
            # self.check_C2T(wave1m)
            # psi_min_a.append(wave1m)
            
            # ########### with the convention that this is the eigenvalue and eigenvector at -k
            # ########### infer from C2 or T symmetry, energies
            # Ene_valley_min_a=np.append(Ene_valley_min_a,E1)
            
            # # infer from C2, Cstar or T symmetry, wavefuncs
            # # wave1m=self.impose_C2(wave1p)
            wave1m=self.impose_T(wave1p)
            # wave1m=self.impose_Cstar(wave1p)
            self.check_C2T(wave1m)
            psi_min_a.append(wave1m)
            
            # ######## checks for symmetry
            self.check_T(wave1p,wave1m)
            self.check_T(wave1m,wave1p)
            self.check_C2(wave1p,wave1m)
            self.check_C2(wave1m,wave1p)
            
    
        e=time.time()
        print("time to diag over MBZ", e-s)
        ##relevant wavefunctions and energies for the + valley
        
        psi_plus=np.array(psi_plus_a)
        psi_min=np.array(psi_min_a)[::-1,:,:]
        
        
        #overlap with the T-reversed generated wavefunction is indeed just a phase
        # psi_min_2=np.array(psi_min_a_2)
        # for k in range(self.latt.Npoi1bz):
        #     self.check_Overlap(psi_min_2[k,:,:],psi_min[k,:,:])
        
        
        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[self.latt.Npoi1bz,self.nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[self.latt.Npoi1bz,self.nbands])

        diff=Ene_valley_min-(-Ene_valley_plus[:,::-1])
        print('\n symmetry in eigens Cstar*Tr', np.mean(np.sqrt(np.diag(diff.T@diff))),"flips the spectrum and momentum space \n")
        
        
        return [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]
    


    ###########Gauge fixing the wavefunctions
    
    def gauge_fix(self, wave1, E1):
        
        #using c2T symmetry to fix the phases of the wavefuncs
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        #lower half of the spectrum
        testw=np.array(wave1[:,:ihalf])
        testw2=self.hpl.c2zT_psi(testw)
        
        # ang_low=np.angle(np.conj(testw.T)@testw2)
        ang_low=np.angle(np.sum(np.conj(testw)*testw2,axis=0)) #phase of the dot product of the columns of wave1  with the C2T transformed columns
        reshape_ang_low=np.vstack([ang_low]*np.shape(testw)[0] )  #making the shapes match with the wavefunciton array
        testw_new=testw*np.exp(1j*reshape_ang_low/2) #"substracting" the phase for each of the columns
        wave1[:,:ihalf]=np.array(testw_new)
        
        #upper half of the spectrum
        testw=np.array(wave1[:,ihalf:])
        testw2=self.hpl.c2zT_psi(testw)
        
        # ang_up=np.angle(np.conj(testw.T)@testw2)
        ang_up=np.angle(np.sum(np.conj(testw)*testw2,axis=0)) #phase of the dot product of the columns of wave1 with the C2T transformed columns
        reshape_ang_up=np.vstack([ang_up]*np.shape(testw)[0] )  #making the shapes match with the wavefunciton array
        testw_new=1j*testw*np.exp(1j*reshape_ang_up/2) #"substracting" the phase for each of the columns, #extra factor of i to make the representation act as n_z
        wave1[:,ihalf:]=np.array(testw_new)
        
        #testing the representation of c2T
        Sewing=np.conj((wave1[:,inde1:inde2]).T)@self.hpl.c2zT_psi((wave1[:,inde1:inde2]))
        pauliz=np.array([[1,0],[0,-1]])
        
        if np.abs(np.mean(Sewing-pauliz))>1e-6:
            print("c2T failed")
            print(Sewing)
            print(E1[inde2]-E1[inde1])
        
        #using C sublattice to fix an additional relative minus sign
        testw=np.array(wave1)
        testw2=self.hpl.Csub_psi(testw)
        
        Sewing2=np.real((np.conj(testw.T)@testw2))
        # print(Sewing2)
        
        #multiplying the sign to the upper half of the spectrum
        Sign=np.real(np.exp(1j*np.angle(Sewing2[0,1]))) #avoids zero in the decoupled limit
        wave1[:,ihalf:]=wave1[:,ihalf:]*Sign

        
        #if we are in the chiral limit the second sewing matrix is a rep of chiral sublattice symmetry
        #should give a paulix in the basis that we chose
        if self.hpl.kappa==0.0:
            Sewing2=np.real( np.conj(wave1[:,inde1:inde2].T)@(self.hpl.Csub_psi(wave1[:,inde1:inde2])) )
            paulix=np.array([[0,1],[1,0]])
            if np.abs(np.mean(Sewing2-paulix))>1e-6:
                print("chiral sublattice failed")
                print(Sewing2)
    
        
        return wave1
    

    def impose_all_Cstar(self, psi_plus):
        
        (Nkpoints, DimV, Nbands)=np.shape(psi_plus)
        II=np.eye(self.Dim)
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        op=np.kron(pauliy,np.kron(II, paulix))
        Wm=np.zeros(np.shape(psi_plus))+0*1j
        Wp=np.array(psi_plus)
        
        TopRange=int(np.ceil(Nkpoints/2))
        kpoints=int(Nkpoints)
        
        #Cstar (Ph) sends plus to minus (only half of BZ)
        for k in range(TopRange):
            Wm[k,:,:]=op@(psi_plus[k,:,::-1])
        
        #Time reversal sends other half of the BZ
        nz=np.kron(pauliz,int(self.nbands/2)) #number of bands has to be even
        for k in range(TopRange):
            Wm[kpoints-1-k,:,:]=np.conj(psi_plus[k,:,:])@nz
        
        #Cstar sends the second half of the BZ back
        for k in range(TopRange):
            Wp[kpoints-1-k,:,:]=op@(Wm[kpoints-1-k,:,::-1])

        for k in range(kpoints):
            
            self.check_C2T(Wp[k,:,:])
            self.check_C2T(Wm[k,:,:])
            
            self.check_Cstar( Wp[k,:,:],Wm[k,:,:])    
             
            #c2 and T send k -> -k        
            self.check_T(Wp[kpoints-1-k,:,:],Wm[k,:,:])
            self.check_C2(Wp[kpoints-1-k,:,:],Wm[k,:,:]) 
        return Wp,Wm
    
    def impose_all_Cstar_2(self, psi_plus, psi_min):
        
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        
        (Nkpoints, DimV, Nbands)=np.shape(psi_plus)
        II=np.fliplr(np.eye(self.Dim)) #this is needed because the order of the basis in the minus valley is flipped
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        op=np.kron(pauliy,np.kron(II, paulix))
        
        Wm=np.zeros(np.shape(psi_plus))+0*1j
        Wp=np.array(psi_plus)
        
        TopRange=int(np.ceil(Nkpoints/2))
        kpoints=int(Nkpoints)
        
        #Cstar (Ph) sends plus to minus (only half of BZ)
        for k in range(kpoints):
            Wm[k,:,inde1:ihalf]=np.array(op@(Wp[k,:,ihalf:inde2])[::-1,::-1])
         
        
        #Time reversal sends other half of the BZ
        nz=np.kron(pauliz,int(self.nbands/2)) #number of bands has to be even
        for k in range(kpoints):
            Wp[kpoints-1-k,:,inde1:ihalf]=-np.array(np.conj(Wm[k,:,inde1:ihalf]))
        
        #Cstar sends the second half of the BZ back
        for k in range(kpoints):
            Wm[k,:,ihalf:inde2]=np.array(op@(Wp[k,:,inde1:ihalf][::-1,::-1]))

        for k in range(kpoints):
            
            self.check_C2T(Wp[k,:,:])
            self.check_C2T(Wm[k,:,:])
            
            self.check_Cstar( Wp[k,:,:],Wm[k,:,:])    
             
            #c2 and T send k -> -k    
            print('cstar',k)    
            self.check_T( Wp[kpoints-1-k,:,:], Wm[k,:,:])
            self.check_C2( Wp[kpoints-1-k,:,:], Wm[k,:,:]) 
        return Wp,Wm
    
  
    
    def check_Cstar(self, wave1p, wave1m):
        
        II=np.eye(self.Dim) #this is needed because the order of the basis in the minus valley is flipped
        
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        #matrices in the micro basis for op and in the band basis for sewing
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        
        op=np.kron(pauliy,np.kron(II, paulix))
        
        Sewing=np.conj((wave1m[:,inde1:inde2]).T)@(op@(wave1p[:,inde1:inde2]))
        if np.abs(np.mean(Sewing-paulix))>1e-6:
            print("C star failed")
            print(Sewing)
        return None
    
    def impose_T(self,wave1p):
     
        ihalf=int(self.nbands/2)
       
        
        wave2=np.array(self.hpl.T_psi(wave1p))
        wave2[:,ihalf:]=-np.array(self.hpl.T_psi(wave1p))[:,ihalf:]
        return wave2
    
    def check_T(self,wave1p,wave1m):
        
        w2=self.hmin.T_psi(wave1m)
        
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        
        pauliz=np.array([[1,0],[0,-1]])
        
        Sewing=np.conj((wave1p[:,inde1:inde2]).T)@(w2[:,inde1:inde2])
        
        if np.abs(np.mean(Sewing-pauliz))>1e-6:
            print("T failed")
            print(Sewing)
        return None
    
    def impose_C2(self,wave1p):
        
        wave2=np.array(self.hpl.T_psi(wave1p))
        
        return wave2
    
    def check_C2(self,wave1p,wave1m):
        
      
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        #matrices in the micro basis for op and in the band basis for sewing
        pauli0=np.array([[1,0],[0,1]])
        
        w2=self.hmin.c2z_psi(wave1m)
        
        Sewing=np.conj((wave1p[:,inde1:inde2]).T)@(w2[:,inde1:inde2])
        
        if np.abs(np.mean(Sewing-pauli0))>1e-6:
            print("C2 failed")
            print(Sewing)
        return None
    
    def check_C2T(self,wave1):
        
        II=np.eye(self.Dim)
        
        
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        #matrices in the micro basis for op and in the band basis for sewing
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        op=np.kron(pauli0,np.kron(II, paulix))
        

        Sewing=np.conj((wave1[:,inde1:inde2]).T)@np.conj(op@(wave1[:,inde1:inde2]))
        if np.abs(np.mean(Sewing-pauliz))>1e-6:
            print("c2T failed")
            print(Sewing)
        
        return None
    
    
    def check_norm(self,wave1p):
        II=np.eye(self.Dim)
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        
        Sewing=np.abs(np.conj((wave1p[:,inde1:inde2]).T)@(wave1p[:,inde1:inde2]))
        if np.abs(np.mean(Sewing-pauli0))>1e-6:
            print("norm failed")
            print(Sewing)
        return None
    
    def check_Overlap(self,wave1,wave2):
        II=np.eye(self.Dim)
        ihalf=int(self.nbands/2)
        inde1=ihalf - 1 #2*self.Dim-int(self.nbands/2)
        inde2=ihalf + 1 #2*self.Dim+int(self.nbands/2)
        
        pauli0=np.array([[1,0],[0,1]])
        paulix=np.array([[0,1],[1,0]])
        pauliy=np.array([[0,-1j],[1j,0]])
        pauliz=np.array([[1,0],[0,-1]])
        
        
        Sewing=np.abs(np.conj((wave2[:,inde1:inde2]).T)@(wave1[:,inde1:inde2]))
        if np.abs(np.mean(Sewing-pauli0))>1e-6:
            print("overlap failed")
            print(Sewing)
            
        else:
            print("ovelap passed")
            print(Sewing)
        return None
    
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
        
        fillings_pre=np.linspace(0,3.9,Nfil)
        fillings=fillings_pre[1:]
        
        if calculate:
            [psi_plus_dos,Ene_valley_plus_dos,psi_min_dos,Ene_valley_min_dos]=self.precompute_E_psi()
        if read:
            print("Loading  ..........")
            with open('dispersions/Edisp_'+str(self.latt.Npoints)+'_theta_'+str(self.latt.theta)+'_kappa_'+str(self.hpl.kappa)+'.npy', 'rb') as f:
                Ene_valley_plus_dos=np.load(f)
            with open('dispersions/Edisp_'+str(self.latt.Npoints)+'_theta_'+str(self.latt.theta)+'_kappa_'+str(self.hpl.kappa)+'.npy', 'rb') as f:
                Ene_valley_min_dos=np.load(f)
    
        if write:
            print("saving  ..........")
            with open('dispersions/Edisp_'+str(self.latt.Npoints)+'_theta_'+str(self.latt.theta)+'_kappa_'+str(self.hpl.kappa)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_plus_dos)
            with open('dispersions/Edism_'+str(self.latt.Npoints)+'_theta_'+str(self.latt.theta)+'_kappa_'+str(self.hpl.kappa)+'.npy', 'wb') as f:
                np.save(f, Ene_valley_min_dos)
                
        

        [earr, dos, f2 ]=self.DOS(Ene_valley_plus_dos,Ene_valley_min_dos)

        mu_values=[]
        mu_values.append(0)        
        for fill in fillings:
            [mu, nfil, es,nn]=self.chem_for_filling( fill, f2, earr)
            mu_values.append(mu)


        
        return [fillings_pre,np.array(mu_values)]
    
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
        print("\n")
        print("band structure across high symmetry directions")
       
        Ene_valley_plus_a=np.empty((0))
        Ene_valley_min_a=np.empty((0))
        psi_plus_a=[]
        psi_min_a=[]

        nbands=2 #Number of bands 
        kpath=self.latt.High_symmetry_path()
        
        kx=kpath[:,0]
        ky=kpath[:,1]
        VV=self.latt.boundary()
        plt.scatter(kx,ky, c='r')
        plt.plot(VV[:,0], VV[:,1], c='b')
        plt.savefig("highsym_path.png")
        plt.close()

        Npoi=np.shape(kpath)[0]
        for l in range(Npoi):
            # h.umklapp_lattice()
            # break
            E1p,wave1p=self.hpl.eigens(kpath[l,0],kpath[l,1],nbands)
            Ene_valley_plus_a=np.append(Ene_valley_plus_a,E1p)
            psi_plus_a.append(wave1p)


            E1m,wave1m=self.hmin.eigens(kpath[l,0],kpath[l,1],nbands)
            Ene_valley_min_a=np.append(Ene_valley_min_a,E1m)
            psi_min_a.append(wave1m)

        Ene_valley_plus= np.reshape(Ene_valley_plus_a,[Npoi,nbands])
        Ene_valley_min= np.reshape(Ene_valley_min_a,[Npoi,nbands])

    

        print("shape of the energies..",np.shape(Ene_valley_plus_a))
        qa=np.linspace(0,1,Npoi)
        for i in range(nbands):
            plt.plot(qa,Ene_valley_plus[:,i] , c='b')
            plt.plot(qa,Ene_valley_min[:,i] , c='r', ls="--")
            if i==int(nbands/2-1):
                min_min  = np.min(Ene_valley_min[:,i])
                min_plus = np.min(Ene_valley_plus[:,i])
            
            elif i==int(nbands/2):
                max_min  = np.max(Ene_valley_min[:,i])
                max_plus = np.max(Ene_valley_plus[:,i])
        maxV=np.max([max_min,max_plus])
        minC=np.max([min_min,min_plus])
        BW=maxV-minC
        print("the bandwidth is ..." ,BW)
        plt.xlim([0,1])
        # plt.ylim([-0.008,0.008])
        plt.savefig("highsym.png")
        plt.close()
        print("finished band structure along high symmetry directions")
        print("\n \n")
        return [Ene_valley_plus, Ene_valley_min]



    
class FormFactors():
    def __init__(self, psi_p, xi, latt, umklapp, ham, Bands=None):
        self.psi_p = psi_p #has dimension #kpoints, 4*N, nbands
        self.cpsi_p=np.conj(psi_p)
        self.latt=latt
        self.xi=xi
        self.Nu=int(np.shape(self.psi_p)[1]/4) #4, 2 for sublattice and 2 for layer
        
        self.kx=self.latt.KQX
        self.ky=self.latt.KQY
 
        #momentum transfer lattice
        kqx1, kqx2=np.meshgrid(self.kx,self.kx)
        kqy1, kqy2=np.meshgrid(self.ky,self.ky)
        self.qx=kqx1-kqx2
        self.qy=kqy1-kqy2
        self.q=np.sqrt(self.qx**2+self.qy**2)+1e-17
        
        self.qmin_x=self.latt.KQX[1]-self.latt.KQX[0]
        self.qmin_y=self.latt.KQY[1]-self.latt.KQY[0]
        self.qmin=np.sqrt(self.qmin_x**2+self.qmin_y**2)
        
        if Bands is None:
            self.Bands = np.shape(self.psi_p)[2]
            inindex=0
            finindex=self.Bands
            print("calculating form factors with all bands is input wavefunction")
        else:
            self.Bands= Bands
            initBands=np.shape(self.psi_p)[2]
            inindex=int(initBands/2)-int(Bands/2)
            finindex=int(initBands/2)+int(Bands/2)
            print(f"truncating wavefunction and calculating form factors with only {Bands} bands")
            
        
        #if we only diagonalized in the FBZ and we need to translate the wavefunctions
        #if umklapp==-1 we don't do this
        if umklapp>=0:
            psi=ham.ExtendPsi(psi_p, umklapp+1)
            self.psi=psi[:,:,inindex:finindex]
            self.cpsi=np.conj(self.psi)
            print("shapes of the wavefunctions in the form factor umklapp class after translation, ",np.shape(self.psi), np.shape(psi_p), np.shape(self.kx))
        else:
            self.psi=psi_p[:,:,inindex:finindex]
            self.cpsi=np.conj(self.psi)
            print("shapes of the wavefunctions in the form factor umklapp class with no translation, ",np.shape(self.psi), np.shape(psi_p), np.shape(self.kx))
            

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
        print(f"calculating tensor that stores the overlaps, layer {layer}, sublattice {sublattice}........")
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
    
    def plotFF(self,Lambda_Tens, msg):
        K=[]
        KP=[]
        cos1=[]
        for s in range(self.latt.Npoi):
            undet=0
            K.append(self.latt.KX[s])
            KP.append(self.latt.KY[s])
            for k in range(self.latt.Npoi1bz):
                k1=np.argmin( (self.latt.KQX-(self.latt.KX1bz[k]+self.latt.KX[s]))**2 +(self.latt.KQY-(self.latt.KY1bz[k]+self.latt.KY[s]))**2)
                k2=np.argmin( (self.latt.KQX-self.latt.KX1bz[k])**2 +(self.latt.KQY-self.latt.KY1bz[k])**2)
                undet=undet+np.abs(np.linalg.det(Lambda_Tens[k1,:,k2,:]))**2  
            cos1.append(undet/self.latt.Npoi1bz)
            

        plt.scatter(K,KP, c=cos1,s=4)
        plt.colorbar()
        plt.savefig("FFplot"+msg+".png")
        plt.close()

        plt.scatter(K,cos1,s=4)
        plt.colorbar()
        plt.savefig("FFplot"+msg+"_kx_cut.png")
        plt.close()

        plt.scatter(KP,cos1,s=4)
        plt.colorbar()
        plt.savefig("FFplot"+msg+"_ky_cut.png")
        plt.close()
        return None
    
       
class HF_BandStruc:
    
    
    def __init__(self, latt,  hpl, hmin, hpl_decoupled, hmin_decoupled, nremote_bands, nbands, substract, cons, mode):

        
        self.latt=latt
        
        
        self.nbands_init=4*hpl.Dim
        self.nbands=nbands
        self.nremote_bands=nremote_bands
        self.tot_nbands=nbands+nremote_bands
        
        self.ini_band=int(self.nbands_init/2)-int(self.tot_nbands/2)
        self.fini_band=int(self.nbands_init/2)+int(self.tot_nbands/2)
        
        self.ini_band_HF=int(self.tot_nbands/2)-int(self.nbands/2)
        self.fini_band_HF=int(self.tot_nbands/2)+int(self.nbands/2)
        
        self.hpl=hpl
        self.hmin=hmin
        
        self.hpl_decoupled=hpl_decoupled
        self.hmin_decoupled=hmin_decoupled
        
        self.subs=substract
        self.mode=mode        

        [self.V0, self.d_screening_norm]=cons
    
        
        
        if self.mode==1:
            
            ################################
            #dispersion attributes
            ################################

            disp=Dispersion( latt, self.nbands_init, hpl, hmin)
            [self.psi_plus_1bz,self.Ene_valley_plus_1bz,self.psi_min_1bz,self.Ene_valley_min_1bz]=disp.precompute_E_psi()

            self.Ene_valley_plus=self.hpl.ExtendE(self.Ene_valley_plus_1bz[:,self.ini_band:self.fini_band] , self.latt.umkl+1)
            self.Ene_valley_min=self.hmin.ExtendE(self.Ene_valley_min_1bz[:,self.ini_band:self.fini_band] , self.latt.umkl+1)

            self.psi_plus=self.hpl.ExtendPsi(self.psi_plus_1bz, self.latt.umkl+1)
            self.psi_min=self.hmin.ExtendPsi(self.psi_min_1bz, self.latt.umkl+1)
            ################################
            #generating form factors
            ################################
            self.FFp=FormFactors(self.psi_plus, 1, latt, -1,self.hpl,self.tot_nbands)
            self.FFm=FormFactors(self.psi_min, -1, latt, -1,self.hmin,self.tot_nbands)
            
            self.LamP=self.FFp.denqFF_s() #no initial transpose
            self.LamP_dag=np.conj(np.transpose(self.LamP,(0,3,2,1) )) #no initial transpose
            self.LamM=self.FFm.denqFF_s() #no initial transpose
            self.LamM_dag=np.conj(np.transpose(self.LamM,(0,3,2,1) )) #no initial transpose
            
            
            ################################
            #reference attributes
            ################################
            
            print('\n')
            print('calculating refernce states...')

            disp_decoupled=Dispersion( latt, self.nbands_init, hpl_decoupled, hmin_decoupled)
            [self.psi_plus_decoupled_1bz,self.Ene_valley_plus_decoupled_1bz,self.psi_min_decoupled_1bz,self.Ene_valley_min_decoupled_1bz]=disp_decoupled.precompute_E_psi()

            self.psi_plus_decoupled=self.hpl.ExtendPsi(self.psi_plus_decoupled_1bz, self.latt.umkl+1)
            self.psi_min_decoupled=self.hmin.ExtendPsi(self.psi_min_decoupled_1bz, self.latt.umkl+1)
            
      
        
            ################################
            #Constructing projector
            ################################
            Pp,Pm=self.Proy_slat_comp(self.psi_plus, self.psi_min)
            Pp0,Pm0=self.Proy_slat_comp(self.psi_plus_decoupled, self.psi_min_decoupled)
            
            if self.subs==1:
                proj=self.Slater_comp(Pp)
            else:
                proj=self.Slater_comp(Pp0)
                
            self.V=self.Vq()
            print('shapes of the operators for HF',np.shape(self.V), np.shape(self.LamP), np.shape(proj))
            
            s=time.time()
            Fock=self.Fock(proj)
            e=time.time()
            print(f'time for Fock {e-s}')

            # HBMp=np.zeros(np.shape(Fock))
            HBMp=np.zeros(np.shape(proj))
            HBMp[:,0,0]=self.Ene_valley_plus[:,0]
            HBMp[:,1,1]=self.Ene_valley_plus[:,1]
            H0=HBMp-Fock*mode
            

            print(f"starting dispersion with {self.latt.Npoi1bz} points..........")
            
            s=time.time()
            EHFp=[]
            U_transf=[]
            EHFm=[]
            U_transfm=[]
            
            paulix=np.array([[0,1],[1,0]])
            
            diagI = np.zeros([int(self.nbands/2),int(self.nbands/2)]);
            for ib in range(int(self.nbands/2)):
                diagI[ib,int(self.nbands/2-1-ib)]=1;
            sx=np.kron(paulix,diagI)
            
            for l in range(self.latt.NpoiQ):
                Hpl=H0[l,:,:]
                Hmin=-sx@Hpl@sx
                # (Eigvals,Eigvect)= np.linalg.eigh(Fock[l,:,:])  #returns sorted eigenvalues
                (Eigvals,Eigvect)= np.linalg.eigh(Hpl)  #returns sorted eigenvalues

                EHFp.append(Eigvals)
                U_transf.append(Eigvect)
                
                (Eigvals,Eigvect)= np.linalg.eigh(Hmin)  #returns sorted eigenvalues
                EHFm.append(Eigvals)
                U_transfm.append(Eigvect)
                
                
            Ik=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
            self.E_HFp=np.array(EHFp)[Ik, self.ini_band_HF:self.fini_band_HF]
            self.E_HFp_K=self.hpl.ExtendE(self.E_HFp, self.latt.umkl)
            self.E_HFp_ex=self.hpl.ExtendE(self.E_HFp, self.latt.umkl+1)
            self.Up=np.array(U_transf)
            
            self.E_HFm=np.array(EHFm)[Ik, self.ini_band_HF:self.fini_band_HF]
            self.E_HFm_K=self.hmin.ExtendE(self.E_HFm, self.latt.umkl)
            self.E_HFm_ex=self.hmin.ExtendE(self.E_HFm, self.latt.umkl+1)
            self.Um=np.array(U_transfm)
            e=time.time()
            print(f'time for Diag {e-s}')
            
        else:
            
            # different dimensions for the number of bands that we keeps since we do not need to keep
            # track of extra remote bands as compared to the HF version
            
            ################################
            #dispersion attributes
            ################################

            disp=Dispersion( latt, self.nbands, hpl, hmin)
            [self.psi_plus_1bz,self.Ene_valley_plus_1bz,self.psi_min_1bz,self.Ene_valley_min_1bz]=disp.precompute_E_psi()

            self.Ene_valley_plus=self.hpl.ExtendE(self.Ene_valley_plus_1bz , self.latt.umkl+1)
            self.Ene_valley_min=self.hmin.ExtendE(self.Ene_valley_min_1bz , self.latt.umkl+1)

            self.psi_plus=self.hpl.ExtendPsi(self.psi_plus_1bz, self.latt.umkl+1)
            self.psi_min=self.hmin.ExtendPsi(self.psi_min_1bz, self.latt.umkl+1)
            ################################
            #generating form factors
            ################################
            
            self.FFp=FormFactors(self.psi_plus, 1, latt, -1,self.hpl)
            self.FFm=FormFactors(self.psi_min, -1, latt, -1,self.hmin)
            
            # self.FFp=FormFactors(self.psi_plus_1bz, 1, latt, latt.umkl,self.hpl)
            # self.FFm=FormFactors(self.psi_min_1bz, -1, latt, latt.umkl,self.hmin)
            
            self.LamP=self.FFp.denqFF_s() #no initial transpose
            self.LamP_dag=np.conj(np.transpose(self.LamP,(0,3,2,1) )) #no initial transpose
            self.LamM=self.FFm.denqFF_s() #no initial transpose
            self.LamM_dag=np.conj(np.transpose(self.LamM,(0,3,2,1) )) #no initial transpose
            
            
            ################################
            #reference attributes
            ################################
            
            print('\n')
            print('calculating refernce states...')

            disp_decoupled=Dispersion( latt, self.nbands, hpl_decoupled, hmin_decoupled)
            [self.psi_plus_decoupled_1bz,self.Ene_valley_plus_decoupled_1bz,self.psi_min_decoupled_1bz,self.Ene_valley_min_decoupled_1bz]=disp_decoupled.precompute_E_psi()

            self.psi_plus_decoupled=self.hpl.ExtendPsi(self.psi_plus_decoupled_1bz, self.latt.umkl+1)
            self.psi_min_decoupled=self.hmin.ExtendPsi(self.psi_min_decoupled_1bz, self.latt.umkl+1)
            
      
            Ik1bz=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
            Ik=self.latt.insertion_index( self.latt.KX,self.latt.KY, self.latt.KQX,self.latt.KQY)
            
            self.E_HFp=self.Ene_valley_plus[Ik1bz, :]
            self.E_HFp_K=self.Ene_valley_plus[Ik, :]
            self.E_HFp_ex=self.Ene_valley_plus[:, :]
            # self.Up=np.array(U_transf)
                
            self.E_HFm=self.Ene_valley_min[Ik1bz, :]
            self.E_HFm_K=self.Ene_valley_min[Ik, :]
            self.E_HFm_ex=self.Ene_valley_min[:, :]
            
            print(np.size(Ik),np.size(self.E_HFm), 'sizes of the energy arrays in HF module')
            # self.Um=np.array(U_transfm)

        #plots of the Bandstructre if needed
        self.plots_bands()
    
    
    def plots_bands(self):
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFp[:,0], s=40)
        plt.colorbar()
        plt.savefig("EHF1p_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFp[:,1], s=40)
        plt.colorbar()
        plt.savefig("EHF2p_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFm[:,0], s=40)
        plt.colorbar()
        plt.savefig("EHF1m_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFm[:,1], s=40)
        plt.colorbar()
        plt.savefig("EHF2m_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        
        #############################
        # high symmetry path
        #############################
        [path,kpath,HSP_index]=self.latt.embedded_High_symmetry_path(self.latt.KQX,self.latt.KQY)
        pth=np.arange(np.size(path))
        plt.plot(self.E_HFm_ex[path,0], ls='--', c='r')
        plt.plot(self.E_HFm_ex[path,1], ls='--', c='r')
        plt.scatter(pth,self.E_HFm_ex[path,0], c='r', s=9)
        plt.scatter(pth,self.E_HFm_ex[path,1], c='r', s=9)
        plt.plot(self.E_HFp_ex[path,0], c='b')
        plt.plot(self.E_HFp_ex[path,1], c='b')
        plt.scatter(pth,self.E_HFp_ex[path,0], c='b', s=9)
        plt.scatter(pth,self.E_HFp_ex[path,1], c='b', s=9)
        plt.savefig("dispHFp_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        plt.plot(self.latt.KQX[path], self.latt.KQY[path])
        VV=self.latt.boundary()
        plt.scatter(self.latt.KQX[path], self.latt.KQY[path], c='r')
        plt.plot(VV[:,0], VV[:,1], c='b')
        plt.savefig("HSPp_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        print("Bandwith,", np.max(self.E_HFp_ex[:,1])-np.min(self.E_HFp_ex[:,0]))
        
        return None
    
    def Proy_slat_comp(self,psi_plus, psi_minus):
        
        Pp=np.zeros([self.latt.NpoiQ, self.nbands_init, self.nbands_init],dtype=type(1j))
        Pm=np.zeros([self.latt.NpoiQ, self.nbands_init, self.nbands_init],dtype=type(1j))
        halfb=int(self.nbands_init/2)
        for k in range(self.latt.NpoiQ):
            vec=psi_plus[k,:,:]
            first=np.conj(vec.T)@(vec[:,:halfb])
            second=np.conj((vec[:,:halfb]).T)@vec
            Pp[k,:,:]=first@second
            
            vec=psi_minus[k,:,:]
            first=np.conj(vec.T)@(vec[:,:halfb])
            second=np.conj((vec[:,:halfb]).T)@vec
            Pm[k,:,:]=first@second
            
        return Pp,Pm
    
    def Slater_comp(self, preP):
        
        Delta=np.zeros([self.latt.NpoiQ, self.tot_nbands, self.tot_nbands],dtype=type(1j))
        Id=np.zeros([self.latt.NpoiQ, self.tot_nbands, self.tot_nbands],dtype=type(1j))

        for k in range(self.latt.NpoiQ):
            Delta[k,:self.ini_band_HF,:self.ini_band_HF]=np.eye(self.ini_band_HF)
            Id[k,self.ini_band_HF:self.fini_band_HF,self.ini_band_HF:self.fini_band_HF]=np.eye(self.nbands)
            
        #isolating bands
        P=preP[:,self.ini_band:self.fini_band,self.ini_band:self.fini_band]
        
        #guaranteeing degeneracy at K and K'
        Dirpoints=self.latt.all_dirac_ind()
        for k in Dirpoints:
            P[k,self.ini_band_HF:self.fini_band_HF,self.ini_band_HF:self.fini_band_HF]=np.eye(self.nbands)/2
            
        proj=P-Delta-Id/2
        
        # for k in Dirpoints:
        #     print(proj[k,self.ini_band_HF:self.fini_band_HF,self.ini_band_HF:self.fini_band_HF])
        return proj
    
    
    def Vq(self):
        V0=self.V0/self.latt.Npoi1bz
        qd=self.d_screening_norm*self.FFp.q
        Vq=V0*np.tanh(qd)/qd
        Vq[np.where(qd==0.0)[0]]=V0
        return Vq
    
    def Hartree(self, M):
        MT=M#np.transpose(M, (0,2,1))
        Numkl=int(self.latt.NpoiQ/self.latt.Npoi1bz) 
        X=np.zeros(np.shape(M),dtype=type(1j))
        origin_ID= np.argmin(self.latt.KQX**2 +self.latt.KQY**2 )
        for G in range(Numkl):
            VG=[origin_ID+G*self.latt.Npoi1bz ,origin_ID ]
            for k in range(self.latt.NpoiQ):
                X[k, :,:]=0
            X[k, :,:]=(X[k, :,:]+np.conj(X[k, :,:].T))/2
        #trace term only 
        return X
    
    def Fock(self, M):
        MT=np.transpose(M, (0,2,1))
        X=np.zeros(np.shape(M),dtype=type(1j))
        for k in range(self.latt.NpoiQ):
            for kq in range(self.latt.NpoiQ):
                # X[k, :,:]=X[k, :, :]+self.V[kq,k]*self.LamP_dag[k, kq ,:,:]@(MT[kq,:,:]@self.LamP[k, kq,:,:]) #initial transpose
                X[k, :,:]=X[k, :, :]+self.V[kq,k]*self.LamP_dag[kq,:,k,:]@(MT[kq,:,:]@self.LamP[kq,:,k,:]) #no initial transpose
            X[k, :,:]=(X[k, :,:]+np.conj(X[k, :,:].T))/2
        return -X
    
    def Form_factor_unitary(self, FormFactor_p, FormFactor_m):

        FormFactor_new_p=FormFactor_p 
        FormFactor_new_m=FormFactor_m
        if self.mode==1:
            for kq in range(self.latt.NpoiQ):
                
                Ukqp=self.Up[kq,:,:]
                Ukqm=self.Um[kq,:,:]
                
                Ukqp_dag=np.conj(Ukqp.T)
                Ukqm_dag=np.conj(Ukqm.T)
                
                for k in range(self.latt.NpoiQ):
                    
                    
                    Ukp=self.Up[k,:,:]
                    Ukm=self.Um[k,:,:]
                    
                    FormFactor_new_p[kq,:,k,:]=Ukqp_dag@((FormFactor_p[kq,:,k,:])@Ukp)
                    FormFactor_new_m[kq,:,k,:]=Ukqm_dag@((FormFactor_m[kq,:,k,:])@Ukm)
            
        return [FormFactor_new_p,FormFactor_new_m]
    


       
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
            self.qins_X=[qins_X]
            self.qins_Y=[qins_Y]
            self.nqs=2
        else:
            self.qins_X=[qins_X,-qins_X]
            self.qins_Y=[qins_Y,-qins_Y]
            self.nqs=2

        [self.alpha_ep, self.beta_ep,  self.Wupsilon, self.agraph, self.mass ]=cons #constants for the exraction of the effective velocity

        
    
        ################################
        #dispersion attributes
        ################################

        disp=Dispersion( latt, self.nbands_init, hpl, hmin)
        [self.psi_plus_1bz,self.Ene_valley_plus_1bz,self.psi_min_1bz,self.Ene_valley_min_1bz]=disp.precompute_E_psi()

        self.Ene_valley_plus=self.hpl.ExtendE(self.Ene_valley_plus_1bz[:,self.ini_band:self.fini_band] , self.latt.umkl+1)
        self.Ene_valley_min=self.hmin.ExtendE(self.Ene_valley_min_1bz[:,self.ini_band:self.fini_band] , self.latt.umkl+1)

        self.psi_plus=self.hpl.ExtendPsi(self.psi_plus_1bz, self.latt.umkl+1)
        self.psi_min=self.hmin.ExtendPsi(self.psi_min_1bz, self.latt.umkl+1)
        
        
        ################################
        #generating form factors
        ################################
        self.FFp=FormFactors(self.psi_plus, 1, latt, -1,self.hpl,self.nbands)
        self.FFm=FormFactors(self.psi_min, -1, latt, -1,self.hmin,self.nbands)
        
        
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
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denqFF_s(), self.FFm.denqFF_s())
                
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denqFF_s(), self.FFm.denqFF_s())
            
        else: # a- mode
            if mode=="L":
                self.L00p=self.FFp.denqFFL_a()
                self.L00m=self.FFm.denqFFL_a()
                self.Lnemp=self.FFp.NemqFFL_a()
                self.Lnemm=self.FFm.NemqFFL_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaL()
                   
            elif mode=="T":
                self.Lnemp=self.FFp.NemqFFT_a()
                self.Lnemm=self.FFm.NemqFFT_a()
                [self.Omega_FFp,self.Omega_FFm]=self.OmegaT()
                

            elif mode=="dens":
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denqFF_s(), self.FFm.denqFF_s())
            else:
                [self.Omega_FFp,self.Omega_FFm]=self.Form_factor_unitary(self.FFp.denqFF_a(), self.FFm.denqFF_a())

        
        
        [mat_s,matp, matm]=self.make_mat_phon()
        [mat_s_eps,matp2_eps, matm_eps]=self.make_mat_phon()
        Hp=matp2_eps+matp+np.conj(matp.T)
        Hm=matm_eps+matm+np.conj(matm.T)

        print(f"starting dispersion with {self.latt.Npoi} points..........")
        
        s=time.time()
        EHFp=[]
        U_transf=[]
        EHFm=[]
        U_transfm=[]
        paulix=np.array([[0,1],[1,0]])
        
        #if we use TRS to reconstruct minus valley
        diagI = np.zeros([int(self.nbands/2),int(self.nbands/2)]);
        for ib in range(int(self.nbands/2)):
            diagI[ib,int(self.nbands/2-1-ib)]=1;
        sx=np.kron(paulix,diagI)
        
        (self.Eigvals_p,self.Eigvect_p)= np.linalg.eigh(Hp) 
        (self.Eigvals_m,self.Eigvect_m)= np.linalg.eigh(Hm) 
        
        ##only works for instabilities that occur at repetitions of Gamma
        for k in range(self.latt.NpoiQ):
            H0p=np.diag(self.Ene_valley_plus[k,:])
            Delt=np.zeros([self.nbands,self.nbands])
            for q in range(self.nqs):
                k1=np.argmin( (self.latt.KQX-(self.latt.KQX[k]+self.qins_X[q]))**2 +(self.latt.KQY-(self.latt.KQY[k]+self.qins_Y[q]))**2)
                Delt=Delt+self.Omega_FFp[k1,:,k,:]*self.field
            Delt=Delt+np.conj(Delt)
            Hpl=H0p+Delt
            (Eigvals,Eigvect)= np.linalg.eigh(Hpl)  #returns sorted eigenvalues

            EHFp.append(Eigvals)
            U_transf.append(Eigvect)
            
            H0min=np.diag(self.Ene_valley_min[k,:])
            Delt=np.zeros([self.nbands,self.nbands])
            for q in range(self.nqs):
                k1=np.argmin( (self.latt.KQX-(self.latt.KQX[k]+self.qins_X[q]))**2 +(self.latt.KQY-(self.latt.KQY[k]+self.qins_Y[q]))**2)
                Delt=Delt+self.Omega_FFm[k1,:,k,:]*self.field
            Delt=Delt+np.conj(Delt)
            Hmin=H0min+Delt
            
            (Eigvals,Eigvect)= np.linalg.eigh(Hmin)  #returns sorted eigenvalues
            EHFm.append(Eigvals)
            U_transfm.append(Eigvect)
                
                
        Ik=self.latt.insertion_index( self.latt.KX1bz,self.latt.KY1bz, self.latt.KQX,self.latt.KQY)
        self.E_HFp=np.array(EHFp)[Ik, :]
        self.E_HFp_K=self.hpl.ExtendE(self.E_HFp, self.latt.umkl)
        self.E_HFp_ex=self.hpl.ExtendE(self.E_HFp, self.latt.umkl+1)
        self.Up=np.array(U_transf)
        
        self.E_HFm=np.array(EHFm)[Ik, :]
        self.E_HFm_K=self.hmin.ExtendE(self.E_HFm, self.latt.umkl)
        self.E_HFm_ex=self.hmin.ExtendE(self.E_HFm, self.latt.umkl+1)
        self.Um=np.array(U_transfm)
        
        #plots of the Bandstructre if needed
        print(np.size(Ik),np.size(self.E_HFp),np.size(self.E_HFm), 'sizes of the energy arrays in HF module')
        print(self.ini_band,self.fini_band)
        self.plots_bands()
        
        e=time.time()
        print(f'time for Diag {e-s}')

        #plots of the Bandstructre if needed
        # self.plots_bands()
    
    
    def plots_bands(self):
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFp[:,0], s=40)
        plt.colorbar()
        plt.savefig("EHF1p_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFp[:,1], s=40)
        plt.colorbar()
        plt.savefig("EHF2p_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFm[:,0], s=40)
        plt.colorbar()
        plt.savefig("EHF1m_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        plt.scatter(self.latt.KX1bz,self.latt.KY1bz, c=self.E_HFm[:,1], s=40)
        plt.colorbar()
        plt.savefig("EHF2m_kappa"+str(self.hpl.kappa)+".png")
        plt.close()
        
        
        #############################
        # high symmetry path
        #############################
        [path,kpath,HSP_index]=self.latt.embedded_High_symmetry_path(self.latt.KQX,self.latt.KQY)
        pth=np.arange(np.size(path))
        plt.plot(self.E_HFm_ex[path,0], ls='--', c='r')
        plt.plot(self.E_HFm_ex[path,1], ls='--', c='r')
        plt.scatter(pth,self.E_HFm_ex[path,0], c='r', s=9)
        plt.scatter(pth,self.E_HFm_ex[path,1], c='r', s=9)
        plt.plot(self.E_HFp_ex[path,0], c='b')
        plt.plot(self.E_HFp_ex[path,1], c='b')
        plt.scatter(pth,self.E_HFp_ex[path,0], c='b', s=9)
        plt.scatter(pth,self.E_HFp_ex[path,1], c='b', s=9)
        plt.savefig("dispHFp_kappa"+str(self.hpl.kappa)+".png")
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
    """
    [summary]
    Tests different methods in the Hamiltonian module
    
    In:
        integer that picks the chemical potential for the calculation
        integer linear number of samples to be used 

        
    Out: 

    
    
    Raises:
        Exception: ValueError, IndexError Input integer in the firs argument to choose chemical potential for desired filling
        Exception: ValueError, IndexError Input int for the number of k-point samples total kpoints =(arg[2])**2
    """
    
    try:
        filling_index=int(sys.argv[1]) #0-25

    except (ValueError, IndexError):
        raise Exception("Input integer in the firs argument to choose chemical potential for desired filling")

    try:
        Nsamp=int(sys.argv[2])

    except (ValueError, IndexError):
        raise Exception("Input int for the number of k-point samples total kpoints =(arg[2])**2")

    try:
        modulation_theta=float(sys.argv[3])

    except (ValueError, IndexError):
        raise Exception("Input double to modulate the twist angle")

    try:
        modulation_kappa=float(sys.argv[4])

    except (ValueError, IndexError):
        raise Exception("Input double to modulate kappa")

    print("\n \n")
    print("lattice sampling...")
    #Lattice parameters 
    #lattices with different normalizations
    theta=modulation_theta*np.pi/180  # magic angle
    c6sym=True
    umkl=1 #the number of umklaps where we calculate an observable ie Pi(q), for momentum transfers we need umkl+1 umklapps when scattering from the 1bz
    l=MoireLattice.MoireTriangLattice(Nsamp,theta,0,c6sym,umkl)
    lq=MoireLattice.MoireTriangLattice(Nsamp,theta,2,c6sym,umkl) #this one
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
    eps_inv = 1/10
    d_screening=20*(1e-9)/a_graphene
    d_screening_norm=d_screening*lq.qnor()
    epsilon_0 = 8.85*1e-12
    ev_conv = e_el
    Vcoul=( e_el*e_el*eps_inv*d_screening/(2*epsilon_0*a_graphene) )
    V0= (  Vcoul/lq.Vol_WZ() )/ev_conv
    print(V0, 'la energia de coulomb en ev')
    
    hpl=Ham_BM(hvkd, alph, 1, lq, kappa, PH,1)
    hmin=Ham_BM(hvkd, alph, -1, lq, kappa, PH,1)

    #CALCULATING FILLING AND CHEMICAL POTENTIAL ARRAYS
    # Ndos=100
    # print("\n \n")
    # print("chem pot determination...")
    # Ndos=18
    # ldos=MoireLattice.MoireTriangLattice(Ndos,theta,2,c6sym,umkl)
    # disp=Dispersion( ldos, nbands, hpl, hmin)
    # Nfils=7
    # # [fillings,mu_values]=disp.mu_filling_array(Nfils, True, False, False) at the magic angle
    # [fillings,mu_values]=disp.mu_filling_array(Nfils, False, False,True)
    # filling_index=int(sys.argv[1]) 
    # mu=mu_values[filling_index]
    # filling=fillings[filling_index]
    # print("\n \n")
    # print("CHEMICAL POTENTIAL AND FILLING", mu, filling)
    # print("\n \n")
    
    ## Dispersion along high symmetry directions
    # disp=Dispersion( lq, nbands, hpl, hmin)
    # disp.High_symmetry()
    
    ## Fermi surface contour plot
    # mu=mu_values[int(Nfils/2)]
    # filling=fillings[int(Nfils/2)]
    # [xFS_dense,yFS_dense]=disp.FS_contour(100, mu, hpl)
    # plt.scatter(xFS_dense,yFS_dense)
    # plt.savefig(f"contour_{mu}.png")
    # plt.close()

    
    """
    [psi_plus,Ene_valley_plus,psi_min,Ene_valley_min]=disp.precompute_E_psi()
    for mu in mu_values:
        print(mu)
        f=Ene_valley_plus[:,0]
        plt.scatter(lq.KX1bz, lq.KY1bz, c=f)
        plt.colorbar()
        eps=np.mean(abs(np.diff(f)))*0.5
        aa=np.where(np.abs(f-mu)<eps)[0]
        # print(aa, eps)
        plt.scatter(lq.KX1bz[aa],lq.KY1bz[aa], c='r', s=3)
        plt.savefig('disp_p1_'+str(modulation_kappa)+'_'+str(modulation_theta)+'_'+str(mu)+'.png')
        plt.close()

        f=Ene_valley_plus[:,1]
        plt.scatter(lq.KX1bz, lq.KY1bz, c=f)
        plt.colorbar()
        eps=np.mean(abs(np.diff(f)))*0.5
        aa=np.where(np.abs(f-mu)<eps)[0]
        # print(aa, eps)
        plt.scatter(lq.KX1bz[aa],lq.KY1bz[aa], c='r', s=3)
        plt.savefig('disp_p2_'+str(modulation_kappa)+'_'+str(modulation_theta)+'_'+str(mu)+'.png')
        plt.close()

        
        f=Ene_valley_min[:,0]
        plt.scatter(lq.KX1bz, lq.KY1bz, c=f)
        plt.colorbar()
        eps=np.mean(abs(np.diff(f)))*0.5
        aa=np.where(np.abs(f-mu)<eps)[0]
        # print(aa, eps)
        plt.scatter(lq.KX1bz[aa],lq.KY1bz[aa], c='r', s=3)
        plt.savefig('disp_m1_'+str(modulation_kappa)+'_'+str(modulation_theta)+'_'+str(mu)+'.png')
        plt.close()
        
        f=Ene_valley_min[:,1]
        plt.scatter(lq.KX1bz, lq.KY1bz, c=f)
        plt.colorbar()
        eps=np.mean(abs(np.diff(f)))*0.5
        aa=np.where(np.abs(f-mu)<eps)[0]
        # print(aa, eps)
        plt.scatter(lq.KX1bz[aa],lq.KY1bz[aa], c='r', s=3)
        plt.savefig('disp_m2_'+str(modulation_kappa)+'_'+str(modulation_theta)+'_'+str(mu)+'.png')
        plt.close()
    """
        
    hpl_decoupled=Ham_BM(hvkd, alph, 1, lq, kappa, PH,0)
    hmin_decoupled=Ham_BM(hvkd, alph, -1, lq, kappa, PH,0)
    
    substract=0 #0 for decoupled layers
    mode=0
    HB=HF_BandStruc( lq, hpl, hmin, hpl_decoupled,hmin_decoupled, nremote_bands, nbands, substract,  [V0, d_screening_norm], mode)
    # plt.scatter(lq.KQX,lq.KQY)
    # VV=lq.all_dirac_ind()
    # print(np.shape(VV))
    # plt.scatter(lq.KQX[VV],lq.KQY[VV])
    # plt.savefig("lattice.png")
    # plt.show()
if __name__ == '__main__':
    import sys
    sys.exit(main())  # next section explains the use of sys.exit
