import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time
import matplotlib.pyplot as plt
 

class MoireTriangLattice:

    def __init__(self, Npoints, theta, normed):

        self.Npoints = Npoints
        self.theta = theta
        self.a =np.array([[1,0],[1/2,np.sqrt(3)/2]])  #original graphene lattice vectors: rows are basis vectors
        self.b =(2*np.pi)*np.array([[1,-1/np.sqrt(3)],[0,2/np.sqrt(3)]]) # original graphene reciprocal lattice vectors : rows are basis vectors
        self.rot_min =np.array([[np.cos(theta/2),np.sin(theta/2)],[-np.sin(theta/2),np.cos(theta/2)]]) #rotation matrix -thet/2
        self.rot_plus =np.array([[np.cos(theta/2),-np.sin(theta/2)],[np.sin(theta/2),np.cos(theta/2)]]) #rotation matrix +thet/2
        self.normed=normed

        #some symmetries:
        #C2z
        th1=np.pi
        self.C2z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C3z
        th1=2*np.pi/3
        self.C3z=np.array([[np.cos(th1),np.sin(th1)],[-np.sin(th1),np.cos(th1)]]) #rotation matrix 
        #C2x inv
        self.C2x=np.array([[1,0],[0,-1]]) #rotation matrix 

        if normed==0:
            self.GMvec=self.GM_vec()
            self.VolMBZ=self.Vol_MBZ()
            self.q=self.qvect()
            self.GMs=self.GM()

        elif normed==1:
            [GM1,GM2]=self.GM_vec()
            [q1,q2,q3]=self.qvect()
            Gnorm=self.GM() #normalized to the reciprocal lattice vector
            self.GMvec=[GM1/Gnorm,GM2/Gnorm]
            self.GMs=self.GM()/Gnorm
            self.VolMBZ=self.Vol_MBZ()/(Gnorm**2)
            self.q=[q1/Gnorm,q2/Gnorm,q3/Gnorm]

        else:
            [GM1,GM2]=self.GM_vec()
            [q1,q2,q3]=self.qvect()
            Gnorm=self.qnor() #normalized to the q1 vector
            # print(Gnorm)
            self.GMvec=[GM1/Gnorm,GM2/Gnorm]
            self.GMs=self.GM()/Gnorm
            self.VolMBZ=self.Vol_MBZ()/(Gnorm**2)
            self.q=[q1/Gnorm,q2/Gnorm,q3/Gnorm]
            
        #G processes
        self.MGS_1=[[0,1],[1,0],[0,-1],[-1,0],[-1,-1],[1,1]] #1G
        self.MGS1=self.MGS_1+[[-1,-2],[-2,-1],[-1,1],[1,2],[2,1],[1,-1]] #1G and possible corners
        self.MGS_2=self.MGS1+[[-2,-2],[0,-2],[2,0],[2,2],[0,2],[-2,0]] #2G
        self.MGS2=self.MGS_2+[[-2,-3],[-1,-3],[1,-2],[2,-1],[3,1],[3,2],[2,3],[1,3],[-1,2],[-2,1],[-3,-1],[-3,-2]] #2G and possible corners
        self.MGS_3=self.MGS2+[[-3,-3],[0,-3],[3,0],[3,3],[0,3],[-3,0]] #3G


    def __repr__(self):
        return "lattice( LX={w}, twist_angle={c})".format(h=self.Npoints, c=self.theta)

    #reciprocal vectors for the superlattice
    def GM_vec(self):
        #rotated vectors
        astar1=self.b@self.rot_min #top layer rotated -theta
        astar2=self.b@self.rot_plus  #bottom layer rotated +theta
        GM1=astar1[0,:]-astar2[0,:]
        GM2=astar1[1,:]-astar2[1,:]
        return [GM1,GM2]
    
    #reciprocal space period for the superlattice
    def GM(self):
        [GM1,GM2]=self.GM_vec()
        return np.sqrt(GM1@GM1)

    #Moire real space superlattice vectors
    def LM_vec(self):

        #rotated vectors
        [GM1,GM2]=self.GM_vec()
        LM1=2*np.pi*np.array([GM2[1],-GM2[0]])/la.det(np.array([GM1,GM2]))
        LM2=2*np.pi*np.array([-GM1[1],GM1[0]])/la.det(np.array([GM1,GM2]))
        return [LM1,LM2]
    
    #moire period in real space
    def LM(self):
        [LM1,LM2]=self.LM_vec()
        return np.sqrt(LM1@LM1)

    #FBZ volume
    def Vol_MBZ(self):
        [GM1,GM2]=self.GM_vec()
        zhat=np.array([0,0,1])
        b_1=np.array([GM1[0],GM1[1],0]) # Moire reciprocal lattice vect extended
        b_2=np.array([GM2[0],GM2[1],0]) # Moire reciprocal lattice vect extended
        Vol_rec=np.cross(b_1,b_2)@zhat
        return Vol_rec
    
    #WZ volume
    def Vol_WZ(self):
        [LM1,LM2]=self.LM_vec()
        zhat=np.array([0,0,1])
        b_1=np.array([LM1[0],LM1[1],0]) # Moire reciprocal lattice vect extended
        b_2=np.array([LM2[0],LM2[1],0]) # Moire reciprocal lattice vect extended
        Vol_rec=np.cross(b_1,b_2)@zhat
        return Vol_rec

    #lowest order q vectors that contribute to the moire potential (plus valley)
    def qvect(self):

        astar1=self.b@self.rot_min #top layer rotated -theta
        astar2=self.b@self.rot_plus  #bottom layer rotated +theta
        Kplus1=+(2*astar1[0,:]+astar1[1,:])/3  #K of top layer
        Kplus2=+(2*astar2[0,:]+astar2[1,:])/3 #K of bottom layer
        q1=Kplus1-Kplus2
        q2=self.C3z@q1
        q3=self.C3z@q2

        return [q1,q2,q3]

    def qnor(self):
        [q1,q2,q3]=self.qvect()
        return la.norm(q1)

    #hexagon where the pointy side is up
    def hexagon1(self,pos,Radius_inscribed_hex):
        y,x = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters #effective rotation
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge
    
    #hexagon where the flat side is up
    def hexagon2(self,pos,Radius_inscribed_hex):
        x,y = map(abs, pos) #taking the absolute value of the rotated hexagon, only first quadrant matters
        return y < np.sqrt(3)* min(Radius_inscribed_hex - x, Radius_inscribed_hex / 2) #checking if the point is under the diagonal of the inscribed hexagon and below the top edge

    #gets high symmetry points
    def FBZ_points(self,b_1,b_2):
        #creating reciprocal lattice
        Np=4
        n1=np.arange(-Np,Np+1)
        n2=np.arange(-Np,Np+1)
        Recip_lat=[]
        for i in n1:
            for j in n2:
                point=b_1*i+b_2*j
                Recip_lat.append(point)

        #getting the nearest neighbours to the gamma point
        Recip_lat_arr=np.array(Recip_lat)
        dist=np.round(np.sqrt(np.sum(Recip_lat_arr**2, axis=1)),decimals=10)
        sorted_dist=np.sort(list(set(dist)) )
        points=Recip_lat_arr[np.where(dist<sorted_dist[2])[0]]

        #getting the voronoi decomposition of the gamma point and the nearest neighbours
        vor = Voronoi(points)
        Vertices=(vor.vertices)

        #ordering the points counterclockwise in the -pi,pi range
        angles_list=list(np.arctan2(Vertices[:,1],Vertices[:,0]))
        Vertices_list=list(Vertices)

        # joint sorting the two lists for angles and vertices for convenience later.
        # the linear plot routine requires the points to be in order
        # atan2 takes into acount quadrant to get the sign of the angle
        angles_list, Vertices_list = (list(t) for t in zip(*sorted(zip(angles_list, Vertices_list))))

        ##getting the M points as the average of consecutive K- Kp points
        Edges_list=[]
        for i in range(len(Vertices_list)):
            Edges_list.append([(Vertices_list[i][0]+Vertices_list[i-1][0])/2,(Vertices_list[i][1]+Vertices_list[i-1][1])/2])

        Gamma=[0,0]
        K=Vertices_list[0::2]
        Kp=Vertices_list[1::2]
        M=Edges_list[0::2]
        Mp=Edges_list[1::2]

        return Vertices_list, Gamma, K, Kp, M, Mp

    #creates two arrays containing X and Y coordinates for the lattice points                                                     
    def Generate_lattice(self):
        [GM1,GM2]=self.GM_vec()
        LM=self.LM()
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        #will filter points that are in a hexagon inscribed in a circle of radius Radius_inscribed_hex
        Radius_inscribed_hex=1.0000005*k_window_sizey


        print("starting sampling in reciprocal space....")
        s=time.time()

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP,LP+1,1)
        nn2=np.arange(-LP,LP+1,1)

        nn_1,nn_2=np.meshgrid(nn1,nn2)

        nn_1p=[]
        nn_2p=[]
        for x in nn1:
            for y in nn2:
                kx=(2*np.pi*x/LP)/LM
                ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))/LM
                if self.hexagon1( ( kx, ky), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)

        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")

        nn_1pp=np.array(nn_1p)
        nn_2pp=np.array(nn_2p)

        KX=(2*np.pi*nn_1pp/LP)/LM
        KY= (2*(2*np.pi*nn_2pp/LP - np.pi*nn_1pp/LP)/np.sqrt(3))/LM

        #Making the sampling lattice commensurate with the MBZ
        fact=K[2][1]/np.max(KY)
        KX=KX*fact
        KY=KY*fact

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM() #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector
        return [KX/Gnorm,KY/Gnorm]

    def Generate_Umklapp_lattice(self, KX, KY, numklaps):
        if numklaps>=0.9:
            Npoi=np.size(KX)
            [GM1,GM2]=self.GMvec
            if numklaps==1:
                GSu=self.MGS_1
            elif numklaps==2:
                GSu=self.MGS_2
            elif numklaps==3:
                GSu=self.MGS_3
            else:
                GSu=[]

            K_um=[]
            #adding original samples
            for i in range(Npoi):
                K_um.append([round(KX[i], 8),round(KY[i], 8)])

            #extending for different moire lattice vectors
            for mg in GSu:
                for i in range(Npoi):
                    K_um.append([round(KX[i]+mg[0]*GM1[0]+mg[1]*GM2[0], 8),round(KY[i]+mg[0]*GM1[1]+mg[1]*GM2[1], 8)])

            unique_data =np.array( [list(i) for i in set(tuple(i) for i in K_um)])
            print("K umkplapp unique grid ",np.shape(unique_data))
            KumX=unique_data[:,0]
            KumY=unique_data[:,1]
            # plt.scatter(KumX,KumY)
            # plt.scatter(KX,KY)
            # plt.show()
            
            return [KumX,KumY]
        else:
            return [KX,KY]

    
    #returns the lattice vectors that span the sampling lattice when multiplied by integers                                              
    def Generating_vec_samp_lattice(self, scale_fac_latt):
        [GM1,GM2]=self.GM_vec()
        LM=self.LM()
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        #will filter points that are in a hexagon inscribed in a circle of radius Radius_inscribed_hex
        Radius_inscribed_hex=1.0000005*k_window_sizey


        print("starting sampling in reciprocal space....")
        s=time.time()

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP,LP+1,1)
        nn2=np.arange(-LP,LP+1,1)

        nn_1,nn_2=np.meshgrid(nn1,nn2)

        nn_1p=[]
        nn_2p=[]
        for x in nn1:
            for y in nn2:
                kx=(2*np.pi*x/LP)/LM
                ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))/LM
                if self.hexagon1( ( kx, ky), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)

        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")

        nn_1pp=np.array(nn_1p)
        nn_2pp=np.array(nn_2p)

        KX=(2*np.pi*nn_1pp/LP)/LM
        KY= (2*(2*np.pi*nn_2pp/LP - np.pi*nn_1pp/LP)/np.sqrt(3))/LM

        #Making the sampling lattice commensurate with the MBZ
        fact=K[2][1]/np.max(KY)
        KX=KX*fact
        KY=KY*fact

        KX_0=(2*np.pi*1/LP)/LM
        KY_0= (2*(2*np.pi*0/LP - np.pi*1/LP)/np.sqrt(3))/LM

        KX_1=(2*np.pi*0/LP)/LM
        KY_1= (2*(2*np.pi*1/LP - np.pi*0/LP)/np.sqrt(3))/LM
    
        #Making the sampling lattice commensurate with the MBZ
        K_0=np.array([KX_0,KY_0])*fact
        K_1=np.array([KX_1,KY_1])*fact

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM() #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector

        return [K_0/Gnorm,K_1/Gnorm]
    
    
    #same as Generate lattice but for the original graphene (FBZ of triangular lattice)
    def Generate_lattice_og(self):

        
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(self.b[0,:],self.b[1,:])

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        #will filter points that are in a hexagon inscribed in a circle of radius Radius_inscribed_hex
        Radius_inscribed_hex=1.0000005*k_window_sizex


        print("starting sampling in reciprocal space....")
        s=time.time()

        #initial grid that will be filtered
        LP=self.Npoints
        nn1=np.arange(-LP,LP+1,1)
        nn2=np.arange(-LP,LP+1,1)

        nn_1,nn_2=np.meshgrid(nn1,nn2)

        nn_1p=[]
        nn_2p=[]
        for x in nn1:
            for y in nn2:
                kx=(2*np.pi*x/LP)
                ky=(2*(2*np.pi*y/LP - np.pi*x/LP)/np.sqrt(3))
                if self.hexagon2( ( kx, ky ), Radius_inscribed_hex ):
                    nn_1p.append(x)
                    nn_2p.append(y)

        e=time.time()
        print("finished sampling in reciprocal space....t=",e-s," s")

        nn_1pp=np.array(nn_1p)
        nn_2pp=np.array(nn_2p)

        KX=(2*np.pi*nn_1pp/LP)
        KY= (2*(2*np.pi*nn_2pp/LP - np.pi*nn_1pp/LP)/np.sqrt(3))

        #Making the sampling lattice commensurate with the MBZ
        fact=K[1][0]/np.max(KX)
        KX=KX*fact
        KY=KY*fact
        
        return [KX,KY]

    def Generate_momentum_transfer_lattice(self, KX, KY):

        Npoi=np.shape(KY)[0]
        KQ=[]
        for i in range(Npoi):
            for j in range(Npoi):
                KQ.append([round(KX[j]+KX[i], 8),round(KY[j]+KY[i], 8)])
        # plt.scatter(KX,KY)
        # plt.show()
        KQarr=np.array(KQ)
        print("Kq non unique grid ",np.shape(KQarr))

        #unique_data =np.array( [list(x) for x in set(tuple(x) for x in KQ)])
        unique_data =np.array( [list(i) for i in set(tuple(i) for i in KQ)])
        print("Kq grid unique",np.shape(unique_data))
        print("K grid ",np.shape(KX))
        KQX=unique_data[:,0]
        KQY=unique_data[:,1]

        Ik=[]
        for j in range(Npoi):
            indmin=np.argmin(np.sqrt((KQX-KX[j])**2+(KQY-KY[j])**2))
            Ik.append(indmin)

        return [KQX, KQY, Ik]
    
    def Generate_momentum_transfer_umklapp_lattice(self, KX, KY,  KXu, KYu):

        Npoi=np.shape(KYu)[0]
        KQ=[]
        for i in range(Npoi):
            for j in range(np.shape(KY)[0]):
                KQ.append([round(KX[j]+KXu[i], 8),round(KY[j]+KYu[i], 8)])
        # plt.scatter(KX,KY)
        # plt.show()
        KQarr=np.array(KQ)
        print("Kq non unique grid ",np.shape(KQarr))

        #unique_data =np.array( [list(x) for x in set(tuple(x) for x in KQ)])
        unique_data =np.array( [list(i) for i in set(tuple(i) for i in KQ)])
        print("Kq grid unique",np.shape(unique_data))
        print("K grid ",np.shape(KX))
        KQX=unique_data[:,0]
        KQY=unique_data[:,1]

        Ik=[]
        for j in range(Npoi):
            indmin=np.argmin(np.sqrt((KQX-KXu[j])**2+(KQY-KYu[j])**2))
            Ik.append(indmin)

        # plt.scatter(KQX, KQY)
        # plt.scatter(KQX[Ik], KQY[Ik])
        # plt.show()
        return [KQX, KQY, Ik]

    #normal linear interpolation to generate samples accross High symmetry points
    def linpam(self,Kps,Npoints_q):
        Npoints=len(Kps)
        t=np.linspace(0, 1, Npoints_q)
        linparam=np.zeros([Npoints_q*(Npoints-1),2])
        for i in range(Npoints-1):
            linparam[i*Npoints_q:(i+1)*Npoints_q,0]=Kps[i][0]*(1-t)+t*Kps[i+1][0]
            linparam[i*Npoints_q:(i+1)*Npoints_q,1]=Kps[i][1]*(1-t)+t*Kps[i+1][1]

        return linparam
        
    def High_symmetry_path(self):
        [GM1,GM2]=self.GM_vec()
        VV, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)
        VV=VV+[VV[0]] #verices

        L=[]
        L=L+[K[0]]+[Gamma]+[M[0]]+[Kp[-1]] ##path in reciprocal space
        # L=L+[K[0]]+[Gamma]+[M[0]]+[K[0]] ##path in reciprocal space Andrei paper

        Nt_points=80
        kp_path=self.linpam(L,Nt_points)

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM() #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector

        return kp_path/Gnorm

    def boundary(self):
        [GM1,GM2]=self.GM_vec()
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)

        if self.normed==0:
            Gnorm=1
        elif self.normed==1:
            Gnorm=self.GM() #normalized to the reciprocal lattice vector
        else:
            Gnorm=self.qnor() #normalized to the q1 vector

        return np.array(Vertices_list+[Vertices_list[0]])/Gnorm

    ### SYMMETRY OPERATIONS ON THE LATTICE
    def C2zLatt(self,KX,KY):
        ##KX and KY are one dimensional arrays
        Npoi = np.size(KX)
        KXc2z=KX*self.C2z[0,0]+KY*self.C2z[0,1]
        KYc2z=KX*self.C2z[1,0]+KY*self.C2z[1,1]
        Indc2z=np.zeros(Npoi) # indices of the initial lattice in the rotated lattice
        for i in range(Npoi):
            #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
            Indc2z[i]=np.argmin( (KX-KXc2z[i])**2 +(KY-KYc2z[i])**2)

        return [KXc2z,KYc2z, Indc2z]

    def C2xLatt(self,KX,KY):
        ##KX and KY are one dimensional arrays
        Npoi = np.size(KX)
        KXc2x=KX*self.C2x[0,0]+KY*self.C2x[0,1]
        KYc2x=KX*self.C2x[1,0]+KY*self.C2x[1,1]
        Indc2x=np.zeros(Npoi) # indices of the initial lattice in the rotated lattice
        for i in range(Npoi):
            #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
            Indc2x[i]=np.argmin( (KX-KXc2x[i])**2 +(KY-KYc2x[i])**2)

        return [KXc2x,KYc2x, Indc2x]

    def C3zLatt(self,KX,KY):
        ##KX and KY are one dimensional arrays
        Npoi = np.size(KX)
        KXc3z=KX*self.C3z[0,0]+KY*self.C3z[0,1]
        KYc3z=KX*self.C3z[1,0]+KY*self.C3z[1,1]
        Indc3z=np.zeros(Npoi) # indices of the initial lattice in the rotated lattice
        for i in range(Npoi):
            #this works well because the rotation is a symmetry of the sampling lattice and the sampling lattice is commensurate
            Indc3z[i]=np.argmin( (KX-KXc3z[i])**2 +(KY-KYc3z[i])**2)

        return [KXc3z,KYc3z, Indc3z]

    #to check whether this is working uncomment the plot statments
    def findpath(self,Kps,KX,KY):

        path=np.empty((0))
        pthK=[]
        HSP_index=[]
        counter_path=0
        nnlist=[[0,1],[1,0],[0,-1],[-1,0],[-1,-1],[1,1]] #specific for the reciprocal lattice vectors that I picked
        HSP_index.append(counter_path)
        NHSpoints=np.shape(Kps)[0]
        

        [k1,k2]=self.Generating_vec_samp_lattice( np.max(KY))


        amin=np.linalg.norm(k1)
        # print(amin, np.sqrt( (KX[1]-KX[0])**2+(KY[1]-KY[0])**2 ))      
        l=np.argmin(  (Kps[0][0]-KX)**2 + (Kps[0][1]-KY)**2 )

        path=np.append(path,int(l)) 
        pthK.append([KX[l],KY[l]])

        
        for indhs in range(NHSpoints-1):

            c=0
            c2=0
            
            
            dist=np.sqrt( (Kps[indhs+1][0]-KX[l])**2 + (Kps[indhs+1][1]-KY[l])**2)
            while ( c2<1  and  dist>=0.8*amin):
                dists=[]
                KXnn=[]
                KYnn=[]

                dist_pre=dist
                # print(Kps[indhs+1], dist, amin)
                # plt.scatter(KX,KY)
                for nn in range(6): #coordination number is 6
                    kxnn= KX[l]+nnlist[nn][0]*k1[0]+nnlist[nn][1]*k2[0]
                    kynn= KY[l]+nnlist[nn][0]*k1[1]+nnlist[nn][1]*k2[1]
                    di=np.sqrt( (Kps[indhs+1][0]-kxnn)**2 + (Kps[indhs+1][1]-kynn)**2)
                    dists.append(di)
                    KXnn.append(kxnn)
                    KYnn.append(kynn)
                    # plt.scatter(kxnn,kynn)
                    # print(kxnn,kynn)
                
                

                
                dist=np.min(np.array(dists))
                ind_min=np.argmin(np.array(dists))

                
                # print(KX[l],KY[l],KXnn[ind_min],KYnn[ind_min])
                
                # plt.scatter(KXnn[ind_min],KYnn[ind_min], marker='x')
                # plt.show()
                l=np.argmin(  np.sqrt((KXnn[ind_min]-KX)**2 + (KYnn[ind_min]-KY)**2 ))
                # print(l)
                # print(KX[l],KY[l],KXnn[ind_min],KYnn[ind_min])

                if dist_pre==dist:
                    c2=c2+1
                

                path=np.append(path,int(l))
                pthK.append([KX[l],KY[l]])
                # print([KX[i,j],KY[i,j]],[Kps[indhs+1][0],Kps[indhs+1][1]], dist)

                c=c+1
                counter_path=counter_path+1
        
            HSP_index.append(counter_path)
            
            
        return path,np.array(pthK),HSP_index

    def embedded_High_symmetry_path(self, KX,KY):
        [GM1,GM2]=self.GMvec
        VV, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)
        VV=VV+[VV[0]] #verices

        Kps=[]
        Kps=Kps+[K[1]]+[Gamma]+[M[0]]+[Kp[2]]
        [path,kpath,HSP_index]=self.findpath(Kps,KX,KY)


        return [path,kpath,HSP_index]

    def kwrap_FBZ(self,kx,ky):
        dmin=kx**2+ky**2
        G=[0,0]
        [GM1,GM2]=self.GMvec
        for MG in self.MGS1:
            d=(kx-MG[0]*GM1[0]-MG[1]*GM2[0])**2+(ky-MG[0]*GM1[1]-MG[1]*GM2[1])**2
            if d<dmin+1e-10: #slightly more stringent to keep terms fixed if they differ by machine precission 
                dmin=d
                G=MG
        kxp=kx-G[0]*GM1[0]-G[1]*GM2[0]
        kyp=ky-G[0]*GM1[1]-G[1]*GM2[1]
        return kxp,kyp

    def mask_KPs(self, KX,KY, thres):
        [GM1,GM2]=self.GMvec
        Vertices_list, Gamma, K, Kp, M, Mp=self.FBZ_points(GM1,GM2)

        k_window_sizey = K[2][1] 
        k_window_sizex = K[1][0] 

        Radius_inscribed_hex=1.0000005*k_window_sizey
        K=np.sqrt(KX**2+KY**2)
        ind=np.where(K<k_window_sizex*thres)
        return [KX[ind],KY[ind], ind]
    
    def Umklapp_List(self, umklapps):
        #G processes
        G=self.GMs
        Gu=[]
        [GM1, GM2]=self.GMvec
        for i in range(-10,10):
            for j in range(-10,10):
                Gp=i*GM1+j*GM2
                Gpn=np.sqrt(Gp.T@Gp)
                if  Gpn<=G*(umklapps+0.1):
                    Gu=Gu+[[i,j]]
        #             plt.scatter(Gp[0], Gp[1], c='r')
        #         else:
        #             plt.scatter(Gp[0], Gp[1], c='b')

        # thetas=np.linspace(0,2*np.pi, 100)
        # xx=umklapps*G*np.cos(thetas)
        # yy=umklapps*G*np.sin(thetas)
        # plt.plot(xx,yy)
        # plt.savefig("ulat.png")
        # plt.close()
        return Gu
    
    def Generate_Umklapp_lattice2(self, KX, KY, numklaps):
        Gu=self.Umklapp_List(numklaps)
        [GM1, GM2]=self.GMvec
        KXu=[]
        KYu=[]
        
        for GG in Gu:
            KXu=KXu+[KX+GG[0]*GM1[0]+GG[1]*GM2[0]]
            KYu=KYu+[KY+GG[0]*GM1[1]+GG[1]*GM2[1]]
        
        KXum=np.concatenate( KXu )
        KYum=np.concatenate( KYu )
        return [KXum, KYum]
    def insertion_index(self, KX,KY, KQX,KQY):
        #list of size Npoi that has the index of K in KQ
        Npoi=np.size(KX)
        Ik=[]
        for j in range(Npoi):
            indmin=np.argmin(np.sqrt((KQX-KX[j])**2+(KQY-KY[j])**2))
            Ik.append(indmin)
        return Ik

