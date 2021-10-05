import numpy as np
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy import linalg as la
import time
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
                if self.hexagon2( ( kx, ky), Radius_inscribed_hex ):
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
        # L=L+[K[0]]+[Gamma]+[M[0]]+[Kp[-1]] ##path in reciprocal space
        L=L+[K[0]]+[Gamma]+[M[0]]+[K[0]] ##path in reciprocal space Andrei paper

        Nt_points=20
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

        
#TODO update save and read lattice update path in reciprocal space for Delafossite