import numpy as np

Nk=10
nbt=5
nac=2
a=np.random.random([Nk,nbt,nac])
b=np.random.random([Nk,nbt,nac])
d=np.zeros([Nk,Nk, nac,nac])
print(np.shape(d), np.shape(a), np.shape(b))
for i in range(Nk):
    for j in range(Nk):
        for n in range(nac):
            for m in range(nac):
                for ba in range(nbt):
                    d[i,j,n,m]=d[i,j,n,m]+a[i,ba,n]*b[j,ba,m]

dp=np.einsum('kin,qim->kqnm',a,b)
dp2=np.swapaxes(np.tensordot(a,b, axes=([1],[1])), 1,2)

print(d-dp2) 
print(np.shape(dp2))