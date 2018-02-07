###==========================================================
###==========================================================
### visualize the cnv-activations of CNN in training 
### act2grdcp: output structure to catalyst pipeline
###==========================================================
###==========================================================

import numpy as np

imageSize=28*28
gimax,gjmax=8,4 #1st layer has 32 filters

def getColor(k,ms,ii):
    rc=-1
    if ii>0:
        for i in range(len(ms)):
            if k in ms[i]:
                rc=i
                break
    return rc*0.3

def act1ScatterPipe(act,ms,kl,tm):
    ii=tm%550
    gi,gj,k=0,0,0
    x_axis=np.linspace(0,1,imageSize)
    z_axis=np.ones(imageSize,dtype=np.int32)
    x_stride,y_stride=1.2,1.2
    lx,ly,lz=[],[],[]
    for a in act.T:
        gi,gj=kl[k]/gjmax,kl[k]%gjmax
        lx.append(gj*x_stride+x_axis)
        ly.append(gi*y_stride+a)
        grpColor=getColor(k,ms,ii)
        lz.append(z_axis*grpColor)
        k+=1
    x=np.asarray(lx).reshape(-1,)
    xx=x.tolist()
    y=np.asarray(ly).reshape(-1,)
    yy=y.tolist()
    z=np.asarray(lz).reshape(-1,)
    zz=z.tolist()
    actScatterPipe=np.asarray(zip(xx,yy,zz))
    return actScatterPipe
