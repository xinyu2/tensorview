###==========================================================
###==========================================================
### visualize the cnv-activations of CNN in training 
### buildSet: similarity and grouping
###==========================================================
###==========================================================

import numpy as np

def makeSet(x):
	s=set()
	for e in x:
		s.add(e)
	return s

def findSimilar(c,alpha):
	l=c.shape[0]
	mask=np.ones([l,l],dtype=np.int32)
	for i in range(l):
		for j in range(l):
			if i>=j:
				mask[i,j]=0
	tmp=c*mask
	z=zip(np.where(tmp>=alpha)[0],np.where(tmp>=alpha)[1])
	rslt=map(lambda x:makeSet(x),z)
	return rslt

def mergeSl(sl):
	rslt=[]
	tmp=sl[0]
	oldlen=len(sl)
	while len(sl)>0:
		if len(tmp.intersection(sl[0]))>0:
			tmp=tmp.union(sl.pop(0))
		else:
			rslt.append(tmp)
			tmp=sl[0]
	if len(tmp)>0:
		rslt.append(tmp)
	newlen=len(rslt)
	if newlen==oldlen:
		return rslt
	else:
		return(mergeSl(rslt))
