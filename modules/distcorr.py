#Distance correlation function distcorr from https://gist.github.com/satra/aa3d19a12b74e9ab7941
from scipy.spatial.distance import pdist, squareform
import numpy as np

def distcorr(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor
	
#code below this line added by JH
def distcorr_array(X):
	"""
	Compute the distance correlation matrix of an array
	"""
	nvar = X.shape[1]
	dcorr = np.zeros((nvar,nvar))
	#fill in the lower triangle
	for i in range(nvar):
		for j in range(nvar):
			if j <= i:
				dcorr[i,j] = distcorr(X[:,i],X[:,j])

	#reflect the lower triangle over the diagonal (symmetric)
	iu = np.triu_indices(nvar) #indices for upper triangle
	dcorr[iu] = dcorr.T[iu] #set upper triangle to upper triangle of transposed matrix
	
	return dcorr