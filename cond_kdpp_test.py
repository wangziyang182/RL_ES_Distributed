import numpy as np
import matplotlib.pyplot as plt
from utils import sample, gaussian_kernelize,get_sympoly


def cond_kdpp(A,X,k):
	n = len(A)
	X_big = np.vstack((A,X))
	L = gaussian_kernelize(X_big)
	print('L.shape=',L.shape)
	I_A_comp = np.eye(len(L))
	I_A_comp[:n] = 0

	t1 = np.linalg.inv(L+I_A_comp)[n:,n:]
	L_A = np.linalg.inv(t1)-np.eye(len(t1))
	print('L_A.shape = ',L_A.shape)
	idx = sample(L_A,k=k)
	print('idx = ',idx)
	return idx


X = np.random.randn(1000,2)
print('X[:10]=',X[:10])
dists = np.linalg.norm(X-np.array([1,1]),axis=1)
# print('dists.shape=',dists.shape)
closest_indices = dists.argsort()[:201]

X_new = np.random.randn(1000,2)+np.array([2.5,2.5])
# print('X[closest_indices].shape=')
print('X_new.shape =',X_new.shape)
idx = cond_kdpp(X[closest_indices],X_new,k=800)

plt.scatter(X[:,0],X[:,1],c='b')
plt.title('first plot')
plt.show()
plt.figure()
plt.scatter(X[closest_indices,0],X[closest_indices,1],c = 'b')
# plt.scatter(X_new[:,0],X_new[:,1],c='r')
plt.scatter(X_new[idx,0],X_new[idx,1],c='r')
plt.title('second plot')
plt.show()



