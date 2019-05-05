import numpy as np
import matplotlib.pyplot as plt
from utils import sample, gaussian_kernelize,get_sympoly


def cond_kdpp(A,X,k):
	if not A:
		L = gaussian_kernelize(X)
		idx = sample(L,k=k)
		return idx
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
rand_idx1 = np.random.choice(len(X),size = 100)
# print('X[:10]=',X[:10])
dists = np.linalg.norm(X-np.array([0,0]),axis=1)
# print('dists.shape=',dists.shape)
closest_indices = dists.argsort()[:31]

X_new = np.random.randn(1000,2) #+np.array([2.5,2.5])
# print('X[closest_indices].shape=')
print('X_new.shape =',X_new.shape)
# idx = cond_kdpp(X[closest_indices]/5,X_new,k=70)
idx = cond_kdpp([],X_new,k=70)

plt.scatter(X[rand_idx1,0],X[rand_idx1,1],c='b')
plt.title('first plot')
plt.show()
plt.figure()
rand_idx = np.random.choice(len(X_new),size = 70)
# plt.scatter(X[closest_indices,0],X[closest_indices,1],c = 'b')
plt.scatter(X[rand_idx,0]/5,X[rand_idx,1]/5,c = 'b')
# plt.scatter(X_new[:,0],X_new[:,1],c='r')
plt.scatter(X_new[idx,0],X_new[idx,1],c='r')
plt.scatter(X_new[rand_idx,0],X_new[rand_idx,1],c='g')

plt.title('second plot')
plt.show()
# plt.figure()
# rand_idx = np.random.choice(len(X_new),size = 50)
# print('len(rand_idx)=',len(rand_idx))
# plt.scatter(X_new[rand_idx,0],X_new[rand_idx,1],c='g')

# plt.show()




