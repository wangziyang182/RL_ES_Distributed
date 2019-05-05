import numpy as np
from scipy.linalg import hadamard,eigh,sqrtm
from scipy.spatial.distance import pdist, squareform

def cond_kdpp(A,X,k):
    if A.size == 0:
        L = gaussian_kernelize(X)
        idx = sample(L,k=k)
        return idx
    n = len(A)
    X_big = np.vstack((A,X))
    L = gaussian_kernelize(X_big)
    # print('L.shape=',L.shape)
    I_A_comp = np.eye(len(L))
    I_A_comp[:n] = 0

    t1 = np.linalg.inv(L+I_A_comp)[n:,n:]
    L_A = np.linalg.inv(t1)-np.eye(len(t1))
    # print('L_A.shape = ',L_A.shape)
    idx = sample(L_A,k=k)
    # print('idx = ',idx)
    return idx

class SGD(object):
    
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v

def softmax(x):
    
    assert (len(list(x.shape)) == 2),'shape dimension error'
    # print('action',x)
    return np.exp(x)/np.sum(np.exp(x),axis = 1)

def get_info_summary(list_of_dict):
    fit = [work_info['fit'] for work_info in list_of_dict]
    fit = np.array([item for sublist in fit for item in sublist])

    anti_fit = [work_info['anti_fit'] for work_info in list_of_dict]
    anti_fit = np.array([item for sublist in anti_fit for item in sublist])

    seed_id = np.array([work_info['seed_id'] for work_info in list_of_dict])

    return fit, anti_fit, seed_id 

def get_noise_matrices(random_seed,number_worker,len_param,sigma,noise_type):
    for i,seed in enumerate(random_seed):
        np.random.seed(seed)
        
        if noise_type == 'Gaussian':
            noise_ele = np.random.randn(number_worker[i],len_param)
        elif noise_type == 'Hadamard':
            h_size = 1<<((max(number_worker[i],len_param)-1).bit_length())
            h = hadamard(h_size)
            noise_ele = (h@np.diag(np.random.choice([-1,1], h_size)))[:number_worker[i],:len_param]
        
        try:
            noise = np.vstack((noise,noise_ele))
        except:
            noise = noise_ele

    return noise * sigma

def compute_weight_decay(weight_decay, model_param_list):
  model_param_grid = np.array(model_param_list)
  return - weight_decay * np.mean(model_param_grid * model_param_grid, axis=1)

def sample(L, k=None, flag_gpu=False):
    print('L =',L)
    D,V = eigh(L)
    E = get_sympoly(D, k, flag_gpu=flag_gpu)
    N = D.shape[0]
    if k is None:
        # general dpp
        D = D / (1 + D)
        V = V[:,np.random.rand(N) < D]
        k = V.shape[1]
    else:
        # k-dpp
        v_idx = sample_k(D, E, k, flag_gpu=flag_gpu)
        V = V[:,v_idx]

    rst = list()

    for i in range(k-1,-1,-1):
        # choose indices

        P = np.sum(V**2, axis=1)

        row_idx = np.random.choice(range(N), p=P/np.sum(P))
        col_idx = np.nonzero(V[row_idx])[0][0]

        rst.append(row_idx)

        # update V
        V_j = np.copy(V[:,col_idx])
        V = V - np.outer(V_j, V[row_idx]/V_j[row_idx])
        V[:,col_idx] = V[:,i]
        V = V[:,:i]
        print('V=',V)
        # reorthogonalize
        if i > 0:
            V = sym(V)

    # rst = np.sort(rst)

    return rst

def sample_k(D, E, k, flag_gpu=False):
    i = D.shape[0]
    remaining = k
    rst = list()

    while remaining > 0:
        if i == remaining:
            marg = 1.
        else:
            marg = D[i-1] * E[remaining-1, i-1] / E[remaining, i]

        if np.random.rand() < marg:
            rst.append(i-1)
            remaining -= 1
        i -= 1

    return np.array(rst)

def get_sympoly(D, k, flag_gpu=False):
    N = D.shape[0]
    if flag_gpu:
        pass
    else:
        E = np.zeros((k+1, N+1))

    E[0] = 1.
    for l in range(1,k+1):
        E[l,1:] = np.copy(np.multiply(D, E[l-1,:N]))
        E[l] = np.cumsum(E[l], axis=0)

    return E

def sym(X):
    # X += 1e-10
    try:
        X.dot(np.linalg.inv(np.real(sqrtm(X.T.dot(X)))))
    except ValueError:
        print('X =',X)
    return X.dot(np.linalg.inv(np.real(sqrtm(X.T.dot(X)))))

def gaussian_kernelize(X):
    pairwise_dists = squareform(pdist(X, 'euclidean'))
    return np.exp(-pairwise_dists ** 2 / 0.5 ** 2)








