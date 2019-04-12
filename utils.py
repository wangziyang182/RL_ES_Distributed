import numpy as np

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

def get_guassian_matrices(random_seed,number_worker,len_param,sigma):
    for i,seed in enumerate(random_seed):
        np.random.seed(seed)
        noise_ele = np.random.randn(number_worker[i],len_param)
        try:
            noise = np.vstack((noise,noise_ele))
        except:
            noise = noise_ele

    return noise * sigma









