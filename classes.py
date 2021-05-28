import numpy as np
from scipy import sparse

class RNN:

    def __init__(self, N_G, output_dim, alpha, tau, g_G_G, g_Gz, p_G_G):

        #Right multiplications
        self.J_G_G = sparse.random(N_G,N_G,density=p_G_G,data_rvs=np.random.randn).toarray()/np.sqrt(p_G_G*N_G) #recurrent weights
        self.w = np.random.randn(N_G,output_dim)/np.sqrt(N_G) #decoder
        #self.w = np.zeros((N_G,output_dim))/np.sqrt(N_G)
        self.J_Gz = (2*np.random.rand(N_G,output_dim) - 1) #feedback weights

        self.g_G_G = g_G_G
        self.g_Gz = g_Gz

        self.x = np.random.randn(N_G)
        self.r = np.tanh(self.x) #neuron state

        self.z = np.random.randn(1) #output np.matmul(self.r,self.w)
        self.P = np.eye(N_G)/alpha

        self.N_G = N_G #number neurons in generator
        self.output_dim = output_dim #1
        self.tau = tau #time bin
        self.alpha = alpha

    def step(self, dt):
        self.x += (- dt/self.tau) *(self.x + self.g_G_G*self.J_G_G.dot(self.r) + self.g_Gz*self.J_Gz.dot(self.z))
        self.r = np.tanh(self.x)
        self.z = self.w.T.dot(self.r).flatten()

        return self.z, self.r

    def update(self, error):

        Pr = self.P.dot(self.r)
        self.P -= np.outer(Pr, self.r).dot(self.P) / (1 + self.r.dot(Pr))

        dw = np.outer(self.P.dot(self.r), error)
        self.w -= dw

        return np.sqrt(np.sum(dw**2))

