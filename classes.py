import numpy as np

class RNN:

    def __init__(self, N_G, output_dim, alpha, tau, g_G_G, g_G_F, g_Gz):

        #Right multiplications
        self.J_G_G = np.random.randn((N_G,N_G)) #recurrent weights
        self.w = np.random.randn((N_G,output_dim)) #decoder
        self.J_Gz = np.random.randn((output_dim,N_G)) #feedback weights

        self.g_G_G = g_G_G
        self.g_Gz = g_Gz

        self.r = np.random.randn(N_G) #neuron state
        self.z = np.matmul(self.r,self.w) #output
        self.P = np.ones((N_G,N_G))/alpha

        self.N_G = N_G #number neurons in generator
        self.output_dim = output_dim #1
        self.tau = tau #time bin
        self.alpha = alpha

    def step(self):

        self.r = np.tanh(self.r + self.tau*(self.g_G_G*self.r @ self.J_G_G + self.g_Gz*self.z @ self.J_Gz))

        self.z = np.matmul(self.r, self.w)

        return self.r, self.z

    def update(self, error, response):

        self.P = self.P - (self.P @ response @ response.transpose() @ self.P)/(1+response.transpose() @ self.P @ response)

        self.w = self.w - error @ self.P @ response

