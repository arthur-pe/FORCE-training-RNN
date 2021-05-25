import numpy as np

class RNN:

    def __init__(self, N_G, output_dim, alpha, tau, g_G_G, g_Gz, p_G_G):

        #Right multiplications
        self.J_G_G = np.random.randn(N_G,N_G)/np.sqrt(p_G_G*N_G)*g_G_G #recurrent weights
        self.w = np.random.randn(N_G,output_dim)/np.sqrt(N_G) #decoder
        self.J_Gz = np.random.uniform(-1,1,(output_dim,N_G))*g_Gz #feedback weights

        self.J_G_G_mask = np.zeros(N_G**2)
        self.J_G_G_mask[:int(N_G**2*p_G_G)] = 1
        np.random.shuffle(self.J_G_G_mask)
        self.J_G_G_mask = self.J_G_G_mask.reshape(N_G,N_G)

        self.g_G_G = g_G_G
        self.g_Gz = g_Gz

        self.x = np.random.randn(N_G)*0.5
        self.r = np.tanh(self.x) #neuron state

        self.z = np.random.randn(1)*0.5 #output np.matmul(self.r,self.w)
        self.P = np.identity(N_G)/alpha

        self.N_G = N_G #number neurons in generator
        self.output_dim = output_dim #1
        self.tau = tau #time bin
        self.alpha = alpha

    def step(self, dt):

        self.x += (dt/self.tau)*(-self.x+self.r @ (self.J_G_G *self.J_G_G_mask) + self.z @ self.J_Gz)

        self.r = np.tanh(self.x)

        self.z = np.matmul(self.r, self.w)

        return self.z, self.r

    def update(self, error):

        self.P -= (((self.P @ self.r) @ self.r.transpose()) * self.P)/(1+self.r.transpose() @ self.P @ self.r)

        w_dot = np.expand_dims((error * self.P @ self.r),1)

        self.w -= w_dot

        return np.sqrt(np.sum(w_dot**2))

