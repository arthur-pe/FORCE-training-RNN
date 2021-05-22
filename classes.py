import numpy as np

class RNN:

    def __init__(self, number_neurons, output_dim, alpha, time_bin_length):

        #Right multiplications
        self.recurrent_weights = np.random.randn((number_neurons,number_neurons))
        self.decoder = np.random.randn((number_neurons,output_dim))
        self.feedback = np.random.randn((output_dim,number_neurons))

        self.neurons_state = np.random.randn(number_neurons)
        self.output = np.matmul(self.neurons_state,self.decoder)
        self.P = np.ones((number_neurons,number_neurons))/alpha

        self.number_neurons = number_neurons
        self.output_dim = output_dim
        self.time_bin_length = time_bin_length
        self.alpha = alpha

    def step(self):

        self.neurons_state = self.neurons_state + \
                             self.time_bin_length*(np.matmul(self.neurons_state,self.recurrent_weights)
                                      +np.matmul(self.output, self.feedback))

        self.output = np.matmul(self.neurons_state, self.decoder)

        return self.neurons_state, self.output

    def update(self, error, response):

        self.P = self.P - (self.P @ response @ response.transpose() @ self.P)/(1+response.transpose() @ self.P @ response)

        self.decoder = self.decoder - error @ self.P @ response

