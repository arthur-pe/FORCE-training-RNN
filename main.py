import numpy as np
from matplotlib import pyplot as plt
from functions import *
from ode import *
from classes import *

def train(net, t, target, delta_t):

    zs = []
    rs = []

    for i in range(len(t)):

        z, r = net.step(0.1)

        error = z-target[i]

        if t[i]%delta_t == 0:

            net.update(error)

        zs.append(z)
        rs.append(r)

        if i%int(len(t)/100)==0:
            print('Iteration:', i, 'Error:', error)

    plt.plot(zs)
    plt.show()

def figure1():
    t = np.arange(0,2000,0.1)
    target = triangle_func(t)
    plt.plot(target)
    net = RNN(N_G=1000, output_dim=1, alpha=1, tau=10, g_G_G=1.5, g_Gz=1, p_G_G=0.1)
    train(net, t, target, 1)

figure1()
plot_triangle_func()