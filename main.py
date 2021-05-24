import numpy as np
from matplotlib import pyplot as plt
from functions import *
from ode import *
from classes import *

def train(net, t, target):

    zs = []
    rs = []

    for i in range(len(t)):

        z, r = net.step(0.01)

        error = z-target[i]

        net.update(error)

        zs.append(z)
        rs.append(r)

        print('Iteration:',i, 'Error:',error)

    plt.plot(zs)
    plt.show()

def figure1():
    t = np.arange(0,2,0.01)
    target = triangle_func(t*1000)
    plt.plot(target)
    net = RNN(N_G=1000, output_dim=1, alpha=1, tau=0.01, g_G_G=1.5, g_Gz=1, p_G_G=0.1)
    train(net,t ,target)

figure1()
plot_triangle_func()