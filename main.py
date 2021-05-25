import numpy as np
from matplotlib import pyplot as plt
from functions import *
from ode import *
from classes import *

def train(net, t, target, delta_t, learning_start, learning_stop, dt):

    zs = []
    rs = []
    len_w_dots = []

    for i in range(len(t)):

        z, r = net.step(dt)

        error = z-target[i]

        len_w_dot = 0

        if t[i]%delta_t == 0 and t[i]>=learning_start and t[i]<learning_stop:

            len_w_dot = net.update(error)

        zs.append(z)
        rs.append(r)
        len_w_dots.append(len_w_dot)

        if i%int(len(t)/100)==0:
            print('Iteration:', i, 'Error:', error)

    return zs, rs, len_w_dots

def figure():

    t_max = 6000
    learning_start = 2000
    learning_stop = 4000
    dt = 0.1
    delta_t = 1

    t = np.arange(0,t_max,dt)
    target = triangle_func(t, freq=1/500)

    net = RNN(N_G=100, output_dim=1, alpha=1., tau=10., g_G_G=1.5, g_Gz=1., p_G_G=0.1)
    zs, rs, len_w_dots = train(net, t, target, delta_t, learning_start=learning_start, learning_stop=learning_stop, dt=dt)

    plt.figure(figsize=(10,5))
    plt.plot(t,target,linestyle='--',label='$f(t)$', zorder=2)
    plt.plot(t,zs, label='$z(t)$', zorder=1)
    plt.plot(t,np.array(len_w_dots) / delta_t - 4,label='$|\dot{w}|$')
    plt.axvline(learning_start,color='black',alpha=0.5)
    plt.axvline(learning_stop, color='black',alpha=0.5)
    plt.xlabel('time (ms)')
    plt.text(learning_start/2, max(target)+0.3,'Pre-Learning', horizontalalignment='center',fontsize=12)
    plt.text((learning_stop+learning_start)/2, max(target)+0.3,'Learning', horizontalalignment='center',fontsize=12)
    plt.text((t_max+learning_stop)/2, max(target)+0.3,'Post-Learning', horizontalalignment='center',fontsize=12)
    plt.box(False)
    plt.yticks([])
    plt.legend()

    plt.tight_layout()
    plt.savefig('f2.png')
    plt.show()

figure()
