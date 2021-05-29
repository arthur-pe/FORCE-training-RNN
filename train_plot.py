import numpy as np
from matplotlib import pyplot as plt
from functions import *
from classes import *
from matplotlib import gridspec

def train(net, t, target, update_freq, learning_start, learning_stop, dt):

    zs = []
    rs = []
    len_w_dots = []

    for i in range(len(t)):

        z, r = net.step(dt)

        error = z-target[i]

        len_w_dot = 0

        if i%update_freq == 0 and t[i]>=learning_start and t[i]<learning_stop:

            len_w_dot = net.update(error)

        zs.append(z)
        rs.append(r)
        len_w_dots.append(len_w_dot)

        if i%int(len(t)/50)==0:
            print('Iteration:', i, 'Error:', error)

    return zs, rs, len_w_dots

def compute_plot(f, t_max = 6000, learning_start = 2000, learning_stop = 4000, dt = 0.1, delta_t = 0.2,
                 N_G=1500, alpha=1., tau=10., g_G_G=1.5, g_Gz=1., p_G_G=0.1, seed=7, save=False):

    np.random.seed(seed)

    update_freq = int(delta_t/dt)

    net = RNN(N_G=N_G, output_dim=1, alpha=alpha, tau=tau, g_G_G=g_G_G, g_Gz=g_Gz, p_G_G=p_G_G)

    #freq is in 100 mHz
    t = np.arange(0,t_max,dt)
    if f=='triangle':
        target = triangle_func(t, freq=1/300, amp=1)
    elif f == 'periodic':
        target = periodic_func(t, freq=1/200, amp=1)
    elif f=='complex_periodic':
        target = complex_periodic_func(t, freq=1/200, amp=1)
    elif f=='noisy_periodic':
        target = periodic_func(t, freq=1/200, amp=1, noise=True)
    elif f=='square':
        target = square_func(t, freq=1/200)
    elif f=='lorenz':
        ode = lorenz(t/4, x0=np.array([10, 10, 10]), rho=28, sigma=10, beta=8 / 3)
        target = ode.integrate()[0]/8
    elif f=='sine_low':
        target = np.sin(2*np.pi*t/4000+np.pi/4)*2
    elif f=='sine_high':
        target = np.sin((2*np.pi*t)/60)*2
    else:
        raise Exception('Incorrect function')

    zs, rs, len_w_dots = train(net, t, target, update_freq=update_freq, learning_start=learning_start, learning_stop=learning_stop, dt=dt)

    if save==True:
        np.save(f+'5.npy',np.array([t,target,zs,len_w_dots]))
        rs = np.array(rs)[:, :10]
        np.save(f+'5_rs.npy',rs)

    plt.figure(figsize=(12,5))
    neuron_sample_size = 5
    rs = np.array(rs)

    rs = rs[:,:neuron_sample_size]
    for i in range(neuron_sample_size):
        rs[:,i] /= max(abs(rs[:,i]))*2
        rs[:,i] -= i+2

        if i == 0:
            plt.plot(t, rs[:,i], color='blue',label='$r_i(t)$')
        else:
            plt.plot(t, rs[:, i], color='blue')

    if f=='noisy_periodic':
        order = 2
    else:
        order = 0

    plt.plot(t,target/max(target),linestyle='--',label='$f(t)$', zorder=2)
    plt.plot(t,np.array(zs)/max(target), label='$z(t)$', zorder=1+order)
    plt.plot(t,np.array(len_w_dots) / max(len_w_dots) - neuron_sample_size-2,label='$|\dot{w}|$')
    plt.axvline(learning_start,color='black',alpha=0.5)
    plt.axvline(learning_stop, color='black',alpha=0.5)
    plt.xlabel('time (ms)')
    plt.text(learning_start/2,1+0.3,'Pre-Learning', horizontalalignment='center',fontsize=12)
    plt.text((learning_stop+learning_start)/2, 1+0.3,'Learning', horizontalalignment='center',fontsize=12)
    plt.text((t_max+learning_stop)/2, 1+0.3,'Post-Learning', horizontalalignment='center',fontsize=12)
    plt.box(False)
    plt.yticks([])
    plt.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(f+'.png')
    #plt.savefig(f+'.pdf')
    plt.show()

def sub_plots(t,target,zs, ax, learning_start, learning_stop, dt):
    ax.plot(t[int(3*learning_start/dt/4):-int(3*learning_start/dt/4)], target[int(3*learning_start/dt/4):-int(3*learning_start/dt/4)], color='#1f77b4', linestyle='--', zorder=2)
    ax.plot(t[int(3*learning_start/dt/4):-int(3*learning_start/dt/4)], zs[int(3*learning_start/dt/4):-int(3*learning_start/dt/4)], color='#ff7f0e', zorder=1)

    ax.axvline(learning_start, color='black', alpha=0.5)
    ax.axvline(learning_stop, color='black', alpha=0.5)

    ax.set_xlabel('time (ms)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_yticks([])

def load():
    periodic = np.load('results/periodic.npy', allow_pickle=True)
    lorenz = np.load('results/lorenz.npy', allow_pickle=True)
    square = np.load('results/square.npy', allow_pickle=True)
    complex_periodic = np.load('results/complex_periodic.npy', allow_pickle=True)
    noisy_periodic = np.load('results/noisy_periodic.npy', allow_pickle=True)
    sine_low = np.load('results/sine-low.npy',allow_pickle=True)
    sine_high = np.load('results/sine-high.npy', allow_pickle=True)
    triangle = np.load('results/triangle.npy', allow_pickle=True)
    triangle_rs = np.load('results/triangle_rs.npy', allow_pickle=True)

    return periodic, lorenz, square, complex_periodic, noisy_periodic, triangle, triangle_rs, sine_low, sine_high

def figure1(periodic, lorenz, square, complex_periodic, noisy_periodic, triangle, triangle_rs,
            sine_low, sine_high, t_max,learning_start, learning_stop,dt):

    fig = plt.figure(figsize=(15,11),tight_layout=True)
    gs = gridspec.GridSpec(5,3)

    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels

    ax = fig.add_subplot(gs[0:2,:])

    #--------------------Triangle------------------------
    rs = triangle_rs
    t = triangle[0]
    target = triangle[1]
    zs = triangle[2]
    len_w_dots = triangle[3]
    neuron_sample_size = 5
    rs = np.array(rs)

    rs = rs[:,:neuron_sample_size]
    for i in range(neuron_sample_size):
        rs[:,i] /= max(abs(rs[:,i]))*2
        rs[:,i] -= i+2

        if i == 0:
            ax.plot(t, rs[:,i], color='blue',label='$r_i(t)$')
        else:
            ax.plot(t, rs[:, i], color='blue')

    ax.plot(t,target/max(target),linestyle='--',label='$f(t)$', zorder=2,color='#1f77b4')
    ax.plot(t,np.array(zs)/max(target), label='$z(t)$', zorder=1,color='#ff7f0e')
    ax.plot(t,np.array(len_w_dots) / max(len_w_dots) - neuron_sample_size-2,label='$|\dot{w}|$',color='green')
    ax.axvline(learning_start,color='black',alpha=0.5)
    ax.axvline(learning_stop, color='black',alpha=0.5)
    ax.set_xlabel('time (ms)')
    ax.text(learning_start/2,1+0.3,'Pre-Learning', horizontalalignment='center',fontsize=14)
    ax.text((learning_stop+learning_start)/2, 1+0.3,'Learning', horizontalalignment='center',fontsize=14)
    ax.text((t_max+learning_stop)/2, 1+0.3,'Post-Learning', horizontalalignment='center',fontsize=14)
    plt.box(False)
    ax.text(-220,  1+0.3, 'A.', fontsize=14, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='lower left')


    #-----------------Periodic------------------
    t = periodic[0]
    target = periodic[1]
    zs = periodic[2]
    ax = fig.add_subplot(gs[2,0])
    ax.set_ylim(min(target)-1,max(target)+1)
    ax.text(t[int(3*learning_start/dt/4)-40], ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'B.', fontsize=14, fontweight='bold')
    sub_plots(t,target,zs,ax,learning_start,learning_stop,dt)


    #-----------------Complex Periodic------------------
    t = complex_periodic[0]
    target = complex_periodic[1]
    zs = complex_periodic[2]
    ax = fig.add_subplot(gs[2,1])
    ax.set_ylim(min(target)-2,max(target)+2)
    ax.text(t[int(3*learning_start/dt/4)-40], ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'C.', fontsize=14, fontweight='bold')
    sub_plots(t,target,zs,ax,learning_start,learning_stop,dt)


    #------------------Noisy Periodic-----------------
    t = noisy_periodic[0]
    target = noisy_periodic[1]
    zs = noisy_periodic[2]
    ax = fig.add_subplot(gs[3,1:])
    ax.plot(t, target, color='#1f77b4', zorder=1)
    ax.plot(t, zs, color='#ff7f0e', zorder=2)

    ax.axvline(learning_start, color='black', alpha=0.5)
    ax.axvline(learning_stop, color='black', alpha=0.5)
    ax.text(-120, ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'F.', fontsize=14, fontweight='bold')

    ax.set_xlabel('time (ms)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks([])


    #------------------Square------------------
    t = square[0]
    target = square[1]
    zs = square[2]
    ax = fig.add_subplot(gs[3,0])
    ax.set_ylim(min(target)-1,max(target)+1)
    ax.text(t[int(3*learning_start/dt/4)-40], ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'E.', fontsize=14, fontweight='bold')
    sub_plots(t,target,zs,ax,learning_start,learning_stop,dt)


    #-------------------Lorenz-----------------
    t = lorenz[0]
    target = lorenz[1]
    zs = lorenz[2]
    ax = fig.add_subplot(gs[2,2])
    sub_plots(t,target,zs,ax,learning_start,learning_stop,dt)
    ax.set_ylim(min(target)-1,max(target)+1)
    ax.text(t[int(3*learning_start/dt/4)-40], ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'D.', fontsize=14, fontweight='bold')

    #-----------------sine high----------------
    t = sine_high[0]
    target = sine_high[1]
    zs = sine_high[2]
    ax = fig.add_subplot(gs[4,2])
    sub_plots(t,target,zs,ax,learning_start,learning_stop,dt)
    ax.set_ylim(min(target)-1,max(target)+1)
    ax.text(t[int(3*learning_start/dt/4)-40], ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'H.', fontsize=14, fontweight='bold')

    #-----------------sine low----------------
    t = sine_low[0]
    target = sine_low[1]
    zs = sine_low[2]
    ax = fig.add_subplot(gs[4,:2])

    ax.plot(t[2200:5000],
            target[2200:5000], color='#1f77b4', linestyle='--',
            zorder=2)
    ax.plot(t[2200:5000],
            zs[2200:5000], color='#ff7f0e', zorder=1)

    ax.axvline(4000, color='black', alpha=0.5)

    ax.set_xlabel('time (ms)')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_yticks([])
    ax.set_ylim(min(target)-1,max(target)+1)
    ax.text(2200-85, ax.get_ylim()[1]-(ax.get_ylim()[1]-ax.get_ylim()[0])/8, 'G.', fontsize=14, fontweight='bold')

    #---------------------------------------
    plt.savefig('figure1.pdf')
    plt.savefig('figure1.png')
    plt.show()
