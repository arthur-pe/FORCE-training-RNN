from scipy import signal, sparse
from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt
import os

def complex_periodic_func(t, n_sinusoids=16, amp=2., freq=1 / 300):
    """returns a complex periodic function with n sinosoids."""
    np.random.seed(4)
    return sum(0.4*amp/i * np.random.randint(1,5) * np.sin(i * np.random.randint(1,2) * np.pi * freq * t)
               for i in range(1, n_sinusoids + 1))


def plot_complex_periodic_func():
    plt.figure(dpi=300)
    t = np.linspace(-20,600,500)
    plt.plot(t, complex_periodic_func(t))
    plt.xlabel('Time (ms)')
    plt.ylabel('y-axis')
    plt.title("plot of complex periodic function")
    plt.savefig(plots_dir + 'complex periodic function.png')
    plt.show()


def triangle_func(t, freq=1/300, amp=3.):
    """triangle wave function."""
    return amp * signal.sawtooth(2 * np.pi * freq * t, 0.5)


def plot_triangle_func():
    plt.figure(dpi=300)
    t = np.linspace(0,1200,500)
    plt.plot(t, triangle_func(t))
    plt.xlabel('Time (ms)')
    plt.ylabel('y-axis')
    plt.title("plot of triangle function")
    plt.savefig(plots_dir + 'triangle function.png')
    plt.show()


def periodic_func(t, n_sinusoids=4, amp=3., freq=1/300, noise=False):
    """returns a periodic function with n sinosoids"""
    my_sum = sum(amp / i * np.sin(i * np.pi * freq * t) for i in range(1, n_sinusoids + 1))
    if noise:
        return my_sum + np.random.randn(len(my_sum))*0.5
    else:
        return my_sum


def plot_periodic_func(noise=False):
    plt.figure(dpi=300)
    t = np.linspace(0,1800,500)
    plt.plot(t, periodic_func(t, noise=noise))
    plt.xlabel('Time (ms)')
    plt.ylabel('y-axis')
    plt.title("plot of periodic function")
    plt.savefig(plots_dir + 'periodic function.png')
    plt.show()


def square_func(t, freq, noise=False):
    """returns a discontinuous function with a square wave"""
    my_sum = signal.square(2 * np.pi * t*freq*0.5)
    if noise:
        return my_sum + np.random.uniform(0, 0.1, size=len(my_sum))
    else:
        return my_sum


def plot_square_func(noise=False):
    plt.figure(dpi=300)
    t = np.linspace(0,400,500)
    plt.plot(t, square_func(t,noise=noise))
    plt.ylim(-2, 2)
    plt.xlabel('Time (ms)')
    plt.ylabel('y-axis')
    plt.title("plot of square function")
    plt.savefig(plots_dir + 'square function.png')
    plt.show()

def plot_sine_waves():
    F = 1  # No. of cycles per second
    T = 60.e-3  # Time period
    Fs = 5.e3  # No. of samples per second
    Ts = 1. / Fs  # Sampling interval
    N = int(T / Ts)  # No. of samples

    t = np.linspace(0, 2, N)
    signal = np.sin(2 * np.pi * F * t/2)
    # signal = signal - np.ones.like(signal)
    plt.figure(dpi=300)
    plt.plot(t, signal)

    F = 120.e2  # No. of cycles per second
    T = 10.e-3  # Time period
    Fs = 50.e3  # No. of samples per second
    Ts = 1. / Fs  # Sampling interval
    N = int(T / Ts)  # No. of samples

    t = np.linspace(0, 0.25, N)
    signal = np.sin(2 * np.pi * F * t)
    plt.plot(t, signal)
    plt.xlabel('Time (s)')
    plt.ylabel('y-axis')
    plt.title("plot of sinus functions")
    plt.savefig(plots_dir + 'sinus functions.png')
    plt.show()

class lorenz:

    def __init__(self,t, sigma, rho, beta, x0):

        self.t = t
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        self.x0 = x0

    def f(self, x, t):

        return [self.sigma*(x[1] - x[0]), x[0]*(self.rho-x[2])-x[1], x[0]*x[1]-self.beta*x[2]]

    def integrate(self):

        return odeint(self.f, self.x0,self.t/30).transpose()

def plot_lorenz():
    t = np.linspace(0,10,10000)
    ode = lorenz(t,x0=np.array([10,10,10]),rho=28,sigma=10,beta=8/3)
    solution = ode.integrate() # 3 x 10000

    #3D plot (for fun)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot(solution[0],solution[1],solution[2])

    #2D plot
    plt.figure()
    plt.plot(t, solution.transpose())
    plt.show()

script_dir = os.path.dirname(__file__)
plots_dir = os.path.join(script_dir, 'plots/')

if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)
