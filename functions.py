from scipy import signal, sparse
from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

# wave target functions

def complex_periodic_func(t, n_sinusoids=16, amp=2., freq=1 / 300):
    """returns a complex periodic function with n sinosoids."""
    np.random.seed(4)
    return sum(amp/i * np.random.randint(1,5) * np.sin(i * np.random.randint(1,2) * np.pi * freq * t)
               for i in range(1, n_sinusoids + 1))


t = np.linspace(-20,600,500)
plt.plot(t, complex_periodic_func(t))
plt.show()

def triangle_func(t, freq=1/300, amp=3.):
    """triangle wave function."""
    return amp * signal.sawtooth(2 * np.pi * freq * t, 0.5)

t = np.linspace(0,1200,500)
# plt.pltlot(t, triangle_func(t))
plt.show()


def periodic_func(t, n_sinusoids=4, amp=3., freq=1/300, noise=False):
    """returns a periodic function with n sinosoids"""
    my_sum = sum(amp / i * np.sin(i * np.pi * freq * t) for i in range(1, n_sinusoids + 1))
    if noise:
        return my_sum + np.random.uniform(0, 0.5, size=len(my_sum))
    else:
        return my_sum

plt.figure(dpi=300)
t = np.linspace(0,1800,500)
plt.plot(t, periodic_func(t))
plt.show()

plt.plot(t, periodic_func(t, noise=True))
plt.show()


def square_func(t, noise=False):
    """returns a discontinuous function with a square wave"""
    my_sum = signal.square(2 * np.pi * 5 * t)
    if noise:
        return my_sum + np.random.uniform(0, 0.1, size=len(my_sum))
    else:
        return my_sum

t = np.linspace(0,400,500)
plt.plot(t, square_func(t))
plt.ylim(-2, 2)
plt.show()

plt.plot(t, square_func(t,noise=True))
plt.ylim(-2, 2)
plt.show()
