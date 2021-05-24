from scipy.integrate import odeint
import numpy as np
from matplotlib import pyplot as plt

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

        return odeint(self.f, self.x0,self.t).transpose()

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
