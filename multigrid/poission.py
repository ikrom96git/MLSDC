import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pdb


class multigrid(object):
    

    def __init__(self, x_0, f):
        self.x_0=x_0
        self.f=f
        self.xn=100
        self.x=np.linspace(0, 1, self.xn)
        self.h=1/self.xn

    def build_A(self):

        diagonals=[(2)*np.ones(self.xn), (-1)*np.ones(self.xn-1), (-1)*np.ones(self.xn-1)]
        A=(1/self.h**2)*sp.diags(diagonals, [0,-1, 1]).toarray()
        print(A)
        return A

    def build_rhs(self):
        F=self.f(self.x)

        return F

    def solve_poisson(self):
        A=self.build_A()
        F=self.build_rhs()
        v=np.linalg.solve(A,F)

        return v


if __name__=='__main__':
    f=lambda x: -x
    p=multigrid(0, f)
    v=p.solve_poisson()
    x=np.linspace(0, 1, 100)
    plt.plot(x, v)
