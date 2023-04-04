import numpy as np
import matplotlib.pyplot as plt
from _Pars import _Pars

class simulation(object):
    def __init__(self, params):
        self.params=_Pars(params)

    def Solution(self, t, a, b):
        D=self.params.k**2/(4*self.params.m**2)-self.params.c/self.params.m

        if D<0:
            omega=np.sqrt(self.params.c/self.params.m-(self.params.k**2/(4*self.params.m**2)))
            x=np.exp(-(0.5*self.params.k*t)/self.params.m)*(a*np.cos(omega*t)+b*np.sin(omega*t))
        else:
            raise ValueError('D<0 only works')

        return x

    def Solution_source(self, t, a, b):
        comega=(self.params.c-self.params.m*self.params.Omega**2)/(self.params.k*self.params.Omega)
        Ap=(self.params.F0)/(self.params.k*self.params.Omega)*(1+comega**2)**(-1)
        Bp=comega*Ap

        x=self.Solution(t, a, b)
        X=x+Ap*np.sin(self.params.Omega*t)+ Bp*np.cos(self.params.Omega*t)
        return X
    def new_params(self, t):
        tau=self.params.Omega*t
        mu=self.params.m*self.params.Omega**2/self.params.c
        kappa=self.params.k*self.params.Omega/self.params.c





def params():
    params=dict()
    params['m']=1.0
    params['k']=1
    params['c']=0
    params['F_0']=0
    params['t_end']=1
    params['Omega']=1
