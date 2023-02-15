import numpy as np

from core.Collocation import CollBase

class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)


class SDC_sweep(object):

    def __init__(self, problem_params, collocation_params, sweeper_params):
        self.prob=_Pars(problem_params)
        self.collocation=_Pars(collocation_params)
        self.sweeper=_Pars(sweeper_params)

    def collocation_matrix(self):
        self.coll=CollBase(num_nodes=self.collocation.num_nodes,quad_type=self.collocation.quad_type)
        Q=self.coll.Qmat[1:,1:]
        S=self.coll.Smat[1:,1:]
        QE = np.zeros(self.coll.Qmat.shape)
        for m in range(self.coll.num_nodes + 1):
            QE[m, 0:m] = self.coll.delta_m[0:m]
        QI = np.zeros(self.coll.Qmat.shape)
        for m in range(self.coll.num_nodes + 1):
            QI[m, 1 : m + 1] = self.coll.delta_m[0:m]
        QI=QI[1:,1:]
        QE=QE[1:,1:]
        return [Q, QI, QE,  S]

    def base_method(self, type=None):
        pass

    def func_E(self, U):
        return self.prob.lambda_E*U

    def func_F(self, U):
        return self.prob.lambda_F*U

    def func(self, U):
        return (self.prob.lambda_E+self.prob.lambda_F)*U

    def sweep(self, U0, tau=None):
        if tau==None:
            tau=0.0

        U=self.base_method()
        Qdelta=self.prob.dt*(self.prob.lambda_F*self.QI+self.prob.lambda_E*self.QE)
        L=np.eye(self.coll.num_nodes)+Qdelta
        R=self.prob.dt*((self.prob.lambda_E+self.prob.lambda_F)*self.Q-Qdelta)

        for ii in range(self.sweep.K_iter):
            U=np.linalg.solve(L,U0+R@U)

        return U
