import numpy as np
import matplotlib.pyplot as plt
import pdb

from core.Collocation import CollBase

class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)


class SDC_sweep(object):

    def __init__(self, problem_params, collocation_params):
        self.prob=_Pars(problem_params)
        self.collocation=_Pars(collocation_params)
        [self.Q, self.Qf, self.Qs,  self.S]=self.collocation_matrix()


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
        QE=np.copy(QI)
        np.fill_diagonal(QE, 0)

        return [Q, QI, QE,  S]

    def base_method(self, type=None):
        if type==None:
            return self.prob.u0*np.ones(self.collocation.num_nodes)
        elif type=="EEuler":
            u=np.ones(self.coll.num_nodes+1)*self.prob.u0
            for ii in range(self.coll.num_nodes):
                u[ii+1]=u[ii]+self.coll.delta_m[ii]*self.func(u[ii])
            return u[1:]





    def func_E(self, U):
        return self.prob.lambda_s*U

    def func_F(self, U):
        return self.prob.lambda_f*U

    def func(self, U):
        return (self.prob.lambda_s+self.prob.lambda_f)*U

    def sweep(self, tau=None):
        if tau==None:
            tau=0.0
        U0=self.base_method()
        U=self.base_method(type="EEuler")
        Qdelta=self.prob.dt*(self.prob.lambda_f*self.Qf+self.prob.lambda_s*self.Qs)
        L=np.eye(self.coll.num_nodes)-Qdelta
        R=self.prob.dt*(self.prob.lambda_s+self.prob.lambda_f)*self.Q-Qdelta

        for ii in range(self.collocation.K_iter):
            U=np.linalg.solve(L,U0+R@U)


        return U

if __name__=='__main__':
    problem_params=dict()
    problem_params['lambda_f']=10j
    problem_params['lambda_s']=1j
    problem_params['u0']=1.0
    problem_params['dt']=0.1
    problem_params['Tend']=2.0

    collocation_params=dict()
    collocation_params['quad_type']='GAUSS'
    collocation_params['num_nodes']=10
    collocation_params['K_iter']=100

    sdc=SDC_sweep(problem_params, collocation_params)
    aa=sdc.sweep()
    plt.plot(aa.real, aa.imag)
    pdb.set_trace()
