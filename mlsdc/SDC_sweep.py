import numpy as np
import matplotlib.pyplot as plt
import pdb

from core.Collocation import CollBase
from core.Lagrange import LagrangeApproximation
class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

class get_level_params(object):
    def __init__(self, num_nodes, quad_type):
        self.num_nodes=num_nodes
        self.quad_type=quad_type
        [self.Q, self.QI,self.QE, self.S]=self.Qmatrix(num_nodes=num_nodes, quad_type=quad_type)

    def Qmatrix(self, num_nodes=None, quad_type=None):
        self.coll=CollBase(num_nodes=num_nodes, quad_type=quad_type)
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


class MLSDC(object):
    def __init__(self, probem_params, collocation_params):
        self.prob=_Pars(problem_params)
        self.coll_params=_Pars(collocation_params)
        if len(self.coll_params.num_nodes)==1:
            self.fine=get_level_params(self.coll_params.num_nodes, self.coll_params.quad_type)
        else:
            self.fine=get_level_params(self.coll_params.num_nodes[0], quad_type=self.coll_params.quad_type)
            self.coarse=get_level_params(self.coll_params.num_nodes[1], quad_type=self.coll_params.quad_type)
            self.transfer=Transfer(self.coll_params.num_nodes[0], self.coll_params.num_nodes[1])







    def func_E(self, U):
        return self.prob.lambda_s*U

    def func_F(self, U):
        return self.prob.lambda_f*U

    def base_method(self, type=None):
        F=dict()
        F['ex']=None
        F['im']=None
        func=_Pars(F)
        if type==None:
            return self.prob.u0*np.ones(self.collocation.num_nodes)
        elif type=="EEuler":
            u=np.ones(self.coll.num_nodes+1)*prob.u0
            for ii in range(self.coll.num_nodes):
                u[ii+1]=u[ii]+self.coll.delta_m[ii]*func(u[ii])
            return u[1:]



    def SDC_sweep(self, U, FI_new, FE_new, tau=None, level=None):
        if tau==None:
            tau=np.zeros(len(U))

        if level==None:
            level=self.fine

        S=self.level.S@(FE_old+FI_old)

        FE_new=self.func_E(U)
        FI_new=self.func_F(U)

        for ii in range(0, self.level.num_nodes-1):
            RHS=U[ii]+self.level.coll.delta_m[ii]*(FE_new[ii]-FE_old[ii]-FI_old[ii+1])+S[ii]+tau[ii]
            U[ii]=RHS/(1-self.level.coll.delta_m[ii]*self.prob.lambda_f)
            FE_old[ii]=self.func_E(U[ii])
            FI_old[ii]=self.func_E(U[ii])

        return U, FE_old, FI_old

    def restrict(self, U):


















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

class Transfer(object):
    def __init__(self,fine_params=None, coarse_params=None):
        self.fine=_Pars(fine_params)
        self.coarse=_Pars(coarse_params)

        self.Rcoll=self.get_transfer_matrix(self.find.num_nodes, self.coarse.num_nodes)
        self.Pcoll=self.get_transfer_matrix(self.coarse.num_nodes, self.fine.num_nodes)

    def get_transfer_matrix(self, fine_nodes, coarse_nodes):

        approx=LagrangeApproximation(coarse_nodes)
        return approx.getInterpolationMatrix(fine_nodes)

class FAS(object):
    def __init__(self, problem_params, collocation_params):
        self.prob=_Pars(problem_params)
        self.coll=_Pars(collocation_params)
        self.problem_params=problem_params
        self.collocation_params=collocation_params

    def fine_params(self):
        fine_prob=dict()
        fine_prob=self.problem_params.copy()
        fine_coll=self.collocation_params.copy()
        fine_coll['num_nodes']=self.collocation_params[0]

    def tau_correction(F_fine, F_coarse, tau=None):
        if tau==None:
            tau=0.0




    def fine_level(self):
        pass








if __name__=='__main__':
    problem_params=dict()
    problem_params['lambda_f']=10j
    problem_params['lambda_s']=1j
    problem_params['u0']=1.0
    problem_params['dt']=0.1
    problem_params['Tend']=2.0

    collocation_params=dict()
    collocation_params['quad_type']='GAUSS'
    collocation_params['num_nodes']=[5,3]
    collocation_params['K_iter']=100




    nn=np.linspace(0, 0.1, 100)
    sdc=SDC_sweep(problem_params, collocation_params)
    aa=sdc.sweep()
    plt.plot(aa.real, aa.imag)
    plt.plot(np.exp(11j*nn).real, np.exp(11j*nn).imag)

    pdb.set_trace()
