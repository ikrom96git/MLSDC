import numpy as np
import matplotlib.pyplot as plt
from core.Collocation import CollBase
from harmonicoscillator import get_collocation_params, Transfer
from scipy.optimize import fsolve
import pdb
class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

class mlsdc_solver(object):

    def __init__(self, params, collocation_params):

        self.params=_Pars(params)
        self.coll=_Pars(collocation_params)
        self.fine=get_collocation_params(self.coll.num_nodes[0], self.coll.quad_type, dt=self.params.dt)
        self.coarse=get_collocation_params(self.coll.num_nodes[1], self.coll.quad_type, dt=self.params.dt)

    def matrix(self):
        pass

    def func(self, x, v, t, eps):
        return (1/eps)*np.array([0, v[2], -v[1]])+np.array([0, np.sin(t/eps), np.cos(t/eps)])


    def SDC_solver(self,xold, vold, level=None):
        if level==None:
            level=self.fine
        else:
            raise ValueError('define level')
        Xold = xold*np.ones([level.num_nodes+1, 3])
        Vold = vold*np.ones([level.num_nodes+1, 3])
        Xnew=np.copy(Xold)
        Vnew=np.copy(Vold)

        for kk in range(10):
            for ii in range(level.num_nodes):
                Sx = np.zeros(3)
                S = np.zeros(3)
                Sq=np.zeros(3)
                for jj, nn in enumerate(level.coll.nodes):
                    nn=self.params.dt*nn
                    Sx+=level.Sx[ii, jj]*(self.func(Xnew[jj,:], Vnew[jj,:], nn, self.params.eps)-self.func(Xold[jj,:], Vold[jj,:], nn, self.params.eps))
                    S+= level.S[ii,jj] * self.func(Xold[jj,:], Vold[jj,:], nn, self.params.eps)
                    Sq+= level.SQ[ii, jj]*self.func(Xold[jj,:], Vold[jj,:], nn, self.params.eps)
                Xnew[ii+1,:]=Xnew[ii,:]+self.params.dt*level.coll.delta_m[ii]*vold+Sx+Sq

                Vrhs=Vnew[ii, :] + 0.5 * self.params.dt*level.coll.delta_m[ii] * (self.func(Xnew[ii, :], Vnew[ii,:], nn, self.params.eps)-self.func(Xold[ii,:], Vold[ii,:], nn, self.params.eps))+S
                Vfunc=lambda V: Vrhs + 0.5*self.params.dt*level.coll.delta_m[ii] * (self.func(Xnew[ii+1,:], V, nn, self.params.eps)-self.func(Xold[ii+1,:], Vold[ii+1,:], nn, self.params.eps))-V
                Vnew[ii+1, :]=fsolve(Vfunc, Vnew[ii,:])
                pdb.set_trace()
            Xold=np.copy(Xnew)
            Vold=np.copy(Vnew)
        pdb.set_trace()
        plt.plot(level.coll.nodes, Xnew[1:,1])
        plt.show()
        return Xnew, Vnew














# =============================================================================
# A uniform time varying electric field
# Link: https://hal.science/hal-03104042
# =============================================================================

class Penning_trap(object):
    def __init__(self, params):
        self.params=_Pars(params)
        self.t=np.linspace(self.params.t0, self.params.tend, 100)

    def Solver(self):

        for ii, jj in enumerate(self.t):
            usol=self.exact_u(self.params.u0, jj, self.params.s, self.params.eps)
            if ii==0:
                u=usol
            else:
                u=np.vstack((u, usol))

        return u

    def reduction_solver(self):

        for ii, jj in enumerate(self.t):
            usol=self.GG(self.params.u0, jj, self.params.s, self.params.eps)
            if ii==0:
                u=usol
            else:
                u=np.vstack((u, usol))

        return u


    def R_mat(self, t):
        R=np.array([[1, 0, 0], [0, np.cos(t), np.sin(t)], [0, -np.sin(t), np.cos(t)]])
        return R

    def RR_mat(self, t):
        RR=np.array([[0, 0, 0], [0, np.sin(t), 1-np.cos(t)], [0, np.cos(t)-1, np.sin(t)]])
        return RR

    def AA_mat(self, t, s, eps):

        P = np.zeros([3,3])
        P[0,0] = 1
        RR=self.RR_mat((t-s)/eps)
        PRR = (t-s)*P+eps*RR
        R=self.RR_mat((t-s)/eps)
        AA=np.block([[np.eye(3), PRR], [np.zeros([3,3]), R]])

        return AA

    def BB_mat(self, t, s, eps):

        vec1=eps*(t-s)*np.array([0, -np.cos(t/eps), np.sin(t/eps)])
        vec2=(eps**2) * np.array([0, np.sin(t/eps) - np.sin(s/eps), np.cos(t/eps)-np.cos(s/eps)])
        vec3=(t-s)*np.array([0, np.sin(t/eps), np.cos(t/eps)])
        BB=np.block([vec1+vec2, vec3])

        return BB

    def exact_u(self,u0,  t, s, eps):

        AA = self.AA_mat(t, s, eps)
        BB = self.BB_mat(t, s, eps)

        u_eps=AA@u0+BB

        return u_eps

    def CC_mat(self, t, s, eps):
        vec1 = eps*(t-s)*np.array([0, -np.cos((t-s)/eps), np.sin((t-s)/eps)])
        vec2 = (t-s)*np.array([0, np.sin((t-s)/eps), np.cos((t-s)/eps)])

        CC=np.block([vec1, vec2])

        return CC

    def GG(self, u0, t, s, eps):

        AA=self.AA_mat(t, s, eps)
        CC=self.CC_mat(t, s, eps)

        GG=AA@u0+CC
        return GG

    def non_uniform_solution(self, t, s, eps, c):
        cos=lambda a, t, s: np.cos(a*(t-s))
        sin=lambda a, t, s: np.sin(a*(t-s))

        a_eps=(1+np.sqrt(1-2*c*eps**2))/(2*eps)
        b_eps=(1-np.sqrt(1-2*c*eps**2))/(2*eps)

        U_eps= np.zeros([6, 6])
        U_eps[0,:]=np.array([0, 0, 0, 0, cos(np.sqrt(c), t, s), sin(np.sqrt(c), t, s)])
        U_eps[1,:]=np.array([sin(a_eps, t, s), -sin(a_eps, t, s), sin(b_eps, t, s), -cos(b_eps, t, s), 0, 0])
        U_eps[2,:]=np.array([cos(a_eps, t, s), sin(a_eps, t, s), cos(b_eps, t, s), sin(b_eps, t, s), 0, 0])
        U_eps[3,:]=np.array([0, 0, 0, 0, -np.sqrt(c)*sin(np.sqrt(c), t, s), np.sqrt(c)*cos(np.sqrt(c), t, s)])
        U_eps[4,:]=np.array([a_eps*cos(a_eps, t, s), -a_eps*cos(a_eps, t, s), b_eps*cos(b_eps, t, s), b_eps*sin(b_eps, t, s), 0, 0])
        U_eps[5,:]=np.array([-a_eps*sin(a_eps, t, s), a_eps*cos(a_eps, t, s), -b_eps*sin(b_eps, t, s), b_eps*cos(b_eps, t, s), 0, 0])



if __name__=='__main__':

    params=dict()
    params['t0']=0.0
    params['tend']=1
    params['dt']=1.0
    params['s']=0.0
    params['eps']=1.0#0.01
    params['u0']= np.array([0, 1, 1, 1, params['eps'], 0])

    collocation_params=dict()
    collocation_params['quad_type']='GAUSS'
    collocation_params['num_nodes']=[4,4]

    # mlsdc=mlsdc_solver(params, collocation_params)
    # X, V=mlsdc.SDC_solver(np.array([0,1,1]), np.array([1, params['eps'], 0]))
    solver=Penning_trap(params)
    U=solver.Solver()
    solver.non_uniform_solution(1, 0, 0.1, 1)
    # U_model=solver.reduction_solver()
