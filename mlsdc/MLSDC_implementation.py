import numpy as np
from core.Collocation import CollBase
from harmonicoscillator import get_collocation_params, Transfer


class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

class mlsdc_solver(object):
    
    def __init__(self, params, collocation_params):
    
        self.params=_Pars(params)
        self.coll=_Pars(collocation_params)
        self.fine=get_collocation_params(self.coll.num_nodes[0], self.coll.quad_type)
        self.coarse=get_collocation_params(self.coll.num_nodes[1], self.coll.quad_type)
    
    def matrix(self):
        pass
    
    def func(self, x, v, t, eps):
        return (1/eps)*np.array([0, v[2], -v[1]])+np.array([0, np.sin(t/eps), np.cos(t/eps)])
    
    
    def SDC_solver(self, level=None):
        if level==None:
            level=self.fine
        else:
            raise ValueError('define level')
        
        for ii in range(level.num_nodes):
            
        
        
        
        
    
    




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

if __name__=='__main__':

    params=dict()
    params['t0']=0.0
    params['tend']=5.0
    params['s']=0.0
    params['eps']=0.00001
    params['u0']= np.array([0, 1, 1, 1, params['eps'], 0])
    
    collocation_params=dict()
    collocation_params['quad_type']='LOBATTO'
    collocation_params['num_nodes']=[5,5]
    

    solver=Penning_trap(params)
    U=solver.Solver()
    U_model=solver.reduction_solver()
