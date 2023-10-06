import numpy as np
import  matplotlib.pyplot as plt
from harmonicoscillator import get_collocation_params, Transfer
class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

class penning_trap(object):
    def __init__(self, params, collocation_params):
        self.params=_Pars(params)
        self.coll=_Pars(collocation_params)

        self.fine=get_collocation_params(self.coll.num_nodes[0], self.coll.quad_type, dt=self.params.dt)
        self.coarse=get_collocation_params(self.coll.num_nodes[1], self.coll.quad_type, dt=self.params.dt)
        self.transfer=Transfer(self.fine.coll.nodes, self.coarse.coll.nodes)
        self.RHS_non_uniform()


    def ndim(self, Q, n):
        return np.kron(np.eye(n), Q)

    def RHS_non_uniform(self):
        self.RHSv=np.zeros((3,3))
        self.RHSv[1,2]=1
        self.RHSv[2,1]=-1
        self.RHSx=-np.eye(3)
        self.RHSx[1,1]=0.5
        self.RHSx[2,2]=0.5
        self.RHSv=(1/self.params.eps)*self.RHSv
        self.RHSx=self.params.c*self.RHSx


    def SDC_solver(self,x0, v0, x, v, level):
        num_nodes=level.coll.num_nodes
        X0=x0*np.ones((num_nodes,3))
        V0=v0*np.ones((num_nodes, 3))
        X0=(X0.T).reshape(3*num_nodes, 1)
        V0=(V0.T).reshape(3*num_nodes, 1)

        X, V=x*np.ones((num_nodes, 3)), v*np.ones((num_nodes, 3))
        X=(X.T).reshape(3*num_nodes, 1)
        V=(V.T).reshape(3*num_nodes, 1)

        level=self.fine
        Q=self.ndim(level.Q, 3)
        QQ=self.ndim(level.QQ,3)
        Qx=self.ndim(level.Qx,3)
        QT=self.ndim(level.QT,3)
        Fvv=self.ndim(self.RHSv, num_nodes)
        Fvx=self.ndim(self.RHSx, num_nodes)
        F=Fvv@V+Fvx@X
        Xsol=X0+Q@V0+(QQ-Qx)@F
        Vsol=np.linalg.inv(np.eye(3*num_nodes)-QT@Fvv)@(V0+(Q-QT)@F+QT@Fvx@Xsol)
        return Xsol, Vsol

    def SDC_iter(self, k):
        x0=self.params.u0[0]
        v0=self.params.u0[1]
        for ii in range(k):
            X, V=self.SDC_solver(x0, v0, x0, v0, self.fine)
















class penningtrap_solution(object):
    def __init__(self, params):
        self.param=_Pars(params)

    def Rtheta(self, theta):
        R=np.zeros([3,3])
        R[0, 0]=1
        R[1, 1]=np.cos(theta)
        R[1,2]=np.sin(theta)
        R[2,1]=-np.sin(theta)
        R[2,2]=np.cos(theta)
        return R

    def RRtheta(self, theta):
        RR=np.zeros([3,3])
        RR[1,1]=np.sin(theta)
        RR[1,2]=1-np.cos(theta)
        RR[2,1]=np.cos(theta)-1
        RR[2,2]=np.sin(theta)
        return RR

    def Amat(self, t, s, eps):
        I=np.eye(3)
        O=np.zeros([3,3])
        P=np.zeros([3,3])
        P[0,0]=1
        A=np.block([[I, (t-s)*P+eps*self.RRtheta((t-s)/eps)], [O, self.Rtheta((t-s)/eps)]])
        return A

    def Bmat(self, t, s, eps):
        b11=eps*(t-s)*np.array([0, -np.cos(t/eps), np.sin(t/eps)])
        b12=eps**2*np.array([0, np.sin(t/eps)-np.sin(s/eps), np.cos(t/eps)-np.cos(s/eps)])
        b2=(t-s)*np.array([0, np.sin(t/eps), np.cos(t/eps)])
        return np.block([b11+b12, b2])

    def exactsol_uniform(self, x, v, t, s, eps):
        Amat=self.Amat(t, s, eps)
        Bmat=self.Bmat(t, s, eps)
        U=np.block([x, v])
        return Amat@U+Bmat

    def Cmat(self, t, s, eps):
        c1=eps*(t-s)*np.array([0, -np.cos((t-s)/eps), np.sin((t-s)/eps)])
        c2=(t-s)*np.array([0, np.sin((t-s)/eps), np.cos((t-s)/eps)])
        return np.block([c1, c2])

    def Gmat_uniform(self, x, v, t, s, eps):
        Amat=self.Amat(t, s, eps)
        U=np.block([x, v])
        Cmat=self.Cmat(t, s, eps)
        return Amat@U+Cmat

    def non_uniform_U(self, t, s, eps, c):
        cos=lambda a, t, s: np.cos(a*(t-s))
        sin=lambda a, t, s: np.sin(a*(t-s))

        a_eps=(1+np.sqrt(1-2*c*eps**2))/(2*eps)
        b_eps=(1-np.sqrt(1-2*c*eps**2))/(2*eps)

        U_eps= np.zeros([6, 6])
        U_eps[0,:]=np.array([0, 0, 0, 0, cos(np.sqrt(c), t, s), sin(np.sqrt(c), t, s)])
        U_eps[1,:]=np.array([sin(a_eps, t, s), -cos(a_eps, t, s), sin(b_eps, t, s), -cos(b_eps, t, s), 0, 0])
        U_eps[2,:]=np.array([cos(a_eps, t, s), sin(a_eps, t, s), cos(b_eps, t, s), sin(b_eps, t, s), 0, 0])
        U_eps[3,:]=np.array([0, 0, 0, 0, -np.sqrt(c)*sin(np.sqrt(c), t, s), np.sqrt(c)*cos(np.sqrt(c), t, s)])
        U_eps[4,:]=np.array([a_eps*cos(a_eps, t, s), a_eps*sin(a_eps, t, s), b_eps*cos(b_eps, t, s), b_eps*sin(b_eps, t, s), 0, 0])
        U_eps[5,:]=np.array([-a_eps*sin(a_eps, t, s), a_eps*cos(a_eps, t, s), -b_eps*sin(b_eps, t, s), b_eps*cos(b_eps, t, s), 0, 0])

        return U_eps

    def non_uniform_E(self, x, c):
        return c*np.array([-x[0], 0.5*x[1], 0.5*x[2]])

    def non_uniform_coeff(self, x, v,  t, s, eps, c):
        U_0=self.non_uniform_U(t, s, eps, c)
        U=np.block([x, v])
        coeff=np.linalg.solve(U_0, U)

        return coeff

    def non_uniform_exact(self,x, v, t, t0, s, eps, c):
        coeff=self.non_uniform_coeff(x, v, t0, s, eps, c)
        if self.params.eps>np.sqrt(1/(2*self.params.c)):
            raise ValueError('Solution does not exist. Need: $\epsilon< \frac{1}{\sqrt(2c))}$')

        U=self.non_uniform_U(t, s, eps, c)@coeff

        return U

    def non_uniform_reduced(self,x, v, t, s, c):
        u0=np.block([x, v])
        yt0=np.array([u0[0]*np.cos(np.sqrt(c)*(t-s))+(u0[3]/(np.sqrt(c)))*np.sin(np.sqrt(c)*(t-s)), u0[1], u0[2]])
        ut0=np.array([-u0[0]*np.sqrt(c)*np.sin(np.sqrt(c)*(t-s))+u0[3]*np.cos(np.sqrt(c)*(t-s)), u0[4], u0[5]])

        yt1=np.array([0, 0.5*c*u0[2]*(t-s), -0.5*c*u0[1]*(t-s)])
        ut1=np.array([0, -0.5*u0[5]*(t-s), 0.5*c*u0[4]*(t-s)])

        return [yt0, ut0, yt1, ut1]

    def GG_non_uniform(self,x, v, t, s, eps, c):
        [yt0, ut0, yt1, ut1]=self.non_uniform_reduced(x, v, t, s, c)
        R0=np.block([yt0, self.R_mat((t-s)/eps)@ut0])
        RR0=np.block([yt1+self.RR_mat((t-s)/eps)@ut0, self.R_mat((t-s)/eps)@ut1+self.RR_mat((t-s)/eps)@self.non_uniform_E(yt0, c)])
        return R0+eps*RR0

    def test(self):
        tn=np.linspace(0, 10, 1000)
        t_len=len(tn)
        x=np.array([1,1, 1])
        v=np.array([1,1, 1])

        s=0
        eps=0.01
        X_ex=np.zeros([3, t_len])
        V_ex=np.zeros([3, t_len])
        X_red=np.zeros([3, t_len])
        V_red=np.zeros([3, t_len])
        for ii in range(t_len):
            U_ex=self.exactsol_uniform(x, v, tn[ii], s, eps)
            U_red=self.Gmat_uniform(x, v, tn[ii], s, eps)
            X_ex[:, ii]=U_ex[:3]
            V_ex[:, ii]=U_ex[3:]
            X_red[:, ii]=U_red[:3]
            V_red[:, ii]=U_red[3:]



        plt.plot(tn, V_ex[2,:], label='Exact solution')
        plt.plot(tn, V_red[2, :], label='reduced problem solution')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__=='__main__':
    params=dict()
    params['t0']=0.0
    params['tend']=5
    params['dt']=0.015625 *1e-3
    dt=0.015625*1e-2
    params['s']=0.0
    params['eps']=1
    params['c']=2.0
    params['u0']= np.array([[1, 2, 3], [1, 1, 1]])
    collocation_params=dict()
    collocation_params['quad_type']='GAUSS'
    collocation_params['num_nodes']=[5,3]
    pt=penning_trap(params, collocation_params)
    pt.SDC_iter(3)
    # reduced_model=penningtrap_solution(params)
    # reduced_model.test()
