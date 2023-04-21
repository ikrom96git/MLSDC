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
        self.transfer=Transfer(self.fine.coll.nodes, self.coarse.coll.nodes)
        self.reduced_model=Penning_trap(params)

    def func(self, x, v, eps):
        return (1/eps)*np.array([0, v[2], -v[1]])+self.params.c*np.array([-x[0], 0.5*x[1], 0.5*x[2]])


    def SDC_solver(self,xold, vold, level=None, tau=[None]):
        if level==None:
            level=self.fine


        if None in tau:
            tau=np.zeros(np.shape(xold))
        Xold=np.copy(xold)
        Vold=np.copy(vold)
        Xnew=np.copy(Xold)
        Vnew=np.copy(Vold)
        M=level.num_nodes
        # print(xold)
        Sq=np.zeros([M, 3])
        S=np.zeros([M, 3])
        for m in range(M):
            for j in range(M+1):

                f = self.func(Xold[j,:], Vold[j,:], self.params.eps)

                Sq[m,:]+=(level.SQ[m+1, j]-level.Sx[m+1, j]) * f

                S[m,:]+=(level.S[m+1, j]) * f




        for m in range(M):

            Sx=np.copy(Sq)
            for j in range(M+1):
                # pdb.set_trace()
                f=self.func(Xnew[j,:], Vnew[j,:], self.params.eps)

                Sx+=level.Sx[m+1, j] * f

            Xnew[m+1,:]=Xnew[m,:]+self.params.dt*level.coll.delta_m[m]*vold[0,:]+Sx[m,:]+tau[m,:]
            Vrhs=Vnew[m, :] + 0.5 * self.params.dt*level.coll.delta_m[m] * (self.func(Xnew[m, :], Vnew[m,:], self.params.eps)-self.func(Xold[m,:], Vold[m,:], self.params.eps))+S[m,:] + tau[m,:]
            Vfunc=lambda V: Vrhs + 0.5*self.params.dt*level.coll.delta_m[m] * (self.func(Xnew[m+1,:], V,self.params.eps)-self.func(Xold[m+1,:], Vold[m+1,:], self.params.eps))-V
            Vnew[m+1, :]=fsolve(Vfunc, Vnew[m,:])

        return Xnew, Vnew

    def FAS_tau(self, Xfine, Vfine, Xcoarse, Vcoarse):
        tau=np.zeros(3)
        ffine=np.zeros(np.shape(Xfine))
        fcoarse=np.zeros(np.shape(Xcoarse))
        tau=np.zeros(np.shape(Xcoarse))

        for ii in range(np.max(np.shape(Xfine))):
            ffine[ii,:] = self.func(Xfine[ii,:], Vfine[ii, :], self.params.eps)
            fcoarse[ii,:] = self.func(Xcoarse[ii, :], Vcoarse[ii, :], self.params.eps)

        for ii in range(3):
            tau[1:,ii]=(self.transfer.Rcoll @ self.fine.coll.Qmat[1:,1:] @ ffine[1:,ii] - self.coarse.coll.Qmat[1:,1:] @ fcoarse[1:,ii])

        # pdb.set_trace()
        # tau=(self.Rcoll @ self.fine.coll.Qcoll @ self.fine.FF @ u_fine - self.coarse.coll.Qcoll @ self.coarse.FF @ u_coarse)
        return tau

    def restriction(self, X, V):
        m=np.min(np.shape(self.transfer.Rcoll))
        Xr=np.ones([m+1, 3])*X[0,:]
        Vr=np.ones([m+1, 3])*V[0,:]
        for ii in range(3):
            Xr[1:,ii]=self.transfer.Rcoll @ X[1:, ii]
            Vr[1:,ii]=self.transfer.Rcoll @ V[1:, ii]

        return Xr, Vr

    def prolongation(self, X, V):
        m=np.min(np.shape(self.transfer.Pcoll))
        Xp=np.ones([m+1, 3])*X[0,:]
        Vp=np.ones([m+1, 3])*V[0,:]
        for ii in range(3):
            Xp[1:,ii]=self.transfer.Pcoll @ X[1:, ii]
            Vp[1:,ii]=self.transfer.Pcoll @ V[1:, ii]

        return Xp, Vp

    def MLSDC_sweep(self, X0, V0):

        X0fine = X0*np.ones([self.fine.num_nodes+1, 3])
        V0fine = V0*np.ones([self.fine.num_nodes+1, 3])


        # Fine to coarse without sweep
        X0coarse, V0coarse=self.restriction(X0fine, V0fine)

        tau_coarse=self.FAS_tau(X0fine , V0fine, X0coarse, V0coarse)

        # Coarse sweep

        Xcoarse, Vcoarse=self.SDC_solver(X0coarse, V0coarse,level=self.coarse ,tau=tau_coarse)

        # Coarse to fine
        Xp, Vp= self.prolongation(Xcoarse- X0coarse, Vcoarse-V0coarse)
        Xfine= X0fine + Xp
        Vfine=V0fine + Vp

        # Relax: fine sweep

        Xfinest, Vfinest=self.SDC_solver(Xfine, Vfine, level=self.fine)

        # MLSDC residual
        # Residual=self.fine.residual(U_finest, U0_fine)
        # pdb.set_trace()

        return Xfinest, Vfinest

    def MLSDC_sweep_reduced(self, X0, V0):

        X0fine = X0*np.ones([self.fine.num_nodes+1, 3])
        V0fine = V0*np.ones([self.fine.num_nodes+1, 3])


        # Fine to coarse without sweep
        X0coarse, V0coarse=self.restriction(X0fine, V0fine)

        tau_coarse=self.FAS_tau(X0fine , V0fine, X0coarse, V0coarse)

        # Coarse sweep

        Xcoarse, Vcoarse=self.SDC_solver(X0coarse, V0coarse,level=self.coarse ,tau=tau_coarse)

        # Coarse to fine
        Xp, Vp= self.prolongation(Xcoarse- X0coarse, Vcoarse-V0coarse)
        Xfine= X0fine + Xp
        Vfine=V0fine + Vp

        # Relax: fine sweep

        Xfinest, Vfinest=self.SDC_solver(Xfine, Vfine, level=self.fine)

        # MLSDC residual
        # Residual=self.fine.residual(U_finest, U0_fine)
        # pdb.set_trace()

        return Xfinest, Vfinest

    def SDC_iter(self, X0, V0, K):
        x0 = X0*np.ones([self.fine.num_nodes+1, 3])
        v0 = V0*np.ones([self.fine.num_nodes+1, 3])
        X0 = X0*np.ones([self.fine.num_nodes+1, 3])
        V0 = V0*np.ones([self.fine.num_nodes+1, 3])

        # X=dict()
        # V=dict()
        Rx=dict()
        Rv=dict()

        for ii in range(K):

            X, V=self.SDC_solver(X0, V0)
            pdb.set_trace()
            Rx[ii], Rv[ii]=self.Residual(x0, v0, X, V)
            X0=X
            V0=V

        return Rx, Rv

    def SDC_0(self, X0, V0, K):
        x0 = X0*np.ones([self.fine.num_nodes+1, 3])
        v0 = V0*np.ones([self.fine.num_nodes+1, 3])

        X0, V0 = self.reduced_model.non_uniform_order0(t=np.append(self.params.t0,self.params.dt*self.fine.coll.nodes))
        # pdb.set_trace()
        # X0=np.transpose(X0)
        # V0=np.transpose(V0)


        X=dict()
        V=dict()
        Rx=dict()
        Rv=dict()

        for ii in range(K):
            X[ii], V[ii]=self.SDC_solver(X0, V0)
            Rx[ii], Rv[ii]=self.Residual(x0, v0, X[ii], V[ii])
            X0=X[ii]
            V0=V[ii]
            # pdb.set_trace()
        return Rx, Rv


    def SDC_1(self, X0, V0, K):
        x0 = X0*np.ones([self.fine.num_nodes+1, 3])
        v0 = V0*np.ones([self.fine.num_nodes+1, 3])

        X0, V0 = self.reduced_model.non_uniform_sol(t=np.append(self.params.t0,self.params.dt*self.fine.coll.nodes))

        # X0=np.transpose(X0)
        # V0=np.transpose(V0)

        X=dict()
        V=dict()
        Rx=dict()
        Rv=dict()
        for ii in range(K):

            X[ii], V[ii]=self.SDC_solver(X0, V0)

            Rx[ii], Rv[ii]=self.Residual(x0, v0, X[ii], V[ii])
            X0=X[ii]
            V0=V[ii]
        return Rx, Rv

    def Residual(self,X0, V0, X, V):
        Rx=np.copy(X0)
        Rv=np.copy(V0)
        func=np.zeros(np.shape(X0))
        for jj in range(self.fine.num_nodes+1):
            func[jj,:]=self.func(X[jj, :], V[jj, :], self.params.eps)
        for ii in range(3):

            Rx[:, ii]= X0[:,ii]+self.params.dt*self.fine.coll.Qmat @ func[:,ii]-X[:,ii]
            Rv[:, ii]= V0[:,ii]+self.params.dt*self.fine.coll.Qmat @ func[:,ii]-V[:,ii]

        return Rx, Rv

    def max_norm(self, Rx, Rv, K):

        rx_norm=np.zeros([3, K])
        rv_norm=np.zeros([3, K])
        for ii in range(3):
            for jj in range(K):
                rx_norm[ii, jj]=np.linalg.norm(Rx[jj][:, ii])
                rv_norm[ii, jj]=np.linalg.norm(Rv[jj][:, ii])
                # pdb.set_trace()
        return rx_norm, rv_norm

    def simulation(self):
        x0, v0=np.split(self.params.u0,2)
        k=5

        Rx, Rv=self.SDC_iter(x0, v0, k)
        Rx0, Rv0=self.SDC_0(x0, v0, k)
        Rx1, Rv1=self.SDC_1(x0, v0, k)
        rx, rv=self.max_norm(Rx, Rv, k)
        rx0, rv0=self.max_norm(Rx0, Rv0, k)
        rx1, rv1=self.max_norm(Rx1, Rv1, k)
        axis=1
        yaxis=np.block([[rx[axis, :]], [rx0[axis, :]], [rx1[axis, :]]])
        xaxis=np.arange(0, k, 1)
        self.plot_resitual(yaxis, xaxis, k)
        pdb.set_trace()





    def plot_resitual(self, yaxis, xaxis, K):
        axis=1
        titles=['SDC', 'SDC $\mathcal{O}(0)$', 'SDC $\mathcal{O}(1)$']
        markers=['o', 's', 'd']
        for ii in range(3):
            plt.semilogy(xaxis, yaxis[ii,:],marker=markers[ii], label=titles[ii])

        plt.legend()
        plt.tight_layout()
        plt.show()





















# =============================================================================
# A uniform time varying electric field
# Link: https://hal.science/hal-03104042
# =============================================================================

class Penning_trap(object):
    def __init__(self, params):
        self.params=_Pars(params)
        self.t=np.linspace(self.params.t0, self.params.tend, 10000)

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

    def non_uniform_coeff(self):
        U_0=self.non_uniform_U(self.params.t0, self.params.s, self.params.eps, self.params.c)
        coeff=np.linalg.solve(U_0, self.params.u0)

        return coeff

    def non_uniform_exact(self, t=[None]):
        if None in t:
            t=self.t
        coeff=self.non_uniform_coeff()
        if self.params.eps>np.sqrt(1/(2*self.params.c)):
            raise ValueError('Solution does not exist. Need: $\epsilon< \frac{1}{\sqrt(2c))}$')
        for ii, tt in enumerate(t):
            if ii==0:
                U=self.non_uniform_U(tt, self.params.s, self.params.eps, self.params.c)@coeff
            else:
                U=np.vstack((U, self.non_uniform_U(tt, self.params.s, self.params.eps, self.params.c)@coeff))

        return U

    def non_uniform_reduced(self, t, s, c):
        u0=self.params.u0
        yt0=np.array([u0[0]*np.cos(np.sqrt(c)*(t-s))+(u0[3]/(np.sqrt(c)))*np.sin(np.sqrt(c)*(t-s)), u0[1], u0[2]])
        ut0=np.array([-u0[0]*np.sqrt(c)*np.sin(np.sqrt(c)*(t-s))+u0[3]*np.cos(np.sqrt(c)*(t-s)), u0[4], u0[5]])

        yt1=np.array([0, 0.5*c*u0[2]*(t-s), -0.5*c*u0[1]*(t-s)])
        ut1=np.array([0, -0.5*u0[5]*(t-s), 0.5*c*u0[4]*(t-s)])

        return [yt0, ut0, yt1, ut1]

    def GG_non_uniform(self, t, s, eps, c):
        [yt0, ut0, yt1, ut1]=self.non_uniform_reduced(t, s, c)
        R0=np.block([yt0, self.R_mat((t-s)/eps)@ut0])
        RR0=np.block([yt1+self.RR_mat((t-s)/eps)@ut0, self.R_mat((t-s)/eps)@ut1+self.RR_mat((t-s)/eps)@self.non_uniform_E(yt0, c)])
        return R0+eps*RR0

    def non_uniform_sol(self, t=[None]):

        if None in t:
            t=self.t


        for ii, tt in enumerate(t):
            if ii==0:
                U=self.GG_non_uniform(tt, self.params.s, self.params.eps, self.params.c)
            else:
                U=np.vstack((U, self.GG_non_uniform(tt, self.params.s, self.params.eps, self.params.c)))
        X, V=np.split(U, 2, axis=1)
        return X, V

    def GG_non_uniform_order0(self, t, s, eps, c):
        [yt0, ut0, yt1, ut1]=self.non_uniform_reduced(t, s, c)
        R0=np.block([yt0, self.R_mat((t-s)/eps)@ut0])
        return R0

    def non_uniform_order0(self, t=[None]):

        if None in t:
            t=self.t


        for ii, tt in enumerate(t):
            if ii==0:
                U=self.GG_non_uniform_order0(tt, self.params.s, self.params.eps, self.params.c)
            else:
                U=np.vstack((U, self.GG_non_uniform_order0(tt, self.params.s, self.params.eps, self.params.c)))
        X, V=np.split(U, 2, axis=1)
        return X, V








if __name__=='__main__':

    params=dict()
    params['t0']=0.0
    params['tend']=5
    params['dt']=0.015625* 0.1
    dt=0.015625
    params['s']=0.0
    params['eps']=0.1
    params['c']=2.0
    params['u0']= np.array([1, 1, 1, 1, 1, 1])

    collocation_params=dict()
    collocation_params['quad_type']='GAUSS'
    collocation_params['num_nodes']=[5,5]

    mlsdc=mlsdc_solver(params, collocation_params)
    X0=np.ones([5+1,3])
    V0=np.ones([5+1,3])
    X, V=mlsdc.SDC_solver(X0, V0)
    solver=Penning_trap(params)
    # u=solver.solver()
    # solver.non_uniform_solution(1, 0, 0.1, 1)
    # x, v=mlsdc.MLSDC_sweep(np.array([1,1,1]), np.array([1, 1, 1]))
    mlsdc.simulation()
    # u_non=solver.non_uniform_exact()
    # u_non_reduced=solver.non_uniform_sol()

    # plt.figure()
    # plt.plot(solver.t, u_non[:,1], label='exact')
    # plt.plot(solver.t, u_non_reduced[:,1], label='reduced')
    # plt.xlabel('time')
    # plt.ylabel('Solution')
    # plt.legend()
    # plt.tight_layout()

    # plt.show()
    # u_model=solver.reduction_solver()
