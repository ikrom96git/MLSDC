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
        [self.Q, self.Qf,self.Qs, self.S]=self.Qmatrix(num_nodes=num_nodes, quad_type=quad_type)

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
        QE=np.copy(QI[:,1:])
        QE=np.hstack((QE, np.zeros(num_nodes).reshape([num_nodes,1])))
        # np.fill_diagonal(QE, 0)

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
            self.transfer=Transfer(self.fine.coll.nodes, self.coarse.coll.nodes)

    def func_E(self, U):
        return self.prob.lambda_s*U

    def func_F(self, U):
        return self.prob.lambda_f*U

    def func(self, u):
        return (self.prob.lambda_s+self.prob.lambda_f)*u

    def base_method(self, u0 ,type=None):
        uE_old=u0*np.ones(self.fine.num_nodes+1, dtype="complex")
        uI_old=u0*np.ones(self.fine.num_nodes+1, dtype="complex")
        # if type==None:

        #     for ii in range(self.fine.num_nodes):
        #         uE_old[ii+1]=uE_old[ii]+self.prob.dt*self.fine.coll.delta_m[ii]*self.func_E(uE_old[ii])
        #         uI_old[ii+1]=uI_old[ii]/(1-self.prob.dt*self.fine.coll.delta_m[ii]*(self.prob.lambda_f+self.prob.lambda_s))#uI_old[ii]+self.fine.coll.delta_m[ii]*self.func_F(uI_old[ii])

        return uI_old#, uE_old

    def SDC_sweep(self, U, FI_new, FE_new, tau=[None], level=None):
        if None in tau:
            tau=np.zeros(len(U), dtype="complex")

        if level==None:
            level=self.fine

        FE_old=np.copy(FE_new)
        FI_old=np.copy(FI_new)
        S=self.prob.dt*level.S@(FE_old[1:]+FI_old[1:])


        # pdb.set_trace()
        for ii in range(0, level.num_nodes):
            U[ii+1]=U[ii]+self.prob.dt*level.coll.delta_m[ii]*(FE_new[ii]-FE_old[ii]+self.func_F(U[ii+1])-FI_old[ii+1])+S[ii]+tau[ii]
            # U[ii]=RHS/(1-self.prob.dt*level.coll.delta_m[ii]*self.prob.lambda_f)

            FE_old[ii]=self.func_E(U[ii])
            FI_old[ii]=self.func_E(U[ii])

        return U, FI_old, FE_old

    def fwsw_sweep(self, U, tau=[None], level=None):
        if None in tau:
            tau=np.zeros(len(U), dtype='complex')

        if level==None:
            level=self.fine

        U_old=np.copy(U)
        S=self.prob.dt*level.S@self.func(U_old[1:]) + tau

        for ii in range(level.num_nodes):
            U[ii+1]=U[ii]+self.prob.dt*level.coll.delta_m[ii]*(self.func_F(U[ii+1])-self.func_F(U_old[ii+1])+self.func_E(U[ii])-self.func_E(U_old[ii]))+S[ii]+tau[ii]

        return U


    def restrict(self, U):
        return self.transfer.Rcoll@U

    def prolongation(self, U):
        return self.transfer.Pcoll@U

    def FAS(self, FI_fine, FE_fine, FI_coarse, FE_coarse, tau=None, level=None):
        if tau==None:
            tau=np.zeros(self.fine.num_nodes)
        if level==None:
            level=self.fine

        tau1=self.prob.dt*(self.transfer.Rcoll@level.Q@(FI_fine+FE_fine)-self.coarse.Q@(FI_coarse+FE_coarse))
        return tau1+self.transfer.Rcoll@tau

    def residual(self, U, U_0, level=None):
        if level==None:
            level=self.fine
        return U_0+self.prob.dt*level.Q@self.func(U)-U

    def sweep(self, U0,U, level=None, tau=[None]):
        if None in tau:
            tau=np.zeros(len(U0), dtype="complex")
        if level==None:
            level=self.fine
        # pdb.set_trace()

        Qdelta=self.prob.dt*(self.prob.lambda_f*level.Qf+self.prob.lambda_s*level.Qs)
        L=np.eye(level.coll.num_nodes)-Qdelta
        R=self.prob.dt*(self.prob.lambda_s+self.prob.lambda_f)*level.Q-Qdelta
        # pdb.set_trace()
        U=np.linalg.solve(L,U0+R@U+tau)


        return U

    def mlsdc_fwsw(self):
        U=self.base_method(self.prob.u0)
        U0=np.ones(self.fine.num_nodes, dtype="complex")*self.prob.u0
        U0c=np.ones(self.coarse.num_nodes, dtype="complex")*self.prob.u0
        U=U[1:]
        for ii in range(1):

            Usol=self.sweep(U0, U)
            U=Usol
            print("fine")

        rf=self.residual(Usol, U0)
        u=U
        # self.coefficient(self.fine.coll.nodes, u)
        U_coarse=self.restrict(u)
        U_tilde=U_coarse.copy()
        # self.coefficient(self.fine.coll.nodes, Usol)
        u_coar=self.best_function(self.coarse.coll.nodes, order_poly=3, filename='order.csv')
        rc=self.restrict(rf)
        # pdb.set_trace()
        tau=self.FAS(self.func_F(u), self.func_E(u), self.func_F(U_coarse), self.func_E(U_coarse))
        tau_minimax=self.FAS(self.func_F(u), self.func_E(u), self.func_F(u_coar), self.func_E(u_coar))
        e=np.zeros(self.coarse.num_nodes, dtype='complex')
        # pdb.set_trace()
        for ii in range(1):
            uiter=self.sweep(U0c, U_coarse, level=self.coarse, tau=tau)
            # rc=esol
            # print("coarse")

        for ii in range(1):
            u_minimax=self.sweep(U0c, u_coar, level=self.coarse, tau=tau_minimax)

        # error=self.prolongation(esol)
        ufinest=U+self.prolongation(-U_coarse+uiter)
        ufine_minimax=U+self.prolongation(u_minimax-u_coar)
        relaxu_fine=self.sweep(U0, ufinest, level=self.fine)
        relaxu_minimax=self.sweep(U0, ufine_minimax, level=self.fine)


        # rfinest=self.residual(ufinest, U0)
        pdb.set_trace()

    def best_function(self, nodes, order_poly=3, filename=None):

        coeff=np.genfromtxt(filename, delimiter=',')
        Real=coeff[0:order_poly+1]
        Imag=coeff[order_poly+1:]
        values=Real+Imag*1j

        u_coar=np.zeros(len(nodes), dtype='complex')
        for jj, nn in enumerate(nodes):

            poly=values[-1]
            for ii in range(order_poly,0,-1):
                print(ii)
                print(poly)
                poly= poly+values[order_poly-ii]*nn**ii
            # pdb.set_trace()
            u_coar[jj]=poly

        return u_coar



    def coefficient(self, x, f):
        X=np.ones(np.size(x))
        for ii in range(1, len(x)):
            X=np.vstack((x**ii, X ))
        d=np.real(np.linalg.det(X))
        D=np.vstack((f, X))
        c=np.zeros(len(x), dtype='complex')
        file = open("coefficient.txt", "w")

        for ii in range(1, len(x)+1):
            # if ii==1:
            # else:
            #     file = open("coefficient.txt", "a")
            print(len(x)-ii+1)
            mat=np.delete(D,len(x)-ii+1 ,0)

            c[ii-1]=(-1)**(len(x)-ii)*(np.linalg.det(mat)/d)
            # pdb.set_trace()
            # if ii==1:
            #     file.write(str(c[ii-1]))
            # else:
            #     file.write(", " + str(c[ii-1]))

        file.write(str(np.real(c))+ "\n"+ str(np.imag(c)))
        file.close()
        c_real=np.flip(np.real(c))
        c_imag=np.flip(np.imag(c))
        C=np.append(c_real, c_imag)
        np.savetxt('coefficient.csv', C)

        pdb.set_trace()








    def iteration(self):

        #base method
        uI_old=self.base_method(self.prob.u0)
        u0=np.ones(self.fine.num_nodes+1, dtype="complex")*self.prob.u0

        # perform fine sweep
        for ii in range(self.coll_params.Kiter):
            u=self.fwsw_sweep(uI_old)
            uI_old=u
            # FI_old=FI
            # FE_old=FE
            # u0=u
        # Cycle from fine to coarse
        # pdb.set_trace()
        U_coarse=np.append(self.prob.u0,self.restrict(u[1:]))
        # FI_coarse=self.func_F(U_coarse)
        # FE_coarse=self.func_E(U_coarse)



        # Compute FAS correction and sweep

        tau=self.FAS(self.func_F(u[1:]), self.func_E(u[1:]), self.func_F(U_coarse[1:]), self.func_E(U_coarse[1:]))

        r=self.residual(u[1:], u0[1:], level=self.fine)
        r_coarse=np.append(0,self.restrict(r))
        for ii in range(100):
            e=self.fwsw_sweep(r_coarse, level=self.coarse)
            r_coarse=e
        e_fine=self.prolongation(e[1:])
        usol=u[1:]+e_fine
        u_sol=np.append(self.prob.u0, usol)
        rr=self.residual(usol, u0[1:])
        plt.plot(u_sol.real, u_sol.imag)
        pdb.set_trace()
        # for ii in range(10):
        #     u_coarse=self.fwsw_sweep(U_coarse, tau=tau, level=self.coarse) #self.SDC_sweep(U_coarse, FI_coarse, FE_coarse, tau, level=self.coarse)
        #     U_coarse=u_coarse
            # FI_coarse=FI_c
            # FE_coarse=FE_c

        # # Cycle from coarse to fine
        # inter=self.prolongation((u_coarse-U_tilde)[1:])
        # u_f=u[1:]+inter
        # usol=np.append(self.prob.u0, u_f)
        # # FI_f=self.func_F(u_f)
        # # FE_f=self.func_E(u_f)
        # pdb.set_trace()
        # for ii in range(self.coll_params.Kiter):
        #     u_cf, FI_cf, FE_cf=self.SDC_sweep(u_f, FI_f, FE_f)
        #     u_f=u_cf
        #     FI_f=FI_cf
        #     FE_f=FE_cf
        # return u_cf, FI_cf, FE_cf


class Transfer(object):
    def __init__(self,fine_num_nodes, coarse_num_nodes):
        pdb.set_trace()
        self.Pcoll=self.get_transfer_matrix(fine_num_nodes, coarse_num_nodes)
        self.Rcoll=self.get_transfer_matrix(coarse_num_nodes, fine_num_nodes)

    def get_transfer_matrix(self, fine_nodes, coarse_nodes):

        approx=LagrangeApproximation(coarse_nodes)
        return approx.getInterpolationMatrix(fine_nodes)








if __name__=='__main__':
    problem_params=dict()
    problem_params['lambda_f']=0+10j
    problem_params['lambda_s']=0+1j
    problem_params['u0']=1.0
    problem_params['dt']=0.1
    problem_params['Tend']=2.0

    collocation_params=dict()
    collocation_params['quad_type']='LOBATTO'
    collocation_params['num_nodes']=[5,3]
    collocation_params['Kiter']=1




    nn=np.linspace(0, 0.1, 100)
    sdc=MLSDC(problem_params, collocation_params)
    sdc.mlsdc_fwsw()
    # u, fi, fe=
    # plt.plot(aa.real, aa.imag)
    # plt.plot(np.exp(11j*nn).real, np.exp(11j*nn).imag)

    pdb.set_trace()
