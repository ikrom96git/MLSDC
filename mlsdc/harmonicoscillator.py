import numpy as np
import matplotlib.pyplot as plt

from core.Collocation import CollBase
from core.Lagrange import LagrangeApproximation
from datetime import datetime
import pdb

class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

class second_order_SDC(object):
    def __init__(self, problem_params, collocation_params):
        self.prob=_Pars(problem_params)
        self.coll=get_collocation_params(collocation_params['num_nodes'], collocation_params['quad_type'])
        I=np.eye(self.coll.num_nodes)
        self.FF=np.block([[I*(-self.prob.kappa), I*(-self.prob.mu)], [I*(-self.prob.kappa), I*(-self.prob.mu)]])
    def func(self, x, v):
        return -self.prob.kappa * x - self.prob.mu * v

    def base_method(self, type='spread'):
        if type=='spread':
            U=np.block([np.ones(self.coll.num_nodes)*self.prob.u0[0], np.ones(self.coll.num_nodes)*self.prob.u0[1]])
        else:
            ValueError("Base method only works for spread")
        return U

    def sweep(self, Uold, U0, tau=[None]):
        if None in tau:
            tau=np.zeros(len(U0))


        IF=np.eye(2*self.coll.num_nodes)-self.coll.Qvv@self.FF
        QF=(self.coll.Qcoll-self.coll.Qvv)@self.FF
        RHS=QF@Uold+self.coll.Ccoll@U0+tau
        U=np.linalg.solve(IF, RHS)
        return U

    def residual(self, U, U0):
        rr=self.coll.Ccoll@U0+self.coll.Qcoll@(self.FF@U)-U
        return np.abs(rr)

    def exact(self, t):
        pass

class MLSDC(object):
    def __init__(self, problem_params, collocation_params):
        self.prob=_Pars(problem_params)

        fine_coll=collocation_params.copy()
        fine_coll['num_nodes']=collocation_params['num_nodes'][0]
        coarse_coll=collocation_params.copy()
        coarse_coll['num_nodes']=collocation_params['num_nodes'][1]
        self.fine=second_order_SDC(problem_params, fine_coll)
        self.coarse=second_order_SDC(problem_params, coarse_coll)
        self.transfer=Transfer(self.fine.coll.coll.nodes, self.coarse.coll.coll.nodes)
        self.Rcoll=np.block([[self.transfer.Rcoll, np.zeros(np.shape(self.transfer.Rcoll))], [np.zeros(np.shape(self.transfer.Rcoll)), self.transfer.Rcoll]])
        self.Pcoll=np.block([[self.transfer.Pcoll, np.zeros(np.shape(self.transfer.Pcoll))], [np.zeros(np.shape(self.transfer.Pcoll)), self.transfer.Pcoll]])


    def FAS_tau(self, u_fine, u_coarse):
        # pdb.set_trace()
        tau=(self.Rcoll @ self.fine.coll.Qcoll @ self.fine.FF @ u_fine - self.coarse.coll.Qcoll @ self.coarse.FF @ u_coarse)
        return tau

    def restriction(self, U):
        return self.Rcoll @ U

    def prolongation(self, U):
        return self.Pcoll @ U

    def MLSDC_sweep(self, U_fine=[None]):
        U0_fine=self.fine.base_method()
        U0_coarse=self.coarse.base_method()

        if None in U_fine:
            U_fine=self.fine.base_method()

        # Fine to coarse without sweep
        U_coarse=self.restriction(U_fine)

        tau_coarse=self.FAS_tau(U_fine, U_coarse)

        # Coarse sweep

        Uc=self.coarse.sweep(U_coarse, U0_coarse, tau=tau_coarse)

        # Coarse to fine

        U_fine=U_fine+self.prolongation(Uc-U_coarse)

        # Relax: fine sweep

        U_finest=self.fine.sweep(U_fine, U0_fine)

        # MLSDC residual
        Residual=self.fine.residual(U_finest, U0_fine)
        # pdb.set_trace()

        return U_finest, Residual

    def SDC_sweep(self, U=[None]):

        U0=self.fine.base_method()

        if None in U:
            U=self.fine.base_method()

        # SDC sweep
        U_sweep=self.fine.sweep(U, U0)

        # SDC residual
        Residual=self.fine.residual(U_sweep, U0)

        return U_sweep, Residual

    def MLSDC_iter(self, Kiter=None):
        Residual_MLSDC=np.zeros([Kiter, 2])
        U=[None]

        # MLSDC iteraiton
        for ii in range(Kiter):
            U_finest, Residual=self.MLSDC_sweep(U_fine=U)
            U=U_finest


            Residual_MLSDC[ii, 0]=np.linalg.norm(Residual[:self.fine.coll.num_nodes], np.inf)
            Residual_MLSDC[ii, 1]=np.linalg.norm(Residual[self.fine.coll.num_nodes:], np.inf)
            # pdb.set_trace()

        return U, Residual_MLSDC

    def SDC_iter(self, Kiter=None):
        Residual_SDC=np.zeros([Kiter, 2])
        U=[None]

        # SDC iteration
        for ii in range(Kiter):
            U_SDC, Residual=self.SDC_sweep(U=U)
            U=U_SDC

            Residual_SDC[ii, 0]=np.linalg.norm(Residual[:self.fine.coll.num_nodes], np.inf)
            Residual_SDC[ii, 1]=np.linalg.norm(Residual[self.fine.coll.num_nodes:], np.inf)



        return U, Residual_SDC

class minimax(MLSDC):
    def __init__(self, problem_params, collocation_params):
        super().__init__(problem_params, collocation_params)
        self.U0_fine=self.fine.base_method()
        self.U0_coarse=self.coarse.base_method()



    def fine_coeff(self, dummy_data=True):

        if dummy_data:

            np.savetxt('U_finest.csv', self.U0_fine)

        nodes=self.fine.coll.coll.nodes
        U=np.genfromtxt('U_finest.csv')
        print('U_finest:', U)
        u_pos, u_vel=np.split(U, 2)
        A=np.ones(np.size(nodes))
        for ii in range(1, len(nodes)):
            A=np.vstack((nodes**ii, A ))
        d=np.real(np.linalg.det(A))
        D_pos=np.vstack((u_pos, A))
        D_vel=np.vstack((u_vel, A))

        c_pos=np.zeros(len(nodes))
        c_vel=np.zeros(len(nodes))


        for ii in range(1, len(nodes)+1):

            mat_pos=np.delete(D_pos,len(nodes)-ii+1 ,0)
            mat_vel=np.delete(D_vel,len(nodes)-ii+1 ,0)

            c_pos[ii-1]=(-1)**(len(nodes)-ii)*(np.linalg.det(mat_pos)/d)

            c_vel[ii-1]=(-1)**(len(nodes)-ii)*(np.linalg.det(mat_vel)/d)

        c_fpos=np.flip(c_pos)
        c_fvel=np.flip(c_vel)
        C=np.append(c_fpos, c_fvel)

        np.savetxt('coefficient_fine.csv', C)

    def FAS_tau_minimax(self, u_fine, u_coarse):

        tau=(u_fine- self.coarse.coll.Qcoll @ self.coarse.FF @ u_coarse)
        return tau

    def coar_coeff(self, order_poly=3):
        nodes=self.coarse.coll.coll.nodes

        coeff_coar=np.genfromtxt('coefficient_coar.csv', delimiter=',')
        coeff_pos, coeff_vel=np.split(coeff_coar, 2)

        u_pos=np.zeros(len(nodes))
        u_vel=np.zeros(len(nodes))
        # pdb.set_trace()
        for jj, nn in enumerate(nodes):

            poly_pos=coeff_pos[-1]
            poly_vel=coeff_vel[-1]

            for ii in range(order_poly, 0, -1):
                poly_pos=poly_pos+coeff_pos[order_poly-ii]*nn**ii
                poly_vel=poly_vel+coeff_vel[order_poly-ii]*nn**ii

            u_pos[jj]=poly_pos
            u_vel[jj]=poly_vel
        return np.block([u_pos, u_vel])

    def fineQ_coeff(self, dummy_data=True):

        if dummy_data:

            QU=self.fine.coll.Qcoll @ self.fine.FF @ self.U0_fine

            np.savetxt('QU_finest.csv', QU)

        nodes=self.fine.coll.coll.nodes
        U=np.genfromtxt('QU_finest.csv')
        print(U)
        u_pos, u_vel=np.split(U, 2)
        A=np.ones(np.size(nodes))
        for ii in range(1, len(nodes)):
            A=np.vstack((nodes**ii, A ))
        d=np.real(np.linalg.det(A))
        D_pos=np.vstack((u_pos, A))
        D_vel=np.vstack((u_vel, A))

        c_pos=np.zeros(len(nodes))
        c_vel=np.zeros(len(nodes))


        for ii in range(1, len(nodes)+1):

            mat_pos=np.delete(D_pos,len(nodes)-ii+1 ,0)
            mat_vel=np.delete(D_vel,len(nodes)-ii+1 ,0)

            c_pos[ii-1]=(-1)**(len(nodes)-ii)*(np.linalg.det(mat_pos)/d)

            c_vel[ii-1]=(-1)**(len(nodes)-ii)*(np.linalg.det(mat_vel)/d)

        c_fpos=np.flip(c_pos)
        c_fvel=np.flip(c_vel)
        C=np.append(c_fpos, c_fvel)
        # pdb.set_trace()
        np.savetxt('coefficient_fineQU.csv', C)

    def coarQU_coeff(self, order_poly=3):
        nodes=self.coarse.coll.coll.nodes

        coeff_coar=np.genfromtxt('coefficient_coarQU.csv', delimiter=',')
        coeff_pos, coeff_vel=np.split(coeff_coar, 2)

        u_pos=np.zeros(len(nodes))
        u_vel=np.zeros(len(nodes))
        # pdb.set_trace()
        for jj, nn in enumerate(nodes):

            poly_pos=coeff_pos[-1]
            poly_vel=coeff_vel[-1]

            for ii in range(order_poly, 0, -1):
                poly_pos=poly_pos+coeff_pos[order_poly-ii]*nn**ii
                poly_vel=poly_vel+coeff_vel[order_poly-ii]*nn**ii

            u_pos[jj]=poly_pos
            u_vel[jj]=poly_vel
        return np.block([u_pos, u_vel])

    def coarse_to_fine(self):
        U_fine=np.genfromtxt('U_finest.csv')
        U_coarse=self.coar_coeff()
        QU=self.coarQU_coeff()

        tau_coarse=self.FAS_tau_minimax(QU, U_coarse)
        # pdb.set_trace()
        # print('fine_value:', U_fine)
        # Coarse sweep

        Uc=self.coarse.sweep(U_coarse, self.U0_coarse, tau=tau_coarse)

        # Coarse to fine

        U_fine=U_fine+self.prolongation(Uc-U_coarse)

        # Relax: fine sweep

        U_finest=self.fine.sweep(U_fine, self.U0_fine)
        QU_finest=self.fine.coll.Qcoll @ self.fine.FF @ U_finest

        np.savetxt('QU_finest.csv', QU_finest)
        np.savetxt('U_finest.csv', U_finest)
        # pdb.set_trace()

        # MLSDC residual
        Residual=self.fine.residual(U_finest, self.U0_fine)
        # save data
        now=str(datetime.now())
        # pdb.set_trace()

        # np.save('data/{}_sol.npy'.format(now[14:19]), U_finest)
        np.save('data/{}_res.npy'.format(now[14:19]), Residual)
        # print(U_finest)
        # print(Residual)
        # pdb.set_trace()

        return U_finest, Residual

    def save_data(self):
        pass





class get_collocation_params(object):
    def __init__(self, num_nodes, quad_type, dt=None):
        if dt==None:
            self.dt=1
        else:
            self.dt=dt
        self.num_nodes=num_nodes
        self.quad_type=quad_type
        [self.Q, self.QI,self.QE, self.S]=self.Qmatrix(num_nodes=num_nodes, quad_type=quad_type)
        self.QQ=self.Q@self.Q
        self.QT=0.5*(self.QI+self.QE)
        self.Qx=self.QE@self.QT + 0.5*np.multiply(self.QE, self.QE)
        self.Ccoll, self.Qcoll, self.Cvv, self.Qvv=self.get_matrix()


    def get_matrix(self):
        c_coll=np.block([[np.eye(self.num_nodes), self.Q], [np.zeros(np.shape(self.Q)), np.eye(self.num_nodes)]])
        q_coll=np.block([[self.QQ, np.zeros(np.shape(self.Q))], [np.zeros(np.shape(self.Q)), self.Q]])
        c_vv=np.block([[np.eye(self.num_nodes), self.QE], [np.zeros(np.shape(self.Q)), np.eye(self.num_nodes)]])
        q_vv=np.block([[self.Qx, np.zeros(np.shape(self.Q))],[ np.zeros(np.shape(self.Q)), self.QT]])
        return c_coll, q_coll, c_vv, q_vv



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


        return [self.dt*Q, self.dt*QI, self.dt*QE,  self.dt*S]


class Transfer(object):
    def __init__(self,fine_num_nodes, coarse_num_nodes):
        self.Pcoll=self.get_transfer_matrix(fine_num_nodes, coarse_num_nodes)
        self.Rcoll=self.get_transfer_matrix(coarse_num_nodes, fine_num_nodes)

    def get_transfer_matrix(self, fine_nodes, coarse_nodes):

        approx=LagrangeApproximation(coarse_nodes)
        return approx.getInterpolationMatrix(fine_nodes)

def Plot_residual(x_axis, y_axis, titles=None):
    plt.figure()
    marker=['s', 'o', 'x']
    for ii in range(np.shape(y_axis)[0]):
        plt.semilogy(x_axis,y_axis[ii, :],marker=marker[ii] , label=titles[ii])

    plt.xlabel("The number of iteration")
    plt.ylabel("$\|R\|_{\infty}$")
    plt.title('Position')
    plt.legend()
    plt.tight_layout()
    plt.show()







if __name__=='__main__':
    problem_params=dict()
    problem_params['kappa']=1
    problem_params['mu']=2
    problem_params['u0']=[1.0, 3]
    problem_params['dt']=0.5
    problem_params['Tend']=2.0

    collocation_params=dict()
    collocation_params['quad_type']='LOBATTO'
    collocation_params['num_nodes']=[5,3]
    iteration=MLSDC(problem_params, collocation_params)
    U_MLSDC, R_MLSDC=iteration.MLSDC_iter(Kiter=4)
    U_SDC, R_SDC=iteration.SDC_iter(Kiter=4)




#     title=['MLSDC', 'SDC']





#     nn=np.linspace(0, 0.1, 100)

#     pdb.set_trace()
