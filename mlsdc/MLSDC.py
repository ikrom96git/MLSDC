import numpy as np
import matplotlib.pyplot as plt
from harmonicoscillator import get_collocation_params, Transfer
import pdb


class _Pars(object):
    def __init__(self, pars):
        for k, v in pars.items():
            setattr(self, k, v)


class penning_trap(object):
    def __init__(self, params, collocation_params):
        self.params = _Pars(params)
        self.coll = _Pars(collocation_params)

        self.fine = get_collocation_params(
            self.coll.num_nodes[0], self.coll.quad_type, dt=self.params.dt
        )
        self.coarse = get_collocation_params(
            self.coll.num_nodes[1], self.coll.quad_type, dt=self.params.dt
        )
        self.transfer = Transfer(self.fine.coll.nodes, self.coarse.coll.nodes)
        self.RHS_non_uniform()
        self.Penning = penningtrap_solution(params)
        # self.Penning.test(self.params.dt*self.fine.coll.nodes)

    def ndim(self, Q, n):
        return np.kron(np.eye(n), Q)

    def RHS_non_uniform(self):
        self.RHSv = np.zeros((3, 3))
        self.RHSv[1, 2] = 1
        self.RHSv[2, 1] = -1
        self.RHSx = -np.eye(3)
        self.RHSx[1, 1] = 0.5
        self.RHSx[2, 2] = 0.5
        self.RHSv = (1 / self.params.eps) * self.RHSv
        self.RHSx = self.params.c * self.RHSx
        return self.RHSx, self.RHSv

    def reduced_model_RHS0(self, x):
        return -self.params.c * x

    def reduced_1order(self):
        F = np.zeros((12, 12))
        F[0, 3] = 1
        F[3, 0] = -self.params.c
        F[6, 9] = 1
        F[7, 2] = 0.5 * self.params.c
        F[8, 1] = -0.5 * self.params.c
        F[9, 6] = -self.params.c
        F[10, 5] = -0.5 * self.params.c
        F[11, 4] = 0.5 * self.params.c
        return F

    def SDC_reduced1(self, U0, U, level):
        num_nodes = level.coll.num_nodes
        Q = self.ndim(level.Q, 12)
        QE = self.ndim(level.QE[1:, 1:], 12)
        QI = self.ndim(level.QI[1:, 1:], 12)
        Fx = self.reduced_1order()
        F = np.kron(Fx, np.eye(num_nodes))
        I = np.eye(np.shape(Q)[0])
        u0_red = np.zeros(np.shape(I)[0])
        u_red = np.copy(u0_red)
        u0_red[: len(U0)] = U0

        # u_red[:len(U)]=U
        A = I - QI @ F

        b = u0_red + Q @ F @ u_red
        Usol = np.linalg.solve(A, b)
        u = Usol.reshape(12, num_nodes)
        y0, u0, y1, u1 = np.split(u, 4)
        U_red = np.zeros((6, num_nodes))

        for ii in range(num_nodes):
            U_red[:, ii] = self.Gt(
                y0[:, ii],
                u0[:, ii],
                y1[:, ii],
                u1[:, ii],
                self.params.dt * level.coll.nodes[ii],
                self.params.s,
                self.params.eps,
            )
        U_red = U_red.reshape(6 * num_nodes)
        return U_red

    def update_step(self, x0, v0, U, level):
        q = self.params.dt * level.coll.weights
        num_nodes = level.coll.num_nodes
        Q = level.Q
        qb = q @ Q
        q = np.kron(np.eye(3), q)
        qb = np.kron(np.eye(3), qb)
        Fx, Fv = self.RHS_non_uniform()
        Fvv = np.kron(Fx, np.eye(num_nodes))
        Fvx = np.kron(Fv, np.eye(num_nodes))
        F = np.block([Fvv, Fvx])
        V0 = v0 * np.ones((num_nodes, 3))
        V0 = (V0.T).reshape(3 * num_nodes, 1)
        Xup = x0 + q @ V0 + qb @ F @ U
        Vup = v0 + q @ F @ U
        return Xup, Vup

    def update_step_reduced_model(self, u0, U, level):
        q = self.params.dt * level.coll.weights
        q = np.kron(np.eye(12), q)
        Fx = self.reduced_1order()
        F = np.kron(Fx, np.eye(level.coll.num_nodes))
        pdb.set_trace()
        qF = q @ F @ U
        return u0 + qF

    def Gt(self, yt0, ut0, yt1, ut1, t, s, eps):
        R0 = np.block([yt0, self.Penning.Rtheta((t - s) / eps) @ ut0])
        RR0 = np.block(
            [
                yt1 + self.Penning.RRtheta((t - s) / eps) @ ut0,
                self.Penning.Rtheta((t - s) / eps) @ ut1
                + self.Penning.RRtheta((t - s) / eps)
                @ self.Penning.non_uniform_E(yt0, self.params.c),
            ]
        )
        return R0 + eps * RR0

    def SDC_solver(self, U0, U, level, A):
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)
        Qx = self.ndim(level.Qx, 3)
        QT = self.ndim(level.QT, 3)
        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ - Qx, O], [O, Q - QT]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])

        # A=np.eye(2*np.shape(Q)[0])-Qmat@F
        b = Dmat @ F @ U + Q0 @ U0
        Usol = np.linalg.solve(A, b)

        return Usol

    def SDC_0D(self, U0, U, level):
        num_nodes = level.coll.num_nodes
        u0 = np.append(U0[:num_nodes], U0[3 * num_nodes : 4 * num_nodes])
        u = 0 * np.copy(u0)
        Q = level.Q
        QQ = level.QQ
        Qx = level.Qx
        QT = level.QT
        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ - Qx, O], [O, Q - QT]])
        Fx = self.reduced_model_RHS0(np.eye(num_nodes))

        F = np.block([[Fx, O], [Fx, O]])
        A = np.eye(2 * np.shape(Q)[0]) - Qmat @ F
        b = Dmat @ F @ u + Q0 @ u0
        Usol = np.linalg.solve(A, b)
        usol = Usol.reshape(2, num_nodes)
        y0 = np.zeros((3, num_nodes))
        u0 = np.zeros((3, num_nodes))
        y0[0, :], u0[0, :] = np.split(usol, 2)
        U_red = np.zeros((6, num_nodes))
        for ii in range(num_nodes):
            U_red[:, ii] = np.block(
                [
                    y0[:, ii],
                    self.Penning.Rtheta(
                        (level.coll.nodes[ii] - self.params.s) / self.params.eps
                    )
                    @ u0[:, ii],
                ]
            )
        U_red = U_red.reshape(6 * num_nodes)

        return U_red

    def SDC_iter(self, k):
        x0 = self.params.u0[0]
        v0 = self.params.u0[1]
        num_nodes = self.fine.coll.num_nodes
        X0 = x0 * np.ones((num_nodes, 3))
        V0 = v0 * np.ones((num_nodes, 3))
        X0 = (X0.T).reshape(3 * num_nodes, 1)
        V0 = (V0.T).reshape(3 * num_nodes, 1)
        U0 = np.append(X0, V0)
        U = np.copy(U0)
        for ii in range(k):
            Usol = self.SDC_solver(U0, U, self.fine)
            # xrm ,vrm =self.SDC_1D(x0, v0, x0, v0, self.fine)
            U = Usol

        # Usol=U.reshape(6, num_nodes)
        # plt.plot(self.params.dt*self.fine.coll.nodes, Usol[3,:], label='SDC solution')
        # plt.legend()

    def MLSDC_iter(self, K_iter):
        level = self.fine
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)
        Qx = self.ndim(level.Qx, 3)
        QT = self.ndim(level.QT, 3)
        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ - Qx, O], [O, Q - QT]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])
        A = np.eye(2 * np.shape(Q)[0]) - Qmat @ F

        x0 = self.params.u0[0]
        v0 = self.params.u0[1]
        num_nodes = self.fine.coll.num_nodes
        X0 = x0 * np.ones((num_nodes, 3))
        V0 = v0 * np.ones((num_nodes, 3))
        X0 = (X0.T).reshape(3 * num_nodes, 1)
        V0 = (V0.T).reshape(3 * num_nodes, 1)
        U0 = np.append(X0, V0)
        U = self.collocation_solution(U0, level=self.fine)
        Rcoll = np.kron(np.eye(6), self.transfer.Rcoll)
        Pcoll = np.kron(np.eye(6), self.transfer.Pcoll)
        U0_coarse = Rcoll @ U0
        A_coarse = Rcoll @ A @ Pcoll
        max_res = np.zeros(6)
        for ii in range(K_iter):
            U_coarse = 0 * U0_coarse
            U0_res = self.Residual(U0, U, level=self.fine)

            U0_res1 = Rcoll @ U0_res

            Usol_coarse = self.SDC_solver(
                U0_res1, U_coarse, level=self.coarse, A=A_coarse
            )
            U_fine = U + Pcoll @ Usol_coarse
            Usol = self.SDC_solver(U0, U_fine, level=self.fine, A=A)
            print(Usol == U)
            U = np.copy(Usol)
            u_res = U0_res.reshape(6, num_nodes)
            U_max_res = np.max(np.abs(u_res), axis=1)
            # pdb.set_trace()
            max_res = np.vstack((max_res, U_max_res))

        # Usol=U.reshape(6, num_nodes)

        # plt.plot(self.params.dt*self.fine.coll.nodes, Usol[3,:], label='MLSDC solution')
        # plt.legend()
        return max_res[1:, :]

    def Residual_reduced(self, U0, U, level):
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        # QQ=self.ndim(level.QQ,3)

        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])

        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[Q, O], [O, Q]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])
        Res = U0 + Dmat @ F @ U - U
        return Res

    def Residual(self, U0, U, level):
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)

        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])

        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ, O], [O, Q]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])
        Res = Q0 @ U0 + Dmat @ F @ U - U
        return Res

    def MLSDC_iter_order1(self, K_iter):
        level = self.fine
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)
        Qx = self.ndim(level.Qx, 3)
        QT = self.ndim(level.QT, 3)
        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ - Qx, O], [O, Q - QT]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])
        A = np.eye(2 * np.shape(Q)[0]) - Qmat @ F

        x0 = self.params.u0[0]
        v0 = self.params.u0[1]
        num_nodes = self.fine.coll.num_nodes
        X0 = x0 * np.ones((num_nodes, 3))
        V0 = v0 * np.ones((num_nodes, 3))
        X0 = (X0.T).reshape(3 * num_nodes, 1)
        V0 = (V0.T).reshape(3 * num_nodes, 1)
        U0 = np.append(X0, V0)
        U = self.collocation_solution(U0, level=self.fine)
        Rcoll = np.kron(np.eye(6), self.transfer.Rcoll)
        Pcoll = np.kron(np.eye(6), self.transfer.Pcoll)
        A_coarse = Rcoll @ A @ Pcoll
        U0_coarse = Rcoll @ U0
        max_res = np.zeros(6)
        for ii in range(K_iter):
            U_coarse = 0 * U0_coarse
            U0_res = self.Residual(U0, U, level=self.fine)
            U0_res1 = Rcoll @ U0_res

            Usol_coarse = self.SDC_reduced1(U0_res1, U_coarse, level=self.coarse)

            U_fine = U + Pcoll @ Usol_coarse

            Usol = self.SDC_solver(U0, U_fine, level=self.fine, A=A)
            U = Usol
            u_res = U0_res.reshape(6, num_nodes)
            U_max_res = np.max(np.abs(u_res), axis=1)

            max_res = np.vstack((max_res, U_max_res))

        # Usol=U.reshape(6, num_nodes)
        # # print(Usol)
        # plt.plot(self.params.dt*self.fine.coll.nodes, Usol[5,:], label='SDC reduced solution')
        # plt.legend()
        return max_res[1:, :]

    def collocation_solution(self, U0, level):
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)
        Qx = self.ndim(level.Qx, 3)
        QT = self.ndim(level.QT, 3)
        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ, O], [O, Q]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])

        A = np.eye(2 * np.shape(Q)[0]) - Dmat @ F
        b = Q0 @ U0
        Usol = np.linalg.solve(A, b)
        return Usol

    def MLSDC_iter_order11(self, K_iter):
        level = self.fine
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)
        Qx = self.ndim(level.Qx, 3)
        QT = self.ndim(level.QT, 3)
        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ - Qx, O], [O, Q - QT]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])
        A = np.eye(2 * np.shape(Q)[0]) - Qmat @ F

        x0 = self.params.u0[0]
        v0 = self.params.u0[1]
        num_nodes = self.fine.coll.num_nodes
        X0 = x0 * np.ones((num_nodes, 3))
        V0 = v0 * np.ones((num_nodes, 3))
        X0 = (X0.T).reshape(3 * num_nodes, 1)
        V0 = (V0.T).reshape(3 * num_nodes, 1)
        U0 = np.append(X0, V0)
        U = np.copy(U0)
        Rcoll = np.kron(np.eye(6), self.transfer.Rcoll)
        Pcoll = np.kron(np.eye(6), self.transfer.Pcoll)
        U0_coarse = Rcoll @ U0
        max_res = np.zeros(6)

        A_coarse = Rcoll @ A @ Pcoll
        for ii in range(K_iter):
            U_coarse = 0 * U0_coarse
            U0_res = self.Residual_reduced(U0, U, level=self.fine)
            U0_res1 = Rcoll @ U0_res

            Usol_coarse = self.SDC_reduced1(U0_res1, U_coarse, level=self.coarse)

            U_fine = U + Pcoll @ Usol_coarse

            Usol = self.SDC_reduced1(U0, U_fine, level=self.fine)
            U = Usol
            u_res = U0_res.reshape(6, num_nodes)
            U_max_res = np.max(np.abs(u_res), axis=1)

            max_res = np.vstack((max_res, U_max_res))
            # pdb.set_trace()
        # self.update_step(x0, v0, U, level=self.fine)
        # Usol=U.reshape(6, num_nodes)
        # print(Usol)
        # plt.plot(self.params.dt*self.fine.coll.nodes, Usol[3,:], label='SDC reduced solution')
        # plt.legend()
        return max_res[1:, :]

    def MLSDC_iter_order0(self, K_iter):
        level = self.fine
        num_nodes = level.coll.num_nodes

        Q = self.ndim(level.Q, 3)
        QQ = self.ndim(level.QQ, 3)
        Qx = self.ndim(level.Qx, 3)
        QT = self.ndim(level.QT, 3)
        Fvv = np.kron(self.RHSv, np.eye(num_nodes))
        Fvx = np.kron(self.RHSx, np.eye(num_nodes))

        O = np.zeros((np.shape(Q)[0], np.shape(Q)[0]))
        I = np.eye(np.shape(Q)[0])
        Qmat = np.block([[Qx, O], [O, QT]])
        Q0 = np.block([[I, Q], [O, I]])
        Dmat = np.block([[QQ - Qx, O], [O, Q - QT]])
        F = np.block([[Fvx, Fvv], [Fvx, Fvv]])
        A = np.eye(2 * np.shape(Q)[0]) - Qmat @ F

        x0 = self.params.u0[0]
        v0 = self.params.u0[1]
        num_nodes = self.fine.coll.num_nodes
        X0 = x0 * np.ones((num_nodes, 3))
        V0 = v0 * np.ones((num_nodes, 3))
        X0 = (X0.T).reshape(3 * num_nodes, 1)
        V0 = (V0.T).reshape(3 * num_nodes, 1)
        U0 = np.append(X0, V0)
        U = self.collocation_solution(U0, level=self.fine)
        Rcoll = np.kron(np.eye(6), self.transfer.Rcoll)
        Pcoll = np.kron(np.eye(6), self.transfer.Pcoll)
        U0_coarse = Rcoll @ U0
        max_res = np.zeros(6)

        A_coarse = Rcoll @ A @ Pcoll
        for ii in range(K_iter):
            U_coarse = 0 * U0_coarse
            U0_res = self.Residual(U0, U, level=self.fine)
            U0_res1 = Rcoll @ U0_res

            Usol_coarse = self.SDC_0D(U0_res1, U_coarse, level=self.coarse)

            U_fine = U + Pcoll @ Usol_coarse

            Usol = self.SDC_solver(U0, U_fine, level=self.fine, A=A)
            U = Usol
            u_res = U0_res.reshape(6, num_nodes)
            U_max_res = np.max(np.abs(u_res), axis=1)

            max_res = np.vstack((max_res, U_max_res))
            # pdb.set_trace()
        # self.update_step(x0, v0, U, level=self.fine)
        # Usol=U.reshape(6, num_nodes)
        # print(Usol)
        # plt.plot(self.params.dt*self.fine.coll.nodes, Usol[3,:], label='SDC reduced solution')
        # plt.legend()
        return max_res[1:, :]

    def compare_plot(self, Kiter):
        MLSDC_iter = self.MLSDC_iter(Kiter)
        MLSDC_iter_order1 = self.MLSDC_iter_order1(Kiter)
        # MLSDC_iter_order11=self.MLSDC_iter_order11(Kiter)
        MLSDC_iter_order11 = self.MLSDC_iter_order0(Kiter)
        print(MLSDC_iter_order11)
        K = np.arange(Kiter)
        axis = 5

        plt.semilogy(K, MLSDC_iter[:, axis], label="MLSDC iteration")
        plt.semilogy(
            K, MLSDC_iter_order1[:, axis], label="Reduced model order 1-coarse"
        )
        plt.semilogx(
            K, MLSDC_iter_order11[:, axis], label="Reduced model order 0 coarse "
        )
        plt.legend()
        plt.xlabel("K iteration")
        plt.ylabel("Max residual")
        plt.tight_layout()


class penningtrap_solution(object):
    def __init__(self, params):
        self.params = _Pars(params)

    def Rtheta(self, theta):
        R = np.zeros([3, 3])
        R[0, 0] = 1
        R[1, 1] = np.cos(theta)
        R[1, 2] = np.sin(theta)
        R[2, 1] = -np.sin(theta)
        R[2, 2] = np.cos(theta)
        return R

    def RRtheta(self, theta):
        RR = np.zeros([3, 3])
        RR[1, 1] = np.sin(theta)
        RR[1, 2] = 1 - np.cos(theta)
        RR[2, 1] = np.cos(theta) - 1
        RR[2, 2] = np.sin(theta)
        return RR

    def Amat(self, t, s, eps):
        I = np.eye(3)
        O = np.zeros([3, 3])
        P = np.zeros([3, 3])
        P[0, 0] = 1
        A = np.block(
            [
                [I, (t - s) * P + eps * self.RRtheta((t - s) / eps)],
                [O, self.Rtheta((t - s) / eps)],
            ]
        )
        return A

    def Bmat(self, t, s, eps):
        b11 = eps * (t - s) * np.array([0, -np.cos(t / eps), np.sin(t / eps)])
        b12 = eps**2 * np.array(
            [0, np.sin(t / eps) - np.sin(s / eps), np.cos(t / eps) - np.cos(s / eps)]
        )
        b2 = (t - s) * np.array([0, np.sin(t / eps), np.cos(t / eps)])
        return np.block([b11 + b12, b2])

    def exactsol_uniform(self, x, v, t, s, eps):
        Amat = self.Amat(t, s, eps)
        Bmat = self.Bmat(t, s, eps)
        U = np.block([x, v])
        return Amat @ U + Bmat

    def Cmat(self, t, s, eps):
        c1 = (
            eps * (t - s) * np.array([0, -np.cos((t - s) / eps), np.sin((t - s) / eps)])
        )
        c2 = (t - s) * np.array([0, np.sin((t - s) / eps), np.cos((t - s) / eps)])
        return np.block([c1, c2])

    def Gmat_uniform(self, x, v, t, s, eps):
        Amat = self.Amat(t, s, eps)
        U = np.block([x, v])
        Cmat = self.Cmat(t, s, eps)
        return Amat @ U + Cmat

    def non_uniform_U(self, t, s, eps, c):
        cos = lambda a, t, s: np.cos(a * (t - s))
        sin = lambda a, t, s: np.sin(a * (t - s))

        a_eps = (1 + np.sqrt(1 - 2 * c * eps**2)) / (2 * eps)
        b_eps = (1 - np.sqrt(1 - 2 * c * eps**2)) / (2 * eps)

        U_eps = np.zeros([6, 6])
        U_eps[0, :] = np.array(
            [0, 0, 0, 0, cos(np.sqrt(c), t, s), sin(np.sqrt(c), t, s)]
        )
        U_eps[1, :] = np.array(
            [
                sin(a_eps, t, s),
                -cos(a_eps, t, s),
                sin(b_eps, t, s),
                -cos(b_eps, t, s),
                0,
                0,
            ]
        )
        U_eps[2, :] = np.array(
            [
                cos(a_eps, t, s),
                sin(a_eps, t, s),
                cos(b_eps, t, s),
                sin(b_eps, t, s),
                0,
                0,
            ]
        )
        U_eps[3, :] = np.array(
            [
                0,
                0,
                0,
                0,
                -np.sqrt(c) * sin(np.sqrt(c), t, s),
                np.sqrt(c) * cos(np.sqrt(c), t, s),
            ]
        )
        U_eps[4, :] = np.array(
            [
                a_eps * cos(a_eps, t, s),
                a_eps * sin(a_eps, t, s),
                b_eps * cos(b_eps, t, s),
                b_eps * sin(b_eps, t, s),
                0,
                0,
            ]
        )
        U_eps[5, :] = np.array(
            [
                -a_eps * sin(a_eps, t, s),
                a_eps * cos(a_eps, t, s),
                -b_eps * sin(b_eps, t, s),
                b_eps * cos(b_eps, t, s),
                0,
                0,
            ]
        )

        return U_eps

    def non_uniform_E(self, x, c):
        return c * np.array([-x[0], 0.5 * x[1], 0.5 * x[2]])

    def non_uniform_coeff(self, x, v, t, s, eps, c):
        U_0 = self.non_uniform_U(t, s, eps, c)
        U = np.block([x, v])
        coeff = np.linalg.solve(U_0, U)

        return coeff

    def non_uniform_exact(self, x, v, t, t0, s, eps, c):
        coeff = self.non_uniform_coeff(x, v, t0, s, eps, c)
        if self.params.eps > np.sqrt(1 / (2 * self.params.c)):
            raise ValueError(
                "Solution does not exist. Need: $\epsilon< \frac{1}{\sqrt(2c))}$"
            )

        U = self.non_uniform_U(t, s, eps, c) @ coeff

        return U

    def non_uniform_reduced(self, x, v, t, s, c):
        u0 = np.block([x, v])
        yt0 = np.array(
            [
                u0[0] * np.cos(np.sqrt(c) * (t - s))
                + (u0[3] / (np.sqrt(c))) * np.sin(np.sqrt(c) * (t - s)),
                u0[1],
                u0[2],
            ]
        )
        ut0 = np.array(
            [
                -u0[0] * np.sqrt(c) * np.sin(np.sqrt(c) * (t - s))
                + u0[3] * np.cos(np.sqrt(c) * (t - s)),
                u0[4],
                u0[5],
            ]
        )

        yt1 = np.array([0, 0.5 * c * u0[2] * (t - s), -0.5 * c * u0[1] * (t - s)])
        ut1 = np.array([0, -0.5 * u0[5] * (t - s), 0.5 * c * u0[4] * (t - s)])

        return [yt0, ut0, yt1, ut1]

    def GG_non_uniform(self, x, v, t, s, eps, c):
        [yt0, ut0, yt1, ut1] = self.non_uniform_reduced(x, v, t, s, c)
        # breakpoint()
        R0 = np.block([yt0, self.Rtheta((t - s) / eps) @ ut0])
        RR0 = np.block(
            [
                yt1 + self.RRtheta((t - s) / eps) @ ut0,
                self.Rtheta((t - s) / eps) @ ut1
                + self.RRtheta((t - s) / eps) @ self.non_uniform_E(yt0, c),
            ]
        )

        return R0

    def test(self, tn):
        # tn=np.linspace(0,1,1000)
        t_len = len(tn)
        x = self.params.u0[0]
        v = self.params.u0[1]
        c = self.params.c
        s = self.params.s
        eps = self.params.eps
        X_ex = np.zeros([3, t_len])
        V_ex = np.zeros([3, t_len])
        X_red = np.zeros([3, t_len])
        V_red = np.zeros([3, t_len])
        for ii in range(t_len):
            U_ex = self.non_uniform_exact(x, v, tn[ii], tn[0], s, eps, c)

            U_red = self.GG_non_uniform(x, v, tn[ii], s, eps, c)
            X_ex[:, ii] = U_ex[:3]
            V_ex[:, ii] = U_ex[3:]
            X_red[:, ii] = U_red[:3]
            V_red[:, ii] = U_red[3:]

        plt.plot(tn, X_ex[2, :], label="Exact solution")
        print(V_red)
        plt.plot(tn, X_red[2, :], label="reduced problem solution")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    params = dict()
    params["t0"] = 0.0
    params["tend"] = 1
    params["dt"] = 0.0015625
    dt = 0.015625 * 1e-2
    params["s"] = 0.0
    params["eps"] = 0.001
    params["c"] = 2.0
    params["u0"] = np.array([[1, 1, 1], [1, 1, 1]])
    collocation_params = dict()
    collocation_params["quad_type"] = "GAUSS"
    collocation_params["num_nodes"] = [5, 3]
    pt = penning_trap(params, collocation_params)
    pt.compare_plot(5)
    # pt.SDC_iter(3)
    # pt.MLSDC_iter(5)
    # pt.MLSDC_iter_order11(1)
    # reduced_model=penningtrap_solution(params)
    # reduced_model.test()
