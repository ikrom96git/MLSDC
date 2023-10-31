import numpy as np
import matplotlib.pyplot as plt
from MLSDC import penning_trap, _Pars


class FAS_correction(penning_trap):
    def __init__(self, params, collocation_params, type_evalf=None):
        super().__init__(params, collocation_params)
        self.type_evalf = type_evalf
    def residual(self):
        pass

    def build_f(self, f, v, t):
        rhs = f["elec"] + np.cross(v, f["magn"])
        return rhs

    def eval_f(self, pos, t):
        f = dict()
        if self.type_evalf=='Harmonic_oscillator':
            Emat = np.diag([-self.params.c, 0, 0])
        else:
            Emat = np.diag([-1, 1 / 2, 1 / 2])
        f["elec"] = self.params.omega_E**2 * np.dot(Emat, pos)
        f["magn"] = self.params.omega_B * np.array([1, 0, 0])

        return f


    def Boris_solver(self, c, dt, old_fields, new_fields, old_parts):
        Emean = 0.5 * (old_fields["elec"] + new_fields["elec"])
        a = 1
        c += dt / 2 * a * np.cross(old_parts, old_fields["magn"] - new_fields["magn"])
        vm = old_parts + dt / 2 * a * Emean + c / 2
        t = dt / 2 * a * new_fields["magn"]
        s = 2 * t / (1 + np.linalg.norm(t, 2) ** 2)
        vp = vm + np.cross(vm + np.cross(vm, t), s)
        vel = vp + dt / 2 * a * Emean + c / 2
        return vel

    def boris_trick(self, x_old, x_new, v_old, c_i, mydt):
        E_np12 = 0.5 * (self.E(x_old) + self.E(x_new)) * self.alpha
        t = 0.5 * mydt * self.B(x_new) * self.alpha
        s = 2.0 * t / (1.0 + np.dot(t, t))
        v_min = v_old + 0.5 * mydt * E_np12 + 0.5 * c_i
        v_star = v_min + np.cross(v_min, t)
        v_plu = v_min + np.cross(v_star, s)
        return v_plu + 0.5 * mydt * E_np12 + 0.5 * c_i

    def restrict(self, U):
        return self.transfer.Rcoll @ U

    def prolong(self, U):
        return self.transfer.Pcoll @ U

    def interpolation(self, U, level):
        Mf=level.coll.num_nodes
        U_fine = {
            "pos": U["pos"][0] * np.ones((Mf + 1, 3)),
            "vel": U["vel"][0] * np.ones((Mf + 1, 3)),
        }
        U_fine["pos"][1:] = self.prolong(U["pos"][1:])
        U_fine["vel"][1:] = self.prolong(U["vel"][1:])

        tau = {
            "pos": np.empty(shape=(Mf + 1, 3), dtype='object'),
            "vel": np.empty(shape=(Mf + 1, 3), dtype='object'),
        }
        return U_fine, tau

    def G_0(self, U):
        pass
    def tau_correction(self, U, level_coarse, level_fine):
        Mc = level_coarse.coll.num_nodes
        Mf = level_fine.coll.num_nodes
        Q_fine = self.params.dt * level_fine.coll.Qmat
        Q_coarse = self.params.dt * level_coarse.coll.Qmat
        QQ_fine = Q_fine @ Q_fine
        QQ_coarse = Q_coarse @ Q_coarse
        U_coarse = {
            "pos": U["pos"][0] * np.ones((Mc + 1, 3)),
            "vel": U["vel"][0] * np.ones((Mc + 1, 3)),
        }

        V0_coarse = U_coarse["vel"]
        V0_fine = U["vel"][0] * np.ones((Mf + 1, 3))
        U_coarse["pos"][1:] = self.restrict(U["pos"][1:])
        U_coarse["vel"][1:] = self.restrict(U["vel"][1:])
        F_coarse = np.zeros((Mc + 1, 3))
        for m in range(Mc + 1):
            fc = self.eval_f(U_coarse["pos"][m], 0)
            F_coarse[m] = self.build_f(fc, U_coarse["vel"][m], self.params.dt)

        F_fine = np.zeros((Mf + 1, 3))
        for m in range(Mf + 1):
            ff = self.eval_f(U["pos"][m], 0)
            F_fine[m] = self.build_f(ff, U["vel"][m], self.params.dt)
        RQF = {"pos": np.zeros((Mc + 1, 3)), "vel": np.zeros((Mc + 1, 3))}

        RQF["vel"][1:] = self.restrict((Q_fine @ F_fine)[1:])
        RQF["pos"][1:] = self.restrict((Q_fine @ V0_fine + QQ_fine @ F_fine)[1:])

        tau = dict()
        tau["pos"] = RQF["pos"] - (QQ_coarse @ F_coarse + Q_coarse @ V0_coarse)
        tau["vel"] = RQF["vel"] - Q_coarse @ F_coarse


        return U_coarse, tau

    def SDC_method(self, U, level, tau):

        M = level.coll.num_nodes
        integ = {"pos": np.zeros((M, 3)), "vel": np.zeros((M, 3))}
        integral = _Pars(integ)

        F = dict()
        for ii in range(M + 1):
            F[ii] = self.eval_f(U["pos"][ii], 0)

        for m in range(M):
            for j in range(M + 1):
                f = self.build_f(
                    F[j], U["vel"][j], self.params.dt * level.coll.nodes[j - 1]
                )
                integral.pos[m] += self.params.dt * (
                    self.params.dt * (level.SQ[m + 1, j] - level.Sx[m + 1, j]) * f
                )

                integral.vel[m] += (
                    self.params.dt * (level.S[m + 1, j] - level.ST[m + 1, j]) * f
                )

            # tau correction part

            if tau['pos'][m].any() is not None:
                integral.pos[m] += tau["pos"][m]
                integral.vel[m] += tau["vel"][m]

                if m > 0:

                    integral.pos[m] -= tau["pos"][m - 1]
                    integral.pos[m] -= tau["vel"][m - 1]

        # do the sweep
        for m in range(0, M):
            tmppos = np.copy(integral.pos[m])
            tmpvel = np.copy(integral.vel[m])
            for j in range(m + 1):
                f = self.build_f(
                    F[j], U["vel"][j], self.params.dt * level.coll.nodes[j - 1]
                )

                tmppos += self.params.dt * (self.params.dt * level.Sx[m + 1, j] * f)

            tmppos += U["pos"][m] + self.params.dt * level.coll.delta_m[m] * U["vel"][0]

            U["pos"][m + 1] = tmppos

            F[m + 1] = self.eval_f(U["vel"][m + 1], self.params.dt * level.coll.nodes)

            ck = tmpvel

            U["vel"][m + 1] = self.Boris_solver(
                ck,
                self.params.dt * np.diag(level.QI)[m + 1],
                F[m],
                F[m + 1],
                U["vel"][m],
            )

        return U

    def SDC_iteration(self, Kiter):
        u0 = self.params.u0

        U = {
            "pos": u0[0] * np.ones((self.fine.coll.num_nodes + 1, 3)),
            "vel": u0[1] * np.ones((self.fine.coll.num_nodes + 1, 3)),
        }

        for ii in range(Kiter):
            U_coarse, tau = self.tau_correction(
                U, level_coarse=self.coarse, level_fine=self.fine
            )



            U_c = self.SDC_method(U_coarse, level=self.coarse, tau=tau)

            U_fine, tau_fine=self.interpolation(U_c, level=self.fine)

            U_f=self.SDC_method(U_fine, level=self.fine, tau=tau_fine)
            U=U_f
        return U_f

    def SDC_reduced_model(self, Kiter):
        
        u0 = self.params.u0

        U = {
            "pos": u0[0] * np.ones((self.fine.coll.num_nodes + 1, 3)),
            "vel": u0[1] * np.ones((self.fine.coll.num_nodes + 1, 3)),
        }

        for ii in range(Kiter):
            U_coarse, tau = self.tau_correction(
                U, level_coarse=self.coarse, level_fine=self.fine
            )



            U_c = self.SDC_method(U_coarse, level=self.coarse, tau=tau)

            U_fine, tau_fine=self.interpolation(U_c, level=self.fine)

            U_f=self.SDC_method(U_fine, level=self.fine, tau=tau_fine)
            print(U_f)


if __name__ == "__main__":
    params = dict()
    params["t0"] = 0.0
    params["tend"] = 1.0
    params["dt"] = 0.0015625
    params["omega_E"] = 1
    params["omega_B"] = 0.0 #1 / 0.1
    params["s"] = 0.0
    params["eps"] = 0.1
    params["c"] = 2.0
    params["u0"] = np.array([[1, 1, 1], [1, 1, 1]])
    collocation_params = dict()
    collocation_params["quad_type"] = "GAUSS"
    collocation_params["num_nodes"] = [5, 3]
    FAS = FAS_correction(params, collocation_params, type_evalf='Harmonic_oscillator')
    FAS.SDC_iteration(5)
