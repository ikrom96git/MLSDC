import numpy as np
import matplotlib.pyplot as plt
from MLSDC import penning_trap

class FAS_correction(penning_trap):

    def __init__(self, params, collocation_params):
        super().__init__(params, collocation_params)

    def residual(self):
        pass

    def build_f(self,f, v, t):
        rhs=f['elec']+np.cross(v, f['magn'])
        return rhs


    def eval_f(self, pos, t):

        f=dict()
        Emat=np.diag([-1, 1/2, 1/2])
        f['elec']=self.params.omega_E**2 * np.dot(Emat, pos)
        f['magn']=self.params.omega_B * np.array([1, 0 , 0])

        return f

    def Boris_solver(self, c, dt, old_fields, new_fields, old_parts):
        Emean=0.5*(old_fields['elec']+new_fields['elec'])
        a=1
        c += dt/2 * a * np.cross(old_parts['vel'], old_fields['magn']-new_fields['magn'])
        vm=old_parts['vel']+dt/2*a* Emean+c/2
        t=dt/2*a*new_fields['magn']
        s=2*t/(1+np.linalg.norm(t, 2)**2)
        vp=vm+np.cross(vm+np.cross(vm, t), s)
        vel=vp+dt/2*a*Emean+c/2
        return vel

    def boris_trick(self, x_old, x_new, v_old, c_i, mydt):
        E_np12=0.5*( self.E(x_old)+self.E(x_new))* self.alpha
        t=0.5*mydt*self.B(x_new)* self.alpha
        s=2.0*t/(1.0+np.dot(t,t))
        v_min=v_old+0.5*mydt*E_np12+0.5*c_i
        v_star =v_min+np.cross(v_min, t)
        v_plu=v_min +np.cross(v_star, s)
        return v_plu+0.5*mydt *E_np12 +0.5*c_i

    def SDC_iteration(self, u0, u, level):
        level=self.fine
        M=level.coll.num_nodes

        for m in range(M):
            for j in range(M+1):
                f=self.build_f(F[j], u[j], self.params.dt*level.coll.nodes[j-1])

                intpos += self.params.dt*(self.params.dt*level.SQ[m+1,j]-level.Sx[m+1, j]*f)

                intvel += self.params.dt* (level.S[m+1, j]-level.ST[m+1, j]) * f

            if tau[m] is not None:
                intpos += tay[m]

                if m>0:
                    intpos -= L.tau[m-1]

        # do the sweep
        for m in range(0, M):
            tmp=intpos

            for j in range(m+1):

                f=self.build_f(F[j], u[j], self.params.dt*self.coll.nodes[j-1])

                tmppos += self.params.dt*(self.params.dt*level.Sx[m+1, j]*f)

            tmppos += upos[m]+ self.params.dt*level.coll.delta_m[m]*uvel[0]

            upos[m+1]=tmppos

            f[m+1]=self.eval_f(u[m+1], self.params.dt*level.coll.nodes)

            ck=tmpvel

            uvel[m+1]=self.boris_solver()
