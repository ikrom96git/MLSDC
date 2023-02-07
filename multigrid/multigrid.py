import numpy as np
import matplotlib.pyplot as plt
import pdb
# short helper class to add params as attributes
class _Pars(object):
    def __init__(self, pars):

        for k, v in pars.items():
            setattr(self, k, v)

        # self._freeze()



class multigrid_2d(object):

    def __init__(self, params):
        self.params=_Pars(params)
        self.x=np.arange(0, self.params.Tend+self.params.dx, self.params.dx)
        self.y=np.arange(0, self.params.Tend+self.params.dx, self.params.dy)
        self.nx=len(self.x)
        self.ny=len(self.y)
        self.nx_inner=self.nx-2
        self.ny_inner=self.ny-2

    def matrix_D(self):
        L=np.diagflat(-np.ones(self.nx_inner-1), -1)
        U=np.diagflat(-np.ones(self.nx_inner-1), 1)
        I=np.diagflat(-np.ones(self.nx_inner-1), -1)+np.diagflat(-np.ones(self.nx_inner-1), 1)
        D=I+4*np.diagflat(np.ones(self.nx_inner), 0)

        return D, I, L, U

    def matrix_A(self):
        D, I, *_ =self.matrix_D()
        A=np.kron(np.eye(self.nx_inner), D)+ np.kron(I, np.eye(self.nx_inner))

        return A

    def func(self):
        F=np.zeros(self.nx_inner*self.ny_inner)
        n=0
        for ii, x in enumerate(self.x[1:-1]):
            for jj, y in enumerate(self.y[1:-1]):
                F[n]=self.params.f(x, y)
                n=n+1

        return F

    def solver(self):
        A=self.matrix_A() #(1/self.params.dx**2)*
        F=(-self.params.dx**2)*self.func()

        V=np.linalg.solve(A, F)
        return V

    def plot_solution(self,V):

        # V=self.solver()

        z=V.reshape(self.nx_inner, self.nx_inner)
        Z=np.zeros([self.nx, self.nx])
        Z[1:-1,1:-1]=z
        X, Y=np.meshgrid(self.x, self.y)
        fig=plt.figure()
        ax=plt.axes(projection='3d')
        ax.plot_surface(X, Y, Z)
        plt.show()


class multigrid_1d(multigrid_2d):
    def __init__(self, params):
        super().__init__(params)
        # self.u=self.base_method()
        # pdb.set_trace()
        self.f=self.func()
        self.A=self.matrix_A()
        self.u=0.0*self.f

    def build_rhs(self):
        pass

    def DLU(self, A):
        D=np.diag(A.diagonal(0))
        L=np.zeros(np.shape(A))
        U=np.zeros(np.shape(A))
        for ii in range(np.shape(A)[0]):
            for jj in range(np.shape(A)[0]):
                if ii<jj:
                    L[ii,jj]=A[ii,jj]
                else:
                    D[ii,jj]=A[ii, jj]
        return D, L, U

    def base_method(self):
        x=np.ones(self.nx_inner)
        y=np.ones(self.ny_inner)
        if self.params.base_method=='copy':
            x=self.params.u0[0]*x
            y=self.params.u0[1]*y
        else:
            raise ValueError('this base method not included')
        u=np.block([x,y])
        return u



    def jacobi_iteration(self, A, v, f, k):
        D=np.diag(A.diagonal(0))
        LU=A-D
        P=np.dot(np.linalg.inv(D), LU)
        # pdb.set_trace()
        for ii in range(k):
            v=P@v+np.linalg.inv(D)@f
        v=v.reshape([self.nx_inner, self.nx_inner])
        return v

    def residual(self, A, v, f):
        return f-A@v

    def restriction(self, x):
        # x=x.reshape(self.nx_inner, self.nx_inner)
        dim=np.shape(x)[1]
        v_2h=np.array([]).reshape(0,int((dim+1)/2)-1)
        v=np.zeros([1,int((dim+1)*0.5)-1])

        if self.params.rest=='full_weighting':
            for jj in range(np.shape(x)[0]):
                for ii in range(0, int((self.nx_inner+1)/2)-1):
                    v[0,ii]=0.25*(x[jj, 2*ii]+2*x[jj, 2*ii+1]+x[jj, 2*ii+2])
                pdb.set_trace()
                v_2h=np.vstack([v_2h, v])
        else:
            raise ValueError('something is wrong')

        return v_2h




    def prolongation(self, x):
        v=np.zeros([self.nx_inner, self.nx_inner])
        for jj in range(self.nx_inner):
            for ii in range(0, int(self.nx_inner/2)-1):
                if (ii+1)%2:
                    v[jj,2*ii+1]=0.5*(x[jj,ii]+x[jj, ii+1])
                else:
                    v[jj, ii]=x[jj, ii]
        return v

    def multigrid(self):

        vv= self.jacobi_iteration(self.A, self.u, self.f, 10)
        x=self.restriction(vv)
        pdb.set_trace()
        # self.plot_solution(vv)




if __name__=='__main__':
    params=dict()
    params['dx']=0.125
    params['dy']=0.125
    params['Tend']=1.0
    params['u0']=[0,0]
    params['base_method']='copy'
    params['rest']='full_weighting'
    params['f']=lambda x, y: -2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)
    aa=multigrid_1d(params)
    aa.multigrid()

    # aa.plot_solution()
