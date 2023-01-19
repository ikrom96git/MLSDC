import numpy as np
import matplotlib.pyplot as plt

class multigrid(object):
    def __init__(self, description):
        self.description=description
        self.nx=description
    def Matrix_A(self):
        diagonals=[(2)*np.ones(self.xn), (-1)*np.ones(self.xn-1), (-1)*np.ones(self.xn-1)]
        A=(1/self.h**2)*sp.diags(diagonals, [0,-1, 1]).toarray()
        return A
