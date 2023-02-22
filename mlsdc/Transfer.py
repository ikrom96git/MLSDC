import numpy as np
import scipy.sparse as sp




class _Pars(object):
    def __init__(self, pars):
        self.finter = False
        for k, v in pars.items():
            setattr(self, k, v)

        self._freeze()
