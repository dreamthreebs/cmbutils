import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from ctypes import *
from pathlib import Path

class CovCalculator:
    @staticmethod
    def is_pos_def(M):
        eigvals = np.linalg.eigvals(M)
        plt.plot(eigvals)
        plt.show()
        print(np.min(eigvals))
        return np.all(eigvals>-1e-5)
    @staticmethod
    def check_symmetric(m, rtol=1e-05, atol=1e-05):
        return np.allclose(m, m.T, rtol=rtol, atol=atol)

    def __init__(self, nside, lmin, lmax, Cl_TT, Cl_EE, Cl_BB, Cl_TE, pixind, calc_opt='scalar', out_pol_opt=None):
        # the input Cl should range from 0 to >lmax
        self.nside = nside
        self.lmin = lmin
        self.lmax = lmax
        self.Cl_TT = Cl_TT[lmin:lmax+1].copy()

        if calc_opt == 'scalar':
            pass
        elif calc_opt == 'polarization':
            self.Cl_EE = Cl_EE[lmin:lmax+1].copy()
            self.Cl_BB = Cl_BB[lmin:lmax+1].copy()
            self.Cl_TE = Cl_TE[lmin:lmax+1].copy()
        else:
            raise ValueError('calc_opt should be scalar or polarization!')

        if np.size(self.Cl_TT) < (lmax+1-lmin):
            raise ValueError('input Cl size < l range')

        self.pixind = pixind
        self.calc_opt = calc_opt
        self.out_pol_opt = out_pol_opt

        self.l = np.arange(lmin, lmax+1).astype(np.float64)

    def Calc_CovMat(self):
        pixind = self.pixind
        nside = self.nside
        l = self.l
        nl = len(l) # number of ells
        print(f'number of l = {nl}')

        nCl = np.zeros((nl,))
        print(f'{nCl.shape=}')

        if self.calc_opt=='scalar':
            Cls = np.array([self.Cl_TT, nCl, nCl, nCl, nCl]) # TT,EE,BB,TE,TB
        elif self.calc_opt=='polarization':
            Cls = np.array([self.Cl_TT, self.Cl_EE, self.Cl_BB, self.Cl_TE, nCl])

        npix = len(pixind)
        vecst = hp.pix2vec(nside, pixind)
        vecs = np.array(vecst).T
        covmat = np.zeros((3*npix, 3*npix), dtype=np.float64)

        # use the c package to calculate the Covmat
        lib = cdll.LoadLibrary('../CovMat.so')
        CovMat = lib.CovMat
        CovMat(c_void_p(vecs.ctypes.data), c_void_p(l.ctypes.data), c_void_p(Cls.ctypes.data), c_void_p(covmat.ctypes.data), c_int(npix), c_int(nl))
        # covert back to 2d
        covmat = covmat.reshape((3*npix, 3*npix))
        return covmat

    def ChooseMat(self, M):
        if self.out_pol_opt is None:
            MP = M[0:len(M):3, 0:len(M):3] # just like TT
            print(f'{MP.shape=}')
        if self.out_pol_opt=='QQ':
            MP = M[1:len(M)+1:3, 1:len(M)+1:3]
        if self.out_pol_opt=='UU':
            MP = M[2:len(M)+2:3, 2:len(M)+2:3]
        if self.out_pol_opt=='QU':
            QQ = M[1:len(M)+1:3, 1:len(M)+1:3]
            print(f'{QQ.shape=}')
            QU = M[1:len(M)+1:3, 2:len(M)+2:3]
            UQ = M[2:len(M)+2:3, 1:len(M)+1:3]
            UU = M[2:len(M)+2:3, 2:len(M)+2:3]
            MP = np.block([[QQ,QU],[UQ,UU]])
            print(f'{MP.shape=}')
        return MP

    def run_calc_cov(self):
        M = self.Calc_CovMat()
        MP = self.ChooseMat(M)
        return MP
