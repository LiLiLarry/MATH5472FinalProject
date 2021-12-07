import os
import time
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg as linalg
from utils.utils import (relative_F_norm_change, get_observed_idx, project2observed, save_results)

__all__ = ['softImpute_ALS']


def init_UVD(m=None, n=None, r=None):
    U = np.random.randn(m, r)
    Q, R = linalg.qr(U)
    U = Q[:, :r]
    D = np.identity(r)
    V = np.zeros((n, r))
    return U, V, D


def update_B(X, omega=None, U=None, V=None, D=None, lamb=None):
    A = U @ D
    B = V @ D
    X_asterisk = project2observed(X, omega=omega) - project2observed(A @ B.T, omega=omega) + A @ B.T
    I_r = np.identity(D.shape[0])

    B_tilde_T = linalg.solve(D @ D + lamb * I_r, D @ U.T @ X_asterisk)
    B_tilde = B_tilde_T.T
    U_tilde, sigma_tilde, V_tilde_T = linalg.svd(B_tilde @ D, full_matrices=False)
    D_tilde = np.diagflat(np.sqrt(sigma_tilde))
    return U_tilde, D_tilde


def softImpute_ALS(X=None, r=None, lamb=300, n_iters=100, threshold=1e-4, savedir='results/', fname='res.txt'):
    m, n = X.shape
    U, V, D = init_UVD(m, n, r)
    omega, omega_T = get_observed_idx(X)

    ABt_pre = U @ D @ D @ V.T
    res = []
    tic = time.time()
    for ite in range(1, n_iters + 1):
        U_tilde, D_tilde = update_B(X, omega=omega, U=U, V=V, D=D, lamb=lamb)
        V, D = U_tilde, D_tilde

        U_tilde, D_tilde = update_B(X.T, omega=omega_T, U=V, V=U, D=D, lamb=lamb)
        U, D = U_tilde, D_tilde
        ABt = U @ D @ D @ V.T

        duration = time.time() - tic
        nabla_F = relative_F_norm_change(ABt, ABt_pre)
        res.append((ite, duration, nabla_F))
        print('Iteration: {}, duration(/s) = {:.3f}, nabla_F = {:.3e}'.format(ite, duration, nabla_F))

        if nabla_F < threshold:
            print('break at iteration: {}'.format(ite))
            break
        ABt_pre = ABt

    A = U @ D
    B = V @ D
    X_full = project2observed(X.toarray(), omega=omega) - project2observed(A @ B.T, omega=omega) + A @ B.T

    res = np.array(res)
    filename = os.path.join(savedir, fname)
    save_results(filename, res)
    return U, V, D, X_full
