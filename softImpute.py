import os
import time
import numpy as np
import scipy
import scipy.linalg as linalg
from utils.utils import (get_observed_idx, project2observed, relative_F_norm_change, save_results)


def soft_thresholding(X, lamb):
    U, Sigma, VT = linalg.svd(X, full_matrices=False)
    Sigma_lamb = Sigma - lamb
    Sigma_lamb[Sigma_lamb < 0] = 0
    out = U @ np.diagflat(Sigma_lamb) @ VT
    return out


def softImpute(X=None, lamb=300, n_iters=100, threshold=1e-4, savedir='results/', fname='res.txt'):
    m, n = X.shape
    omega, _ = get_observed_idx(X)

    M_hat_pre = np.zeros((m, n))
    res = []
    tic = time.time()
    for ite in range(1, n_iters + 1):
        X_hat = X + M_hat_pre - project2observed(M_hat_pre, omega)
        M_hat = soft_thresholding(X_hat, lamb)

        duration = time.time() - tic
        nabla_F = relative_F_norm_change(M_hat, M_hat_pre)
        res.append((ite, duration, nabla_F))
        print('Iteration: {}, duration(/s) = {:.3f}, nabla_F = {:.3e}'.format(ite, duration, nabla_F))

        if nabla_F < threshold:
            print('break at iteration: {}'.format(ite))
            break
        M_hat_pre = M_hat

    X_full = X_hat

    res = np.array(res)
    filename = os.path.join(savedir, fname)
    save_results(filename, res)
    return X_full
