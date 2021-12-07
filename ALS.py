# refer to: https://blog.csdn.net/ddydavie/article/details/83020600
import os
import time
import numpy as np
import scipy.linalg as linalg
from utils.utils import (relative_F_norm_change, get_observed_idx, save_results)


def ALS(X=None, A=None, B=None, r=10, lamb=0, n_iters=100, threshold=1e-4, savedir='results/', fname='res.txt'):
    m, n = X.shape
    if A is None:
        A = np.random.randn(m, r)
    if B is None:
        B = np.random.randn(n, r)
    assert A.shape[0] == m and A.shape[1] == r and B.shape[0] == n and B.shape[1] == r
    I = np.eye(r)
    omega, _ = get_observed_idx(X)
    num_observed = len(omega)
    ABt_pre = A @ B.T
    res = []
    tic = time.time()
    for ite in range(1, n_iters + 1):
        # # <<<<<<<<<<<<<<<<<<< DEPRECATED
        # A = (linalg.inv(B.T @ B + lamb * I) @ B.T @ X.toarray().T).T
        # B = (linalg.inv(A.T @ A + lamb * I) @ A.T @ X.toarray()).T
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        for i in range(m):
            tmp = linalg.inv(B.T @ B + lamb * I) @ B.T @ X[i, :].toarray().T
            A[i, :] = tmp.squeeze()

        for j in range(n):
            tmp = linalg.inv(A.T@A + lamb*I)@A.T@X[:, j].toarray()
            B[j, :] = tmp.squeeze()

        ABt = A @ B.T

        duration = time.time() - tic
        nabla_F = relative_F_norm_change(ABt, ABt_pre)
        res.append((ite, duration, nabla_F))
        print('Iteration: {}, duration(/s) = {:.3f}, nabla_F = {:.3e}'.format(ite, duration, nabla_F))

        if nabla_F < threshold:
            print('break at iteration: {}'.format(ite))
            break
        ABt_pre = ABt

    X_full = A @ B.T

    res = np.array(res)
    filename = os.path.join(savedir, fname)
    save_results(filename, res)
    return A, B, X_full
