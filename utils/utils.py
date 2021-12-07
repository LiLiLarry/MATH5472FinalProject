import os
import copy
import numpy as np
import scipy
import scipy.sparse


def generate_matrix(shape=None, ratio=None):
    data = np.random.randint(1, 6, size=shape)
    mask = np.random.binomial(1, p=1 - ratio, size=shape)
    data = data * mask
    X = scipy.sparse.coo_matrix(data)
    X = X.todok()
    return X


def read_res(filename=None):
    X = np.loadtxt(filename)
    return X


def relative_F_norm_change(X, X0):
    out = np.linalg.norm(X - X0, ord='fro') ** 2 / (np.linalg.norm(X0, ord='fro') ** 2 + np.finfo(np.float64).eps)
    return out


def get_observed_idx(X):
    omega = X.keys()
    omega_T = X.transpose().keys()
    return omega, omega_T


def project2observed(X0, omega=None):
    X = copy.deepcopy(X0)
    out = np.zeros(X.shape, dtype=X.dtype)
    for r, c in omega:
        out[r, c] = X[r, c]
    return out


def save_results(filename, X):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    np.savetxt(filename, X)
    return
