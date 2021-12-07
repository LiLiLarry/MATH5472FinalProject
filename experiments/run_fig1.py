# implementation for figure 1.

import sys

if './' not in sys.path:
    sys.path.append('./')
import os
from ALS import ALS
from softImpute import softImpute
from softImpute_ALS import softImpute_ALS
from utils.utils import generate_matrix
from utils.loading import (load_ml100k, convert2matrix)

SAVE_DIR = 'results/res/'
os.makedirs(SAVE_DIR, exist_ok=True)


def run_subfig1():
    data = generate_matrix(shape=[300, 200], ratio=0.7)

    lamb = 120
    r = 25
    n_iters = 150
    threshold = 1e-5
    _ = softImpute_ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                       fname='fig1_simulation1_softimputeALS.txt')
    _ = softImpute(X=data, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                   fname='fig1_simulation1_softimpute.txt')
    _ = ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
            fname='fig1_simulation1_ALS.txt')
    return


def run_subfig2():
    data = generate_matrix(shape=[800, 600], ratio=0.9)

    lamb = 140
    r = 50
    n_iters = 150
    threshold = 1e-5
    _ = softImpute_ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                       fname='fig1_simulation2_softimputeALS.txt')
    _ = softImpute(X=data, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                   fname='fig1_simulation2_softimpute.txt')
    _ = ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
            fname='fig1_simulation2_ALS.txt')
    return


def run_subfig3():
    data = generate_matrix(shape=[1200, 900], ratio=0.8)

    lamb = 300
    r = 50
    n_iters = 150
    threshold = 1e-5
    _ = softImpute_ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                       fname='fig1_simulation3_softimputeALS.txt')
    _ = softImpute(X=data, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                   fname='fig1_simulation3_softimpute.txt')
    _ = ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
            fname='fig1_simulation3_ALS.txt')
    return


def run_subfig4():
    user_movie_rating = load_ml100k(root='data/ml-100k/')
    data = convert2matrix(user_movie_rating)

    lamb = 20
    r = 40
    n_iters = 150
    threshold = 1e-5
    _ = softImpute_ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                       fname='fig1_ml100k_softimputeALS.txt')
    _ = softImpute(X=data, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR,
                   fname='fig1_ml100k_softimpute.txt')
    _ = ALS(X=data, r=r, lamb=lamb, n_iters=n_iters, threshold=threshold, savedir=SAVE_DIR, fname='fig1_ml100k_ALS.txt')
    return


def main():
    run_subfig1()
    run_subfig2()
    run_subfig3()
    run_subfig4()
    return


if __name__ == '__main__':
    main()
    print('done!')
