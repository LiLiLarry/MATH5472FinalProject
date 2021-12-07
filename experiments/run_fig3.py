# implementation for figure 3.

import sys
if './' not in sys.path:
    sys.path.append('./')
import os
from ALS import ALS
from softImpute_ALS import softImpute_ALS
from utils.loading import (load_ml10m, convert2matrix)

SAVE_DIR = 'results/res/'
os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    user_movie_rating = load_ml10m(root='data/ml-10m/ml-10M100K/', preserved_ratio=0.01)
    data = convert2matrix(user_movie_rating)
    _ = softImpute_ALS(X=data, r=100, lamb=50, n_iters=100, threshold=1e-5, savedir=SAVE_DIR, fname='fig3_ml10m_softimputeALS.txt')
    _ = ALS(X=data, r=100, lamb=50, n_iters=100, threshold=1e-5, savedir=SAVE_DIR, fname='fig3_ml10m_ALS.txt')
    return


if __name__ == '__main__':
    main()
    print('done!')
