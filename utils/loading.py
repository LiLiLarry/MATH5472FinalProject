import os
import numpy as np
import scipy
import scipy.sparse


def load_ml100k(root=None):
    fname = os.path.join(root, 'u.data')
    data = np.genfromtxt(fname)
    return data[:, :3]


def load_ml10m(root=None, preserved_ratio=1):
    file = os.path.join(root, 'ratings.dat')
    with open(file, encoding='utf-8') as f:
        lines = f.readlines()

    user_movie_rating = []
    for line in lines:
        user_id, movie_id, rating, _ = line.split('::')
        item = (int(user_id), int(movie_id), float(rating))
        user_movie_rating.append(item)

    user_movie_rating = np.array(user_movie_rating)
    size = int(preserved_ratio*len(user_movie_rating))
    preserved_idx = np.random.choice(len(user_movie_rating), size=size)
    user_movie_rating = user_movie_rating[preserved_idx]
    return user_movie_rating


def id2idx(identities):
    ids = list(set(identities))
    ids.sort()
    id_idx_dict = {ele: i for i, ele in enumerate(ids)}
    idxs = np.array([id_idx_dict[key] for key in identities])
    return idxs, id_idx_dict


def convert2matrix(user_movie_rating):
    user_ids = user_movie_rating[:, 0]
    movie_ids = user_movie_rating[:, 1]
    ratings = user_movie_rating[:, 2]

    user_idxs, user_dict = id2idx(user_ids)
    movie_idxs, movie_dict = id2idx(movie_ids)
    shape = (len(user_dict), len(movie_dict))
    X = scipy.sparse.coo_matrix((ratings, (user_idxs, movie_idxs)), shape=shape)
    X = X.todok()
    return X
