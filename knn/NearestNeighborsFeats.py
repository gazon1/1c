from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import copy
from tqdm import tqdm

import numpy as np


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    '''
        This class should implement KNN features extraction
    '''

    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric

        if n_neighbors is None or n_neighbors < max(k_list):
            self.n_neighbors = max(k_list)
        else:
            self.n_neighbors = n_neighbors

        self.eps = eps
        self.n_classes_ = n_classes

    def fit(self, X, y):
        '''
            Set's up the train set and self.NN object
        '''
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function
        self.NN = NearestNeighbors(n_neighbors=self.n_neighbors,
                                   metric=self.metric,
                                   n_jobs=1,
                                   algorithm='brute' if self.metric == 'cosine' else 'auto')
        self.NN.fit(X)

        # Store labels
        self.y_train = y

        # Save how many classes we have
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_

    def predict(self, X):
        '''
            Produces KNN features for every object of a dataset X
        '''
        if self.n_jobs == 1:
            test_feats = []
            for i in tqdm(range(X.shape[0]), total=X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i + 1]))
        else:
            '''
                 *Make it parallel*
                     Number of threads should be controlled by `self.n_jobs`  


                     You can use whatever you want to do it
                     For Python 3 the simplest option would be to use 
                     `multiprocessing.Pool` (but don't use `multiprocessing.dummy.Pool` here)
                     You may try use `joblib` but you will most likely encounter an error, 
                     that you will need to google up (and eventually it will work slowly)

                     For Python 2 I also suggest using `multiprocessing.Pool` 
                     You will need to use a hint from this blog 
                     http://qingkaikong.blogspot.ru/2016/12/python-parallel-method-in-class.html
                     I could not get `joblib` working at all for this code 
                     (but in general `joblib` is very convenient)

            '''
            # http://python-3.ru/page/multiprocessing
            pool = Pool(processes=self.n_jobs)
            ar = [X[i:i + 1] for i in range(X.shape[0])]
            test_feats = pool.map(self.get_features_for_one, ar)

            # Comment out this line once you implement the code
        #             assert False, 'You need to implement it for n_jobs > 1'

        return np.vstack(test_feats)

    def get_features_for_one(self, x):
        '''
            Computes KNN features for a single object `x`
        '''

        NN_output = self.NN.kneighbors(x)

        # Vector of size `n_neighbors`
        # Stores indices of the neighbors
        neighs = NN_output[1][0]

        # Vector of size `n_neighbors`
        # Stores distances to corresponding neighbors
        neighs_dist = NN_output[0][0]

        # Vector of size `n_neighbors`
        # Stores labels of corresponding neighbors
        neighs_y = self.y_train[neighs]

        ## ========================================== ##
        ##              YOUR CODE BELOW
        ## ========================================== ##

        # We will accumulate the computed features here
        # Eventually it will be a list of lists or np.arrays
        # and we will use np.hstack to concatenate those
        return_list = []

        # mean, max, min target for every k
        for k in self.k_list:
            feats = []
            q = neighs_y[:k]
            feats.append(q.mean())
            feats.append(q.min())
            feats.append(q.max())
            return_list += feats

        # merge
        knn_feats = np.hstack(return_list)

        return knn_feats