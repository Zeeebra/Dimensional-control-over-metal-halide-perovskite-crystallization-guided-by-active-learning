# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:11:10 2021

@author: Zhi Li
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import entropy
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, recall_score, f1_score
import xgboost as xgb 
from modAL.batch import uncertainty_batch_sampling
from modAL.utils.combination import make_linear_combination, make_product
from modAL.uncertainty import classifier_uncertainty, classifier_margin, classifier_entropy, entropy_sampling
from modAL.density import information_density
from modAL.utils.selection import multi_argmax

from modAL.disagreement import vote_entropy_sampling, vote_entropy, consensus_entropy, KL_max_disagreement
import time
from tqdm import tqdm



def minibatch_AL (pool, X_label, y_label, model, numb_periter = 24):
    model.fit(np.array(X_label), np.array(y_label).ravel())
    uncernlst = classifier_uncertainty(model, pool).reshape((len(pool),1))
    pool['uncertainty'] = uncernlst
    
    sampling_numb = 0
    for i in uncernlst.ravel():
        if i>0:
            sampling_numb += 1
                
    # Use diverse mini-batch active learning
    beta = (sampling_numb)//numb_periter # beta factor selection
    minbatch = pool.nlargest(n = beta*numb_periter, columns = 'uncertainty') # pick the top k*beta points based on uncertainty
    
    # use k-means clustering to find k centorid points out of k*beta points
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = numb_periter, random_state=42)
    kmeans.fit(minbatch.iloc[:,:-1],sample_weight=minbatch.iloc[:,-1])
    centers = kmeans.cluster_centers_ # k centorid points (not necessary to be within k*beta points)
    
    # Find the nearest neighbor in the pool to the centorid points of k-means clustering
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=1, algorithm='ball_tree') # set neighbor number to be 1
    neigh.fit(np.array(minbatch.iloc[:,:-1])) # fit the model with top k*beta points
    query_idx = neigh.kneighbors(centers)[1] # find the index of nearest neighbor in the pool
    
    pool_query = pool.loc[minbatch.iloc[query_idx.ravel()].index]
    
    return pool_query