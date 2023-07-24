# This module includes ML models, active learning models, and 
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:23:15 2020

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

# Pearson VII universal function kernel for SVC
def PearsonVII_kernel(X1,X2, sigma=1.0, omega=1.0):
    if X1 is X2 :
        kernel = squareform(pdist(X1, 'euclidean'))
    else:
        kernel = cdist(X1, X2, 'euclidean')

    kernel = (1 + (kernel * 4 * np.sqrt(2**(1.0/omega)-1)) / sigma**2) ** omega
    kernel = 1/kernel
    return kernel

# define different query strategies
def custom_query_strategy(classifier, X, n_instances=1):
    query_idx = multi_argmax(classifier_uncertainty(classifier, X), n_instances=n_instances)
    return query_idx, X[query_idx]

# random search query working for activelearner and committee
def random_sampling(classifier, X, n_instances=1):
    n_samples = len(X)
    query_idx = np.random.choice(range(n_samples), size = n_instances)
    return query_idx, X[query_idx]

# querying strategy based on standard devidation for regressor
def regression_std(regressor, X, n_instances=1):
    _, std = regressor.predict(X, return_std=True)
    query_idx = multi_argmax (std, n_instances=n_instances)
    return query_idx, X[query_idx]

# A function to calculate prediction accuracy of a committee (a list of acitve learner), the learner in learner list must be pretrained Active-Learner
def committee_score (committee, X, y):
    y_pred = []
    pred_matrix = np.zeros(shape=(np.shape(X)[0],len(committee)))
    for i in range(len(committee)):
        pred_matrix[:,i]=committee[i].predict(X).reshape(-1,)
    for j in range(np.shape(X)[0]):
        y_pred.append(max(list(pred_matrix[j]), key = list(pred_matrix[j]).count))
    count = 0
    for k in range(np.shape(X)[0]):
        if y_pred[k]==y[k]:
            count += 1
        else:
            pass
    score = count/np.shape(X)[0]
    return score

# a function calculating active learner performance (accuracy) along the learnign process.
def actlearn_perf (learner, X, y, X_training, y_training, X_test, y_test, n_queries, n_instances):
    train_size = [X_training.shape[0]] 
    train_pool_size = train_size[0]
    X_training_pool = X_training.copy()
    y_training_pool = y_training.copy()
    X_test_pool = X_test.copy()
    y_test_pool = y_test.copy()
    
    accuracy = [learner.score(X,y)]
    precision = [precision_score(y, learner.predict(X), pos_label=1)]
    recall = [recall_score(y, learner.predict(X), pos_label=1)]
    f1 = [f1_score(y, learner.predict(X), pos_label=1)]
    
    for idx in tqdm(range(n_queries)):
        query_idx, query_instances = learner.query(X_test_pool, n_instances=n_instances)
        learner.teach(X_test_pool[query_idx], y_test_pool[query_idx])
        X_training_pool = np.vstack((X_training_pool, X_test_pool[query_idx]))
        y_training_pool = np.hstack((y_training_pool, y_test_pool[query_idx]))
        X_test_pool = np.delete(X_test_pool, query_idx, axis=0)
        y_test_pool = np.delete(y_test_pool, query_idx)
        
        accuracy.append(learner.score(X,y))
        precision.append(precision_score(y, learner.predict(X), pos_label=1))
        recall.append(recall_score(y, learner.predict(X), pos_label=1))
        f1.append(f1_score(y, learner.predict(X), pos_label=1))
        
        
        train_pool_size += n_instances
        train_size.append(train_pool_size)
        time.sleep(0)
    
    return train_size, accuracy, precision, recall, f1

# a function calculating committee performance (accuracy) along the learnign process.
def comm_perf (committee, X, y, X_training, y_training, X_test, y_test, n_queries, n_instances):
    # initializing training and testing pools
    train_size = X_training.shape[0] 
    train_size_list = [train_size]
    X_test_pool = X_test.copy()
    y_test_pool = y_test.copy()
    scores = [committee_score(committee, X, y)] # initial prediction accuracy
    
    for idx in tqdm(range(n_queries)):
        # Calculating prediction matrix of all dataset in X_test_pool for all learners in the committee
        # row: each data in X_test_pool. column: prediction results from different active-learners in the committee
        pred_matrix = np.zeros(shape=(np.shape(X_test_pool)[0],len(committee)))
        for i in range(len(committee)):
            pred_matrix[:,i]=committee[i].predict(X_test_pool).reshape(-1,)
        
        # define a list for prediction distributions (fullfilled by committee) for each data in X_test_pool.
        # Items of the list are prediction distribution for different data in X_test_pool
        
        pred_matrix_dist = []
        
        # Calculate prediction distribution
        # Please note: items in 'pred_matrix_dist' normally don't have same size, e.g., [0.3,0.7], [0.1,0.3,0.6]
        for j in range(np.shape(pred_matrix)[0]):
            dist_list = []
            for k in list(set(list(pred_matrix[j]))):
                freq = list(pred_matrix[j]).count(k)
                frac = freq/len(pred_matrix[j])
                dist_list.append(frac)
            pred_matrix_dist.append(dist_list)
        
        # Calculate vote entropy for each item (data in X_test_pool) in 'pred_matrix_dist'
        entropy_list = []
        for j in range(len(pred_matrix_dist)):
            S = entropy(pred_matrix_dist[j])
            entropy_list.append(S)
            
        # Calculate query index based on 'entropy_list'. Find the n_instances of largest vote entropy in the list
        query_idx = np.array(entropy_list).argsort()[-n_instances:][::-1]
        
        # Teach every learner in the committee with X_test_pool[query_idx]
        for learner in committee:
            learner.teach(X_test_pool[query_idx], y_test_pool[query_idx])
        
        # Calculate new scores for newly trained committee
        scores.append(committee_score(committee, X, y))
        
        # Remove trained n_instances of trained data from X_test_pool
        X_test_pool = np.delete(X_test_pool, query_idx, axis=0)
        y_test_pool = np.delete(y_test_pool, query_idx)
        
        # update training size and training number list
        train_size += n_instances
        train_size_list.append(train_size)
        time.sleep(0)

    train_size_list = np.array(train_size_list)
    scores = np.array(scores)
    
    return train_size_list, scores
    
    

# Define different sklearning machine learning models, which are considered to have good performance based on RAPID1 paper.
SVC_rbf = SVC(C=1,gamma=1,cache_size=6000,max_iter=-1,kernel='rbf', \
                        decision_function_shape='ovr', probability=True, \
                        class_weight='balanced', random_state=42)
SVC_Pearson = SVC(C=1,cache_size=6000,max_iter=-1,kernel=PearsonVII_kernel, \
                        decision_function_shape='ovr', probability=True, \
                        class_weight='balanced', random_state=42)
RF = RandomForestClassifier(criterion='entropy', n_estimators = 100, max_depth = 9,\
                            min_samples_leaf = 1)
kNN = KNeighborsClassifier(n_neighbors = 3, weights = "distance", p = 2)
GPC = GaussianProcessClassifier(1.0*RBF(1.0))
GPR = GaussianProcessRegressor(kernel=Matern)

# ensample above estimators as a dictionary                  
estimator = {'SVC_rbf': SVC_rbf,\
             'SVC_Pearson': SVC_Pearson,\
             'RF': RF,\
             'kNN': kNN,\
             'GPC': GPC,\
             'GPR': GPR}

# Define different sklearning machine learning models with flexible hyper-parameters.
def SVC_rbf_hyper (C=1.0, gamma=1.0):
    SVC_rbf_hyper = SVC(C=C,gamma=gamma,cache_size=6000,max_iter=-1,kernel='rbf', \
                  decision_function_shape='ovr', probability=True, \
                  class_weight='balanced', random_state=42)
    return SVC_rbf_hyper

def roc (y_test, y_score, name):
    fpr = []
    tpr = []
    thresholds = []
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc=auc(fpr,tpr)

    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.savefig(name,dpi = 1000)
    plt.legend(loc="lower right")
    plt.show()