# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:20:14 2020

@author: Zhi Li
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Function to save any obj in 'obj' folder
def save_obj(obj, name ):
    with open('../Active-learning phase-Morph Phase Mapping/obj/'+ name + '.pkl', 'wb') as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('../Active-learning phase-Morph Phase Mapping/obj/' + name + '.pkl', 'rb') as file:
        return pickle.load(file)

def data_preprocess(Xy):
    X = Xy.filter(['_rxn_M_inorganic','_rxn_M_organic','_rxn_M_acid'], axis =1)
    y = Xy.filter(['crystal score'], axis =1)
    # Standarize the input
    x = StandardScaler().fit_transform(X) # dataframe turned into array and it is reformed as dataframe in the below line of code
    X = pd.DataFrame(x, index = X.index, columns = X.columns)
    return X, y

# Generate meshgrid points of certain size and location
def gridgen6D (n=10, x=[-1,1]):
    from scipy.spatial import distance
    a = np.linspace(x[0],x[1],n)
    b = np.linspace(x[0],x[1],n)
    c = np.linspace(x[0],x[1],n)
    d = np.linspace(x[0],x[1],n)
    e = np.linspace(x[0],x[1],n)
    f = np.linspace(x[0],x[1],n)
    
    points = np.zeros((n**6,6))
    x1v,x2v,x3v,x4v,x5v,x6v = np.meshgrid(a,b,c,d,e,f)
    x1v = x1v.flatten()
    x2v = x2v.flatten()
    x3v = x3v.flatten()
    x4v = x4v.flatten()
    x5v = x5v.flatten()
    x6v = x6v.flatten()

    for i in tqdm(range(n**6)):
        points[i] = [x1v[i], x2v[i], x3v[i], x4v[i], x5v[i], x6v[i]]
    points = pd.DataFrame(columns = ['a','b','c','d','e','f'], data = points)
    
    center = np.array([(x[0]+x[1])/2]*6)
    center = pd.DataFrame(columns = ['a','b','c','d','e','f'], data = center.reshape(1,6))
    
    distance = distance.cdist(XA = points, XB = center, metric = 'euclidean')
    
    return points, center, distance

# Generate meshgrid points of certain size and location
def gridgen2D (n=10, x=[-1,1]):
    from scipy.spatial import distance
    a = np.linspace(x[0],x[1],n)
    b = np.linspace(x[0],x[1],n)
    
    points = np.zeros((n**2,2))
    x1v,x2v = np.meshgrid(a,b)
    x1v = x1v.flatten()
    x2v = x2v.flatten()

    for i in tqdm(range(n**2)):
        points[i] = [x1v[i], x2v[i]]
    points = pd.DataFrame(columns = ['a','b'], data = points)
        
    center = np.array([(x[0]+x[1])/2]*2)
    center = pd.DataFrame(columns = ['a','b'], data = center. reshape(1,2))
    
    distance = distance.cdist(XA = points, XB = center, metric = 'euclidean')
    
    return points, center, distance




