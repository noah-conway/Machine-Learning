# Noah Conway
# CS 422 project 1
# problem 3: clustering 

import numpy as np

def K_Means(X, K, mu):

    if len(mu) == 0:
        rand_ids = np.random.choice(np.shape(X)[0], K, replace=False)
        mu = [X[i] for i in rand_ids]

    clusters = [[] for _ in range(K)]

    for data_idx in X:
        distances = [np.linalg.norm(np.subtract(data_idx, c)) for c in mu] # populate a "distances" array with the distances from point i to each of the centers
        center_idx = np.argmin(distances) 
        clusters[center_idx].append(data_idx)
    
    mu_idx = 0
    for clusters_idx in clusters:
        new_center = np.mean(clusters_idx, axis = 0)
        mu[mu_idx] = new_center
        mu_idx=mu_idx+1


    return mu

