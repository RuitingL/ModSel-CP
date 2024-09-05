#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from random import sample
import time

import mtds_func_classification as cfm




# data generating
def data_gene_class(n_sample, Beta):
    d = len(Beta[:,0]) # dim of the feature
    X = np.random.randn(n_sample,d)
    
    probabilities = [0.2, 0.8]  # Probabilities for 1 and -8
    values = [1, -8]            # The possible values
    X1 = np.random.choice(values, size=n_sample, p=probabilities)
    X[:,0] = X1
    
    Z = np.exp(X@Beta) # n_sample*K array
    
    Y = np.zeros(n_sample)
    for i in range(n_sample):
        weights = Z[i,:]/np.sum(Z[i,:])
        single_sample = np.random.multinomial(1, weights)
        Y[i] = np.where(single_sample > 0.5)[0]
    
    return X,Y


def Mmodels_card(Mmodels, X_cal, Y_cal, x_test, alpha):
    M = len(Mmodels); n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    card = np.zeros(M)
    for m in range(M):
        mdl = Mmodels[m]
        Pred_cond = mdl.predict_proba(X_cal)
        pred_prob = mdl.predict_proba(x_test)
        
        Scores = np.zeros(n)
        for i in range(n):
            Scores[i] = 1-Pred_cond[i,int(Y_cal[i])]
        
        q = np.sort(Scores)[k-1]
        card[m] = np.sum(pred_prob >= (1-q))
    
    return card




# experiment
def experiment_class_RF(N_rep, n_tr, n_cal, alpha, d, K, N_est, split_portion):
    Beta = np.random.randn(d,K)
    X_tr, Y_tr = data_gene_class(n_tr, Beta)
    while len(np.unique(Y_tr))<K:
        X_tr, Y_tr = data_gene_class(n_tr, Beta)
    #train the models
    Mmodels = []
    N_ESTIMATOR = np.linspace(10, 100, N_est, dtype=int)
    for i in N_ESTIMATOR:
        Mmodels.append(RandomForestClassifier(n_estimators=i, criterion='gini').fit(X_tr,Y_tr))
        Mmodels.append(RandomForestClassifier(n_estimators=i, criterion='entropy').fit(X_tr,Y_tr))

    
    Coverage = np.zeros((5,N_rep)); Length = np.zeros((5,N_rep))
    benchL = np.zeros((len(Mmodels),N_rep))
    
    M = 2*N_est
    til_alpha = n_cal*alpha/(n_cal+1) + 1/(n_cal+1) - n_cal*(1/(3*np.sqrt(n_cal)) 
                                                             + np.sqrt(np.log(2*M)/(2*n_cal)))/(n_cal+1)
    
    n_tmp = n_cal+1
    start_time = time.time()
    for t in range(N_rep):
        X, Y = data_gene_class(n_tr, Beta)
        while len(np.unique(Y))<K:
            X, Y = data_gene_class(n_tr, Beta)
            
        X_cal = X[:n_cal,:]; Y_cal = Y[:n_cal]
        x_test= X[n_cal:n_tmp,:]; y_test = Y[n_cal:n_tmp]
        
        coverE, lengthE, _, _ = cfm.ModSel_class_def(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverL, lengthL, _ = cfm.ModSelLOO_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverEFCP, lengthEFCP, _ = cfm.YKbaseline_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverVFCP, lengthVFCP, _ = cfm.YKsplit_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha, split_portion)
        if til_alpha <= 0: # return the entire Y
            coverEFCP_adj = 1
            lengthEFCP_adj = K
        else:
            coverEFCP_adj, lengthEFCP_adj, _ = cfm.YK_adj_class(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        
        Coverage[0,t] = coverE; Coverage[1,t] = coverL
        Coverage[2,t] = coverEFCP; Coverage[3,t] = coverVFCP; Coverage[4,t] = coverEFCP_adj
        
        Length[0,t] = lengthE; Length[1,t] = lengthL
        Length[2,t] = lengthEFCP; Length[3,t] = lengthVFCP; Length[4,t] = lengthEFCP_adj
        
        benchL[:,t] = Mmodels_card(Mmodels, X_cal, Y_cal, x_test, alpha)
        
    # calculate mean and std
    cov = np.zeros((5,2)); leng = np.zeros((5,2))
    for j in range(5):
        cov[j,0] = np.mean(Coverage[j,:]); cov[j,1] = np.std(Coverage[j,:])
        leng[j,0] = np.mean(Length[j,:]); leng[j,1] = np.std(Length[j,:])
    min_single_md_len = np.min(np.mean(benchL, axis=1))
    
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time)/60} minutes")
    
    return cov, leng, min_single_md_len




N_rep = 5000; n_tr = 300; n_cal = 150; alpha = 0.1; d = 50; split_portion = 0.5
K = 10
N_EST = [7,11,16,19]

# Initialize an empty DataFrame
results_df = pd.DataFrame(columns=["M", "ModSelc", "ModSelLOOc","YKbaselinec","YKsplitc","YK_adjc",
                                  "ModSell","ModSelLOOl","YKbaselinel","YKsplitl","YK_adjl","Min_Length"])
for N_est in N_EST:
    cov, leng, min_single_md_len = experiment_class_RF(N_rep, n_tr, n_cal, alpha, d, K, N_est,split_portion)
    
    # Create a new DataFrame for the current result
    new_result_df = pd.DataFrame({"M": [2*N_est], "ModSelc": [cov[0,0]], "ModSelLOOc": [cov[1,0]],
                                  "YKbaselinec": [cov[2,0]], "YKsplitc": [cov[3,0]], "YK_adjc":[cov[4,0]],
                                  "ModSell":[leng[0,0]],"ModSelLOOl":[leng[1,0]],"YKbaselinel":[leng[2,0]], 
                                  "YKsplitl":[leng[3,0]],"YK_adjl":[leng[4,0]], "Min_Length": [min_single_md_len]
                                 })
    # Append the new result to the main DataFrame
    results_df = pd.concat([results_df, new_result_df], ignore_index=True)

# define the file name
filename = f"Classification_results.csv"
# Save the DataFrame to a CSV file
results_df.to_csv(filename, index=False)

