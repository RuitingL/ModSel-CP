#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats
import pandas as pd

from sklearn.linear_model import Ridge

from random import sample
import time

import mtds_func_residual as rfm



# model training
class feature_projection_pred_with_intercept:
    def __init__(self, coef, intercept, proj_ind):
        self.coef_ = coef
        self.intercept_ = intercept
        self.projind_ = proj_ind
    def predict(self,X):
        X_nonzero = X[:,self.projind_]
        pred = X_nonzero@self.coef_ + self.intercept_
        return pred


def train_model_feature_proj_ridge(M, X_tr, Y_tr, d_proj_portion, lamda):
    d = len(X_tr[0,:])
    d_proj = math.ceil(d*d_proj_portion)
    Mmodels = []
    
    for m in range(M):
        ind_sub = sample(range(d),d_proj)
        X_tmp = X_tr[:,ind_sub]
        modl = Ridge(alpha=lamda).fit(X_tmp, Y_tr)
        new_modl = feature_projection_pred_with_intercept(modl.coef_, modl.intercept_, ind_sub)
        Mmodels.append(new_modl)
    return Mmodels




# distributions
def easyX_sparse_normal(n_sample):
    d =300; s=18
    X = np.random.randn(n_sample,d) # easy X
    
    #coef
    b = np.array([max(0, (j-1)%20-s) for j in range(1, d+1)])
    
    Y = np.zeros(n_sample)
    Y += np.random.normal(size=n_sample)
    Y += np.matmul(X, b)

    return X, Y

def easyX_sparse_simpleT(n_sample):
    d =300; s=18; nu = 3
    X = np.random.randn(n_sample,d) # easy X
    
    #coef
    b = np.array([max(0, (j-1)%20-s) for j in range(1, d+1)])
    
    Y = np.zeros(n_sample)
    Y += np.random.standard_t(df=nu, size=n_sample)
    Y += np.matmul(X, b)

    return X, Y

def easyX_dense(n_sample):
    d=300
    X = np.random.randn(n_sample,d) # easy X
    Y = np.mean(X, axis=1) + (np.random.randn(n_sample) * (1 / d))
    return X, Y

def easytX_sparse_normal(n_sample):
    d =300; s=18; nu = 3
    #generate X (t-distribution)
    X = np.zeros((n_sample,d))
    Id = np.eye(d)
    mean = np.zeros(d)
    X += stats.multivariate_t.rvs(mean, Id, df=nu, size=n_sample)
    
    #coef
    b = np.array([max(0, (j-1)%20-s) for j in range(1, d+1)])
    
    Y = np.zeros(n_sample)
    Y += np.random.normal(size=n_sample)
    Y += np.matmul(X, b)

    return X, Y




# experiments
def Mmodels_length(Mmodels, X_cal, Y_cal, alpha):
    M = len(Mmodels); n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    S = np.zeros(M)
    for m in range(M):
        mdl = Mmodels[m]
        S[m] = np.sort(np.abs(Y_cal - mdl.predict(X_cal)))[k-1]
    
    length = 2*S
    return length

def experiment_residual(distribution_type, N_rep, n_tr, n_cal, M, alpha, d_proj_portion, split_portion, lamda):
    # Dictionary to the corresponding data distribution
    task_dict = {
        'sparse_normal': easyX_sparse_normal,
        'sparse_simpleT': easyX_sparse_simpleT,
        'dense': easyX_dense,
        'tsparse_normal': easytX_sparse_normal,
    }
    
    # Get the function based on the input difficulty and call it
    task_func = task_dict.get(distribution_type)
    if task_func:
        data_gene = task_func
    else:
        print("Invalid distribution_type level")
    
    # start the experiment
    X_tr, Y_tr = data_gene(n_tr)
    Mmodels = train_model_feature_proj_ridge(M, X_tr, Y_tr, d_proj_portion, lamda)
    
    Coverage = np.zeros((5,N_rep)); Length = np.zeros((5,N_rep))
    benchL = np.zeros((M,N_rep))
    
    til_alpha = n_cal*alpha/(n_cal+1) + 1/(n_cal+1) - n_cal*(1/(3*np.sqrt(n_cal)) 
                                                             + np.sqrt(np.log(2*M)/(2*n_cal)))/(n_cal+1)
    
    start_time = time.time()
    for t in range(N_rep):
        X_cal, Y_cal = data_gene(n_cal)
        x_test, y_test = data_gene(1)
        
        coverE, lengthE, _, _, _ = rfm.ModSel_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverL, lengthL, _, _ = rfm.ModSelLOO_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverEFCP, lengthEFCP, _ = rfm.YKbaseline_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverVFCP, lengthVFCP, _ = rfm.YKsplit_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha, split_portion)
        # if YK_adj will return infinity, just don't run it (replace with cov=1, length=inf later)
        if til_alpha <= 0:
            coverEFCP_adj = 0; lengthEFCP_adj = 0
        else:
            coverEFCP_adj, lengthEFCP_adj, _ = rfm.YK_adj_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        
        Coverage[0,t] = coverE; Coverage[1,t] = coverL
        Coverage[2,t] = coverEFCP; Coverage[3,t] = coverVFCP; Coverage[4,t] = coverEFCP_adj
        
        Length[0,t] = lengthE; Length[1,t] = lengthL
        Length[2,t] = lengthEFCP; Length[3,t] = lengthVFCP; Length[4,t] = lengthEFCP_adj
        
        benchL[:,t] = Mmodels_length(Mmodels, X_cal, Y_cal, alpha)
    
    # calculate mean and std
    cov = np.zeros((5,2)); leng = np.zeros((5,2))
    for j in range(5):
        cov[j,0] = np.mean(Coverage[j,:]); cov[j,1] = np.std(Coverage[j,:])
        leng[j,0] = np.mean(Length[j,:]); leng[j,1] = np.std(Length[j,:])
    
    min_single_md_len = np.min(np.mean(benchL, axis=1))
    
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time)/60} minutes")
    
    return cov, leng, min_single_md_len



N_rep = 5000; n_tr = 300; alpha = 0.1; d_proj_portion = 0.1; lamda = 0.1; split_portion = 0.5
DISTR = ['sparse_normal','sparse_simpleT','dense','tsparse_normal']

# fix n, vary M
N_cal = [100]
M_size = [2, 50, 100, 200, 400, 800, 1600]
for distribution_type in DISTR:
    for n_cal in N_cal:
        # Initialize an empty DataFrame
        results_df = pd.DataFrame(columns=["M", "ModSelc", "ModSelLOOc","YKbaselinec","YKsplitc","YK_adjc",
                                  "ModSell","ModSelLOOl","YKbaselinel","YKsplitl","YK_adjl","Min_Length"])
        
        for M in M_size:
            cov, leng, min_single_md_len = experiment_residual(distribution_type,N_rep, n_tr, 
                                                                            n_cal, M, alpha, d_proj_portion,
                                                                            split_portion, lamda)
            
            # Create a new DataFrame for the current result
            new_result_df = pd.DataFrame({"M": [M], "ModSelc": [cov[0,0]], "ModSelLOOc": [cov[1,0]],
                                          "YKbaselinec": [cov[2,0]], "YKsplitc": [cov[3,0]], "YK_adjc":[cov[4,0]],
                                  "ModSell":[leng[0,0]],"ModSelLOOl":[leng[1,0]],"YKbaselinel":[leng[2,0]], 
                                  "YKsplitl":[leng[3,0]],"YK_adjl":[leng[4,0]], "Min_Length": [min_single_md_len]
                                 })
            # Append the new result to the main DataFrame
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)
        
        # define the file name
        filename = f"Residual_{distribution_type}_results_of_n_{n_cal}.csv"
        # Save the DataFrame to a CSV file
        results_df.to_csv(filename, index=False)

        
# fix M, vary n
M_size = [200]
N_cal = [600, 500, 400, 300, 200, 100, 50]
for distribution_type in DISTR:
    for M in M_size:
        # Initialize an empty DataFrame
        results_df = pd.DataFrame(columns=["n", "ModSelc", "ModSelLOOc","YKbaselinec","YKsplitc","YK_adjc",
                                  "ModSell","ModSelLOOl","YKbaselinel","YKsplitl","YK_adjl","Min_Length"])
        
        for n_cal in N_cal:
            cov, leng, min_single_md_len = experiment_residual(distribution_type,N_rep, n_tr, 
                                                                            n_cal, M, alpha, d_proj_portion,
                                                                            split_portion, lamda)
            
            # Create a new DataFrame for the current result
            new_result_df = pd.DataFrame({"n": [n_cal], "ModSelc": [cov[0,0]], "ModSelLOOc": [cov[1,0]],
                                          "YKbaselinec": [cov[2,0]], "YKsplitc": [cov[3,0]], "YK_adjc":[cov[4,0]],
                                  "ModSell":[leng[0,0]],"ModSelLOOl":[leng[1,0]],"YKbaselinel":[leng[2,0]], 
                                  "YKsplitl":[leng[3,0]],"YK_adjl":[leng[4,0]], "Min_Length": [min_single_md_len]
                                 })
            # Append the new result to the main DataFrame
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)
        
        # define the file name
        filename = f"Residual_{distribution_type}_results_of_M_{M}.csv"
        # Save the DataFrame to a CSV file
        results_df.to_csv(filename, index=False)

