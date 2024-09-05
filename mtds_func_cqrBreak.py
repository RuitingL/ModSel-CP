#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats



# YK-baseline
def YKbaseline_CQRcal(Models, Betas, X_cal, Y_cal, alpha):
    n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    M_models = len(Models); M_betas = len(Betas)
    Q = np.zeros((M_models, M_betas))
    sel_len_factor = np.zeros((M_models, M_betas))
    
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            Q_lo = mdl.predict(X_cal, quantiles = beta)
            Q_hi = mdl.predict(X_cal, quantiles = (1-beta))
            
            Scores = np.maximum((Q_lo- Y_cal), (Y_cal -Q_hi))
            Q[m,b] = np.sort(Scores)[k-1] # S^{(m)}_{(k)}
            sel_len_factor[m,b] = Q[m,b] + ((np.mean((Q_hi-Q_lo)))*n/(2*(n+1)))
    
    return Q, sel_len_factor

# single test points
def YKbaseline_CQRtest(Models, Betas, Q, sel_len_factor, n, x_test, y_test):
    M_models = len(Models); M_betas = len(Betas)
    L = np.zeros((M_models, M_betas))
    
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            q_lo = mdl.predict(x_test, quantiles=beta)
            q_hi = mdl.predict(x_test, quantiles=(1-beta))
            L[m,b] = sel_len_factor[m,b] + ((q_hi-q_lo)/(2*(n+1)))
    
    # select the model
    min_position = np.unravel_index(np.argmin(L), L.shape)
    mdl = Models[min_position[0]]
    beta = Betas[min_position[1]]
    
    q_lo = mdl.predict(x_test, quantiles=beta)
    q_hi = mdl.predict(x_test, quantiles=(1-beta))
    
    left = q_lo - Q[min_position]; right = q_hi + Q[min_position]
    
    # check coverage
    if (y_test - left)*(y_test - right)<=0:
        cover = 1
    else:
        cover = 0
    
    # calculate the length
    length = right - left
    
    return cover, length





# YK_adj (only need to adj alpha)
def YK_adj_CQRcal(Models, Betas, X_cal, Y_cal, alpha):
    n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    M_models = len(Models); M_betas = len(Betas)
    M = M_models*M_betas
    til_alpha = n*alpha/(n+1) + 1/(n+1) - n*(1/(3*np.sqrt(n)) + np.sqrt(np.log(2*M)/(2*n)))/(n+1)
    if math.ceil((n+1)*(1-til_alpha)) > n: # the quantile is taken to be +inf
        Q = np.inf*np.ones((M_models, M_betas))
        sel_len_factor = np.inf*np.ones((M_models, M_betas))
    else:
        Q, sel_len_factor = YKbaseline_CQRcal(Models, Betas, X_cal, Y_cal, til_alpha)
    return Q, sel_len_factor




# YK-split
def YKsplit_CQR(Models, Betas, X_cal, Y_cal, X_test, Y_test, alpha, split_portion):
    n1 = math.ceil(len(Y_cal)*split_portion); n2 = len(Y_cal) - n1
    k1 = math.ceil((n1 + 1)*(1-alpha)); k2 = math.ceil((n2 + 1)*(1-alpha))
    
    X_sel = X_cal[:n1, :]; Y_sel = Y_cal[:n1]
    X_c = X_cal[n1:, :]; Y_c = Y_cal[n1:]
    
    M_models = len(Models); M_betas = len(Betas)
    L = np.zeros((M_models, M_betas))
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            Q_lo = mdl.predict(X_sel, quantiles = beta)
            Q_hi = mdl.predict(X_sel, quantiles = (1-beta))
            Scores = np.maximum((Q_lo- Y_sel), (Y_sel -Q_hi))
            L[m,b] = np.sort(Scores)[k1-1] # S^{(m)}_{(k)}
            L[m,b] += (np.mean((Q_hi - Q_lo)))/2
    
    # select the model
    min_position = np.unravel_index(np.argmin(L), L.shape)
    mdl = Models[min_position[0]]
    beta = Betas[min_position[1]]
    
    Q_lo = mdl.predict(X_c, quantiles = beta)
    Q_hi = mdl.predict(X_c, quantiles = (1-beta))
    S = np.maximum((Q_lo-Y_c), (Y_c-Q_hi))
    Q = np.sort(S)[k2-1]
    
    # multiple test points
    q_lo = mdl.predict(X_test, quantiles = beta)
    q_hi = mdl.predict(X_test, quantiles = (1-beta))
    lefts = q_lo - Q
    rights = q_hi + Q
    
    # check coverage 
    test_cov = np.mean((Y_test - lefts) * (Y_test - rights) <= 0)
    
    # calculate the length
    ave_length = 2*Q + np.mean((q_hi-q_lo))
    
    return test_cov, ave_length




# extra compare: single model CP (directly given test_ave length for each model)
def Mmodels_length_CQR(Models, Betas, X_cal, Y_cal, X_test, alpha):
    M_models = len(Models); M_betas = len(Betas)
    n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    L = np.zeros((M_models, M_betas))
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            Qcal_lo = mdl.predict(X_cal, quantiles = beta)
            Qcal_hi = mdl.predict(X_cal, quantiles = (1-beta))
            S = np.maximum(Qcal_lo-Y_cal, Y_cal-Qcal_hi)
            Q = np.sort(S)[k-1]
            
            Qtest_lo = mdl.predict(X_test, quantiles = beta)
            Qtest_hi = mdl.predict(X_test, quantiles = (1-beta))
            L[m,b] = 2*Q + np.mean((Qtest_hi-Qtest_lo))
            
    L = L.flatten()
    return L





# ModSel-CP
def ModSel_CQRcal(Models, Betas, X_cal, Y_cal, alpha):
    n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    M_models = len(Models); M_betas = len(Betas)
    Q = np.zeros((2, M_models, M_betas))
    sel_len_factor = np.zeros((M_models, M_betas))
    
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            Q_lo = mdl.predict(X_cal, quantiles = beta)
            Q_hi = mdl.predict(X_cal, quantiles = (1-beta))
            
            sort_Scores = np.sort(np.maximum((Q_lo- Y_cal), (Y_cal -Q_hi)))
            Q[1,m,b] = sort_Scores[k-1] # S^{(m)}_{(k)}
            Q[0,m,b] = sort_Scores[k-2] # S^{(m)}_{(k-1)}
            sel_len_factor[m,b] = (np.mean((Q_hi-Q_lo)))*n/(2*(n+1))
    
    return Q, sel_len_factor

# single test point
def ModSel_CQRtest(Models, Betas, Q, sel_len_factor, n, x_test, y_test):
    M_models = len(Models); M_betas = len(Betas)
    L_factor = np.zeros((M_models, M_betas))
    
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            q_lo = mdl.predict(x_test, quantiles=beta)
            q_hi = mdl.predict(x_test, quantiles=(1-beta))
            L_factor[m,b] = sel_len_factor[m,b] + ((q_hi-q_lo)/(2*(n+1)))
    
    # filter models
    min_sel_L = np.min((Q[1,:,:]+L_factor))
    find_ind = np.where((Q[0,:,:]+L_factor)<=min_sel_L)
    calM = len(find_ind[0])
    
    L = np.zeros(calM)
    center = np.zeros(calM)
    
    for t in range(calM):
        m = find_ind[0][t]; b = find_ind[1][t]
        
        mdl = Models[m]; beta = Betas[b]
        q_lo = mdl.predict(x_test, quantiles=beta)
        q_hi = mdl.predict(x_test, quantiles=(1-beta))
        center[t] = (q_lo+q_hi)/2
        L[t] = L_factor[m,b] + ((q_lo-q_hi)/2)
    
    # check coverage
    if np.min(np.abs(y_test - center) + L)<=min_sel_L:
        cover = 1
    else:
        cover = 0
    
    # calculate the length (reduce to residual type)
    quantl = min_sel_L - L
    Lefts = center - quantl; Rights = center + quantl
        
    sort_ind = np.argsort(Lefts)
    Lefts = Lefts[sort_ind]; Rights = Rights[sort_ind]
        
    loc = 0
    length = 0; connect = 1
    while loc<len(Lefts):
        find_ind = np.where(Lefts - Rights[loc]>0)[0]
        if len(find_ind) == 0:
            length += np.max(Rights) - Lefts[loc]
            loc = len(Lefts)
        else:
            if find_ind[0] - 1 == loc:
                connect = 0
                length += Rights[loc] - Lefts[loc]
                loc = find_ind[0]
            else:
                length += Lefts[find_ind[0]-1] - Lefts[loc]
                loc = find_ind[0] - 1
    
    return cover, length, connect, calM


# ModSel-CP-LOO

def CALMi(i, S_all, S_q, pred_center, pred_level):
    M = len(S_all[0,:])
    Lo = np.zeros(M); Up = np.zeros(M)
    pointer = np.zeros(M)
    
    for m in range(M):
        if S_all[i,m] < S_q[1,m]:
            Lo[m] = S_q[1,m]
            Up[m] = S_q[2,m]
        else:
            Lo[m] = S_q[0,m]
            if S_all[i,m] > S_q[1,m]:
                Up[m] = S_q[1,m]
            else:
                Up[m] = S_q[2,m]
        
        if pred_level[m]>= Up[m]:
            Lo[m] = Up[m]
            pointer[m] = -1 # constant function
        else:
            if pred_level[m]>= Lo[m]:
                Lo[m] = pred_level[m]
                pointer[m] = 0 # V shape
            else:
                pointer[m] = 1 # U shape
    
    
    # filter models
    find_ind = np.where(Lo <= np.min(Up))[0]
    Li = Lo[find_ind]; Ui = Up[find_ind]
    Mu_i = pred_center[find_ind]; Level_i = pred_level[find_ind]
    pointer = pointer[find_ind]
    Sis = S_all[i,find_ind]
    M = len(find_ind)
    
    ui = np.min(Ui)
    m_ui = np.where(Ui == ui)[0]
    Si_ui = max(Sis[m_ui]) # transformed
    
    Upleft = Mu_i - (ui - Level_i)
    Upright = Mu_i + (ui - Level_i)
    # further filter models
    columns_to_delete = set()
    for j in range(M):
        if Upleft[j]>=Upright[j]: # it's constant type
            if Sis[j] < Si_ui:
                columns_to_delete.add(j)
        else:
            for k in range(M):
                if j != k:
                    if Upleft[j]>=Upleft[k] and Upright[j]<=Upright[k] and Li[j]>=Li[k]: # auto exclude const case
                        if Upleft[j]>Upleft[k] and Upright[j]<Upright[k]:
                            columns_to_delete.add(j)
                        else:
                            if Sis[j] <= Sis[k]:
                                columns_to_delete.add(j)

    mask = [j not in columns_to_delete for j in range(M)]
    # Apply the mask to each vector and return the filtered models
    Li = Li[mask]; Mu_i = Mu_i[mask]; Level_i = Level_i[mask]; Sis = Sis[mask]; pointer = pointer[mask]
            
    
    return Li, ui, Mu_i, Level_i, pointer, Sis, Si_ui


# calculate intersections
def crossV(centerc, centern, lvlc, lvln, Lc, Ln):
    pt = (centerc+centern + lvln-lvlc)/2
    pt_lvl = abs(pt-centerc)+lvlc
    
    if pt_lvl >= max(Lc, Ln):
        intersec = [pt]
    else:
        if pt_lvl < min(Lc,Ln):
            if Ln >= Lc:
                intersec = [centerc + (Ln - lvlc)]
            else:
                intersec = [centern - (Lc - lvln)]
        else:
            if Ln > pt_lvl:
                intersec = [centerc + (Ln - lvlc)]
            else:
                intersec = [centern - (Lc - lvln)]
    return intersec
            
def slope_overlap(ct_large, ct_small, lvl_large, lvl_small, Ll, Ls, overlap_end):
    if ct_large <= ct_small: # overlape on right
        pt = ct_small + (Ls-lvl_small)
        if Ll<Ls:
            intersec = [pt, overlap_end]
        else:
            pt1 = ct_small - (Ll - lvl_small)
            if Ll>Ls:
                intersec = [pt1, ct_small + (Ll - lvl_small), overlap_end]
            else:
                intersec = [pt1, overlap_end]
    else: # overlap on left
        pt = ct_small - (Ls-lvl_small)
        if Ll<Ls:
            intersec = [pt, overlap_end]
        else:
            pt1 = ct_small + (Ll - lvl_small)
            if Ll>Ls:
                intersec = [pt1, ct_small - (Ll - lvl_small), overlap_end]
            else:
                intersec = [pt1, overlap_end]
    return intersec


def Intersec(centerc, centern, lvlc, lvln, Lc, Ln, Ulc, Urc, Uln, Urn):
    if Ulc<Uln:
        if Urc<Urn: # cross V
            intersec = crossV(centerc, centern, lvlc, lvln, Lc, Ln)
        else:
            if Urc>Urn: # contain V
                intersec = [centern - (Lc-lvln), centern + (Lc+lvln)]
                if Lc<Ln: # quick check
                    print("mistake on model filter")
            else: #right slope overlap (current model larger)
                intersec = slope_overlap(centerc, centern, lvlc, lvln, Lc, Ln, Urc)
    else: # left slope overlap
        if Urn<Urc: #current model larger
            intersec = slope_overlap(centerc, centern, lvlc, lvln, Lc, Ln, Ulc)
        else: # next model larger
            intersec = slope_overlap(centern, centerc, lvln, lvlc, Ln, Lc, Ulc)
    
    return intersec


# each model's min
def min_val(Center, Level, Lo, lendpt, rendpt):
    min_each_mdl = np.zeros(len(Center))
    
    for m in range(len(Center)):
        Ll = Center[m] - (Lo[m] - Level[m])
        Lr = Center[m] + (Lo[m] - Level[m])
        if rendpt < Ll or lendpt > Lr:
            min_each_mdl[m] = min(abs(lendpt-Center[m])+Level[m], abs(rendpt-Center[m])+Level[m])
        else:
            min_each_mdl[m] = Lo[m]    
    return min_each_mdl

def qi_y(Li, ui, Mu_i, Level_i, pointer, Sis, Si_ui):
    seg_i = []
    if len(Mu_i) == 1:
        Si_y = Si_ui
    else:
        # rid of const type (if existed, only as ui)
        ind_non_const = np.where(pointer > -0.5)[0]
        Lo = Li[ind_non_const]
        Center = Mu_i[ind_non_const]; Level = Level_i[ind_non_const]
        LSis = Sis[ind_non_const]
        
        Upleft = Center - (ui - Level); Upright = Center + (ui - Level)
        # sort from left to right by upper left endpoint
        sort_ind = np.argsort(Upleft)
        Lo = Lo[sort_ind]; LSis = LSis[sort_ind]
        Center = Center[sort_ind]; Level = Level[sort_ind]
        Upleft = Upleft[sort_ind]; Upright = Upright[sort_ind]
        
        # identify indep chunks
        chunk_l = []; chunk_r = []
        chunk_l.append(Upleft[0])
        loc = 0
        while loc<len(Center):
            find_ind = np.where(Upleft - Upright[loc]>0)[0]
            if len(find_ind) == 0:
                chunk_r.append(np.max(Upright))
                loc = len(Center)
            else:
                if find_ind[0] - 1 == loc:
                    chunk_r.append(Upright[loc])
                    chunk_l.append(Upleft[find_ind[0]])
                    loc = find_ind[0]
                else:
                    loc = find_ind[0] - 1
        
        # calculate intercepts inside each chunk
        for c in range(len(chunk_l)):
            seg_i.append(chunk_l[c])
            seg_i.append(chunk_r[c])
            # take the models inside this chunk
            md_in_chunk = np.where((Center-chunk_l[c])*(Center-chunk_r[c]) <=0)[0]
            ct_sub = Center[md_in_chunk]; lv_sub = Level[md_in_chunk]
            S_sub = LSis[md_in_chunk]; L_sub = Lo[md_in_chunk]
            Ul_sub = Upleft[md_in_chunk]; Ur_sub = Upright[md_in_chunk]
            
            #should naturally ordered by upright endpt from left to right
            if len(md_in_chunk)>1: # quick check
                if Ul_sub[0]>Ul_sub[1]:
                    print("mistake in order")
            
            for j in range(len(md_in_chunk)):
                k = j+1
                while k<len(md_in_chunk) and Upleft[k]<=Upright[j]:
                    # calculate intersection between mdl j and k
                    if Upleft[k] < Upright[j]:
                        intercpt = Intersec(ct_sub[j], ct_sub[k],lv_sub[j],lv_sub[k], L_sub[j],L_sub[k], 
                                        Ul_sub[j], Ur_sub[j], Ul_sub[k], Ur_sub[k])
                        for p in intercpt:
                            seg_i.append(p)
                    else:
                        seg_i.append(Upleft[k])
                    k = k+1
            
        
        # calculate step function Si_y
        seg_i = np.unique(seg_i)
        Si_y = np.zeros(len(seg_i)+1)
        for j in range(len(seg_i)):
            if j == 0:
                Si_y[0] = Si_ui; Si_y[-1] = Si_ui
            else:
                if np.min(np.abs(seg_i[j] - np.array(chunk_l)))>0: #interval inside a chunk
                    # find min of each model between seg_i[j-1] and seg_i[j]
                    min_each_mdl = min_val(Center, Level, Lo, seg_i[j-1], seg_i[j])
                    find_ind = np.where(min_each_mdl == np.min(min_each_mdl))[0]
                    Si_y[j] = max(LSis[find_ind])
                else: #its corresponding interval is outside-chunk
                    Si_y[j] = Si_ui
        
    
    return seg_i, Si_y

def quantl_y(k, S_and_seg):
    seg = []; S_out_chunk = []
    # unfold info
    for i in S_and_seg:
        seg_i = i[0]
        tmp_Sis = i[1]
        if len(seg_i)>0:
            S_out_chunk.append(tmp_Sis[0])
            for j in seg_i:
                seg.append(j)
        else:
            S_out_chunk.append(tmp_Sis)
    
    # calculate the step function of quantile
    if len(seg) == 0:
        quantl_y = np.sort(S_out_chunk)[k-1]
    else:
        seg = np.unique(seg)
        quantl_y = np.zeros(len(seg)+1)
        for j in range(len(seg)):
            if j == 0:
                quantl_y[0] = np.sort(S_out_chunk)[k-1]
                quantl_y[-1] = np.sort(S_out_chunk)[k-1]
            else:
                midpt = (seg[j-1]+seg[j])/2
                tmp_S = np.array(S_out_chunk)
                for i in range(len(S_and_seg)):
                    seg_i = S_and_seg[i][0]
                    S_cor = S_and_seg[i][1]
                    if len(seg_i)>0:
                        loc = np.where(seg_i-midpt>0)[0]
                        if len(loc)>0:
                            tmp_S[i] = S_cor[loc[0]]
                
                quantl_y[j] = np.sort(tmp_S)[k-1]
    
    return quantl_y, seg

def pred_set(mu_hat, y_test, quantl, seg):
    true_res = abs(y_test - mu_hat)
    if len(seg) == 0:
        Interv = [mu_hat-quantl, mu_hat+quantl]
        length = 2*quantl
        connect = 1
        if true_res <= quantl:
            cover = 1
        else:
            cover = 0
    else:
        left = np.zeros(len(seg)+1); right = np.zeros(len(seg)+1)
        for j in range(len(seg)):
            if j == 0:
                left[-1] = max(seg[-1], mu_hat - quantl[-1]); right[-1] = mu_hat + quantl[-1]
                left[0] = mu_hat - quantl[0]; right[0] = min(seg[0], mu_hat + quantl[0])
                if y_test > seg[-1]:
                    if true_res <= quantl[-1]:
                        cover = 1
                    else:
                        cover = 0
                if y_test < seg[0]:
                    if true_res <= quantl[0]:
                        cover = 1
                    else:
                        cover = 0
            else:
                left[j] = max(seg[j-1], mu_hat-quantl[j]); right[j] = min(seg[j], mu_hat+quantl[j])
                if (y_test - seg[j-1])*(y_test - seg[j])<0:
                    if true_res <= quantl[j]:
                        cover = 1
                    else:
                        cover = 0
            
            if y_test == seg[j]:
                if true_res <= max(quantl[j], quantl[j+1]):
                    cover = 1
                else:
                    cover = 0
        
        # calculate the length and output prediction set
        valid_ind = np.where(left<right)[0]
        left = left[valid_ind]; right = right[valid_ind]
        length = np.sum(right-left)
        if len(left) == 1:
            connect = 1
            Interv = [left[0], right[0]]
        else:
            breakpt = np.where(left[1:] - right[-1]>0)[0]
            if len(breakpt)>0:
                connect = 0
                l = np.append(left[0], left[breakpt+1])
                r = np.append(right[breakpt], right[-1])
                Interv = [l, r]
            else:
                connect = 1
                Interv = [min(left), max(right)]
        
    return cover, length, connect, Interv

def ModSelLOO_CQR(Models, Betas, X_cal, Y_cal, x_test, y_test, alpha):
    n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    M_models = len(Models); M_betas = len(Betas)
    
    S = np.zeros((n,M_models, M_betas))
    pred_center = np.zeros((M_models, M_betas)); pred_level = np.zeros((M_models, M_betas))
    for m in range(M_models):
        mdl = Models[m]
        for b in range(M_betas):
            beta = Betas[b]
            Q_lo = mdl.predict(X_cal, quantiles = beta)
            Q_hi = mdl.predict(X_cal, quantiles = (1-beta))
            q_lo = mdl.predict(x_test, quantiles = beta)
            q_hi = mdl.predict(x_test, quantiles = (1-beta))
            A = (np.sum((Q_hi-Q_lo)) + (q_hi-q_lo))/(2*(n+1))
            
            S[:,m,b] = np.maximum((Q_lo-Y_cal),(Y_cal-Q_hi)) + A # transformed score
            pred_center[m,b] = (q_lo+q_hi)/2
            pred_level[m,b] = ((q_lo-q_hi)/2) + A # transformed

    # flatten above
    pred_center = pred_center.flatten()
    pred_level = pred_level.flatten()
    S_all = S.reshape(S.shape[0], -1)
    
    S_q = np.sort(S_all, axis=0)[(k-2):(k+1),:]

    # initial filtering
    find_ind = np.where(S_q[0,:] <= np.min(S_q[2,:]))[0]
    S_all = S_all[:,find_ind]; S_q = S_q[:,find_ind]
    pred_center = pred_center[find_ind]; pred_level = pred_level[find_ind]
    M = len(find_ind)
    
    # select the model
    mhat = np.argmin(S_q[1,:])
    
    if M == 1: #only mhat remains
        if np.abs(y_test - pred_center[mhat]) + pred_level[mhat] <= S_q[1,mhat]:
            cover = 1
        else:
            cover = 0
        
        length = 2*(S_q[1,mhat] - pred_level[mhat])
        connect = 1
        Interv = [pred_center[mhat] - (length/2), pred_center[mhat]+ (length/2)]
    else:
        S_and_seg = []
        for i in range(n):
            Li, ui, Mu_i, Level_i, pointer, Sis, Si_ui = CALMi(i, S_all, S_q, pred_center, pred_level)
            seg_i, Si_y = qi_y(Li, ui, Mu_i, Level_i, pointer, Sis, Si_ui)
            S_and_seg.append([seg_i, Si_y])
        
        # combine all segs and get quantile's step function
        quantl, seg = quantl_y(k, S_and_seg)
        rescale_q = quantl - pred_level[mhat] #reduce to residual type
        cover, length, connect, Interv = pred_set(pred_center[mhat], y_test, rescale_q, seg)
    return cover, length, connect, Interv

