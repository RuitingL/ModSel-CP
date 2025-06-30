import numpy as np
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd

from random import sample
from tqdm import tqdm

import mtds_func_residual as rfm

def Mmodels_length(Mmodels, X_cal, Y_cal, alpha):
    M = len(Mmodels); n = len(Y_cal)
    k = math.ceil((n+1)*(1-alpha))
    
    S = np.zeros(M)
    for m in range(M):
        mdl = Mmodels[m]
        S[m] = np.sort(np.abs(Y_cal - mdl.predict(X_cal)))[k-1]
    
    length = 2*S
    return length

class pred_by_add_intercept:
    def __init__(self, C):
        self.C = C
    def predict(self,X):
        X_back = X.reshape(-1)
        pred = X_back + self.C
        return pred



def train_model(C):
    Mmodels = []
    
    for m in range(2):
        c = pow(-1, m+1) * C
        new_modl = pred_by_add_intercept(c)
        Mmodels.append(new_modl)
    return Mmodels




def experiment_2models(N_rep, C, noise_mu, n_cal, alpha, split_portion):
    Mmodels = train_model(C)
    M = len(Mmodels)
    Coverage = np.zeros((5,N_rep)); Length = np.zeros((5,N_rep))
    Connect = np.zeros(2); effeM = np.zeros(N_rep)
    benchL = np.zeros((M,N_rep))
    
    for t in tqdm(range(N_rep)):
        X = np.random.normal(size=n_cal+1)
        W = np.random.normal(loc=noise_mu, size=n_cal+1) # noise with mean noise_mu
        Y = X + W
        X = X.reshape(n_cal+1, 1)
        X_cal = X[:-1, :]; Y_cal = Y[:-1]
        x_test = X[-1, :]; y_test = Y[-1]
        
        coverE, lengthE, _, connectE, calM = rfm.ModSel_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverL, lengthL, connectL, _ = rfm.ModSelLOO_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverEFCP, lengthEFCP, _ = rfm.YKbaseline_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        coverVFCP, lengthVFCP, _ = rfm.YKsplit_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha, split_portion)
        coverEFCP_adj, lengthEFCP_adj, _ = rfm.YKbaseline_adj_res(Mmodels, X_cal, Y_cal, x_test, y_test, alpha)
        
        Coverage[0,t] = coverE; Coverage[1,t] = coverL
        Coverage[2,t] = coverEFCP; Coverage[3,t] = coverVFCP; Coverage[4,t] = coverEFCP_adj
        
        Length[0,t] = lengthE; Length[1,t] = lengthL
        Length[2,t] = lengthEFCP; Length[3,t] = lengthVFCP; Length[4,t] = lengthEFCP_adj
        
        Connect[0] += connectE; Connect[1] += connectL
        effeM[t] = calM
        
        benchL[:,t] = Mmodels_length(Mmodels, X_cal, Y_cal, alpha)
    
    # calculate mean and se
    cov = np.zeros((5,2)); leng = np.zeros((5,2))
    for j in range(5):
        cov[j,0] = np.mean(Coverage[j,:]); cov[j,1] = np.std(Coverage[j,:])/np.sqrt(N_rep)
        leng[j,0] = np.mean(Length[j,:]); leng[j,1] = np.std(Length[j,:])/np.sqrt(N_rep)
    Connect = Connect/N_rep
    min_single_md_len = np.min(np.mean(benchL, axis=1))
    
    
    return cov, leng, Connect, effeM, min_single_md_len




alpha = 0.1; N_rep = 5000; split_portion = 0.5
n_cal = 200
Clist = [0.5, 1, 5]
Noise_mu = [-1, -0.5, 0, 0.5, 1]




for C in Clist:
    print(f"C= {C}")
    method_names = ["ModSel-CP", "ModSel-CP-LOO", "YK-baseline", "YK-split", "YK-adjust"]
    # List to collect all results
    all_results = []
    for mu in Noise_mu:
        cov, leng, Connect, effeM, min_single_len = experiment_2models(N_rep, C, mu, 
                                                                          n_cal, alpha, split_portion)
        
        for metric_name, data in [("coverage", cov), ("length", leng)]:
            for method, (mean, se) in zip(method_names, data):
                all_results.append({
                    "mu": mu,
                    "metric": metric_name,
                    "method": method,
                    "mean": mean,
                    "se": se,
                    "min_single_len": min_single_len
                })
    # Convert to DataFrame and save
    df_results = pd.DataFrame(all_results)
    filename = f"results_ of_C_{C}_n_{n_cal}.csv"
    df_results.to_csv(filename, index=False)


# Plot with se bar
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Functions to plot data
def plot_coverage(ax, df):
    row_id_col = "mu"
    metric_id = "coverage" if "coverage" in df["metric"].unique() else "Coverage"
    df_cov = df[df["metric"] == metric_id]
    row_vals = sorted(df_cov[row_id_col].unique())
    x_vals = range(len(row_vals))

    methods = ["ModSel-CP", "ModSel-CP-LOO", "YK-baseline", "YK-split"]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    markers = ['o','v','h','p','>']

    for method, color, marker in zip(methods, colors, markers):
        df_m = df_cov[df_cov["method"] == method].sort_values(by=row_id_col)
        mean = df_m["mean"].values
        se = df_m["se"].values

        ax.plot(x_vals, mean, label=method, linestyle='-', marker=marker, 
                color=color, linewidth=line_width, markersize=marker_size)
        ax.fill_between(x_vals, mean - se, mean + se, color=color, alpha=0.3)

    ax.axhline(y=0.9, color='black', linestyle='--', linewidth=2)
    #ax.set_xlabel('$|\Lambda|$', fontsize=label_size)
    ax.set_ylabel('Coverage', fontsize=label_size)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in row_vals], rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)



def plot_length(ax, df):
    row_id_col = "mu"
    df_len = df[df["metric"] == "length"]
    row_vals = sorted(df_len[row_id_col].unique())
    x_vals = range(len(row_vals))

    methods = ["ModSel-CP", "ModSel-CP-LOO", "YK-baseline", "YK-split"]
    colors = ['blue', 'green', 'red', 'purple','orange']
    markers = ['o','v','h','p','>']

    for method, color, marker in zip(methods, colors, markers):
        df_m = df_len[df_len["method"] == method].sort_values(by=row_id_col)
        mean = df_m["mean"].values
        se = df_m["se"].values

        ax.plot(x_vals, mean, label=method, linestyle='-', marker=marker,
                color=color, linewidth=line_width, markersize=marker_size)
        ax.fill_between(x_vals, mean - se, mean + se, color=color, alpha=0.3)

    # Plot the line for min_length
    ax.plot(x_vals, df["min_single_len"].unique(), 
            label=r"$\min_{\lambda \in \Lambda} |\widehat{C}^\lambda_{\hat{q}(\lambda)}(X_{n+1})|$", 
            linestyle="-", color="brown", linewidth=2)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in row_vals], rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)
    #ax.set_xlabel('$|\Lambda|$', fontsize=label_size)
    #ax.set_ylabel('Length', fontsize=label_size)



def plot2by3_with_se(dfs, titles, filename, x_axis_name="M"):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 13), sharex=True)

    for i in range(3):
        df = dfs[i]
        title = titles[i]

        axes[0, i].set_title(title, fontsize=title_size, fontweight='bold')
        plot_coverage(axes[0, i], df)
        plot_length(axes[1, i], df)

        x_label = '$\mu$' if x_axis_name == "M" else '$n$'
        axes[1, i].set_xlabel(x_label, fontsize=label_size)

    # Set row titles
    for ax, row in zip(axes[:, 0], ['Coverage', 'Length ratio']):
        ax.set_ylabel(row, fontsize=label_size)

    # Universal legend
    handles, labels = ax.get_legend_handles_labels()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=legend_size)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and show
    plt.savefig(filename, bbox_inches='tight')
    plt.show()




plt.style.use('seaborn')
    
# Adjusting settings for visual enhancements
label_size = 20 # Increased label size
legend_size = 20  # Increased legend size
title_size = 21  # Increased title size
line_width = 1.5  # Bolder lines
marker_size = 12   # Larger dots
y_axis_value_label_size = label_size  # Increased y-axis value label size for better visibility



df_C_half = pd.read_csv('results_ of_C_0.5_n_200.csv')
df_C_1 = pd.read_csv('results_ of_C_1_n_200.csv')
df_C_5 = pd.read_csv('results_ of_C_5_n_200.csv')


# List of dataframes in the order they should be plotted
dfs = [df_C_half, df_C_1, df_C_5]

# Titles for the plots
titles = ['C = 0.5', 'C = 1','C = 5']
filename = f"ModSel_conserv_with_se.pdf"
plot2by3_with_se(dfs, titles, filename, x_axis_name="M")
