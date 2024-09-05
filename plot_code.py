#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd



plt.style.use('seaborn')
    
# Adjusting settings for visual enhancements
label_size = 18 # Increased label size
legend_size = 19  # Increased legend size
title_size = 17  # Increased title size
line_width = 1  # Bolder lines
marker_size = 8   # Larger dots
y_axis_value_label_size = label_size  # Increased y-axis value label size for better visibility




# process length to ratio
def process_ratio(df):
    num_row = len(df["ModSelc"])
    df_ratio = df
    
    # process to ratio
    # Columns to divide
    columns_to_divide = ["ModSell", "ModSelLOOl", "YKbaselinel", "YKsplitl", "YK_adjl"]
    for cln in columns_to_divide:
        df_ratio[cln] = df_ratio[cln]/df_ratio["Min_Length"]
    
    df_ratio["Min_Length"] = 1
    
    # process YK-adj
    if df_ratio["YK_adjc"].max() == 0: # if every YK-adj return cal{Y}
        df_ratio["YK_adjc"] = 1
        df_ratio["YK_adjl"] = 1.35
    else:
        mr = df_ratio["YK_adjl"].max()
        for i in range(num_row):
            if df_ratio.loc[i, "YK_adjc"] == 0:
                df_ratio.loc[i, "YK_adjc"] = 1
                df_ratio.loc[i, "YK_adjl"] = mr+0.1
    
    return df_ratio

# Functions to plot data
def plot_coverage(ax, df):
    M_cat_indices_equal = range(len(df["ModSelc"]))
    M_values_str = df.iloc[:,0].astype(str).tolist()
    for method, color, marker in zip(["ModSelc", "ModSelLOOc", "YKbaselinec", "YKsplitc", "YK_adjc"], 
                                     ['blue', 'green', 'red', 'purple', 'orange'], 
                                     ['o','v','h','p','>']):
        new_label = "ModSel-CP" if "ModSelc" in method else "ModSel-CP-LOO" if "ModSelLOOc" in method else "YK-baseline" if "YKbaselinec" in method else "YK-adjust" if "YK_adjc" in method else "YK-split" if "YKsplitc" in method else method.split('_')[0]
        ax.plot(M_cat_indices_equal, df[method], label=new_label, marker=marker, linestyle='-', 
                color=color, linewidth=line_width, markersize=marker_size)
    ax.axhline(y=0.9, color='black', linestyle='--', linewidth=2)
    ax.set_xticks(M_cat_indices_equal)
    ax.set_xticklabels(M_values_str, rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)

def plot_length(ax, df):
    M_cat_indices_equal = range(len(df["ModSelc"]))
    M_values_str = df.iloc[:,0].astype(str).tolist()
    for method, color, marker in zip(["ModSell", "ModSelLOOl", "YKbaselinel", "YKsplitl", "YK_adjl", "Min_Length"], 
                                     ['blue', 'green', 'red', 'purple','orange', 'brown'], 
                                     ['o','v','h','p','>','s']):
        new_label = "ModSel-CP" if "ModSell" in method else "ModSel-CP-LOO" if "ModSelLOOl" in method else "YK-baseline" if "YKbaselinel" in method else "YK-adjust" if "YK_adjl" in method else "YK-split" if "YKsplitl" in method else "$\min_{\lambda \in \Lambda}  |\widehat{C}^{\lambda}_{\hat{q}(\lambda)}(X_{n+1})|$" if "Min_Length" in method else method.split('_')[0]
        ax.plot(M_cat_indices_equal, df[method], label=new_label, marker = marker, linestyle='-', 
                color=color, linewidth=line_width, markersize=marker_size)
    ax.set_xticks(M_cat_indices_equal)
    ax.set_xticklabels(M_values_str, rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    
    # Customize y-axis to show 'Infinity'
    max_y_val = df["YK_adjl"].max()
    ticks = ax.get_yticks()  # Get current ticks
    new_ticks = np.unique(np.append(ticks, max_y_val)) # Append to the list of ticks
    if np.max(new_ticks)>max_y_val:
        new_ticks = new_ticks[new_ticks<=max_y_val]
    new_ticklabels = [f"{item:.1f}" if item < max_y_val else r'$\infty$' for item in new_ticks]
    ax.set_yticks(new_ticks)  # Set new ticks
    ax.set_yticklabels(new_ticklabels)  # Set new tick labels
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)

def plot4by2(dfs, titles):
    # Create a figure with subplots
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True)
    
    # Fill each subplot
    for i in range(4):
        axes[0, i].set_title(titles[i], fontsize=title_size, fontweight='bold')
        plot_coverage(axes[0, i], dfs[i])
        plot_length(axes[1, i], dfs[i])
        if dfs[i].columns[0] == "M":
            axes[1, i].set_xlabel('$|\Lambda|$', fontsize=label_size)
        else:
            axes[1, i].set_xlabel('$n$', fontsize=label_size)
    
    # Set row titles
    for ax, row in zip(axes[:,0], ['Coverage', 'Length ratio']):
        ax.set_ylabel(row, fontsize=label_size)
    
    # set the universal legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=legend_size)
    
    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if dfs[0].columns[0] == "M":
        filename = f"R_{titles[0]}_n.pdf"
    else:
        filename = f"R_{titles[0]}_M.pdf"
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    

