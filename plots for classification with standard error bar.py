import pandas as pd

import numpy as np
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches




# preprocess to get length ratio
def get_length_ratio(df):
    df_mod = df.copy()
    mask = df_mod["metric"] == "length"
    
    df_mod.loc[mask, "mean"] = df_mod.loc[mask, "mean"] / df_mod.loc[mask, "min_single_len"]
    df_mod.loc[mask, "se"] = df_mod.loc[mask, "se"] / df_mod.loc[mask, "min_single_len"]
    
    df_mod["metric"] = df_mod["metric"].replace({"length": "length_ratio"})

    return df_mod




# Functions to plot data
def plot_coverage(ax, df):
    row_id_col = "n_cal" if "n_cal" in df.columns else "M" if "M" in df.columns else "n"
    metric_id = "coverage" if "coverage" in df["metric"].unique() else "Coverage"
    df_cov = df[df["metric"] == metric_id]
    row_vals = sorted(df_cov[row_id_col].unique())
    x_vals = range(len(row_vals))

    methods = ["ModSel-CP", "ModSel-CP-LOO", "YK-baseline", "YK-split", "YK-adjust"]
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
    ax.set_xlabel('$|\Lambda|$', fontsize=label_size)
    ax.set_ylabel('Coverage', fontsize=label_size)
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in row_vals], rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)




def plot_length_ratio(ax, df):
    row_id_col = "n_cal" if "n_cal" in df.columns else "M" if "M" in df.columns else "n"
    df_len = df[df["metric"] == "length_ratio"]
    row_vals = sorted(df_len[row_id_col].unique())
    x_vals = range(len(row_vals))

    methods = ["ModSel-CP", "ModSel-CP-LOO", "YK-baseline", "YK-split", "YK-adjust"]
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
    ax.plot(x_vals, [1]*len(x_vals), 
            label=r"$\min_{\lambda \in \Lambda} |\widehat{C}^\lambda_{\hat{q}(\lambda)}(X_{n+1})|$", 
            linestyle="-", color="brown", linewidth=2)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in row_vals], rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)
    ax.set_xlabel('$|\Lambda|$', fontsize=label_size)
    ax.set_ylabel('Cardinality ratio', fontsize=label_size)



def plot1by2_with_se_ratio(df):
    fig, axes = plt.subplots(1, 2, figsize=(26, 12))
    
    plot_coverage(axes[0], df)
    plot_length_ratio(axes[1], df)
    
    # Title
    fig.suptitle('Classification experiment', fontsize=title_size, fontweight='bold', y=title_position)
    
    # set the universal legend
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=legend_size)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('classification_exp_ratio.pdf', bbox_inches='tight')
    plt.show()



plt.style.use('seaborn')
    
# Adjusting settings for visual enhancements
label_size = 30  # Increased label size
legend_size = 24  # Increased legend size
title_size = 28  # Increased title size
line_width = 1  # Bolder lines
marker_size = 13   # Larger dots
y_axis_value_label_size = label_size  # Increased y-axis value label size for better visibility
title_position = 0.93  # Adjusting title position closer to the plot



# Load the saved result CSV
df = pd.read_csv("classification_results.csv")
df_mod = get_length_ratio(df)
plot1by2_with_se_ratio(df_mod)

