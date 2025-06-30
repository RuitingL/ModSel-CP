import pandas as pd

import numpy as np
import math
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.patches as patches




# pre-process YK-adjust for better visualization
def adjust_YKadjust(df):
    df_mod = df.copy()
    metric_id = "length_ratio" if "length_ratio" in df["metric"].unique() else "Length ratio"
    mask_yk_adjust = (df_mod["method"] == "YK-adjust") & (df_mod["metric"] == metric_id)
    
    # Check if any are inf
    inf_mask = mask_yk_adjust & df_mod["mean"].eq(np.inf)
    
    if df_mod.loc[inf_mask].shape[0] == df_mod.loc[mask_yk_adjust].shape[0]:
        # Case: all YK-adjust length ratios are inf: set all to 1.35 and se = 0
        df_mod.loc[mask_yk_adjust, "mean"] = 1.35
        df_mod.loc[mask_yk_adjust, "se"] = 0
    else:
        # Case: some inf, some not: set infs to max(non-inf) + 0.1, se = 0
        max_val = df_mod.loc[mask_yk_adjust & ~df_mod["mean"].eq(np.inf), "mean"].max()
        df_mod.loc[inf_mask, "mean"] = max_val + 0.1
        df_mod.loc[inf_mask, "se"] = 0

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
    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in row_vals], rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)




def plot_length(ax, df):
    row_id_col = "n_cal" if "n_cal" in df.columns else "M" if "M" in df.columns else "n"
    metric_id = "length_ratio" if "length_ratio" in df["metric"].unique() else "Length ratio"
    df_len = df[df["metric"] == metric_id]
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

    # Plot the constant line for min_length = 1
    ax.plot(x_vals, [1.0] * len(x_vals), 
            label=r"$\min_{\lambda \in \Lambda} |\widehat{C}^\lambda_{\hat{q}(\lambda)}(X_{n+1})|$", 
            linestyle="-", color="brown", linewidth=2)

    ax.set_xticks(x_vals)
    ax.set_xticklabels([str(v) for v in row_vals], rotation=45, fontsize=label_size)
    ax.margins(x=0.05, y=0.05)

    # Customize y-axis to show 'Infinity'
    max_y_val = df_len["mean"].max()
    ticks = ax.get_yticks()  # Get current ticks
    new_ticks = np.unique(np.append(ticks, max_y_val)) # Append to the list of ticks
    if np.max(new_ticks)>max_y_val:
        new_ticks = new_ticks[new_ticks<=max_y_val]
    new_ticklabels = [f"{item:.1f}" if item < max_y_val else r'$\infty$' for item in new_ticks]
    ax.set_yticks(new_ticks)  # Set new ticks
    ax.set_yticklabels(new_ticklabels)  # Set new tick labels
    ax.tick_params(axis='y', labelsize=y_axis_value_label_size)




def plot4by2_with_se(dfs, titles, filename, x_axis_name="M"):
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True)

    for i in range(4):
        df = dfs[i]
        title = titles[i]

        axes[0, i].set_title(title, fontsize=title_size, fontweight='bold')
        plot_coverage(axes[0, i], df)
        plot_length(axes[1, i], df)

        x_label = '$|\Lambda|$' if x_axis_name == "M" else '$n$'
        axes[1, i].set_xlabel(x_label, fontsize=label_size)

    # Set row titles
    for ax, row in zip(axes[:, 0], ['Coverage', 'Length ratio']):
        ax.set_ylabel(row, fontsize=label_size)

    # Universal legend
    handles, labels = ax.get_legend_handles_labels() #axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=legend_size)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save and show
    plt.savefig(filename, bbox_inches='tight')
    plt.show()




plt.style.use('seaborn')
    
# Adjusting settings for visual enhancements
title_size = 17  # Increased title size
line_width = 1  # Bolder lines
marker_size = 8   # Larger dots
label_size = 18 # Increased label size
legend_size = 19  # Increased legend size
y_axis_value_label_size = label_size  # Increased y-axis value label size for better visibility




df1 = pd.read_csv("ReasyX_sparse_normal_n100.csv")
df2 = pd.read_csv("ReasyX_sparse_simpleT_n100.csv")
df3 = pd.read_csv("ReasyX_dense_normal_n100.csv")
df4 = pd.read_csv("RtX_sparse_normal_n100.csv")

dfs = [df1, df2, df3, df4]
mod_dfs = []
for df in dfs:
    mod_df = adjust_YKadjust(df)
    mod_dfs.append(mod_df)
titles = ['NormalX + sparse + Gaussian noise', 'NormalX + sparse + heavy-tail noise', 
          'NormalX + dense + Gaussian noise', 'tX + sparse + Gaussian noise']
filename = "Residual_n100.pdf"
plot4by2_with_se(mod_dfs, titles, filename, x_axis_name="M")




df1 = pd.read_csv("ReasyX_sparse_normal_M200.csv")
df2 = pd.read_csv("ReasyX_sparse_simpleT_M200.csv")
df3 = pd.read_csv("ReasyX_dense_normal_M200.csv")
df4 = pd.read_csv("RtX_sparse_normal_M200.csv")

dfs = [df1, df2, df3, df4]
mod_dfs = []
for df in dfs:
    mod_df = adjust_YKadjust(df)
    mod_dfs.append(mod_df)
titles = ['NormalX + sparse + Gaussian noise', 'NormalX + sparse + heavy-tail noise', 
          'NormalX + dense + Gaussian noise', 'tX + sparse + Gaussian noise']
filename = "Residual_M200.pdf"
plot4by2_with_se(mod_dfs, titles, filename, x_axis_name="n_cal")




df1 = pd.read_csv("RReasyX_sparse_normal_n100.csv")
df2 = pd.read_csv("RReasyX_sparse_simpleT_n100.csv")
df3 = pd.read_csv("RReasyX_dense_normal_n100.csv")
df4 = pd.read_csv("RRtX_sparse_normal_n100.csv")

dfs = [df1, df2, df3, df4]
mod_dfs = []
for df in dfs:
    mod_df = adjust_YKadjust(df)
    mod_dfs.append(mod_df)
titles = ['NormalX + sparse + Gaussian noise', 'NormalX + sparse + heavy-tail noise', 
          'NormalX + dense + Gaussian noise', 'tX + sparse + Gaussian noise']
filename = "Rescaled_residual_n100.pdf"
plot4by2_with_se(mod_dfs, titles, filename, x_axis_name="M")




df1 = pd.read_csv("RReasyX_sparse_normal_M200.csv")
df2 = pd.read_csv("RReasyX_sparse_simpleT_M200.csv")
df3 = pd.read_csv("RReasyX_dense_normal_M200.csv")
df4 = pd.read_csv("RRtX_sparse_normal_M200.csv")

dfs = [df1, df2, df3, df4]
mod_dfs = []
for df in dfs:
    mod_df = adjust_YKadjust(df)
    mod_dfs.append(mod_df)
titles = ['NormalX + sparse + Gaussian noise', 'NormalX + sparse + heavy-tail noise', 
          'NormalX + dense + Gaussian noise', 'tX + sparse + Gaussian noise']
filename = "Rescaled_residual_M200.pdf"
plot4by2_with_se(mod_dfs, titles, filename, x_axis_name="n")






