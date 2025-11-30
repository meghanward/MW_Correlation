import os
import re
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np
import matplotlib.gridspec as gridspec
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# ---- App Title and Instructions ----
st.title("Interactive Correlation Clustering Dashboard")
st.write(
    "**Cluster branch colouring:** Branches are coloured at the 75th percentile of the linkage distances—"
    "only the most significant splits in the dendrogram are highlighted."
)

# ---- Folder & Sheet Setup ----
FOLDER = "./Correlation data"
SHEET_NAME = "Correlation Matrix"

if not os.path.isdir(FOLDER):
    st.error(f"Folder '{FOLDER}' was not found. Please create it and add your Excel files.")
    st.stop()

files = [f for f in os.listdir(FOLDER) if f.endswith(".xlsx")]

strategy_period_map = defaultdict(dict)
for filename in files:
    match = re.match(r"(.+)_Correlation Matrix(?: - (\d+AY))?\.xlsx$", filename)
    if match:
        strategy = match.group(1).strip()
        period = match.group(2) if match.group(2) else "1AY"
        strategy_period_map[strategy][period] = filename

if not strategy_period_map:
    st.error("No valid correlation matrix files found in the folder.")
    st.stop()

strategy_names = sorted(strategy_period_map.keys())
selected_strategy = st.selectbox("Select strategy", strategy_names)
available_periods = sorted(strategy_period_map[selected_strategy].keys())
selected_period = st.selectbox("Select period", available_periods)
excel_file = os.path.join(FOLDER, strategy_period_map[selected_strategy][selected_period])

st.header(f"{selected_strategy} ({selected_period})")

try:
    corr = pd.read_excel(excel_file, sheet_name=SHEET_NAME, header=2, index_col=0)
except Exception as e:
    st.error(f"Error reading {excel_file}: {e}")
    st.stop()

corr.index = corr.index.astype(str).str.strip()
corr.columns = corr.columns.astype(str).str.strip()
corr = corr.loc[corr.index.intersection(corr.columns), corr.columns.intersection(corr.index)]
corr = corr.loc[~corr.index.isnull(), ~corr.columns.isnull()]

linkage_method = "ward"
metric = "euclidean"

linkage_matrix = linkage(corr.abs(), method=linkage_method, metric=metric)
distances = linkage_matrix[:, 2]
distance_threshold = float(np.percentile(distances, 75))

# ---- Friendly View Names ----
view_option = st.radio("Choose view", [
    "Clustergram (Matrix & Dendrograms)",
    "Heatmap (Cluster Order)",
    "Dendrogram (Coloured Branches)",
    "Side-by-Side: Dendrogram & Heatmap"
])

show_heatmap_yticklabels = False
if view_option == "Side-by-Side: Dendrogram & Heatmap":
    show_heatmap_yticklabels = st.checkbox(
        "Show heatmap y-axis labels in joint view (recommended off for large matrices)",
        value=False
    )

def plot_and_download(fig):
    st.pyplot(fig)
    pdf_buffer = io.BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    pdf_buffer.seek(0)
    st.download_button(
        label="Download as PDF",
        data=pdf_buffer,
        file_name=f"{selected_strategy}_{selected_period}_correlation_plot.pdf",
        mime="application/pdf"
    )

# ---- 1. Clustergram (Matrix & Dendrograms) ----
if view_option == "Clustergram (Matrix & Dendrograms)":
    sns.set(font_scale=0.8)
    clustermap_fig = sns.clustermap(
        corr.abs(),
        method=linkage_method,
        metric=metric,
        cmap=sns.light_palette("navy", as_cmap=True),
        linewidths=0.5,
        figsize=(13, 13),
        cbar_kws={'shrink': 0.5}
    )
    clustermap_fig.fig.suptitle(f"Clustergram (Ward linkage)", y=1.02, fontsize=15)
    plot_and_download(clustermap_fig.fig)

# ---- 2. Heatmap (Cluster Order) ----
elif view_option == "Heatmap (Cluster Order)":
    fig, ax = plt.subplots(figsize=(14, 16), constrained_layout=True)
    dendro = dendrogram(
        linkage_matrix, labels=corr.index.tolist(), orientation='left',
        color_threshold=distance_threshold, leaf_font_size=9.5, no_plot=True
    )
    labels_reordered = [corr.index.tolist()[i] for i in dendro['leaves']]
    corr_reordered = corr.loc[labels_reordered, labels_reordered]
    sns.heatmap(
        corr_reordered.abs(),
        cmap=sns.light_palette("navy", as_cmap=True),
        linewidths=0.5,
        square=True,
        ax=ax,
        xticklabels=True,
        yticklabels=labels_reordered,
        cbar_kws={'shrink': 0.5}
    )
    ax.set_title(
        f'Heatmap (Ward linkage, cluster order from dendrogram)',
        fontsize=12
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y', labelsize=9, pad=1)
    plot_and_download(fig)

# ---- 3. Dendrogram (Coloured Branches) ----
elif view_option == "Dendrogram (Coloured Branches)":
    fig, ax = plt.subplots(figsize=(14, 16), constrained_layout=True)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.yaxis.grid(True, color='#e0e0e0', linewidth=0.7, linestyle='-', alpha=0.7)
    ax.xaxis.grid(True, color='#e0e0e0', linewidth=0.7, linestyle='-', alpha=0.7)
    dendro = dendrogram(
        linkage_matrix,
        labels=corr.index.tolist(),
        orientation='left',
        color_threshold=distance_threshold,
        leaf_font_size=9.5,
        ax=ax
    )
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.2)
    ax.set_ylabel('Security', fontsize=10)
    ax.set_xlabel('Distance', fontsize=10)
    ax.set_title(
        f'Dendrogram (Ward linkage, 75th percentile threshold = {distance_threshold:.2f})',
        fontsize=12
    )
    ax.tick_params(axis='y', labelsize=9.5, pad=1)
    plot_and_download(fig)

# ---- 4. Side-by-Side: Dendrogram & Heatmap ----
elif view_option == "Side-by-Side: Dendrogram & Heatmap":
    fig = plt.figure(figsize=(36, 16), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 9], wspace=0.18)
    ax0 = fig.add_subplot(gs[0])
    ax0.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax0.yaxis.grid(True, color='#e0e0e0', linewidth=0.7, linestyle='-', alpha=0.7)
    ax0.xaxis.grid(True, color='#e0e0e0', linewidth=0.7, linestyle='-', alpha=0.7)
    dendro = dendrogram(
        linkage_matrix,
        labels=corr.index.tolist(),
        orientation='left',
        color_threshold=distance_threshold,
        leaf_font_size=9.5,
        ax=ax0
    )
    for spine in ax0.spines.values():
        spine.set_edgecolor('#333333')
        spine.set_linewidth(1.2)
    ax0.set_ylabel('Security', fontsize=10)
    ax0.set_xlabel('Distance', fontsize=10)
    ax0.set_title(
        f'Dendrogram (Ward linkage, 75th percentile threshold = {distance_threshold:.2f})',
        fontsize=12
    )
    ax0.tick_params(axis='y', labelsize=9.5, pad=1)
    labels_reordered = [tick.get_text() for tick in ax0.get_yticklabels()][::-1]
    corr_reordered = corr.loc[labels_reordered, labels_reordered]
    ax1 = fig.add_subplot(gs[1])
    sns.heatmap(
        corr_reordered.abs(),
        cmap=sns.light_palette("navy", as_cmap=True),
        linewidths=0.5,
        square=True,
        ax=ax1,
        xticklabels=True,
        yticklabels=labels_reordered if show_heatmap_yticklabels else False,
        cbar_kws={'shrink': 0.5}
    )
    ax1.set_title(
        f'Heatmap (Ward linkage, cluster order from dendrogram)',
        fontsize=12
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=9, rotation=90)
    if show_heatmap_yticklabels:
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=9)
    ax1.yaxis.set_label_position("right")
    ax1.tick_params(axis='y', labelsize=9, pad=1)
    plot_and_download(fig)

# ---- Interactive Info Panel ----
row = st.selectbox("Select first security (row)", corr.index.tolist())
col = st.selectbox("Select second security (column)", corr.columns.tolist())
value = corr.loc[row, col]
st.info(f"**Correlation between `{row}` and `{col}`:** {value:.3f}")
