import os
import re
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import numpy as np
import matplotlib.gridspec as gridspec
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages

# ---- App Title and Instructions ----
st.title("Interactive Correlation Clustering Dashboard (Folder Version)")
st.write(
    "This app scans the 'Correlation data' folder for Excel files named like "
    "`Strategy_Correlation Matrix.xlsx` or `Strategy_Correlation Matrix - 3AY.xlsx`."
)
st.write("**How to use:** Put your correlation Excel files in the 'Correlation data' folder before starting.")

# ---- Folder & Sheet Setup ----
FOLDER = "Correlation data"
SHEET_NAME = "Correlation Matrix"

# ---- Scan Folder for Excel Files ----
if not os.path.isdir(FOLDER):
    st.error(f"Folder '{FOLDER}' was not found. Please create it and add your Excel files.")
    st.stop()

files = [f for f in os.listdir(FOLDER) if f.endswith(".xlsx")]

# ---- Parse File Names for Strategy & Period ----
strategy_period_map = defaultdict(dict)
for filename in files:
    match = re.match(r"(.+)_Correlation Matrix(?: - (\d+AY))?\.xlsx$", filename)
    if match:
        strategy = match.group(1).strip()
        period = match.group(2) if match.group(2) else "1AY"
        strategy_period_map[strategy][period] = filename

if not strategy_period_map:
    st.error("No valid correlation matrix files found in the 'Correlation data' folder.")
    st.stop()

# ---- User Selectors ----
strategy_names = sorted(strategy_period_map.keys())
selected_strategy = st.selectbox("Select strategy", strategy_names)
available_periods = sorted(strategy_period_map[selected_strategy].keys())
selected_period = st.selectbox("Select period", available_periods)
excel_file = os.path.join(FOLDER, strategy_period_map[selected_strategy][selected_period])

st.header(f"{selected_strategy} ({selected_period})")

# ---- Load and Clean Data ----
try:
    corr = pd.read_excel(excel_file, sheet_name=SHEET_NAME, header=2, index_col=0)
except Exception as e:
    st.error(f"Error reading {excel_file}: {e}")
    st.stop()

corr.index = corr.index.astype(str).str.strip()
corr.columns = corr.columns.astype(str).str.strip()
corr = corr.loc[corr.index.intersection(corr.columns), corr.columns.intersection(corr.index)]
corr = corr.loc[~corr.index.isnull(), ~corr.columns.isnull()]

# ---- Interactive Controls ----
linkage_method = st.selectbox("Linkage Method", [
    'ward', 'average', 'single', 'complete', 'weighted', 'centroid', 'median'
])
max_clusters = min(12, len(corr))
n_clusters = st.slider("Number of clusters", 2, max_clusters, 7)
view_option = st.radio("Choose view", [
    "A: Dendrogram with coloured branches",
    "B: Heatmap with coloured cluster bar",
    "A & B: Both side-by-side"
])
show_heatmap_yticklabels = False
if view_option == "A & B: Both side-by-side":
    show_heatmap_yticklabels = st.checkbox(
        "Show heatmap y-axis labels in joint view (recommended off for large matrices)",
        value=False
    )

metric = 'euclidean'  # You can change this if needed

def get_threshold(linkage_matrix, n_clusters):
    distances = linkage_matrix[:, 2]
    return np.sort(distances)[-(n_clusters-1)] if n_clusters > 1 else np.max(distances) + 1

# ---- Clustering ----
linkage_matrix = linkage(corr.abs(), method=linkage_method, metric=metric)
threshold = get_threshold(linkage_matrix, n_clusters)
palette = sns.color_palette("Set2", n_clusters)
clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
cluster_colors = {cluster: palette[i % len(palette)] for i, cluster in enumerate(sorted(set(clusters)))}

# ---- Plot and Export ----
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

if view_option == "A: Dendrogram with coloured branches":
    fig, ax = plt.subplots(figsize=(14, 16), constrained_layout=True)
    dendro = dendrogram(
        linkage_matrix,
        labels=corr.index.tolist(),
        orientation='left',
        color_threshold=threshold,
        leaf_font_size=9.5,
        ax=ax
    )
    ax.set_ylabel('Security', fontsize=10)
    ax.set_xlabel('Distance', fontsize=10)
    ax.set_title(
        f'Dendrogram ({linkage_method.capitalize()} linkage, {n_clusters} clusters)',
        fontsize=12
    )
    ax.tick_params(axis='y', labelsize=9.5, pad=1)
    plot_and_download(fig)

elif view_option == "B: Heatmap with coloured cluster bar":
    fig, ax = plt.subplots(figsize=(14, 16), constrained_layout=True)
    dendro = dendrogram(linkage_matrix, labels=corr.index.tolist(), orientation='left',
                        color_threshold=threshold, leaf_font_size=9.5, no_plot=True)
    labels_reordered = [corr.index.tolist()[i] for i in dendro['leaves']]
    corr_reordered = corr.loc[labels_reordered, labels_reordered]
    row_colors = pd.Series(clusters, index=corr.index).map(cluster_colors)[labels_reordered]
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
    for y, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-1, y), 0.5, 1, color=color, lw=0))
    ax.set_title(
        f'Correlation Matrix Heatmap ({linkage_method.capitalize()} linkage)',
        fontsize=12
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis='y', labelsize=9, pad=1)
    plot_and_download(fig)

else:  # "A & B: Both side-by-side"
    fig = plt.figure(figsize=(36, 16), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 9], wspace=0.18)
    # Dendrogram
    ax0 = fig.add_subplot(gs[0])
    dendro = dendrogram(
        linkage_matrix,
        labels=corr.index.tolist(),
        orientation='left',
        color_threshold=threshold,
        leaf_font_size=9.5,
        ax=ax0
    )
    ax0.set_ylabel('Security', fontsize=10)
    ax0.set_xlabel('Distance', fontsize=10)
    ax0.set_title(
        f'Dendrogram ({linkage_method.capitalize()} linkage, {n_clusters} clusters)',
        fontsize=12
    )
    ax0.tick_params(axis='y', labelsize=9.5, pad=1)
    labels_reordered = [tick.get_text() for tick in ax0.get_yticklabels()][::-1]
    corr_reordered = corr.loc[labels_reordered, labels_reordered]
    row_colors = pd.Series(clusters, index=corr.index).map(cluster_colors)[labels_reordered]
    # Heatmap
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
    for y, color in enumerate(row_colors):
        ax1.add_patch(plt.Rectangle((-1, y), 0.5, 1, color=color, lw=0))
    ax1.set_title(
        f'Correlation Matrix Heatmap ({linkage_method.capitalize()} linkage)',
        fontsize=12
    )
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=9, rotation=90)
    if show_heatmap_yticklabels:
        ax1.set_yticklabels(ax1.get_yticklabels(), fontsize=9)
    ax1.yaxis.set_label_position("right")
    ax1.tick_params(axis='y', labelsize=9, pad=1)
    plot_and_download(fig)
