"""
Two-Stage Weighted Graph Visualization
========================================
Stage 1: Global force-directed layout using ALL S weights (S > 0.1 only)
         → positions are fixed for all threshold experiments
Stage 2: Sub-graph filtering at THRESHOLD → Louvain community detection → color by community

Change THRESHOLD at line 9 to explore different cutoffs.
"""

THRESHOLD = 0.7      # 改這行來調整視覺化門檻
MIN_EDGE  = 0.1      # Stage 1: 忽略低於此的弱連接（不影響物理模擬大局）

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from collections import defaultdict

INBOUND  = Path('/home/ubuntu/.openclaw/media/inbound')
W_PATH   = INBOUND / 'symmetrized_matrix---dec21f33-b895-4377-a969-a4a6b7f7493c.npy'
OUT_PNG  = INBOUND / f'similarity_graph_thresh{THRESHOLD}.png'

# ═══════════════════════════════════════════════════════
# Stage 1: Physical Simulation (全權重力導向佈局)
# ═══════════════════════════════════════════════════════
print("Stage 1: Loading W → computing S (full matrix)...")

W = np.load(W_PATH)
W = (W + W.T) / 2.0
np.fill_diagonal(W, 0)

row_sums = W.sum(axis=1).reshape(-1, 1)
row_sums[row_sums == 0] = 1.0
M_prob = W / row_sums

from sklearn.metrics.pairwise import cosine_similarity
S_full = cosine_similarity(M_prob)
np.fill_diagonal(S_full, 0)
print(f"S_full shape={S_full.shape}, range=[{S_full.min():.4f}, {S_full.max():.4f}]")

# Build full weighted graph (S > MIN_EDGE) for layout
G_full = nx.Graph()
n = S_full.shape[0]
G_full.add_nodes_from(range(n))
i_idx, j_idx = np.triu_indices(n, k=1)
mask = S_full[i_idx, j_idx] > MIN_EDGE
edges_with_weight = [(int(i_idx[k]), int(j_idx[k]), float(S_full[i_idx[k], j_idx[k]]))
                     for k in range(len(i_idx)) if mask[k]]
G_full.add_weighted_edges_from(edges_with_weight)
print(f"G_full for layout: nodes={G_full.number_of_nodes()}, edges={G_full.number_of_edges()}")

# Force-directed layout (full weighted graph)
# Use spring_layout with weight (attraction proportional to S)
print("Computing force-directed layout (this may take a minute)...")
pos = nx.spring_layout(G_full, k=0.15, iterations=100, seed=42)
print("Layout done.")

# ═══════════════════════════════════════════════════════
# Stage 2: Visual Rendering (Threshold + Community)
# ═══════════════════════════════════════════════════════
print(f"\nStage 2: Filtering at THRESHOLD={THRESHOLD}...")

G_thresh = nx.Graph()
G_thresh.add_nodes_from(range(n))
mask_thresh = S_full[i_idx, j_idx] > THRESHOLD
edges_thresh = list(zip(i_idx[mask_thresh], j_idx[mask_thresh]))
G_thresh.add_edges_from(edges_thresh)

isolated = list(nx.isolates(G_thresh))
G_thresh.remove_nodes_from(isolated)
print(f"After removing isolated: nodes={G_thresh.number_of_nodes()}, edges={G_thresh.number_of_edges()}")

# Louvain community detection on the thresholded sub-graph
from community import community_louvain
partition = community_louvain.best_partition(G_thresh, weight='weight', resolution=1.0)
n_communities = len(set(partition.values()))
print(f"Communities detected: {n_communities}")

# Color map
cmap = plt.get_cmap('tab20' if n_communities <= 20 else 'hsv')
color_map = {node: cmap(partition[node] / max(n_communities - 1, 1)) for node in G_thresh.nodes()}
node_colors = [color_map[n] for n in G_thresh.nodes()]

# Degrees for node size
degrees = dict(G_thresh.degree())
node_sizes = [20 + degrees[n] * 5 for n in G_thresh.nodes()]

# ═══════════════════════════════════════════════════════
# Draw
# ═══════════════════════════════════════════════════════
plt.figure(figsize=(20, 20))
fig_facecolor = '#111111'
ax = plt.gca()
ax.set_facecolor(fig_facecolor)

nx.draw_networkx_nodes(G_thresh, pos,
                       nodelist=list(G_thresh.nodes()),
                       node_size=node_sizes,
                       node_color=node_colors,
                       alpha=0.85,
                       ax=ax)

# Draw only edges above threshold (thin, low alpha)
edge_weights_viz = [S_full[u, v] for u, v in G_thresh.edges()]
nx.draw_networkx_edges(G_thresh, pos,
                       edge_color='#555555',
                       width=[0.3 + (S_full[u, v] - THRESHOLD) * 0.5 for u, v in G_thresh.edges()],
                       alpha=0.4,
                       ax=ax)

# Component summary
component_sizes = defaultdict(int)
for node in G_thresh.nodes():
    component_sizes[partition[node]] += 1

ax.set_title(
    f'Similarity Graph (S > {THRESHOLD}) | Nodes={G_thresh.number_of_nodes()}  '
    f'Edges={G_thresh.number_of_edges()}  Communities={n_communities}',
    color='white', fontsize=14
)
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=fig_facecolor)
plt.close()
print(f"Saved: {OUT_PNG}")

# Print community summary
print("\nCommunity sizes:")
for comm_id, size in sorted(component_sizes.items(), key=lambda x: -x[1]):
    print(f"  Community {comm_id}: {size} nodes")
print("Done.")