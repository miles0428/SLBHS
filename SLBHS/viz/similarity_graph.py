"""
Similarity Graph Visualization
Threshold = 0.7 — 只畫 S > 0.7 的邊，無 neighbor 的 node 不畫
"""

THRESHOLD = 0.7   # 改門檻值只改這行

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

INBOUND = Path('/home/ubuntu/.openclaw/media/inbound')
W_PATH  = INBOUND / 'symmetrized_matrix---dec21f33-b895-4377-a969-a4a6b7f7493c.npy'
OUT_PNG = INBOUND / f'similarity_graph_thresh{THRESHOLD}.png'

# ── Step 1: Load W ──────────────────────────────────────────
print("Loading W...")
W = np.load(W_PATH)

# ── Step 2: Symmetrize → Remove self-transitions (zero diagonal) ──
W = (W + W.T) / 2.0
np.fill_diagonal(W, 0)
print(f"W diagonal after fill_diagonal: {np.diag(W)[:5]} (should be 0)")

# ── Step 3: Row normalize → M_prob ─────────────────────────
row_sums = W.sum(axis=1).reshape(-1, 1)
row_sums[row_sums == 0] = 1.0
M_prob = W / row_sums
print(f"M_prob diagonal: {np.diag(M_prob)[:5]} (should be 0)")
print(f"M_prob row sums: {M_prob.sum(axis=1)[:5]} (should be ~1.0)")

# ── Step 4: Cosine Similarity ──────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity
S = cosine_similarity(M_prob)
print(f"S shape={S.shape}, diag all 1={np.allclose(np.diag(S), 1.0)}")

# ── Step 5: Build graph (S > THRESHOLD, exclude diagonal) ─
G = nx.Graph()
n = S.shape[0]
G.add_nodes_from(range(n))

i_idx, j_idx = np.triu_indices(n, k=1)
mask = S[i_idx, j_idx] > THRESHOLD
edges = list(zip(i_idx[mask], j_idx[mask]))
G.add_edges_from(edges)

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
isolated = list(nx.isolates(G))
print(f"Isolated (no edge): {len(isolated)} nodes")

# Remove isolated nodes for cleaner visualization
G.remove_nodes_from(isolated)
print(f"After removing isolated — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

if G.number_of_edges() == 0:
    print("No edges above threshold.")
    plt.figure(figsize=(12, 12))
    plt.text(0.5, 0.5, f"No edges at threshold={THRESHOLD}", ha='center', va='center', fontsize=16)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close()
else:
    # ── Step 6: Layout ─────────────────────────────────────
    n_nodes = G.number_of_nodes()
    print(f"Computing layout for {n_nodes} nodes...")

    if n_nodes < 300:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # ── Step 7: Draw ───────────────────────────────────────
    plt.figure(figsize=(16, 16))
    fig_facecolor = '#1a1a1a'
    ax = plt.gca()
    ax.set_facecolor(fig_facecolor)

    degrees = dict(G.degree())
    node_sizes = [30 + degrees[n] * 8 for n in G.nodes()]
    edge_weights = [S[u, v] for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos,
                           nodelist=list(G.nodes()),
                           node_size=node_sizes,
                           node_color='#00e5ff',
                           alpha=0.85,
                           ax=ax)

    nx.draw_networkx_edges(G, pos,
                           edge_color='#ff9800',
                           width=[0.5 + (w - THRESHOLD) * 2 for w in edge_weights],
                           alpha=0.6,
                           ax=ax)

    plt.title(f'Similarity Graph (S > {THRESHOLD})  |  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}',
              color='white', fontsize=14)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight', facecolor=fig_facecolor)
    plt.close()
    print(f"Saved: {OUT_PNG}")

print("Done.")