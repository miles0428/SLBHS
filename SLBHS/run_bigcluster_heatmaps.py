"""
BigClusterPipeline heatmap generation script
Runs delta_t in [1, 3, 5] and generates heatmaps
"""
import sys, os
import numpy as np
import h5py
import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("heatmap_gen")

# ── paths ──────────────────────────────────────────────────────────────
H5 = "/home/ubuntu/.openclaw/media/inbound/2022_12_12_14_00_中央流行疫情指揮中心嚴重特殊傳染性肺炎記者會_crop---87414a0f-15ac-4b38-8710-15c43fa52793.h5"
RESULTS_DIR = "/home/ubuntu/.openclaw/workspace-coding/SLBHS/SLBHS/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from SLBHS.clustering.super_cluster_pipeline import BigClusterPipeline

# ── load H5 ───────────────────────────────────────────────────────────
logger.info("Loading H5...")
with h5py.File(H5, "r") as f:
    X   = f["aligned_63d"][:]
    xv  = f["x_vec"][:]
    yv  = f["y_vec"][:]
    zv  = f["z_vec"][:]
logger.info(f"X={X.shape}, x_vec={xv.shape}")

# ── params ───────────────────────────────────────────────────────────
k             = 512
delta_ts      = [1, 3, 5]
tau           = 0.9
cosine_feats  = True
symmetrize    = True

# ── run pipeline for each delta_t ────────────────────────────────────
C_matrices = {}  # delta_t → C matrix
results_summary = {}

for dt in delta_ts:
    logger.info(f"\n{'='*60}\nRunning BigClusterPipeline delta_t={dt}\n{'='*60}")
    pipeline = BigClusterPipeline(
        k=k, tau=tau, delta_t=dt,
        cosine_features=cosine_feats,
        symmetrize=symmetrize,
        results_dir=RESULTS_DIR
    )
    pipeline.fit(
        X, xv, yv, zv,
        cosine_features=cosine_feats,
        symmetrize=symmetrize
    )

    C = pipeline.transition_counter.get_matrix()
    S = pipeline.similarity_matrix.S
    n_clusters = pipeline.big_clusterer.n_clusters

    C_matrices[dt] = C
    results_summary[dt] = {
        "C_nnz":  int(np.sum(C > 0)),
        "C_max":  float(C.max()),
        "N_clusters": n_clusters,
        "S_nan": int(np.sum(np.isnan(S))),
        "k": pipeline._k_used,
        "cosine_features": pipeline.cosine_features,
        "symmetrize": symmetrize,
        "tau": tau,
        "delta_t": dt
    }
    logger.info(f"  delta_t={dt}: C.nnz={results_summary[dt]['C_nnz']}, "
                f"N_clusters={n_clusters}")

# ── save C matrices for each delta_t ────────────────────────────────
for dt, C in C_matrices.items():
    np.save(Path(RESULTS_DIR) / f"transition_matrix_dt{dt}.npy", C)

# ── save S (cosine similarity) once ─────────────────────────────────
S = pipeline.similarity_matrix.S
np.save(Path(RESULTS_DIR) / "similarity_matrix.npy", S)

# ── generate heatmaps ────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_heatmap(M, path, title, log_scale=False, diagonal_zero=True):
    """Plot symmetric heatmap, zero diagonal."""
    if diagonal_zero:
        M = M.copy()
        np.fill_diagonal(M, 0)
    vmax = M.max() if M.max() > 0 else 1
    plt.figure(figsize=(10, 8))
    if log_scale:
        # avoid log(0)
        M_plot = np.ma.log10(M + 1e-12)
        plt.imshow(M_plot, cmap="viridis", interpolation="nearest")
        plt.colorbar(label="log10(value)")
    else:
        plt.imshow(M, cmap="viridis", interpolation="nearest", vmin=0, vmax=vmax)
        plt.colorbar(label="value")
    plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {path}")

# C heatmaps for each delta_t
for dt, C in C_matrices.items():
    plot_heatmap(C, Path(RESULTS_DIR) / f"C_heatmap_dt{dt}.png",
                 f"Transition Matrix C (delta_t={dt}, log scale)",
                 log_scale=True, diagonal_zero=True)

# S heatmap (cosine similarity) — no log scale, keep values 0-1
M_s = S.copy()
np.fill_diagonal(M_s, 0)
plt.figure(figsize=(10, 8))
plt.imshow(M_s, cmap="magma", interpolation="nearest", vmin=0, vmax=1)
plt.colorbar(label="cosine similarity")
plt.title("Cosine Similarity Matrix S")
plt.savefig(Path(RESULTS_DIR) / "S_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"  Saved: {RESULTS_DIR}/S_heatmap.png")

# M heatmap (symmetrized W = (C+C.T)/2) for last delta_t
W = (C + C.T) / 2.0
np.fill_diagonal(W, 0)
W_log = np.ma.log10(W + 1e-12)
plt.figure(figsize=(10, 8))
plt.imshow(W_log, cmap="viridis", interpolation="nearest")
plt.colorbar(label="log10(symmetrized W)")
plt.title(f"Symmetrized W Matrix (delta_t={dt}, log scale)")
plt.savefig(Path(RESULTS_DIR) / "M_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
logger.info(f"  Saved: {RESULTS_DIR}/M_heatmap.png")

# ── final summary ─────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for dt, info in results_summary.items():
    print(f"  delta_t={dt}: C.nnz={info['C_nnz']}, N_clusters={info['N_clusters']}, "
          f"S.nan={info['S_nan']}, k={info['k']}")
print(f"\nHeatmaps saved to: {RESULTS_DIR}/")
for f in sorted(Path(RESULTS_DIR).glob("*heatmap*.png")):
    print(f"  {f.name}")
