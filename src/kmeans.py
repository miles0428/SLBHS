"""
TWSLT K-Means Clustering
對 aligned_63d landmark 資料做 K-Means 聚類
"""

import h5py
import numpy as np
import os
import glob
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


def load_all_features(data_dir, sample_per_file=10000, max_files=None):
    """
    載入所有 H5 檔案的 aligned_63d 特徵
    sample_per_file: 每個檔案最多取多少幀（記憶體考量）
    """
    # Try both with and without .h5 extension
    files_h5 = sorted(glob.glob(os.path.join(data_dir, '*.h5')))
    files_noext = sorted(glob.glob(os.path.join(data_dir, '*_crop---*')))
    files = sorted(set(files_h5 + files_noext))
    if max_files:
        files = files[:max_files]

    all_features = []
    all_labels = []  # (filename, frame_idx)
    file_counts = []

    for fpath in files:
        with h5py.File(fpath, 'r') as f:
            data = f['aligned_63d'][:]
            n = len(data)

            if n > sample_per_file:
                # 均勻取樣
                indices = np.linspace(0, n-1, sample_per_file, dtype=int)
                data = data[indices]

            all_features.append(data)
            fname = Path(fpath).stem
            all_labels.extend([fname] * len(data))
            file_counts.append(len(data))

    X = np.vstack(all_features).astype(np.float32)
    labels = np.array(all_labels)

    print(f'載入完成: {len(files)} 個檔案, 共 {len(X)} 幀')
    print(f'維度: {X.shape[1]} (21 joints × 3)')
    print(f'每檔案幀數: {file_counts}')

    return X, labels, files, file_counts


def find_optimal_k(X, k_range=(5, 50), step=5, sample_size=20000):
    """
    用多個指標找最佳 K 值
    - Inertia (Elbow Method)
    - Silhouette Score
    - Calinski-Harabasz Index
    - Davies-Bouldin Index
    """
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    results = []

    for k in range(k_range[0], k_range[1] + 1, step):
        print(f'  測試 k={k}...', end=' ')
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3)
        labels = km.fit_predict(X_sample)

        inertia = km.inertia_
        sil = silhouette_score(X_sample, labels, sample_size=min(5000, len(X_sample)))
        ch = calinski_harabasz_score(X_sample, labels)
        db = davies_bouldin_score(X_sample, labels)

        results.append({
            'k': k,
            'inertia': inertia,
            'silhouette': sil,
            'calinski_harabasz': ch,
            'davies_bouldin': db
        })
        print(f'sil={sil:.4f}, ch={ch:.1f}, db={db:.4f}')

    return results


def run_kmeans(X, k, random_state=42):
    """
    執行 K-Means
    """
    print(f'執行 K-Means (k={k})...')
    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=4096,
        n_init=10,
        max_iter=300
    )
    labels = km.fit_predict(X)
    centers = km.cluster_centers_
    inertia = km.inertia_

    print(f'  完成。Inertia: {inertia:.2f}')

    # 計算各 cluster 大小
    unique, counts = np.unique(labels, return_counts=True)
    print(f'  Cluster 大小分佈:')
    for c, n in zip(unique, counts):
        print(f'    Cluster {c:3d}: {n:6d} ({n/len(labels)*100:5.1f}%)')

    return km, labels, centers


def save_results(km, labels, centers, output_dir):
    """
    儲存聚類結果
    """
    import joblib
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    np.save(os.path.join(output_dir, 'centers.npy'), centers)
    joblib.dump(km, os.path.join(output_dir, 'kmeans_model.pkl'))

    print(f'結果儲存至: {output_dir}')


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description='TWSLT K-Means Clustering')
    parser.add_argument('--data_dir', type=str, default='~/手語資料/features_h5',
                        help='H5 檔案目錄')
    parser.add_argument('--output_dir', type=str, default='~/手語資料/kmeans_results',
                        help='結果輸出目錄')
    parser.add_argument('--k', type=int, default=None,
                        help='直接指定 K 值（若不指定則搜尋最佳）')
    parser.add_argument('--k_range', type=str, default='5,50',
                        help='K 值搜尋範圍 (start,end)')
    parser.add_argument('--k_step', type=int, default=5,
                        help='K 值搜尋步進')
    parser.add_argument('--sample_per_file', type=int, default=10000,
                        help='每檔案最大取樣幀數')
    parser.add_argument('--max_files', type=int, default=None,
                        help='最多處理幾個檔案')
    parser.add_argument('--random_state', type=int, default=42)
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 1. 載入資料
    print('=== 載入資料 ===')
    X, labels, files, file_counts = load_all_features(
        data_dir, sample_per_file=args.sample_per_file, max_files=args.max_files
    )

    # 2. 標準化
    print('\n=== 標準化 ===')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f'標準化完成: mean={X_scaled.mean():.4f}, std={X_scaled.std():.4f}')

    # 3. 找最佳 K 或直接跑指定 K
    if args.k:
        km, cluster_labels, centers = run_kmeans(X_scaled, args.k, args.random_state)
        results = None
    else:
        print('\n=== 搜尋最佳 K ===')
        k_start, k_end = map(int, args.k_range.split(','))
        results = find_optimal_k(X_scaled, k_range=(k_start, k_end), step=args.k_step)
        print('\n=== 建議 ===')
        best_sil = max(results, key=lambda r: r['silhouette'])
        best_db = min(results, key=lambda r: r['davies_bouldin'])
        print(f'  最高 silhouette: k={best_sil["k"]} ({best_sil["silhouette"]:.4f})')
        print(f'  最低 davies_bouldin: k={best_db["k"]} ({best_db["davies_bouldin"]:.4f})')
        print('\n執行 --k 參數以指定 K 值並儲存結果')
        return

    # 4. 儲存
    print('\n=== 儲存結果 ===')
    save_results(km, cluster_labels, centers, output_dir)

    # 5. 儲存 config
    config = {
        'data_dir': data_dir,
        'output_dir': output_dir,
        'k': args.k,
        'n_frames': len(X),
        'n_files': len(files),
        'sample_per_file': args.sample_per_file,
        'random_state': args.random_state,
        'results': results
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print('\n完成！')


if __name__ == '__main__':
    main()
