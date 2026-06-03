"""
Test CosineSimilarity.compute() performance bottleneck with smaller sizes.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '/home/ubuntu/repos/SLBHS')

from SLBHS.similarity import TransitionCounter, CosineSimilarity

# Build matrix
n_frames = 10000
n_classes = 1024
labels = np.random.randint(0, n_classes, n_frames)
hand_labels = np.random.choice(['L', 'R'], n_frames)

t0 = time.time()
counter = TransitionCounter(k=n_classes, delta_t=1)
counter.fit(labels, hand_labels)
C = counter.get_matrix()
t1 = time.time()
nnz = np.sum(C > 0)
print(f"Build transition matrix: {t1-t0:.3f}s, nnz={nnz}")

sim = CosineSimilarity()

# Test incrementally
for size in [8, 16, 32, 64, 128]:
    C_sub = C[:size, :size]
    t0 = time.time()
    S = sim.compute(C_sub)
    t1 = time.time()
    print(f"  {size}x{size}: {t1-t0:.3f}s")
    if size == 128:
        print(f"  (1024x1024 would be ~{(t1-t0) * (1024/128)**2:.0f}s based on O(k^2))")
