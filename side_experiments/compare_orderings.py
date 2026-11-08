"""
Compares angular-continuity ordering against Chinese Postman ordering on
the same skeleton, using two concrete metrics:

1. Total angular variation: sum of direction-change angles along each
   stroke. Lower = smoother, more plausible as a single continuous pen
   motion.

2. Retracing: Chinese Postman duplicates edges to force an Eulerian
   path/circuit when odd-degree nodes exist. Every duplicated edge is a
   pixel-length of ink the pen would have to re-trace that never
   physically happened.
"""

import numpy as np
from skeleton_angular_experiment import (
    binarize_sauvola, thin_guo_hall, segment_rows, prune_short_branches,
    traverse_angular_continuity,
)
from skeleton_cpp_baseline import build_networkx_graph, chinese_postman_order
from skan import Skeleton
import cv2


def angular_variation(path_coords: list) -> float:
    """Sum of |angle change| in radians between consecutive segments."""
    if len(path_coords) < 3:
        return 0.0
    pts = np.array(path_coords, dtype=float)
    vecs = np.diff(pts, axis=0)
    norms = np.linalg.norm(vecs, axis=1)
    valid = norms > 1e-6
    vecs = vecs[valid] / norms[valid, None]

    total = 0.0
    for i in range(len(vecs) - 1):
        cos_a = np.clip(np.dot(vecs[i], vecs[i + 1]), -1.0, 1.0)
        total += np.arccos(cos_a)
    return total


def run(input_path: str):
    gray = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    binary = binarize_sauvola(gray)
    thinned = thin_guo_hall(binary)
    row_bands = segment_rows(thinned) or [(0, thinned.shape[0] - 1)]

    angular_strokes = []
    cpp_strokes = []
    cpp_retraced_edges = 0
    cpp_original_edges = 0

    for (r_start, r_end) in row_bands:
        row_mask = np.zeros_like(thinned)
        row_mask[r_start:r_end + 1, :] = thinned[r_start:r_end + 1, :]
        if row_mask.sum() < 10:
            continue

        skeleton = Skeleton(row_mask)
        pruned = prune_short_branches(skeleton, min_length=5)
        if pruned.sum() < 10:
            continue

        pruned_skeleton = Skeleton(pruned)

        strokes = traverse_angular_continuity(pruned_skeleton, pruned)
        angular_strokes.extend(strokes)

        nxg = build_networkx_graph(pruned_skeleton)
        cpp_original_edges += nxg.number_of_edges()
        node_paths = chinese_postman_order(nxg)
        coords = pruned_skeleton.coordinates
        for node_path in node_paths:
            cpp_strokes.append([tuple(coords[n]) for n in node_path])
            edges_in_path = len(node_path) - 1
            cpp_retraced_edges += max(0, edges_in_path - nxg.number_of_edges())

    angular_total_variation = sum(angular_variation(s) for s in angular_strokes)
    cpp_total_variation = sum(angular_variation(s) for s in cpp_strokes)

    print("=== Comparison ===")
    print(f"Angular-continuity: {len(angular_strokes)} strokes, "
          f"total angular variation = {angular_total_variation:.2f} rad"i
iiiiiiiii(f"Chinese Postman:    {len(cpp_strokes)} strokes, "
          f"total angular variation = {cpp_total_variation:.2f} rad")
    print(f"CPP retraced edges (pen movement that never physically occurred): "
          f"~{cpp_retraced_edges} / {cpp_original_edges} original edges")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "test_cursive.png")
