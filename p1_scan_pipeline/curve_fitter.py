from dataclasses import dataclass
from typing import Protocol
import numpy as np
import cv2
from skimage.morphology import skeletonize, medial_axis
from scipy.ndimage import distance_transform_edt, binary_dilation
from scipy.interpolate import splprep, splev


@dataclass(eq=False)
class Curve:
    dense_points: np.ndarray        # N x 2, evenly-sampled points -- for visualization
    spline_params: tuple | None     # (t, c, k) from splprep -- None only if degenerate
    origin_offset: tuple[int, int]  # page-coordinate placement, inherited from Component
    degenerate: bool = False        # True = single-point segment, no real spline fit


@dataclass(frozen=True)
class RecenterParams:
    """Controls used by the hybrid/prune/recenter experiment pipeline.

    Kept separate from the public function defaults so an experiment can vary
    related values together while existing callers retain their behaviour.
    Values expressed as ratios are scaled by the component half-width.
    """
    hybrid_width_percentile: float = 73.23055871712805
    hybrid_dilation_iterations: int = 1
    max_spur_length: int = 2
    max_shift_ratio: float = 0.31080310858995536
    tangent_lookback: int = 3
    ray_limit_ratio: float = 1.9301960287900635
    red_profile_radius: int = 2
    displacement_profile_radius: int = 1
    min_red_region_area_factor: float = 1.270458780815832
    outline_red_dilation_extra: int = 0
    existing_skeleton_moat: int = 1
    allowed_new_red_pixels: int = 0


class CenterlineExtractor(Protocol):
    def extract(self, mask: np.ndarray) -> np.ndarray:
        """mask: uint8 ink blob (0/255) -> uint8 skeleton (0/255), same shape, 1px wide."""
        ...


class ZhangSuenExtractor:
    """Default centerline extractor. Thin wrapper around skimage's
    skeletonize(), which implements Zhang-Suen thinning -- the same
    algorithm confirmed safe (doesn't introduce gaps) during the old P1
    postmortem."""

    def extract(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D binary array")
        if mask.dtype != np.uint8:
            raise ValueError("mask must be dtype uint8 (0/255)")

        if not np.any(mask):
            return np.zeros_like(mask)

        skeleton_bool = skeletonize(mask > 0)
        return (skeleton_bool.astype(np.uint8)) * 255


class MedialAxisExtractor:
    """Alternative centerline extractor, for comparison against Zhang-Suen.
    Based on the distance transform: finds the ridge of points locally
    farthest from the ink boundary, rather than iterative pixel thinning.
    Tends to behave differently at tapered stroke ends -- worth comparing
    directly on real handwriting rather than assuming either is better."""

    def extract(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D binary array")
        if mask.dtype != np.uint8:
            raise ValueError("mask must be dtype uint8 (0/255)")

        if not np.any(mask):
            return np.zeros_like(mask)

        skeleton_bool = medial_axis(mask > 0)
        return (skeleton_bool.astype(np.uint8)) * 255


class HybridExtractor:
    """
    Combines both approaches based on local stroke width: Zhang-Suen is
    cleaner on simple, uniform-width strokes; medial axis is more
    geometrically accurate at junctions and loop interiors (which are
    locally WIDER than straight strokes). Uses the distance transform to
    find those wide regions, takes medial axis pixels there and
    Zhang-Suen pixels elsewhere, then re-skeletonizes the combined
    result once to guarantee a clean, fully-connected 1px-wide output
    (the two source skeletons don't perfectly align pixel-for-pixel, so
    this cleanup step is required, not optional).
    """

    def __init__(self, width_percentile: float = 73.23055871712805,
                 dilation_iterations: int = 1):
        self.width_percentile = width_percentile
        self.dilation_iterations = dilation_iterations
        self._zs = ZhangSuenExtractor()
        self._ma = MedialAxisExtractor()

    def extract(self, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D binary array")
        if mask.dtype != np.uint8:
            raise ValueError("mask must be dtype uint8 (0/255)")
        if not np.any(mask):
            return np.zeros_like(mask)

        dist = distance_transform_edt(mask > 0)
        ink_distances = dist[mask > 0]
        threshold = np.percentile(ink_distances, self.width_percentile)

        wide_region = dist >= threshold
        wide_region = binary_dilation(wide_region, iterations=self.dilation_iterations)

        skel_zs = self._zs.extract(mask) > 0
        skel_ma = self._ma.extract(mask) > 0

        hybrid = np.where(wide_region, skel_ma, skel_zs)
        if not np.any(hybrid):
            return np.zeros_like(mask)

        hybrid_clean = skeletonize(hybrid)  # fixes seams/doubled-width at region boundaries
        return (hybrid_clean.astype(np.uint8)) * 255


def _neighbor_count(skeleton: np.ndarray, y: int, x: int) -> int:
    """Counts 8-connected skeleton neighbors of pixel (y, x)."""
    h, w = skeleton.shape
    count = 0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx] > 0:
                count += 1
    return count


def _get_skeleton_neighbors(skel_set: set, y: int, x: int) -> list:
    """8-connected skeleton neighbors of (y, x), using a set for O(1) lookup."""
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            p = (y + dy, x + dx)
            if p in skel_set:
                neighbors.append(p)
    return neighbors


def _find_branch_and_endpoints(skeleton: np.ndarray) -> tuple[set, set]:
    """
    branch_points: pixels with 3+ skeleton neighbors (junctions, loop attachments)
    endpoints: pixels with exactly 1 skeleton neighbor (stroke tips)
    """
    ys, xs = np.where(skeleton > 0)
    skel_set = set(zip(ys.tolist(), xs.tolist()))

    branch_points, endpoints = set(), set()
    for p in skel_set:
        degree = len(_get_skeleton_neighbors(skel_set, *p))
        if degree == 1:
            endpoints.add(p)
        elif degree >= 3:
            branch_points.add(p)

    return branch_points, endpoints


def _split_into_segments(skeleton: np.ndarray, branch_points: set, endpoints: set) -> list:
    """
    Splits a skeleton into branch-free paths. Each returned array is an
    ordered sequence of (y, x) points from one node to another node --
    which may be the SAME node (a loop attached to the rest of the shape,
    e.g. the top of a cursive 'b' or 'o'). Fully isolated closed loops
    (no branch/endpoint anywhere) are handled separately at the end.

    Returns: list of np.ndarray, each shape (N, 2).
    """
    ys, xs = np.where(skeleton > 0)
    skel_set = set(zip(ys.tolist(), xs.tolist()))
    nodes = branch_points | endpoints

    visited_edges = set()
    segments = []

    def walk(start, first_step):
        path = [start, first_step]
        visited_edges.add(frozenset((start, first_step)))
        prev, current = start, first_step

        while current not in nodes:
            candidates = [n for n in _get_skeleton_neighbors(skel_set, *current) if n != prev]
            if len(candidates) != 1:
                break  # dead end or ambiguity -- stop safely rather than guess
            nxt = candidates[0]
            visited_edges.add(frozenset((current, nxt)))
            prev, current = current, nxt
            path.append(current)

        return path

    for node in nodes:
        for neighbor in _get_skeleton_neighbors(skel_set, *node):
            edge = frozenset((node, neighbor))
            if edge in visited_edges:
                continue
            segments.append(np.array(walk(node, neighbor)))

    # fully isolated closed loops: pixels no node-based walk ever reached
    used_pixels = {tuple(pt) for seg in segments for pt in seg}
    remaining = skel_set - used_pixels

    while remaining:
        start = next(iter(remaining))
        path = [start]
        remaining.discard(start)
        prev, current = None, start

        while True:
            candidates = [n for n in _get_skeleton_neighbors(skel_set, *current) if n != prev]
            if not candidates:
                break  # isolated single pixel
            nxt = candidates[0]
            if nxt == start:
                path.append(start)  # close the loop
                break
            path.append(nxt)
            remaining.discard(nxt)
            prev, current = current, nxt

        segments.append(np.array(path))

    return segments


class NoiseSuppressor(Protocol):
    def suppress(self, points: np.ndarray) -> np.ndarray:
        """points: N x 2 ordered raw points -> fewer, denoised N' x 2 points."""
        ...


def _perpendicular_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    """Distance from `point` to the line segment start-end (used by RDP)."""
    if np.array_equal(start, end):
        return float(np.linalg.norm(point - start))
    line_vec = end - start
    point_vec = point - start
    line_len = np.linalg.norm(line_vec)
    proj = np.dot(point_vec, line_vec) / (line_len ** 2)
    proj = np.clip(proj, 0.0, 1.0)
    closest = start + proj * line_vec
    return float(np.linalg.norm(point - closest))


def _rdp(points: np.ndarray, epsilon: float) -> np.ndarray:
    """Standard recursive Ramer-Douglas-Peucker simplification."""
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    max_dist = 0.0
    max_idx = 0

    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], start, end)
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > epsilon:
        left = _rdp(points[:max_idx + 1], epsilon)
        right = _rdp(points[max_idx:], epsilon)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


class RDPSuppressor:
    """Default noise suppressor. Strips pixel-level jitter from a raw
    centerline walk before spline fitting -- this is the main lever
    against the squiggly-line problem: fitting a spline through every
    noisy pixel produces a wobbly curve, RDP first reduces the point
    count to only the geometrically significant ones."""

    def __init__(self, epsilon: float = 2.0):
        self.epsilon = epsilon  # max allowed deviation, in pixels

    def suppress(self, points: np.ndarray) -> np.ndarray:
        if len(points) < 3:
            return points  # nothing to simplify
        return _rdp(points, self.epsilon)


def prune_spurs(skeleton: np.ndarray, max_spur_length: int = 2) -> np.ndarray:
    """
    Removes short dangling branches (spurs) from a skeleton -- both at
    outer stroke tips (thinning-artifact curls) and near true branch
    points (boundary-noise forks at junctions like a cursive 't' or
    'b'-'e' connection). Iterative: removing one spur can expose a new
    short one, so this repeats until no more short spurs remain or
    max_spur_length iterations pass (safety cap, not a tuning parameter).

    skeleton: uint8, 0/255, 1px wide.
    max_spur_length: spurs shorter than this (in pixels) are removed.
                      Longer branches are assumed to be real strokes,
                      not artifacts, and are left alone.
    """
    skel = (skeleton > 0).copy()

    for _ in range(max_spur_length):
        ys, xs = np.where(skel)
        if len(ys) == 0:
            break

        degree = {}
        for y, x in zip(ys, xs):
            degree[(y, x)] = _neighbor_count(skel, y, x)

        endpoints = [p for p, d in degree.items() if d == 1]
        if not endpoints:
            break  # closed loop or no endpoints left -- nothing more to prune

        removed_any = False
        for start in endpoints:
            if not skel[start]:
                continue  # already removed as part of an earlier spur this pass

            path = [start]
            current = start
            prev = None

            for _ in range(max_spur_length + 1):
                y, x = current
                neighbors = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if (ny, nx) != prev and 0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1] and skel[ny, nx]:
                            neighbors.append((ny, nx))

                if len(neighbors) != 1:
                    break

                prev = current
                current = neighbors[0]
                path.append(current)

                if _neighbor_count(skel, current[0], current[1]) >= 3:
                    break

            if len(path) <= max_spur_length and _neighbor_count(skel, current[0], current[1]) >= 3:
                for (py, px) in path[:-1]:
                    skel[py, px] = False
                removed_any = True

        if not removed_any:
            break

    return (skel.astype(np.uint8)) * 255


def estimate_stroke_width(mask: np.ndarray, skeleton: np.ndarray) -> int:
    """
    ONE representative stroke-width radius for the whole component,
    used to uniformly dilate the skeleton for fidelity comparison.

    Deliberately NOT sampled per-pixel at each skeleton point -- that
    would trivially "succeed," since every skeleton pixel sits inside
    real ink by construction and would always get a locally-correct
    radius there regardless of whether the skeleton's PATH is actually
    well-centered. A single global width makes the test sensitive to
    real positional/path deviations instead of self-correcting around
    them.
    """
    dist = distance_transform_edt(mask > 0)
    ys, xs = np.where(skeleton > 0)
    if len(ys) == 0:
        return 1
    widths = dist[ys, xs]
    return max(1, int(round(np.median(widths))))


def reconstruct_from_skeleton(skeleton: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Reconstructs the filled shape by dilating the skeleton UNIFORMLY,
    using one representative stroke-width radius for the whole
    component (see estimate_stroke_width) -- like stroking a fixed-width
    pen along the traced path -- not by custom-fitting a radius at each
    individual point. Used purely for measuring skeleton fidelity, not
    part of the production fitting pipeline.
    """
    radius = estimate_stroke_width(mask, skeleton)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    skel_u8 = (skeleton > 0).astype(np.uint8) * 255
    recon = cv2.dilate(skel_u8, kernel)
    return recon


def skeleton_fidelity(mask: np.ndarray, skeleton: np.ndarray) -> dict:
    """
    Scores how faithfully a skeleton represents the original ink shape.

    Returns:
        iou: intersection-over-union between original mask and the
             skeleton's disk-reconstruction. 1.0 = perfect, lower = worse.
        missed_pixels: real ink the skeleton's reconstruction failed to
                        cover -- indicates a gap or a path that deviated
                        away from where the ink actually is.
        extra_pixels: reconstruction area with no real ink underneath --
                      indicates the skeleton path deviated outside the
                      real stroke (e.g. into a spur or noise artifact).
        reconstruction: the actual reconstructed image, for visualization.
    """
    recon = reconstruct_from_skeleton(skeleton, mask)
    mask_bool = mask > 0
    recon_bool = recon > 0

    intersection = np.sum(mask_bool & recon_bool)
    union = np.sum(mask_bool | recon_bool)
    iou = float(intersection / union) if union > 0 else 1.0

    missed = int(np.sum(mask_bool & ~recon_bool))
    extra = int(np.sum(recon_bool & ~mask_bool))

    return {
        "iou": iou,
        "missed_pixels": missed,
        "extra_pixels": extra,
        "reconstruction": recon,
    }


def render_fidelity_diff(mask: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
    """
    Color-coded diff for visual debugging:
      GREEN = mask and reconstruction agree (correctly represented ink)
      RED   = mask has ink, reconstruction missed it (skeleton gap/deviation)
      BLUE  = reconstruction claims ink that isn't really there (skeleton
              path went somewhere it shouldn't have)
    """
    result = skeleton_fidelity(mask, skeleton)
    recon_bool = result["reconstruction"] > 0
    mask_bool = mask > 0

    h, w = mask.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[mask_bool & recon_bool] = (0, 200, 0)      # green: agree
    diff[mask_bool & ~recon_bool] = (0, 0, 255)     # red: missed
    diff[~mask_bool & recon_bool] = (255, 0, 0)     # blue: extra

    return diff


def refine_skeleton(mask: np.ndarray, skeleton: np.ndarray,
                     max_iterations: int = 3, min_iou_improvement: float = 0.001,
                     max_spur_length: int = 6,
                     apply_correction_loop: bool = False) -> np.ndarray:
    """
    Iteratively improves a skeleton by using skeleton_fidelity() to find
    where it under-represents real ink (missed_pixels / "red" regions),
    then patches those exact spots with medial-axis skeleton pixels
    (which tend to sit closer to the true local centerline at junctions
    and loops). Re-thins and re-prunes after every patch, since adding
    raw medial-axis pixels can reintroduce short spurs of its own.

    Stops when: fidelity stops meaningfully improving, there's nothing
    left to patch, or max_iterations is reached (safety cap).

    Set ``apply_correction_loop`` to also run the newer red/blue geometry
    correction pass after the legacy medial-axis patching step.  It defaults
    to False so existing callers keep the original refinement behaviour.

    Works on the output of ANY CenterlineExtractor -- not tied to a
    specific one, same as prune_spurs.
    """
    current = skeleton.copy()
    ma_extractor = MedialAxisExtractor()
    ma_skel = ma_extractor.extract(mask) > 0

    prev_iou = skeleton_fidelity(mask, current)["iou"]

    for i in range(max_iterations):
        fid = skeleton_fidelity(mask, current)

        if fid["missed_pixels"] == 0:
            break  # nothing left to fix

        mask_bool = mask > 0
        recon_bool = fid["reconstruction"] > 0
        missed_region = mask_bool & ~recon_bool
        missed_region_dilated = binary_dilation(missed_region, iterations=1)

        patch_pixels = ma_skel & missed_region_dilated
        if not np.any(patch_pixels):
            break  # medial axis has nothing useful to offer here either -- stop

        combined = (current > 0) | patch_pixels
        current = skeletonize(combined)
        current = (current.astype(np.uint8)) * 255
        current = prune_spurs(current, max_spur_length=max_spur_length)

        new_iou = skeleton_fidelity(mask, current)["iou"]
        if new_iou - prev_iou < min_iou_improvement:
            break  # converged -- further iterations aren't helping
        prev_iou = new_iou

    if apply_correction_loop:
        return correct_skeleton_from_fidelity(
            mask, current, max_iterations=max_iterations,
            min_iou_improvement=min_iou_improvement,
            max_spur_length=max_spur_length,
        )
    return current


def _find_bulge_runs(mask: np.ndarray, skeleton: np.ndarray, width_ratio_threshold: float = 1.6):
    """
    Finds contiguous stretches of the skeleton passing through locally
    "bulged" ink -- places where real ink is noticeably wider than the
    component's overall stroke width. Most often this means two strands
    of a loop have pinched/merged together (very common in thin
    handwriting, where a loop's interior hole can vanish entirely).

    Returns (list of pixel-sets, global_half_width).
    """
    dist = distance_transform_edt(mask > 0)
    global_half_width = estimate_stroke_width(mask, skeleton)

    ys, xs = np.where(skeleton > 0)
    skel_set = set(zip(ys.tolist(), xs.tolist()))

    bulge_pixels = {p for p in skel_set if dist[p] > width_ratio_threshold * global_half_width}

    runs = []
    remaining = set(bulge_pixels)
    while remaining:
        start = next(iter(remaining))
        run = {start}
        frontier = [start]
        remaining.discard(start)
        while frontier:
            p = frontier.pop()
            for n in _get_skeleton_neighbors(bulge_pixels, *p):
                if n in remaining:
                    run.add(n)
                    remaining.discard(n)
                    frontier.append(n)
        runs.append(run)

    return runs, global_half_width


def _order_run(run_pixels: set) -> list:
    """Orders an unordered set of connected run pixels into a walkable path."""
    neighbors_map = {p: _get_skeleton_neighbors(run_pixels, *p) for p in run_pixels}
    endpoints = [p for p, n in neighbors_map.items() if len(n) == 1]
    start = endpoints[0] if endpoints else next(iter(run_pixels))

    ordered = [start]
    visited = {start}
    current = start
    while True:
        nbrs = [n for n in neighbors_map[current] if n not in visited]
        if not nbrs:
            break
        nxt = nbrs[0]
        ordered.append(nxt)
        visited.add(nxt)
        current = nxt
    return ordered


def _split_bulge_region(mask: np.ndarray, run_pixels: set, global_half_width: int):
    """
    Leaves the original line completely untouched (no shifting -- this
    is what was causing the self-intersecting loop artifacts). Instead:

      1. Find the red (uncovered) region next to the original line.
      2. ERODE that red region by the same global_half_width used
         everywhere else. Erosion guarantees the result is a SUBSET of
         the input -- so when this eroded shape is later skeletonized
         and re-inflated by that same global_half_width, it is
         mathematically guaranteed to land back inside the original red
         region, never outside it. No per-point width tracking needed,
         no overflow possible, because both directions (erode then
         later re-dilate) use the exact same fixed number.
      3. If the red is narrower than one full pen-width anywhere,
         erosion correctly removes that spot entirely -- a fixed-width
         pen genuinely cannot represent ink thinner than itself.

    Returns (path_a, path_b) as lists of (y, x) global-coordinate
    points, or None if there's no real second strand to add.
    """
    ordered = _order_run(run_pixels)
    pts = np.array(ordered)
    n = len(pts)
    if n < 2:
        return None

    ys, xs = pts[:, 0], pts[:, 1]
    pad = global_half_width * 3
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(mask.shape[0], int(ys.max()) + pad + 1)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(mask.shape[1], int(xs.max()) + pad + 1)

    local_mask = mask[y0:y1, x0:x1] > 0
    local_pts = pts - np.array([y0, x0])

    line_canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    for i in range(n - 1):
        p1 = tuple(local_pts[i][::-1])
        p2 = tuple(local_pts[i + 1][::-1])
        cv2.line(line_canvas, p1, p2, 255, 1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (global_half_width * 2 + 1, global_half_width * 2 + 1))
    recon_original = cv2.dilate(line_canvas, kernel) > 0

    red = local_mask & ~recon_original
    if not np.any(red):
        return None

    # side_a: the ORIGINAL line, completely unchanged
    side_a = line_canvas > 0

    # side_b: erode red by the same fixed pen radius, then take the
    # OUTLINE of whatever survives -- no thinning algorithm involved
    red_u8 = (red.astype(np.uint8)) * 255
    red_eroded = cv2.erode(red_u8, kernel)

    if not np.any(red_eroded):
        return None  # red patch is narrower than one pen-width everywhere -- nothing to add

    contours, _ = cv2.findContours(red_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    contour_pts = largest.reshape(-1, 2)  # cv2 gives (x, y)
    path_b_local = [(int(y), int(x)) for x, y in contour_pts]

    def _to_global_points(skel_local):
        pys, pxs = np.where(skel_local)
        return list(zip((pys + y0).tolist(), (pxs + x0).tolist()))

    path_a = _to_global_points(side_a) if np.any(side_a) else ordered
    path_b = [(y + y0, x + x0) for (y, x) in path_b_local]
    return path_a, path_b


def split_bulge_regions(mask: np.ndarray, skeleton: np.ndarray,
                         width_ratio_threshold: float = 1.6,
                         max_iterations: int = 1) -> np.ndarray:
    """
    Finds places where the skeleton passes through locally bulged ink
    (likely a pinched/merged loop) and adds a second line to cover the
    red (uncovered) region alongside it -- using erosion by the fixed
    global pen-width so the new line can never, by construction, extend
    beyond the red region it was built to cover.
    """
    working = (skeleton > 0).copy()

    for iteration in range(max_iterations):
        runs, global_half_width = _find_bulge_runs(mask, working, width_ratio_threshold)
        if not runs:
            break

        any_change = False
        for run in runs:
            result = _split_bulge_region(mask, run, global_half_width)
            if result is None:
                continue

            path_a, path_b = result
            for p in run:
                working[p] = False
            for p in path_a:
                working[p] = True
            for p in path_b:
                working[p] = True
            any_change = True

        working = skeletonize(working)

        if not any_change:
            break

    return (working.astype(np.uint8)) * 255


def _fit_smooth_closed_curve(points_yx: np.ndarray, num_samples: int = 200,
                              smoothing: float = None) -> np.ndarray:
    """
    Fits a PERIODIC (closed) SMOOTHING spline (s > 0) through ordered
    (y, x) points and samples it densely.

    Deliberately NOT an interpolating spline (s=0) -- forcing a curve
    through every point exactly, especially a SPARSE set of them (e.g.
    after aggressive RDP simplification), causes bad overshoot/bulging
    between distant points, particularly near genuine sharp corners
    where two strokes meet. A smoothing spline instead balances fidelity
    against smoothness using `smoothing`, which is far more stable.

    smoothing=None picks a reasonable default proportional to point
    count (scipy's own convention: roughly the number of points).
    """
    if len(points_yx) < 4:
        return points_yx

    pts = np.array(points_yx, dtype=float)

    keep = [0]
    for i in range(1, len(pts)):
        if not np.allclose(pts[i], pts[keep[-1]]):
            keep.append(i)
    pts = pts[keep]

    if len(pts) < 4:
        return points_yx

    y, x = pts[:, 0], pts[:, 1]
    if smoothing is None:
        smoothing = len(pts)  # scipy's own suggested starting heuristic

    try:
        k = 3 if len(pts) > 3 else len(pts) - 1
        tck, u = splprep([y, x], s=smoothing, per=1, k=k)
        u_dense = np.linspace(0, 1, num_samples)
        y_smooth, x_smooth = splev(u_dense, tck)
        return np.stack([y_smooth, x_smooth], axis=1)
    except Exception:
        return points_yx


def _turning_angles(pts: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Angle (radians) between the incoming and outgoing chord vectors at
    each point, using points `window` steps before/after (wrapping,
    since the contour is closed). Using a window rather than immediate
    neighbors suppresses pixel-level raster noise when estimating local
    direction -- a single-pixel step is too small a baseline to measure
    genuine curvature reliably. 0 = perfectly straight, pi = full
    reversal.
    """
    n = len(pts)
    angles = np.zeros(n)
    for i in range(n):
        prev_pt = pts[(i - window) % n]
        next_pt = pts[(i + window) % n]
        v1 = pts[i] - prev_pt
        v2 = next_pt - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            continue
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    return angles


def _detect_corners(pts: np.ndarray, window: int = 5, angle_threshold_deg: float = 45.0,
                     min_separation: int = 8) -> list:
    """
    Finds robust corner points: locations where the turning angle
    exceeds angle_threshold_deg, with non-max suppression so a cluster
    of nearby high-angle points (one real corner spread across a few
    pixels) collapses to a single representative corner instead of
    several.
    """
    angles = _turning_angles(pts, window=window)
    threshold = np.deg2rad(angle_threshold_deg)
    n = len(pts)

    candidates = [i for i in range(n) if angles[i] > threshold]
    if not candidates:
        return []

    candidates.sort(key=lambda i: -angles[i])
    kept = []
    for i in candidates:
        if all(min(abs(i - k), n - abs(i - k)) >= min_separation for k in kept):
            kept.append(i)

    return sorted(kept)


def _fit_smooth_closed_curve_preserving_corners(
    points_yx: np.ndarray, num_samples: int = 200, smoothing: float = None,
    corner_window: int = 5, corner_angle_threshold_deg: float = 45.0,
    corner_min_separation: int = 8,
) -> np.ndarray:
    """
    Smooths a closed contour while preserving genuine sharp corners.

    A single global smoothing value can't tell "this is noise, smooth
    it" apart from "this is a real corner, keep it" -- both RDP epsilon
    and a global spline's `s` trade off fidelity vs smoothness UNIFORMLY
    across the whole curve. This instead detects robust high-curvature
    points first (resistant to raster pixel noise via the windowed
    turning-angle check), keeps those fixed as hard breakpoints, and
    fits an independent smoothing spline on each arc BETWEEN corners --
    so noise gets smoothed out on the curved stretches while real
    corners stay sharp.
    """
    if len(points_yx) < 8:
        return points_yx

    pts = np.array(points_yx, dtype=float)
    keep_idx = [0]
    for i in range(1, len(pts)):
        if not np.allclose(pts[i], pts[keep_idx[-1]]):
            keep_idx.append(i)
    pts = pts[keep_idx]
    n = len(pts)
    if n < 8:
        return pts

    corners = _detect_corners(pts, window=corner_window,
                               angle_threshold_deg=corner_angle_threshold_deg,
                               min_separation=corner_min_separation)

    if not corners:
        return _fit_smooth_closed_curve(pts, num_samples=num_samples, smoothing=smoothing)

    all_samples = []
    for c_idx in range(len(corners)):
        start = corners[c_idx]
        end = corners[(c_idx + 1) % len(corners)]

        if end > start:
            arc = pts[start:end + 1]
        else:
            arc = np.vstack([pts[start:], pts[:end + 1]])  # wraps around the closure point

        if len(arc) < 4:
            all_samples.append(arc)
            continue

        y, x = arc[:, 0], arc[:, 1]
        arc_smoothing = smoothing if smoothing is not None else len(arc)
        try:
            k = 3 if len(arc) > 3 else len(arc) - 1
            tck, u = splprep([y, x], s=arc_smoothing, k=k)  # NOT periodic -- each arc is open, corners are its fixed ends
            samples_this_arc = max(4, int(num_samples * len(arc) / n))
            u_dense = np.linspace(0, 1, samples_this_arc)
            y_smooth, x_smooth = splev(u_dense, tck)
            all_samples.append(np.stack([y_smooth, x_smooth], axis=1))
        except Exception:
            all_samples.append(arc)

    return np.vstack(all_samples)


def extract_smooth_outlines(mask: np.ndarray, num_samples: int = 200,
                             smoothing: float = None,
                             corner_angle_threshold_deg: float = 45.0) -> dict:
    """
    Extracts smooth, 1-pixel-wide outline(s) of a binary shape, including
    the INNER outline of any true hole (e.g. a fully closed loop like an
    'o', or a 'b' whose loop never actually pinched shut). For your
    typical thin, pinched handwriting where the hole has collapsed
    entirely, "inner" will simply come back empty -- that's correct,
    not a bug, matching what you already confirmed about your loops.

    Pipeline: raw contour (every boundary pixel) -> corner-preserving
    smoothing (detects genuine sharp corners, keeps them fixed, smooths
    only the arcs between them) -> dense resample.

    Returns:
        {"outer": [ (y,x) point arrays, one per outer boundary ],
         "inner": [ (y,x) point arrays, one per hole -- empty if none ]}
    """
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    outer, inner = [], []
    if hierarchy is not None and len(contours) > 0:
        hierarchy = hierarchy[0]  # cv2 wraps it in an extra dimension
        for i, contour in enumerate(contours):
            pts_xy = contour.reshape(-1, 2)
            pts_yx = np.array([(int(y), int(x)) for x, y in pts_xy])

            smooth_curve = _fit_smooth_closed_curve_preserving_corners(
                pts_yx, num_samples=num_samples, smoothing=smoothing,
                corner_angle_threshold_deg=corner_angle_threshold_deg,
            )

            parent = hierarchy[i][3]  # -1 = outer boundary, else = it's a hole
            if parent == -1:
                outer.append(smooth_curve)
            else:
                inner.append(smooth_curve)

    return {"outer": outer, "inner": inner}


def to_origin_frame(points_yx: np.ndarray) -> tuple:
    """
    Transforms points into a coordinate frame anchored at the
    bottom-left of their bounding box (leftmost x, lowermost y in image
    terms) -- guarantees every transformed point is non-negative, and
    makes directional shifts simpler to reason about (pure positive
    quadrant, "up"/"right" both mean "bigger number").

    Returns (transformed_points, origin) where origin=(min_x, max_y_img)
    in original image coordinates, needed to convert back.
    """
    ys = points_yx[:, 0]
    xs = points_yx[:, 1]
    min_x = xs.min()
    max_y_img = ys.max()

    x_new = xs - min_x
    y_new = max_y_img - ys  # flip so "up" is positive, not image-down
    return np.stack([y_new, x_new], axis=1), (min_x, max_y_img)


def from_origin_frame(points_transformed: np.ndarray, origin: tuple) -> np.ndarray:
    """Inverse of to_origin_frame -- back to original image (y, x) coordinates."""
    min_x, max_y_img = origin
    y_new = points_transformed[:, 0]
    x_new = points_transformed[:, 1]
    ys = max_y_img - y_new
    xs = x_new + min_x
    return np.stack([ys, xs], axis=1)


def _reconstruct_with_local_widths(skeleton: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Rasterize a skeleton with one half-width per skeleton pixel.

    This is deliberately private: the existing public fidelity functions retain
    their fixed-width semantics.  The correction loop needs this local model so
    symmetric blue error can reduce width without moving a correctly centred
    line.
    """
    reconstruction = np.zeros(skeleton.shape, dtype=np.uint8)
    ys, xs = np.where(skeleton > 0)
    for y, x in zip(ys, xs):
        cv2.circle(reconstruction, (int(x), int(y)), max(1, int(round(widths[y, x]))), 255, -1)
    return reconstruction > 0


def _ray_ink_extent(mask_bool: np.ndarray, point: np.ndarray, direction: np.ndarray,
                    max_distance: float, step: float = 0.5) -> float | None:
    """Measure the contiguous ink run from ``point`` along one normal ray.

    A ray is intentionally one-dimensional.  Unlike the previous square
    neighbourhood measurement, it cannot mistake a nearby loop or a parallel
    stroke for error belonging to this particular centreline point.
    """
    h, w = mask_bool.shape
    last_inside = 0.0
    seen_ink = False
    for distance in np.arange(0.0, max_distance + step, step):
        y, x = np.rint(point + direction * distance).astype(int)
        if not (0 <= y < h and 0 <= x < w) or not mask_bool[y, x]:
            return last_inside if seen_ink else None
        seen_ink = True
        last_inside = distance
    return last_inside


def _smooth_profile(values: np.ndarray, radius: int) -> np.ndarray:
    """Edge-preserving moving-average smoothing for a segment profile."""
    if radius <= 0 or len(values) < 3:
        return values.copy()
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(2 * radius + 1, 1.0 / (2 * radius + 1))
    return np.convolve(padded, kernel, mode="valid")


def _smooth_open_path(points: np.ndarray, smoothing_per_point: float = 0.35) -> np.ndarray:
    """Return a dense, endpoint-preserving smoothing spline for one path.

    The skew is computed in floating point, but drawing its rounded raster
    samples still leaves stair-steps.  This final geometric smoothing happens
    before drawing, rather than trying to blur a path after it has become
    pixels again.
    """
    if len(points) < 6:
        return points
    keep = np.r_[True, np.any(np.diff(points, axis=0) != 0, axis=1)]
    pts = points[keep]
    if len(pts) < 6:
        return points
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if arc[-1] == 0:
        return points
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=arc / arc[-1],
                         s=len(pts) * smoothing_per_point, k=3)
        u = np.linspace(0.0, 1.0, max(len(pts) * 3, 24))
        y, x = splev(u, tck)
        smoothed = np.column_stack((y, x))
        smoothed[0], smoothed[-1] = pts[0], pts[-1]
        return smoothed
    except Exception:
        return points


def _shift_segments_from_normal_rays(mask: np.ndarray, skeleton: np.ndarray,
                                     fallback_half_width: float,
                                     max_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Re-centre complete paths from measured normal-ray boundary distances.

    For every point on a branch-free path, the two ray lengths provide a direct
    geometric estimate: offset is half their difference and local half-width
    is half their sum.  Those profiles are smoothed along arc length before
    moving any point.  Segment endpoints (including junction pixels) remain
    anchored, preventing corrections on one path from tearing apart a join.
    """
    mask_bool = mask > 0
    branches, ends = _find_branch_and_endpoints(skeleton)
    segments = _split_into_segments(skeleton, branches, ends)
    working = np.zeros(mask.shape, dtype=np.uint8)
    local_widths = np.zeros(mask.shape, dtype=float)
    ray_limit = max(2.0, fallback_half_width * 3.0)

    for segment in segments:
        count = len(segment)
        if count < 3:
            for y, x in segment:
                working[y, x] = 1
                local_widths[y, x] = fallback_half_width
            continue

        offsets = np.zeros(count, dtype=float)
        radii = np.full(count, fallback_half_width, dtype=float)
        normals = np.zeros((count, 2), dtype=float)
        valid = np.zeros(count, dtype=bool)
        for i, point in enumerate(segment):
            before, after = max(0, i - 3), min(count - 1, i + 3)
            tangent = (segment[after] - segment[before]).astype(float)
            tangent_length = np.linalg.norm(tangent)
            if tangent_length == 0:
                continue
            normal = np.array([-tangent[1], tangent[0]]) / tangent_length
            positive = _ray_ink_extent(mask_bool, point, normal, ray_limit)
            negative = _ray_ink_extent(mask_bool, point, -normal, ray_limit)
            if positive is None or negative is None:
                continue
            normals[i] = normal
            offsets[i] = np.clip((positive - negative) / 2.0, -max_shift, max_shift)
            radii[i] = max(1.0, (positive + negative) / 2.0)
            valid[i] = True

        # Do not blend missing measurements into valid ones.  This matters at
        # narrow tips and crossings, where the skeleton may already be outside
        # the mask by a fraction of a pixel.
        if np.any(valid):
            smooth_offsets = _smooth_profile(offsets, radius=3)
            smooth_radii = _smooth_profile(radii, radius=3)
            smooth_offsets[~valid] = 0.0
            smooth_radii[~valid] = fallback_half_width
        else:
            smooth_offsets, smooth_radii = offsets, radii

        shifted = segment.astype(float).copy()
        # Anchor both path ends.  They are endpoints or branch nodes and must
        # not be dragged sideways by a neighbouring segment's measurement.
        for i in range(1, count - 1):
            if valid[i]:
                shifted[i] += normals[i] * smooth_offsets[i]
        shifted = _smooth_open_path(shifted)
        shifted = np.rint(shifted).astype(int)
        shifted[:, 0] = np.clip(shifted[:, 0], 0, mask.shape[0] - 1)
        shifted[:, 1] = np.clip(shifted[:, 1], 0, mask.shape[1] - 1)
        cv2.polylines(working, [shifted[:, ::-1]], False, 1, 1)
        for point, radius in zip(shifted, smooth_radii):
            local_widths[tuple(point)] = max(local_widths[tuple(point)], radius)

    local_widths[working > 0] = np.where(local_widths[working > 0] > 0,
                                           local_widths[working > 0], fallback_half_width)
    return (skeletonize(working > 0).astype(np.uint8) * 255), local_widths


def _insert_missing_centerlines(mask: np.ndarray, skeleton: np.ndarray, reconstruction: np.ndarray,
                                half_width: float, min_region_area: int | None = None) -> np.ndarray:
    """Add medial-axis paths through sizeable red components far from a line.

    The medial axis of the original ink is retained only where it crosses an
    uncovered component and is at least roughly half a stroke-width from the
    boundary.  Thus an inserted path sits approximately half a width inside
    the ink rather than tracing the red component's outer edge.
    """
    red = (mask > 0) & ~reconstruction
    if min_region_area is None:
        min_region_area = max(4, int(np.pi * half_width * half_width / 2))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(skeleton)
    distance_to_line = distance_transform_edt(skeleton == 0)
    ink_distance = distance_transform_edt(mask > 0)
    medial = medial_axis(mask > 0)
    inserted = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] < min_region_area:
            continue
        if np.min(distance_to_line[component]) <= half_width:
            continue  # this red is a local alignment issue, handled by shifting
        safe_interior = ink_distance >= max(1.0, half_width * 0.5)
        candidates = medial & binary_dilation(component, iterations=1) & safe_interior
        inserted |= candidates
    return inserted.astype(np.uint8) * 255


def correct_skeleton_from_fidelity(mask: np.ndarray, skeleton: np.ndarray, max_iterations: int = 3,
                                   min_iou_improvement: float = 0.001,
                                   max_spur_length: int = 6) -> np.ndarray:
    """Iteratively correct position, local width, and missing centreline paths.

    Assumptions: masks are binary ink images; a single representative stroke
    half-width is a useful starting point; and new paths must be supported by a
    substantial uncovered ink region.  Width is an internal correction state:
    the returned value remains a normal 1-pixel skeleton, preserving callers'
    existing API.
    """
    current = (skeleton > 0).astype(np.uint8) * 255
    if not np.any(current) or not np.any(mask):
        return current
    half_width = float(estimate_stroke_width(mask, current))
    widths = np.zeros(mask.shape, dtype=float)
    widths[current > 0] = half_width
    previous_iou = skeleton_fidelity(mask, current)["iou"]

    for _ in range(max_iterations):
        shifted, widths = _shift_segments_from_normal_rays(
            mask, current, half_width, max_shift=half_width * 0.5
        )
        seeded_widths = widths.copy()
        seeded_widths[(shifted > 0) & (seeded_widths == 0)] = half_width
        reconstruction = _reconstruct_with_local_widths(shifted, seeded_widths)
        inserted = _insert_missing_centerlines(mask, shifted, reconstruction, half_width)
        combined = (shifted > 0) | (inserted > 0)
        candidate = skeletonize(combined).astype(np.uint8) * 255
        candidate = prune_spurs(candidate, max_spur_length=max_spur_length)
        new_iou = skeleton_fidelity(mask, candidate)["iou"]
        if new_iou < previous_iou:
            break  # retain the last known-good geometry
        improvement = new_iou - previous_iou
        # Re-thinning can move a pixel by one cell.  Carry the width from the
        # nearest shifted pixel, then initialise genuinely inserted paths.
        _, nearest = distance_transform_edt(shifted == 0, return_indices=True)
        candidate_widths = seeded_widths[nearest[0], nearest[1]]
        candidate_widths[candidate == 0] = 0
        current, widths, previous_iou = candidate, candidate_widths, new_iou
        widths[current > 0] = np.where(widths[current > 0] > 0,
                                       widths[current > 0], half_width)
        if improvement < min_iou_improvement and not np.any(inserted):
            break
    return current


def skew_left_red_first(mask: np.ndarray, skeleton: np.ndarray,
                        global_half_width: int = None, max_shift: float = None,
                        tangent_lookback: int = 2, ray_limit_ratio: float = 2.0,
                        red_profile_radius: int = 6,
                        displacement_profile_radius: int = 5) -> np.ndarray:
    """Diagnostic one-sided correction: move only toward red on the left.

    This deliberately ignores blue error, local width, and red on the right.
    ``left`` means the left-hand normal after each path is oriented consistently
    from lower-x to higher-x (or top-to-bottom for vertical paths).  It exists
    to make one proposed rule visible in isolation before combining it with any
    other correction mechanism.  By default it shifts affected points by one
    full estimated half-width; pass ``max_shift`` to make that explicit.
    """
    half_width = float(global_half_width or estimate_stroke_width(mask, skeleton))
    if max_shift is None:
        # This is intentionally an all-the-way one-sided experiment: a red
        # region found on the left may move its centreline by a whole pen
        # radius, rather than the earlier one-pixel diagnostic nudge.
        max_shift = half_width
    reconstruction = reconstruct_from_skeleton(skeleton, mask) > 0
    red = (mask > 0) & ~reconstruction
    branches, ends = _find_branch_and_endpoints(skeleton)
    working = np.zeros(mask.shape, dtype=np.uint8)
    ray_limit = max(2, int(np.ceil(half_width * ray_limit_ratio)))

    for segment in _split_into_segments(skeleton, branches, ends):
        if len(segment) < 3:
            for y, x in segment:
                working[y, x] = 1
            continue
        # _split_into_segments walks from an unordered set of graph nodes.  A
        # stable direction makes "left" mean the same thing on every pass.
        if (segment[-1, 1], segment[-1, 0]) < (segment[0, 1], segment[0, 0]):
            segment = segment[::-1]
        left_red_profile = np.zeros(len(segment), dtype=float)
        normals = np.zeros((len(segment), 2), dtype=float)
        for i in range(len(segment)):
            tangent = (segment[min(len(segment) - 1, i + tangent_lookback)] -
                       segment[max(0, i - tangent_lookback)]).astype(float)
            length = np.linalg.norm(tangent)
            if length == 0:
                continue
            left_normal = np.array([-tangent[1], tangent[0]]) / length
            normals[i] = left_normal
            found_red = False
            for distance in range(max(1, int(np.floor(half_width))), ray_limit + 1):
                y, x = np.rint(segment[i] + left_normal * distance).astype(int)
                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and red[y, x]:
                    found_red = True
                    break
            if found_red:
                left_red_profile[i] = 1.0

        # The raw decision is binary per raster pixel, which is what created
        # the saw-tooth paths.  Blur the *amount of movement* along the path,
        # then apply it once; this keeps the same left-red rule while making
        # neighbouring points move by compatible amounts.
        smooth_profile = _smooth_profile(left_red_profile, radius=red_profile_radius)
        raw_displacement = normals * (smooth_profile * max_shift)[:, None]
        # Raster tangents turn in one-pixel increments.  Smooth the vector,
        # not merely its magnitude, so that a tiny normal-direction flip does
        # not turn a otherwise smooth skew into a zig-zag.
        displacement = np.column_stack((
            _smooth_profile(raw_displacement[:, 0], radius=displacement_profile_radius),
            _smooth_profile(raw_displacement[:, 1], radius=displacement_profile_radius),
        ))
        shifted = segment.astype(float).copy()
        for i in range(len(segment)):
            if np.any(normals[i]):
                shifted[i] += displacement[i]
        shifted = _smooth_open_path(shifted)
        shifted = np.rint(shifted).astype(int)
        shifted[:, 0] = np.clip(shifted[:, 0], 0, mask.shape[0] - 1)
        shifted[:, 1] = np.clip(shifted[:, 1], 0, mask.shape[1] - 1)
        cv2.polylines(working, [shifted[:, ::-1]], False, 1, 1)
    # ``working`` is already made of explicit 1px polylines.  Re-thinning the
    # union can turn a harmless near-touch between separate segments into a
    # branch/joint, so preserve the rendered paths exactly.
    return (working > 0).astype(np.uint8) * 255


def _offset_outline_runs_near_red(ink_core: np.ndarray, red_region: np.ndarray,
                                  half_width: int,
                                  dilation_extra: int = 1) -> list[np.ndarray]:
    """Return ordered inward-outline arcs facing one remaining-red component.

    A contour is ordered by OpenCV.  Selecting contiguous runs from that order
    is crucial: masking a contour as an unordered bitmap and re-skeletonizing
    it was what joined nearby, otherwise independent lines into spaghetti.
    """
    near_red = binary_dilation(red_region, iterations=half_width + dilation_extra).astype(np.uint8)
    contours, _ = cv2.findContours(ink_core.astype(np.uint8), cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    runs = []
    for contour in contours:
        points_xy = contour.reshape(-1, 2)
        if len(points_xy) < 3:
            continue
        selected = np.array([near_red[y, x] > 0 for x, y in points_xy])
        if not np.any(selected):
            continue
        if np.all(selected):
            runs.append(points_xy)
            continue
        # Rotate after a non-selected point, making a wrapped contour behave
        # like a simple linear sequence for run extraction.
        start = int(np.flatnonzero(~selected)[0])
        points_xy = np.roll(points_xy, -(start + 1), axis=0)
        selected = np.roll(selected, -(start + 1))
        run_start = None
        for index, keep in enumerate(selected):
            if keep and run_start is None:
                run_start = index
            elif not keep and run_start is not None:
                if index - run_start >= 3:
                    runs.append(points_xy[run_start:index])
                run_start = None
        if run_start is not None and len(points_xy) - run_start >= 3:
            runs.append(points_xy[run_start:])
    return runs


def _reconstruct_with_fixed_half_width(skeleton: np.ndarray, half_width: int) -> np.ndarray:
    """Fixed-width reconstruction used for comparable candidate experiments."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (half_width * 2 + 1, half_width * 2 + 1))
    return cv2.dilate((skeleton > 0).astype(np.uint8) * 255, kernel) > 0


def _split_candidate_components(candidate: np.ndarray) -> list[np.ndarray]:
    """Keep every disconnected inserted arc as an independently testable path."""
    count, labels = cv2.connectedComponents((candidate > 0).astype(np.uint8), 8)
    paths = []
    for label in range(1, count):
        path = labels == label
        if np.any(path):
            paths.append(path)
    return paths


def _prune_paths_without_red_regression(mask: np.ndarray, base_skeleton: np.ndarray,
                                        candidate_paths: list[np.ndarray],
                                        half_width: int,
                                        allowed_new_red_pixels: int = 1) -> np.ndarray:
    """Greedily remove inserted paths that uniquely cover no real ink.

    Candidates are tested from smallest to largest.  A path is retained only
    when removing it would make at least ``allowed_new_red_pixels + 1`` ink
    pixels become newly red.  The fixed reconstruction width prevents a
    candidate removal from changing the measuring ruler itself.
    """
    active = [path.copy() for path in candidate_paths if np.any(path)]
    for index in sorted(range(len(active)), key=lambda i: np.count_nonzero(active[i])):
        if not np.any(active[index]):
            continue
        current = base_skeleton > 0
        for path in active:
            current |= path
        trial = base_skeleton > 0
        for other_index, path in enumerate(active):
            if other_index != index:
                trial |= path
        current_reconstruction = _reconstruct_with_fixed_half_width(current, half_width)
        trial_reconstruction = _reconstruct_with_fixed_half_width(trial, half_width)
        newly_red = int(np.sum((mask > 0) & current_reconstruction & ~trial_reconstruction))
        if newly_red <= allowed_new_red_pixels:
            active[index][:] = False

    kept = np.zeros(mask.shape, dtype=bool)
    for path in active:
        kept |= path
    return kept


def add_centerlines_in_missing_red(mask: np.ndarray, skeleton: np.ndarray,
                                   global_half_width: int = None,
                                   min_region_area: int = None,
                                   params: RecenterParams | None = None) -> np.ndarray:
    """Add paths rather than bending paths into unsupported red ink.

    A candidate centreline is made from the skeleton of the red region eroded
    by one pen radius.  Every candidate pixel is therefore at least a
    half-width inside the red boundary: re-outlining it with that same pen
    radius cannot spill outside the red region.  Existing skeleton pixels are
    never moved, which avoids the zig-zag produced by forced skewing.
    """
    half_width = int(round(global_half_width or estimate_stroke_width(mask, skeleton)))
    half_width = max(1, half_width)
    reconstruction = _reconstruct_with_fixed_half_width(skeleton, half_width)
    red = (mask > 0) & ~reconstruction
    params = params or RecenterParams()
    if min_region_area is None:
        # Remaining-red crescents can be narrow but long.  Their area is not
        # necessarily a complete pen disk, so do not discard them on that
        # basis alone.
        min_region_area = max(8, int(round(half_width * params.min_red_region_area_factor)))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(red.astype(np.uint8), 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (half_width * 2 + 1, half_width * 2 + 1))
    # This is the locus of points exactly one half-width inside the original
    # ink edge.  Stroking it by that same radius puts its outside outline on
    # the red/ink boundary while allowing its inside outline to overlap the
    # already reconstructed stroke.
    ink_core = cv2.erode((mask > 0).astype(np.uint8), kernel) > 0
    # OpenCV drawing requires an integer image; a boolean canvas raises an
    # overload error in cv2.polylines.
    candidate_paths = []
    for label in range(1, count):
        region = labels == label
        if stats[label, cv2.CC_STAT_AREA] < min_region_area:
            continue
        core = cv2.erode(region.astype(np.uint8), kernel) > 0
        if np.any(core):
            candidate_paths.extend(_split_candidate_components(skeletonize(core)))
            continue
        # A residual red crescent is too thin to contain an entire new stroke
        # by itself.  Its outer edge is still legitimate ink, though: add the
        # matching inward offset of the ORIGINAL ink outline.  This is the
        # requested "new line whose outline sits on the red outline" case.
        for run_xy in _offset_outline_runs_near_red(
            ink_core, region, half_width, params.outline_red_dilation_extra,
        ):
            # Draw only this ordered contour arc.  It remains an independent
            # path; no bitmap thinning is allowed to invent a connection.
            candidate = np.zeros(mask.shape, dtype=np.uint8)
            cv2.polylines(candidate, [run_xy], False, 1, 1)
            candidate_paths.extend(_split_candidate_components(candidate))

    if not candidate_paths:
        return (skeleton > 0).astype(np.uint8) * 255
    # Leave a one-pixel moat around the old skeleton.  Two paths that merely
    # touch diagonally are 8-connected to downstream topology code, so keeping
    # this separation is what lets them remain genuinely disconnected.
    existing_neighbourhood = binary_dilation(
        skeleton > 0, iterations=params.existing_skeleton_moat,
    )
    separated_paths = []
    for path in candidate_paths:
        separated = path & ~existing_neighbourhood
        separated_paths.extend(_split_candidate_components(separated))
    inserted = _prune_paths_without_red_regression(
        mask, skeleton, separated_paths, half_width,
        allowed_new_red_pixels=params.allowed_new_red_pixels,
    )
    return (((skeleton > 0) | inserted).astype(np.uint8)) * 255


def recenter_skeleton(mask: np.ndarray, skeleton: np.ndarray,
                       global_half_width: int = None, max_shift: float = None,
                       params: RecenterParams | None = None) -> np.ndarray:
    """Skew toward left-side red, then add paths for red still uncovered.

    The second stage measures red against the *skewed* skeleton, not the input
    skeleton.  It therefore only adds a line where skewing did not already
    cover the ink, and positions that line a half-width inside the remaining
    red region.
    """
    params = params or RecenterParams()
    half_width = float(global_half_width or estimate_stroke_width(mask, skeleton))
    if max_shift is None:
        max_shift = half_width * params.max_shift_ratio
    skewed = skew_left_red_first(
        mask, skeleton, global_half_width=global_half_width, max_shift=max_shift,
        tangent_lookback=params.tangent_lookback,
        ray_limit_ratio=params.ray_limit_ratio,
        red_profile_radius=params.red_profile_radius,
        displacement_profile_radius=params.displacement_profile_radius,
    )
    return add_centerlines_in_missing_red(
        mask, skewed, global_half_width=global_half_width, params=params,
    )


# ---------------------------------------------------------------------------
# Evidence-based gap bridging: join endpoints only when the gap between them
# is demonstrably a missing stroke, never just because they happen to be near
# each other.
# ---------------------------------------------------------------------------

def _segment_membership(segments: list) -> dict:
    """Maps each pixel (as a tuple) to the index of the segment it belongs to."""
    membership = {}
    for idx, seg in enumerate(segments):
        for pt in seg:
            membership[tuple(int(v) for v in pt)] = idx
    return membership


def _endpoint_tangent(segment: np.ndarray, at_start: bool, lookback: int = 5) -> np.ndarray:
    """
    Outward-pointing unit tangent at one end of a branch-free path -- the
    direction the stroke would naturally continue if extended past that
    endpoint. Uses a short lookback window rather than the immediate
    neighbor, since a single-pixel step is too noisy on raster data to
    give a reliable direction (same lesson as elsewhere in this file).
    """
    n = len(segment)
    lb = min(lookback, n - 1)
    if at_start:
        p0, p1 = segment[lb], segment[0]
    else:
        p0, p1 = segment[n - 1 - lb], segment[n - 1]
    v = (p1 - p0).astype(float)
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.array([0.0, 0.0])
    return v / norm


def _turn_angle_deg(tangent_a: np.ndarray, tangent_b: np.ndarray) -> float:
    """Angle in degrees between two vectors (0 = pointing the same way)."""
    na, nb = np.linalg.norm(tangent_a), np.linalg.norm(tangent_b)
    if na == 0 or nb == 0:
        return 180.0
    cos_a = np.clip(np.dot(tangent_a, tangent_b) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def _bezier_bridge(p1: np.ndarray, tangent1: np.ndarray, p2: np.ndarray, tangent2: np.ndarray,
                    num_points: int = 24) -> np.ndarray:
    """
    A gently curved bridge between two endpoints, departing p1 along its
    own outward tangent and arriving at p2 along ITS outward tangent --
    so the bridge continues each stroke's existing direction rather than
    forcing a straight-line connection that could require a sharp,
    unnatural turn at either end.
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    dist = float(np.linalg.norm(p2 - p1))
    if dist == 0:
        return np.array([p1])
    c1 = p1 + tangent1 * (dist / 3.0)
    c2 = p2 - tangent2 * (dist / 3.0)
    ts = np.linspace(0.0, 1.0, num_points)
    return np.array([
        (1 - t) ** 3 * p1 + 3 * (1 - t) ** 2 * t * c1 +
        3 * (1 - t) * t ** 2 * c2 + t ** 3 * p2
        for t in ts
    ])


def bridge_missing_strokes(mask: np.ndarray, skeleton: np.ndarray,
                            max_gap: float = None, half_width: int = None,
                            min_red_reduction: int = 3, max_turn_deg: float = 70.0,
                            min_spur_segment_length: int = 6) -> np.ndarray:
    """
    Joins two nearby endpoints ONLY when there is real evidence the gap
    between them is a missing stroke -- proximity alone is never
    sufficient, since two sides of a loop that pass close to each other,
    or nearby strokes from separate letters, can also produce close
    endpoints with no real connection between them.

    For each candidate endpoint pair within max_gap, a gently curved
    bridge is proposed (following each endpoint's own outward stroke
    direction) and accepted only if ALL hold, checked against the same
    fixed-width reconstruction ruler used everywhere else in this file:

      1. The bridge lies mostly inside real ink.
      2. Adding it measurably reduces missed (red) pixels.
      3. It does not meaningfully increase extra (blue) pixels.
      4. It does not touch any existing, unrelated path (which would
         silently create an unwanted junction).
      5. Neither endpoint's own stroke direction disagrees sharply with
         the direction toward the other endpoint (rejects forced sharp
         turns / bridges that would have to leave the ink to connect).
      6. Neither endpoint belongs to a very short segment -- a short
         segment is more likely a spur/tip than a genuine stroke end
         that should be extended (mirrors prune_spurs' reasoning).

    Candidates are tried smallest-gap-first and re-evaluated against the
    skeleton as it stands after each accepted bridge, since accepting
    one bridge changes what red/blue mean for the next candidate.
    """
    if half_width is None:
        half_width = estimate_stroke_width(mask, skeleton)
    if max_gap is None:
        max_gap = half_width * 4  # untuned starting guess -- adjust after testing

    current = (skeleton > 0).astype(np.uint8) * 255
    mask_bool = mask > 0

    while True:
        branch_pts, end_pts = _find_branch_and_endpoints(current)
        segments = _split_into_segments(current, branch_pts, end_pts)
        membership = _segment_membership(segments)

        real_endpoints = []
        for idx, seg in enumerate(segments):
            if len(seg) < min_spur_segment_length:
                continue  # too short to be a genuine stroke tip, not a bridging candidate
            real_endpoints.append((tuple(int(v) for v in seg[0]), idx, True))
            real_endpoints.append((tuple(int(v) for v in seg[-1]), idx, False))

        candidates = []
        for i in range(len(real_endpoints)):
            for j in range(i + 1, len(real_endpoints)):
                p1, seg_i, start_i = real_endpoints[i]
                p2, seg_j, start_j = real_endpoints[j]
                if seg_i == seg_j:
                    continue
                d = float(np.linalg.norm(np.array(p1) - np.array(p2)))
                if d <= max_gap:
                    candidates.append((d, p1, seg_i, start_i, p2, seg_j, start_j))

        if not candidates:
            break
        candidates.sort(key=lambda c: c[0])

        accepted_this_pass = False
        for d, p1, seg_i, start_i, p2, seg_j, start_j in candidates:
            tangent1 = _endpoint_tangent(segments[seg_i], at_start=start_i)
            tangent2 = _endpoint_tangent(segments[seg_j], at_start=start_j)

            chord = np.array(p2, dtype=float) - np.array(p1, dtype=float)
            chord_norm = np.linalg.norm(chord)
            if chord_norm == 0:
                continue
            chord_dir = chord / chord_norm

            # reject 5: forced sharp turns
            if _turn_angle_deg(tangent1, chord_dir) > max_turn_deg:
                continue
            if _turn_angle_deg(tangent2, -chord_dir) > max_turn_deg:
                continue

            bridge = _bezier_bridge(np.array(p1, dtype=float), tangent1,
                                     np.array(p2, dtype=float), tangent2)
            bridge_int = np.rint(bridge).astype(int)
            bridge_int[:, 0] = np.clip(bridge_int[:, 0], 0, mask.shape[0] - 1)
            bridge_int[:, 1] = np.clip(bridge_int[:, 1], 0, mask.shape[1] - 1)

            # accept 1: bridge lies mostly inside real ink
            inside = sum(1 for (y, x) in bridge_int if mask_bool[y, x])
            if inside / len(bridge_int) < 0.8:
                continue

            # reject 4 / "unwanted junction": bridge must not touch any
            # OTHER existing path
            own_pixels = {tuple(int(v) for v in p) for p in segments[seg_i]}
            own_pixels |= {tuple(int(v) for v in p) for p in segments[seg_j]}
            touches_other = False
            for (y, x) in bridge_int:
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if (ny, nx) in membership and (ny, nx) not in own_pixels:
                            touches_other = True
                            break
                    if touches_other:
                        break
                if touches_other:
                    break
            if touches_other:
                continue

            # accept 2 & 3: measurable red reduction, no meaningful blue growth
            before_fid = skeleton_fidelity(mask, current)
            bridge_canvas = np.zeros(mask.shape, dtype=np.uint8)
            for k in range(len(bridge_int) - 1):
                cv2.line(bridge_canvas, tuple(bridge_int[k][::-1]),
                          tuple(bridge_int[k + 1][::-1]), 255, 1)
            trial = ((current > 0) | (bridge_canvas > 0)).astype(np.uint8) * 255
            after_fid = skeleton_fidelity(mask, trial)

            red_reduced = before_fid["missed_pixels"] - after_fid["missed_pixels"]
            blue_increase = after_fid["extra_pixels"] - before_fid["extra_pixels"]

            if red_reduced < min_red_reduction:
                continue
            if blue_increase > min_red_reduction:
                continue

            current = trial
            accepted_this_pass = True
            break  # topology changed -- re-derive segments/endpoints before the next candidate

        if not accepted_this_pass:
            break

    return current


def finalize_topology_smoothing(skeleton: np.ndarray, smoothing_per_point: float = 0.35) -> np.ndarray:
    """
    Final cleanup pass: smooths raster jitter WITHOUT touching topology.

    Run this LAST, after bridging/insertion decisions are final. Splits
    the skeleton into its branch-free segments, keeps every true
    endpoint and junction pixel EXACTLY fixed, and smooths only each
    segment's INTERIOR points (reusing the same endpoint-preserving
    smoothing spline used elsewhere in this file). This removes pixel
    jitter without deleting or altering which points connect to which --
    the topology already decided is treated as final and untouchable.

    Deliberately does NOT re-skeletonize the combined result afterward:
    doing so risks merging separate segments that happen to pass close
    together post-smoothing, exactly the "spaghetti" bug already fixed
    elsewhere in this file by working with ordered paths directly
    instead of re-thinning a bitmap.
    """
    branch_pts, end_pts = _find_branch_and_endpoints(skeleton)
    segments = _split_into_segments(skeleton, branch_pts, end_pts)

    working = np.zeros(skeleton.shape, dtype=np.uint8)
    for seg in segments:
        if len(seg) < 3:
            for y, x in seg:
                working[y, x] = 255
            continue

        smoothed = _smooth_open_path(seg.astype(float), smoothing_per_point=smoothing_per_point)
        smoothed_int = np.rint(smoothed).astype(int)
        smoothed_int[:, 0] = np.clip(smoothed_int[:, 0], 0, skeleton.shape[0] - 1)
        smoothed_int[:, 1] = np.clip(smoothed_int[:, 1], 0, skeleton.shape[1] - 1)
        for i in range(len(smoothed_int) - 1):
            cv2.line(working, tuple(smoothed_int[i][::-1]), tuple(smoothed_int[i + 1][::-1]), 255, 1)

    return working
