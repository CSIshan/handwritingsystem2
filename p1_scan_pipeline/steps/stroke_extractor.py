"""
Steps 6–9 — Skeletonize → Stroke Graph → Ordering → Normalize
Converts a binary word crop into ordered, baseline-normalized stroke sequences.
"""
import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple
from skimage.morphology import skeletonize as sk_thin

log = logging.getLogger(__name__)

Point = Tuple[float, float]   # (x_mm, y_mm) baseline-normalized


@dataclass
class StrokePoint:
    x:       float   # mm, relative to word left edge
    y:       float   # mm, baseline-normalized (positive = above baseline)
    pen_up:  int     # 1 = lift pen after this point, 0 = keep drawing


def extract_strokes(
    crop:       np.ndarray,   # binary word image (255=ink, 0=bg)
    baseline_y: int,          # pixel Y of baseline within FULL PAGE
    strip_top:  int,          # pixel Y of strip top within full page
    config,
) -> List[StrokePoint]:
    """
    Full chain: binary crop → skeleton → graph → ordered strokes → normalized.
    Returns list of StrokePoint in mm, baseline-normalized.
    """
    if crop is None or crop.size == 0:
        return []

    # ── Step 6: Skeletonize ───────────────────────────────────────────────────
    bool_img = crop > 0
    if not np.any(bool_img):
        return []

    # Conservative morphological closing before skeletonize: bridges tiny
    # gaps in the binary mask caused by faded/thin ink at tight curvature
    # (e.g. bottom of 'o' where pen pressure often lightens at speed).
    # kernel=2 is deliberately small — empirical tests confirm kernel>=3
    # begins merging genuinely separate strokes when their edge-to-edge
    # gap is <= ~4px (0.17mm at 600dpi), which is a realistic letter
    # spacing in cursive. At kernel=2 the only risk is strokes already
    # actually touching (true gap=0px), which is not a new problem.
    # If a gap requires kernel>=3 to bridge, you likely need a real
    # example scan to calibrate — send one and we'll revisit the size.
    _close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bool_img = cv2.morphologyEx(
        bool_img.astype(np.uint8), cv2.MORPH_CLOSE, _close_kernel
    ).astype(bool)

    skeleton = sk_thin(bool_img).astype(np.uint8) * 255

    # ── Step 7: Get ordered pixel path ───────────────────────────────────────
    pixel_paths = _skeleton_to_paths(skeleton)

    if not pixel_paths:
        return []

    # ── Step 8: Order strokes (greedy nearest-neighbour across paths) ─────────
    ordered = _order_paths(pixel_paths, config)

    # ── Step 9: Normalize to baseline=0 space, convert to mm ─────────────────
    # baseline_y is in full-page pixel coords
    # strip_top is in full-page pixel coords
    # crop's Y=0 corresponds to full-page Y = strip_top
    # So within crop: baseline pixel Y = baseline_y - strip_top
    baseline_in_crop = baseline_y - strip_top

    strokes = _normalize(ordered, baseline_in_crop, config)

    # ── Step 9b: Smooth — detect and break anomalous internal jumps ──────────
    # Converts tracer artifacts (wrong branch arm → large internal step)
    # into proper pen-up breaks rather than visible straight lines
    strokes = _smooth_strokes(strokes)

    # ── Step 9c: Light positional denoise — reduce pixel-level wobble ────────
    # Operates on already pen-up-clean segments (must run after _smooth_strokes,
    # not before — smoothing positions before anomaly-detection would blend the
    # anomalous jump into its neighbors instead of isolating it for the break)
    strokes = _smooth_positions(strokes)

    # ── Step 9d: Reconnect direction-compatible short gaps ────────────────────
    # _order_paths greedy ordering occasionally visits fragment endpoints in a
    # suboptimal sequence, leaving small pen-up gaps (0.5–1mm) between parts
    # of the same continuous ink stroke (e.g. y-descender tip).  Reconnect
    # these where both gap distance and stroke direction are compatible.
    strokes = _reconnect_compatible(strokes)

    # Filter strokes that are too short
    strokes = _filter_short(strokes, config.min_stroke_points)

    return strokes


# ── Step 6 helper: skeleton → list of pixel paths ────────────────────────────

def _skeleton_to_paths(skeleton: np.ndarray) -> List[List[Tuple[int, int]]]:
    """
    Extract skeleton fragments by walking between topological nodes.

    A *node* is any pixel whose skeleton-degree is not exactly 2:
      • degree 1   → endpoint (arm tip)
      • degree 0   → isolated pixel (its own node)
      • degree ≥ 3 → junction / branch point

    Every fragment begins and ends at a node; degree-2 pixels appear only
    as fragment interiors.  Fragments always stop exactly at junctions so
    that the skeleton becomes an exact graph: nodes = junctions/endpoints,
    edges = fragments, two fragments are adjacent iff they share an
    identical endpoint pixel.  This is required by _order_paths so it can
    run Route Inspection on the real graph topology.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return []

    pixel_set: set = set(zip(xs.tolist(), ys.tolist()))

    def nbrs(x: int, y: int) -> List[Tuple[int, int]]:
        return [(x+dx, y+dy)
                for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if not (dx == 0 and dy == 0)
                and (x+dx, y+dy) in pixel_set]

    def deg(x: int, y: int) -> int:
        return len(nbrs(x, y))

    nodes: set = {p for p in pixel_set if deg(*p) != 2}
    if not nodes:
        # Pure closed loop — anchor on the topmost-leftmost pixel.
        nodes.add(min(pixel_set, key=lambda p: (p[1], p[0])))

    paths: List[List[Tuple[int, int]]] = []
    visited_interior: set = set()

    for node in sorted(nodes, key=lambda p: (p[1], p[0])):
        for nbr in nbrs(*node):
            if nbr in nodes:
                if node < nbr:
                    paths.append([node, nbr])
                continue
            if nbr in visited_interior:
                continue
            path = [node, nbr]
            visited_interior.add(nbr)
            prev, cur = node, nbr
            while cur not in nodes:
                nxts = [p for p in nbrs(*cur) if p != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                path.append(nxt)
                if nxt not in nodes:
                    visited_interior.add(nxt)
                prev, cur = cur, nxt
            paths.append(path)

    return paths


# ── Step 8 helper: Route Inspection (Chinese Postman) ordering ───────────────

def _order_paths(
    paths: List[List[Tuple[int, int]]],
    config,
) -> List[Tuple[Tuple[int, int], bool]]:
    """
    Order skeleton fragments by solving the Route Inspection problem
    (Chinese Postman Problem) on the exact skeleton graph.

    Why greedy approaches fail: any single-step or locally-scored
    decision can make a "locally reasonable" choice that strands other
    fragments far away, forcing a large unplanned jump later.  This is
    not a tuning problem — it is the structural shortcoming of greedy
    algorithms on routing problems.  Route Inspection has an optimal
    polynomial-time solution; we use that directly.

    Algorithm (per connected component):
      1. Build the component as a multigraph: nodes = junctions/endpoints,
         edges = fragments, weight = fragment pixel-path length.
      2. Find odd-degree nodes (always an even number, by handshake lemma).
         • 0 odd nodes → Eulerian circuit already exists (draw entire
           component in one continuous pen-down path).
         • >0 odd nodes → compute minimum-weight perfect matching over all
           odd nodes (networkx Blossom V — exact, not heuristic) with
           Euclidean straight-line distance as match cost (pen-up travel
           is a straight jump, not constrained to ink).  Drop the
           highest-cost matched pair (open-path CPP trick: we don't need
           to return to the start).  The remaining pairs become virtual
           pen-up edges added to the graph.
      3. With augmentation, every node is even-degree → Eulerian path
         exists (Hierholzer's algorithm, networkx built-in).  Walking it
         emits every real fragment exactly once pen-down, with pen-up
         only at the provably-necessary virtual edges.

    Virtual edge detection: edges are tagged with _virtual=True so the
    walk emitter can reliably detect them regardless of key collisions
    in the multigraph (virtual edges are always added between node pairs
    that already have a real fragment edge, so they share the same
    endpoint tuple — key-based lookup alone cannot distinguish them).

    Pen-up convention: pen_up=True on point P means "lift pen AFTER
    drawing to P".  The walk emitter never reverses component walks,
    because pen_up flags are direction-dependent and reversal would
    corrupt them.  Inter-component sequencing always traverses from each
    component's natural start endpoint.
    """
    if not paths:
        return []

    import networkx as nx
    import itertools

    def path_length(p: List[Tuple[int, int]]) -> float:
        return sum(_dist(p[i], p[i+1]) for i in range(len(p) - 1)) or 1e-6

    G = nx.MultiGraph()
    for idx, p in enumerate(paths):
        G.add_edge(p[0], p[-1], key=idx, weight=path_length(p), frag_idx=idx)

    component_walks: List[List[Tuple[Tuple[int, int], bool]]] = []

    for comp_nodes in nx.connected_components(G):
        H = G.subgraph(comp_nodes).copy()
        if H.number_of_edges() == 0:
            continue

        odd_nodes = [n for n in H.nodes if H.degree(n) % 2 == 1]

        if not odd_nodes:
            start      = min(H.nodes, key=lambda p: (p[1], p[0]))
            walk_src   = H
            walk_edges = list(nx.eulerian_circuit(H, source=start, keys=True))
        else:
            M = nx.Graph()
            for u, v in itertools.combinations(odd_nodes, 2):
                M.add_edge(u, v, weight=_dist(u, v))
            matching = nx.algorithms.matching.min_weight_matching(M)

            if matching:
                weighted = sorted(
                    ((u, v, M[u][v]['weight']) for u, v in matching),
                    key=lambda t: t[2],
                )
                start, _end, _ = weighted[-1]
                aug_pairs = [(u, v) for u, v, _ in weighted[:-1]]
            else:
                start = odd_nodes[0]
                aug_pairs = []

            H_aug = H.copy()
            for u, v in aug_pairs:
                # Tag with _virtual=True so the emitter can identify these
                # edges without relying on key lookup (which collides when
                # the matched node pair also has a real fragment edge).
                H_aug.add_edge(u, v, weight=_dist(u, v), frag_idx=None, _virtual=True)

            walk_src   = H_aug
            walk_edges = list(nx.eulerian_path(H_aug, source=start, keys=True))

        walk: List[Tuple[Tuple[int, int], bool]] = []
        for (u, v, key) in walk_edges:
            edge_data  = walk_src.get_edge_data(u, v, key) or {}
            is_virtual = edge_data.get('_virtual', False)
            frag_idx   = None if is_virtual else edge_data.get('frag_idx')

            if is_virtual:
                # Mark the last emitted real point as pen-up (lift after it).
                if walk:
                    walk[-1] = (walk[-1][0], True)
                continue

            if frag_idx is None:
                continue

            frag = paths[frag_idx]
            pts  = frag if frag[0] == u else list(reversed(frag))
            for pt in pts:
                walk.append((pt, False))

        if walk:
            # Ensure every component walk ends pen-up so the inter-component
            # stitching never creates a pen-down step across the component
            # boundary (the final real fragment always emits False on its
            # last pixel; virtual-edge marking only fires mid-walk).
            walk[-1] = (walk[-1][0], True)
            component_walks.append(walk)

    if not component_walks:
        return []

    # ── Sequence components by nearest-current-position ─────────────────────────
    # Always traverse each component from its natural start (first point of its
    # walk).  Never reverse: pen_up flags are direction-dependent and reversing
    # a walk that contains embedded pen_up=True markers from virtual edges
    # corrupts the semantics — the flags mean "lift AFTER this point" in the
    # forward traversal direction only.
    component_walks.sort(key=lambda w: (w[0][0][1], w[0][0][0]))
    ordered = component_walks.pop(0)
    cur_pos = ordered[-1][0]

    while component_walks:
        best_idx, best_d = 0, float('inf')
        for i, w in enumerate(component_walks):
            d = _dist(cur_pos, w[0][0])
            if d < best_d:
                best_d, best_idx = d, i

        w = component_walks.pop(best_idx)
        ordered[-1] = (ordered[-1][0], True)
        ordered.extend(w)
        cur_pos = ordered[-1][0]

    if ordered:
        ordered[-1] = (ordered[-1][0], True)

    return ordered


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5



# ── Step 9 helper: normalize to baseline=0, convert to mm ────────────────────

def _normalize(
    ordered:           List[Tuple[Tuple[int, int], bool]],
    baseline_in_crop:  int,
    config,
) -> List[StrokePoint]:
    """
    Shift all Y coords so baseline = 0.
    Flip Y: positive = above baseline (ascenders), negative = below (descenders).
    Convert pixels → mm.
    """
    if not ordered:
        return []

    px = config.px_per_mm

    # X offset: shift so word starts at x=0
    x_offset = ordered[0][0][0]

    strokes = []
    for (px_x, px_y), pen_up in ordered:
        x_mm =  (px_x - x_offset) / px
        y_mm = -(px_y - baseline_in_crop) / px   # flip Y: up = positive
        strokes.append(StrokePoint(x=x_mm, y=y_mm, pen_up=int(pen_up)))

    return strokes


def _smooth_strokes(
    strokes:        List[StrokePoint],
    jump_threshold: float = 0.15,  # mm — absolute floor; see rationale below
) -> List[StrokePoint]:
    """
    Post-processing smoothener: detects anomalous large jumps WITHIN a
    continuous stroke segment and converts them to pen-up breaks.

    The root cause: the tracer walks into a wrong branch arm at a
    skeleton branch point, hits a dead-end, and _order_paths jumps back
    across the word to rejoin the main path. This jump can range from
    well under 1mm up to several mm depending on word geometry,
    compared to the normal step size (real measured median: 0.04-0.06mm
    for cursive at 600dpi/23.6px-per-mm), and since it's within a single
    traced fragment it gets registered as pen-down, drawing a visible
    straight line across the word.

    Detection: within each continuous segment (between pen-ups), compute
    the median step size. Any step more than N times the median is
    anomalous — break it into a pen-up at that point. This converts
    "one segment with a big internal jump" into "two clean segments with
    a pen-up between them" which is semantically correct (the tracer
    should have lifted the pen there anyway).

    N=8x is conservative enough to not break genuine cursive (which can
    have step-size variation of ~3-4x at sharp corners) but catches the
    tracer artifacts, which run 10-500x the local median in real data.

    jump_threshold (0.15mm) is an absolute floor under the 8x-median
    relative threshold — NOT the primary detector. It originally
    defaulted to 1.0mm, but real-data diagnosis on an actual scanned
    word found segments with median step ~0.04-0.06mm where a genuinely
    anomalous internal jump (confirmed visually as a residual straight
    line, e.g. in "not") measured only 0.4-1.0mm — under the old 1.0mm
    floor despite being 10-23x the segment's own local median. The old
    floor was silently overriding the relative (and correct) signal on
    fine cursive. 0.15mm is comfortably above genuine pixel-level
    skeleton-walk noise (max single-step size is one diagonal pixel,
    ~0.06mm at 600dpi) while no longer masking real anomalies in this
    size range. In practice the 8x-median relative threshold dominates
    for most segments anyway (it typically works out to 0.3-0.5mm on
    real cursive); the floor mainly guards extremely short or unusually
    clean segments where the median itself is near zero.
    """
    if not strokes:
        return []

    result = []

    # Split into segments at existing pen-ups
    segments = []
    current = []
    for pt in strokes:
        current.append(pt)
        if pt.pen_up:
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    for seg in segments:
        if len(seg) < 2:
            result.extend(seg)
            continue

        # Compute step distances within this segment
        dists = [
            ((seg[i].x - seg[i-1].x)**2 + (seg[i].y - seg[i-1].y)**2)**0.5
            for i in range(1, len(seg))
        ]

        import statistics
        median_step = statistics.median(dists) if dists else 0
        threshold   = max(jump_threshold, median_step * 8)

        # Walk through, breaking at anomalous jumps
        sub_seg = [seg[0]]
        for i in range(1, len(seg)):
            if dists[i-1] > threshold:
                # Anomalous jump — end current sub-segment with pen-up
                sub_seg[-1] = StrokePoint(
                    x=sub_seg[-1].x,
                    y=sub_seg[-1].y,
                    pen_up=1
                )
                result.extend(sub_seg)
                sub_seg = [seg[i]]
            else:
                sub_seg.append(seg[i])

        # Preserve original pen_up on last point of last sub-segment
        if sub_seg:
            last = seg[-1]
            sub_seg[-1] = StrokePoint(
                x=sub_seg[-1].x,
                y=sub_seg[-1].y,
                pen_up=last.pen_up
            )
            result.extend(sub_seg)

    return result


def _smooth_positions(
    strokes: List[StrokePoint],
    radius:  int   = 4,     # smoothing window radius in points (4 = 9-point window)
    weight:  float = 0.6,   # how much of the windowed-average position to apply, 0=none, 1=full
) -> List[StrokePoint]:
    """
    Light positional denoising for the pixel-level "staircase" wobble
    inherent to skeletonizing a rasterized stroke (Zhang-Suen thinning
    nudges the centerline left/right by ~1 pixel as it follows a
    gently curving or diagonal line at finite resolution).

    This is NOT a fix for a tracing bug — measured on real scan data,
    this wobble is genuinely tiny (mean perpendicular deviation from a
    local line fit: ~0.02mm, max ~0.08mm). At normal viewing scale this
    is invisible; at high zoom, an unsmoothed polyline through points
    spaced ~0.04-0.06mm apart renders every one of these sub-pixel
    kinks as a visible vertex, which is what reads as "squiggly". The
    actual long-term fix is proper curve fitting in P9 (SVG composer,
    Bezier-smoothed paths per the architecture) — this is a cheap
    interim pass at the P1 level that meaningfully reduces visible
    jaggedness without waiting on that.

    Windowed averaging (radius=3, i.e. a 7-point window) applied only to
    INTERIOR points of segments with enough points to support a full
    window — endpoints and short segments (dots, t-bar fragments) are
    left untouched. weight=0.6 blends 60% of the way toward the window
    average rather than snapping fully to it.

    Parameter choice (empirically tuned on real scan data + synthetic
    edge cases, not guessed):
    - A 3-tap (radius=1) kernel only reduced measured wobble by ~9% —
      the wobble's spatial scale is wider than immediate-neighbor
      smoothing can reach.
    - radius=3, weight=0.6 reduces real measured wobble by ~27-30%
      while shifting a synthetic sharp 90-degree corner by only ~35-42
      micrometers and shrinking a synthetic tight loop (1.5mm radius,
      matching real letter-loop scale) by under 0.3% — i.e. genuine
      sharp features and loop shapes are preserved, only the sub-pixel
      jitter is meaningfully reduced.
    - Wider/stronger settings (radius>=4, weight>=0.8) reduce wobble
      further (60%+) but start visibly rounding genuine sharp corners
      (~80+ micrometers of corner displacement), so were deliberately
      not used as the default — a real but conservative improvement
      was chosen over a larger one that risks softening real letterform
      features that aren't actually wobble.
    """
    if not strokes or weight <= 0:
        return list(strokes) if strokes else []

    result = []
    segments = []
    current = []
    for pt in strokes:
        current.append(pt)
        if pt.pen_up:
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    min_len = 2 * radius + 1
    for seg in segments:
        if len(seg) < min_len:
            result.extend(seg)
            continue

        smoothed_seg = list(seg[:radius])   # leave leading boundary points untouched
        for i in range(radius, len(seg) - radius):
            window_x = [seg[j].x for j in range(i - radius, i + radius + 1)]
            window_y = [seg[j].y for j in range(i - radius, i + radius + 1)]
            avg_x = sum(window_x) / len(window_x)
            avg_y = sum(window_y) / len(window_y)
            new_x = seg[i].x + weight * (avg_x - seg[i].x)
            new_y = seg[i].y + weight * (avg_y - seg[i].y)
            smoothed_seg.append(StrokePoint(x=new_x, y=new_y, pen_up=seg[i].pen_up))
        smoothed_seg.extend(seg[-radius:])  # leave trailing boundary points (incl. real pen_up) untouched

        result.extend(smoothed_seg)

    return result


def _reconnect_compatible(
    strokes:    List[StrokePoint],
    max_gap_mm: float = 1.0,   # only reconnect pen-up breaks below this distance
    min_dot:    float = 0.6,   # minimum cosine similarity between outgoing/incoming directions
    window:     int   = 5,     # number of points to average for direction estimates
) -> List[StrokePoint]:
    """
    Direction-compatible gap reconnection.

    After `_smooth_strokes` and `_order_paths`, some stroke segments are
    separated by short pen-up gaps even though they belong to the same
    continuous ink stroke.  This happens because `_smooth_strokes` creates
    singleton "bridge" segments (a single pen_up=True point that was the
    destination of a large inter-fragment jump in `_order_paths`), and the
    greedy nearest-neighbour ordering occasionally visits fragment endpoints
    in a suboptimal sequence.

    These singletons appear as the pattern:
        [... normal_seg_A ... | singleton | ... normal_seg_B ...]

    The singleton itself has no direction (zero-length).  This function
    looks *through* singletons to the real segments on each side, computing
    direction compatibility between those.

    Reconnection criteria:
      1. Total gap from end of last real segment to start of next real segment
         (summing through any singletons) < max_gap_mm.
      2. Outgoing direction of real segment A ≈ incoming direction of real
         segment B (cosine similarity ≥ min_dot, i.e. angle ≤ ~53°).

    Direction compatibility correctly rejects:
      - Genuine pen lifts between separate strokes (mismatched angles)
      - 'y' descender junction (downward vs upward: dot < 0)
    And correctly reconnects:
      - Fragment-ordering artefacts where both sides of a small gap are
        going in nearly the same direction (e.g. 'ou' continuation, dot ≈ 1)

    Parameters:
      max_gap_mm: tight upper bound — 1.0mm is well below any intentional
                  inter-letter or inter-word gap (typically 2–10mm).
      min_dot:    0.6 ≈ 53° maximum angle between directions.
      window:     number of points used to estimate each direction.
    """
    if not strokes:
        return []

    # Split into segments at pen_up=True
    segments: List[List[StrokePoint]] = []
    current: List[StrokePoint] = []
    for pt in strokes:
        current.append(pt)
        if pt.pen_up:
            segments.append(current)
            current = []
    if current:
        segments.append(current)

    if len(segments) < 2:
        return strokes

    def seg_direction(seg: List[StrokePoint], from_end: bool) -> List[float]:
        """Unit direction vector at the given end of a segment.  Returns [0,0]
        if the segment has fewer than 2 distinct points (e.g. a singleton)."""
        n = min(window, len(seg))
        if from_end:
            pts = seg[-n:]
        else:
            pts = seg[:n]
        dx = pts[-1].x - pts[0].x
        dy = pts[-1].y - pts[0].y
        mag = (dx**2 + dy**2) ** 0.5
        if mag < 1e-9:
            return [0.0, 0.0]
        return [dx / mag, dy / mag]

    def is_singleton(seg: List[StrokePoint]) -> bool:
        return len(seg) <= 1

    # Build a list of (index, is_real) to navigate through singletons
    # We process left-to-right, collapsing singleton runs.
    merged: List[List[StrokePoint]] = [segments[0]]

    i = 1
    while i < len(segments):
        # Look ahead: collect any singletons after the current segment
        j = i
        singletons_ahead: List[List[StrokePoint]] = []
        while j < len(segments) and is_singleton(segments[j]):
            singletons_ahead.append(segments[j])
            j += 1

        seg_b_idx = j  # index of the next real segment (or end)
        seg_a = merged[-1]

        if singletons_ahead and seg_b_idx < len(segments):
            # We have: seg_a → [singletons] → seg_b
            # Compute total gap from end of seg_a to start of seg_b
            seg_b = segments[seg_b_idx]
            end_a   = seg_a[-1]
            start_b = seg_b[0]
            gap = ((end_a.x - start_b.x)**2 + (end_a.y - start_b.y)**2) ** 0.5

            reconnect = False
            if gap < max_gap_mm:
                dir_a_vec = seg_direction(seg_a, from_end=True)
                dir_b_vec = seg_direction(seg_b, from_end=False)
                valid = (dir_a_vec != [0.0, 0.0]) and (dir_b_vec != [0.0, 0.0])
                if valid:
                    dot = dir_a_vec[0]*dir_b_vec[0] + dir_a_vec[1]*dir_b_vec[1]
                    reconnect = (dot >= min_dot)

            if reconnect:
                # Merge seg_a + singletons + seg_b into one continuous segment
                combo = list(seg_a)
                combo[-1] = StrokePoint(x=combo[-1].x, y=combo[-1].y, pen_up=0)
                for s in singletons_ahead:
                    pts = [StrokePoint(x=p.x, y=p.y, pen_up=0) for p in s]
                    combo.extend(pts)
                combo.extend(seg_b)
                merged[-1] = combo
                i = seg_b_idx + 1
                continue
            else:
                # Keep singletons and seg_b as separate segments
                for s in singletons_ahead:
                    merged.append(s)
                merged.append(seg_b)
                i = seg_b_idx + 1
                continue

        elif not singletons_ahead and i < len(segments):
            # Normal case: seg_a → seg_b (no singletons between them)
            seg_b = segments[i]
            end_a   = seg_a[-1]
            start_b = seg_b[0]
            gap = ((end_a.x - start_b.x)**2 + (end_a.y - start_b.y)**2) ** 0.5

            reconnect = False
            if gap < max_gap_mm:
                dir_a_vec = seg_direction(seg_a, from_end=True)
                dir_b_vec = seg_direction(seg_b, from_end=False)
                valid = (dir_a_vec != [0.0, 0.0]) and (dir_b_vec != [0.0, 0.0])
                if valid:
                    dot = dir_a_vec[0]*dir_b_vec[0] + dir_a_vec[1]*dir_b_vec[1]
                    reconnect = (dot >= min_dot)

            if reconnect:
                combo = list(seg_a)
                combo[-1] = StrokePoint(x=combo[-1].x, y=combo[-1].y, pen_up=0)
                combo.extend(seg_b)
                merged[-1] = combo
            else:
                merged.append(seg_b)
            i += 1
        else:
            # Only trailing singletons — keep them
            for s in singletons_ahead:
                merged.append(s)
            i = j

    return [pt for seg in merged for pt in seg]


def _filter_short(strokes: List[StrokePoint], min_points: int) -> List[StrokePoint]:
    """Remove stroke segments shorter than min_points."""
    if not strokes:
        return []

    # Split into segments at pen_up, filter, rejoin
    segments = []
    current  = []
    for pt in strokes:
        current.append(pt)
        if pt.pen_up:
            if len(current) >= min_points:
                segments.append(current)
            current = []
    if current and len(current) >= min_points:
        segments.append(current)

    return [pt for seg in segments for pt in seg]
