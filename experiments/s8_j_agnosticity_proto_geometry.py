#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S8 — J-Agnosticity Test + Extended Proto-Geometry
==================================================

S8 extends S7 with three scientific questions:

1. J-AGNOSTICITY (core test):
   S7's proto-geometry might be an artifact of J = exp(-Δσ - Δχ),
   which is already metric-like. S8 tests whether nontrivial geometric
   structure emerges for fundamentally different J functions:
     - exponential (S7 baseline, smooth, metric-compatible)
     - step        (discontinuous, hard threshold)
     - cosine      (oscillating, non-monotone)
     - power       (polynomial decay, not exponential)
     - random      (null model — no structure → geometry must NOT emerge)
     - asymmetric  (J(x,y) ≠ J(y,x) → directed proto-topology)

   Verdict: if step/cosine/power all produce nontrivial geometry and
   random does NOT, then S7 geometry is a genuine structural consequence
   of persistent identity networks, not an artifact of the J form.

2. DIMENSIONALITY:
   What effective dimension does the emergent proto-space have?
   Estimated via MDS eigenvalue spectrum (requires numpy).

3. CURVATURE:
   Flat, positive, or negative curvature?
     - Clustering coefficient → positive curvature proxy
     - Gromov δ-hyperbolicity → tree-likeness (δ ≈ 0: hyperbolic/tree-like)

S8 also fixes two S7 weaknesses:
   - representative_ids: weighted by final state persistence (w) instead
     of arbitrary last-id ordering
   - Asymmetric K option: uses directed K_asym(T_i → T_j) ≠ K_asym(T_j → T_i)

Requires:
   s7_emergent_track_proto_geometry.py in the same directory.
   numpy (optional, for MDS dimension estimation).

This is NOT proof of FUT/CoMath.
It tests whether the CoMath claim — geometry as emergent relational structure —
holds independently of the specific coupling function.
"""

import csv
import math
import random
import heapq
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[S8] numpy not found — MDS dimension estimation disabled.")

from s7_emergent_track_proto_geometry import (
    State,
    angle_diff,
    recurse,
    split,
    jaccard_overlap,
    assign_clusters_to_tracks,
    track_lifetime,
    apply_track_memory_feedback,
    select_stable_tracks,
    build_id_state_map,
    network_components_from_geometry,
    degree_centrality,
    closeness_centrality,
    dijkstra,
    adjacency_from_edges,
    ensure_output_dir,
)


# ---------------------------------------------------------------------------
# 1. J-Family
# ---------------------------------------------------------------------------

def j_exponential(x, y):
    """
    S7 baseline.
    Smooth, metric-compatible, exponential decay in angle differences.
    """
    return math.exp(
        -angle_diff(x.sigma, y.sigma)
        - angle_diff(x.chi, y.chi)
    )


def j_step(x, y):
    """
    Hard threshold: 1.0 inside cone, 0.05 outside.
    Discontinuous — not metric-compatible by construction.
    """
    threshold = math.pi / 4
    if (angle_diff(x.sigma, y.sigma) < threshold
            and angle_diff(x.chi, y.chi) < threshold):
        return 1.0
    return 0.05


def j_cosine(x, y):
    """
    Oscillating, non-monotone.
    cos²(Δσ/2) · cos²(Δχ/2): peaks at alignment, troughs at π/2.
    """
    a = math.cos(angle_diff(x.sigma, y.sigma) / 2) ** 2
    b = math.cos(angle_diff(x.chi, y.chi) / 2) ** 2
    return a * b


def j_power(x, y):
    """
    Polynomial decay: (1 - Δσ/π)² · (1 - Δχ/π)².
    No exponential — algebraic, not smooth at origin.
    """
    a = max(0.0, 1.0 - angle_diff(x.sigma, y.sigma) / math.pi) ** 2
    b = max(0.0, 1.0 - angle_diff(x.chi, y.chi) / math.pi) ** 2
    return a * b


def j_random(x, y):
    """
    Null model: deterministic hash-based coupling.
    Symmetric, but carries no structural information.
    If this produces geometry, S7/S8 is broken.
    """
    a = min(x.id, y.id)
    b = max(x.id, y.id)
    # Linear congruential hash — deterministic and process-stable
    h = (1013904223 * a + 1664525 * b + 1013904223) % (2 ** 32)
    return h / (2 ** 32)


def j_asymmetric(x, y):
    """
    Directed coupling: J(x→y) ≠ J(y→x).
    Uses directed σ-difference and symmetric χ-difference.
    Produces a directed proto-topology.
    """
    d_sigma_directed = (y.sigma - x.sigma) % (2 * math.pi)  # 0..2π, not symmetric
    d_chi = angle_diff(x.chi, y.chi)
    return math.exp(-d_sigma_directed / math.pi - d_chi)


J_FAMILY = {
    "exponential": j_exponential,
    "step":        j_step,
    "cosine":      j_cosine,
    "power":       j_power,
    "random":      j_random,
    "asymmetric":  j_asymmetric,
}


# ---------------------------------------------------------------------------
# 2. J-injectable core functions (rewrites of S7 J-dependent functions)
# ---------------------------------------------------------------------------

def I_j(x, y, j_func):
    """Interference using injectable j_func."""
    return j_func(x, y) * math.cos(angle_diff(x.chi, y.chi))


def local_persistence_j(x, states, j_func):
    """Persistence without memory, using injectable j_func."""
    others = [y for y in states if y is not x]
    if not others:
        return 0.0
    return x.w * sum(
        j_func(x, y) * max(0.0, I_j(x, y, j_func))
        for y in others
    ) / len(others)


def persistence_with_memory_j(x, states, j_func, memory_lambda=0.08):
    """Persistence with memory feedback, using injectable j_func."""
    return local_persistence_j(x, states, j_func) + memory_lambda * x.memory


def find_clusters_j(states, j_func, threshold=0.6):
    """Connected-component clustering using injectable j_func."""
    visited = set()
    clusters = []

    for i in range(len(states)):
        if i in visited:
            continue
        stack = [i]
        cluster = []

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            cluster.append(current)

            for j in range(len(states)):
                if j not in visited and j_func(states[current], states[j]) > threshold:
                    stack.append(j)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# 3. S7 fix: persistence-weighted representative IDs
# ---------------------------------------------------------------------------

def representative_ids_weighted(track, id_state, max_ids=10):
    """
    S7 fix: selects representative IDs weighted by final state weight w.

    S7 used last_ids[:max_ids] (arbitrary ordering).
    S8 prefers states with higher w — more persistent, more representative.
    Falls back to id order if states are not in final snapshot.
    """
    ids = list(track["last_ids"])
    weighted = [(id_state[i].w if i in id_state else 0.0, i) for i in ids]
    weighted.sort(reverse=True)
    return set(i for _, i in weighted[:max_ids])


# ---------------------------------------------------------------------------
# 4. J-injectable simulation runner
# ---------------------------------------------------------------------------

def run_simulation_j(
    j_func,
    initial_count=40,
    steps=300,
    delta_min=0.001,
    delta_split=0.08,
    max_states=120,
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    memory_lambda=0.08,
    memory_gain=0.04,
    memory_decay=0.985,
    max_memory=1.0,
    min_track_updates=3,
    verbose=False,
):
    """
    S7 simulation core with injectable j_func.

    Identical dynamics to S7, but J is replaceable.
    """
    next_id = 0
    states = []

    for _ in range(initial_count):
        states.append(State(
            next_id,
            random.uniform(0.2, 1.0),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi),
            memory=0.0,
        ))
        next_id += 1

    history = []
    memory_history = []
    snapshots = []
    active_tracks = {}
    finished_tracks = []
    next_track_id = 0

    for step in range(steps):
        states = [recurse(x, memory_decay=memory_decay) for x in states]

        scored = [
            (x, persistence_with_memory_j(x, states, j_func, memory_lambda=memory_lambda))
            for x in states
        ]

        survivors = [x for x, p in scored if p >= delta_min]

        offspring = []
        for x, p in scored:
            if p >= delta_split and len(survivors) + len(offspring) < max_states:
                offspring.append(split(x, next_id))
                next_id += 1

        states = survivors + offspring

        avg_p = sum(p for _, p in scored) / len(scored) if scored else 0.0
        avg_mem = mean([s.memory for s in states]) if states else 0.0
        max_mem = max([s.memory for s in states]) if states else 0.0

        history.append((step, len(states), avg_p))
        memory_history.append((step, avg_mem, max_mem))

        if step % sample_every == 0 and states:
            clusters = find_clusters_j(states, j_func, threshold=cluster_threshold)

            cluster_records = []
            for cluster in clusters:
                ids = frozenset(states[i].id for i in cluster)
                cluster_records.append({"step": step, "ids": ids, "size": len(ids)})

            assign_clusters_to_tracks(
                cluster_records,
                active_tracks,
                finished_tracks,
                overlap_threshold,
                next_track_id_ref=[next_track_id],
            )

            if active_tracks:
                next_track_id = max(active_tracks.keys()) + 1
            elif finished_tracks:
                next_track_id = max(t["track_id"] for t in finished_tracks) + 1
            else:
                next_track_id = 0

            apply_track_memory_feedback(
                states, active_tracks,
                sample_every=sample_every,
                memory_gain=memory_gain,
                max_memory=max_memory,
                min_track_updates=min_track_updates,
            )

            snapshots.append({
                "step": step,
                "clusters": cluster_records,
                "active_track_ids": list(active_tracks.keys()),
            })

        if verbose and step % 50 == 0:
            print(f"  step {step:3d}  states {len(states):3d}  avg_p {avg_p:.4f}  avg_mem {avg_mem:.4f}")

        if not states:
            break

    finished_tracks.extend(active_tracks.values())
    return states, history, finished_tracks, snapshots, memory_history


# ---------------------------------------------------------------------------
# 5. J-injectable track geometry
# ---------------------------------------------------------------------------

def track_coupling_j(track_a, track_b, id_state, j_func, max_ids=10):
    """
    K(T_a, T_b) with injectable j_func and weighted representative IDs.
    """
    ids_a = representative_ids_weighted(track_a, id_state, max_ids=max_ids)
    ids_b = representative_ids_weighted(track_b, id_state, max_ids=max_ids)

    states_a = [id_state[i] for i in ids_a if i in id_state]
    states_b = [id_state[i] for i in ids_b if i in id_state]

    if not states_a or not states_b:
        return 0.0

    values = [j_func(a, b) for a in states_a for b in states_b if a.id != b.id]
    return mean(values) if values else 0.0


def build_track_geometry_j(
    stable_tracks,
    final_states,
    j_func,
    coupling_threshold=0.25,
    max_ids_per_track=10,
    directed=False,
):
    """
    Track coupling matrix with injectable j_func.

    If directed=True (for j_asymmetric), K is not symmetrized.
    Note: geodesic computation treats the graph as undirected even for
    directed K — full directed geodesics are a future S9 extension.
    """
    id_state = build_id_state_map(final_states)
    track_ids = [t["track_id"] for t in stable_tracks]
    n = len(stable_tracks)

    K_matrix = [[0.0] * n for _ in range(n)]
    D_matrix = [[math.inf] * n for _ in range(n)]
    edges = []

    for i in range(n):
        K_matrix[i][i] = 1.0
        D_matrix[i][i] = 0.0

    for i, track_a in enumerate(stable_tracks):
        for j in range(i + 1, n):
            track_b = stable_tracks[j]

            k_ij = track_coupling_j(track_a, track_b, id_state, j_func, max_ids_per_track)

            if directed:
                k_ji = track_coupling_j(track_b, track_a, id_state, j_func, max_ids_per_track)
            else:
                k_ji = k_ij

            K_matrix[i][j] = k_ij
            K_matrix[j][i] = k_ji

            # Symmetrize for undirected edge weight
            k_sym = (k_ij + k_ji) / 2
            if k_sym > 0:
                d = -math.log(max(k_sym, 1e-12))
                D_matrix[i][j] = d
                D_matrix[j][i] = d

            if k_sym >= coupling_threshold:
                edges.append({
                    "i": i, "j": j,
                    "a": track_ids[i], "b": track_ids[j],
                    "K": k_sym,
                    "K_ij": k_ij, "K_ji": k_ji,
                    "asymmetry": abs(k_ij - k_ji),
                    "d": -math.log(max(k_sym, 1e-12)),
                })

    return {
        "tracks": stable_tracks,
        "track_ids": track_ids,
        "K_matrix": K_matrix,
        "D_matrix": D_matrix,
        "edges": edges,
        "coupling_threshold": coupling_threshold,
        "directed": directed,
    }


def geodesic_distance_matrix_from_geometry(geometry):
    """Geodesic matrix via Dijkstra (reuses S7 dijkstra + adjacency_from_edges)."""
    n = len(geometry["track_ids"])
    adjacency = adjacency_from_edges(n, geometry["edges"])
    G = [[math.inf] * n for _ in range(n)]

    for i in range(n):
        distances = dijkstra(adjacency, i)
        for j, d in distances.items():
            G[i][j] = d

    return G


# ---------------------------------------------------------------------------
# 6. Extended geometry analysis
# ---------------------------------------------------------------------------

def estimate_dimension_mds(geodesic_matrix, variance_threshold=0.90):
    """
    MDS-based effective dimensionality estimate.

    Classical MDS on geodesic distance matrix:
    1. Replace inf with 2·max_finite
    2. Double-center squared distance matrix
    3. Eigendecompose → count eigenvalues needed for variance_threshold

    Returns effective_dim = min k s.t. top-k explains ≥ variance_threshold of variance.
    """
    if not HAS_NUMPY:
        return {"effective_dim": None, "note": "numpy required"}

    n = len(geodesic_matrix)
    if n < 3:
        return {"effective_dim": 0, "note": "too few nodes"}

    finite_vals = [v for row in geodesic_matrix for v in row
                   if math.isfinite(v) and v > 0]
    if not finite_vals:
        return {"effective_dim": 0, "note": "no finite distances"}

    max_finite = max(finite_vals)
    G = np.array([
        [v if math.isfinite(v) else 2 * max_finite for v in row]
        for row in geodesic_matrix
    ], dtype=float)

    D2 = G ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H

    eigenvalues = np.linalg.eigvalsh(B)
    eigenvalues = sorted(eigenvalues.tolist(), reverse=True)

    pos_eigs = [e for e in eigenvalues if e > 1e-8]
    if not pos_eigs:
        return {"effective_dim": 0, "n_positive": 0, "variance_explained": []}

    total = sum(pos_eigs)
    cumvar = []
    running = 0.0
    for e in pos_eigs:
        running += e
        cumvar.append(running / total)

    effective_dim = next(
        (i + 1 for i, v in enumerate(cumvar) if v >= variance_threshold),
        len(pos_eigs)
    )

    return {
        "effective_dim": effective_dim,
        "n_positive": len(pos_eigs),
        "variance_explained_top10": [round(v, 4) for v in cumvar[:10]],
        "eigenvalues_top10": [round(e, 4) for e in pos_eigs[:10]],
    }


def clustering_coefficient(geometry):
    """
    Per-node clustering coefficient and global mean.
    High CC → positive curvature proxy (nodes cluster into triangles).
    """
    n = len(geometry["track_ids"])
    adjacency = {i: set() for i in range(n)}

    for edge in geometry["edges"]:
        adjacency[edge["i"]].add(edge["j"])
        adjacency[edge["j"]].add(edge["i"])

    coeffs = []
    for i in range(n):
        neighbors = adjacency[i]
        k = len(neighbors)
        if k < 2:
            coeffs.append(0.0)
            continue
        triangles = sum(
            1 for u in neighbors for v in neighbors
            if u < v and v in adjacency[u]
        )
        possible = k * (k - 1) / 2
        coeffs.append(triangles / possible)

    return coeffs, (mean(coeffs) if coeffs else 0.0)


def gromov_delta_hyperbolicity(geodesic_matrix, max_samples=300, seed=42):
    """
    Estimates Gromov δ-hyperbolicity by sampling 4-tuples.

    For 4 points a,b,c,d:
        S1 = d(a,b)+d(c,d), S2 = d(a,c)+d(b,d), S3 = d(a,d)+d(b,c)
        δ(a,b,c,d) = (max - second_max) / 2

    δ ≈ 0: tree-like / hyperbolic space
    δ large: Euclidean-like / spherical

    Returns max δ over samples (worst-case hyperbolicity).
    """
    n = len(geodesic_matrix)
    rng = random.Random(seed)

    connected = [
        i for i in range(n)
        if sum(1 for j in range(n) if math.isfinite(geodesic_matrix[i][j]) and i != j) >= 3
    ]

    if len(connected) < 4:
        return {"delta_max": float("nan"), "delta_mean": float("nan"), "samples": 0}

    deltas = []
    attempts = 0

    while len(deltas) < max_samples and attempts < max_samples * 20:
        attempts += 1
        a, b, c, d = rng.sample(connected, 4)

        s1 = geodesic_matrix[a][b] + geodesic_matrix[c][d]
        s2 = geodesic_matrix[a][c] + geodesic_matrix[b][d]
        s3 = geodesic_matrix[a][d] + geodesic_matrix[b][c]

        if not all(math.isfinite(s) for s in [s1, s2, s3]):
            continue

        sums = sorted([s1, s2, s3], reverse=True)
        deltas.append((sums[0] - sums[1]) / 2.0)

    return {
        "delta_max": max(deltas) if deltas else float("nan"),
        "delta_mean": mean(deltas) if deltas else float("nan"),
        "samples": len(deltas),
    }


def extended_geometry_summary(geometry, geodesic_matrix, j_name=""):
    """
    Full S8 geometry summary: S7 metrics + dimension + curvature.
    """
    from s7_emergent_track_proto_geometry import summarize_geometry
    base = summarize_geometry(geometry, geodesic_matrix)

    cc_per_node, cc_mean = clustering_coefficient(geometry)
    gromov = gromov_delta_hyperbolicity(geodesic_matrix)
    mds = estimate_dimension_mds(geodesic_matrix)

    # Asymmetry measure (only meaningful for j_asymmetric)
    asym_vals = [e["asymmetry"] for e in geometry["edges"]]
    mean_asymmetry = mean(asym_vals) if asym_vals else 0.0

    return {
        **base,
        "j_name": j_name,
        "clustering_coefficient": cc_mean,
        "gromov_delta_max": gromov["delta_max"],
        "gromov_delta_mean": gromov["delta_mean"],
        "gromov_samples": gromov["samples"],
        "effective_dim": mds.get("effective_dim"),
        "mds_n_positive": mds.get("n_positive"),
        "mds_variance_top10": mds.get("variance_explained_top10", []),
        "mean_K_asymmetry": mean_asymmetry,
        "cc_per_node": cc_per_node,
    }


# ---------------------------------------------------------------------------
# 7. J-agnosticity run engine
# ---------------------------------------------------------------------------

def run_one_j(
    j_name,
    j_func,
    seed=0,
    output_dir="s8_outputs",
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    memory_lambda=0.08,
    memory_gain=0.04,
    memory_decay=0.985,
    min_lifetime=100,
    min_updates=10,
    coupling_threshold=0.25,
    verbose=False,
):
    """
    Full S8 pipeline for one J function.
    Returns summary dict for cross-J comparison.
    """
    random.seed(seed)

    states, history, tracks, snapshots, memory_history = run_simulation_j(
        j_func,
        cluster_threshold=cluster_threshold,
        sample_every=sample_every,
        overlap_threshold=overlap_threshold,
        memory_lambda=memory_lambda,
        memory_gain=memory_gain,
        memory_decay=memory_decay,
        verbose=verbose,
    )

    stable_tracks = select_stable_tracks(
        tracks,
        sample_every=sample_every,
        min_lifetime=min_lifetime,
        min_updates=min_updates,
    )

    directed = (j_name == "asymmetric")
    geometry = build_track_geometry_j(
        stable_tracks, states, j_func,
        coupling_threshold=coupling_threshold,
        directed=directed,
    )

    geodesic_matrix = geodesic_distance_matrix_from_geometry(geometry)
    summary = extended_geometry_summary(geometry, geodesic_matrix, j_name=j_name)

    final_avg_mem = memory_history[-1][1] if memory_history else 0.0
    final_max_mem = memory_history[-1][2] if memory_history else 0.0

    return {
        "j_name": j_name,
        "seed": seed,
        "final_states": len(states),
        "stable_tracks": len(stable_tracks),
        **{k: v for k, v in summary.items()
           if not isinstance(v, list)},  # skip per-node lists for CSV
        "final_avg_memory": round(final_avg_mem, 6),
        "final_max_memory": round(final_max_mem, 6),
        # keep these separately for plots
        "_geometry": geometry,
        "_geodesic": geodesic_matrix,
        "_history": history,
        "_memory_history": memory_history,
        "_summary": summary,
    }


def run_s8_j_agnosticity(
    seeds=(0, 1, 2, 3, 4),
    j_names=None,
    output_dir="s8_outputs",
    coupling_threshold=0.25,
    min_lifetime=100,
    min_updates=10,
):
    """
    Main S8 entry point: test all J functions across multiple seeds.

    Produces:
    - Per-J summary CSVs
    - Cross-J comparison plots
    - J-agnosticity verdict
    """
    if j_names is None:
        j_names = list(J_FAMILY.keys())

    print("=== S8 J-Agnosticity Test ===")
    print(f"J functions: {j_names}")
    print(f"Seeds: {list(seeds)}")
    print(f"Coupling threshold: {coupling_threshold}")
    print()

    all_results = []  # flat list: one dict per (j_name, seed)

    for j_name in j_names:
        j_func = J_FAMILY[j_name]
        print(f"--- J: {j_name} ---")

        for seed in seeds:
            result = run_one_j(
                j_name, j_func,
                seed=seed,
                output_dir=output_dir,
                coupling_threshold=coupling_threshold,
                min_lifetime=min_lifetime,
                min_updates=min_updates,
                verbose=False,
            )
            all_results.append(result)

            print(
                f"  seed {seed}  states {result['final_states']:3d}"
                f"  nodes {result['node_count']:2d}"
                f"  edges {result['edge_count']:3d}"
                f"  components {result['component_count']:2d}"
                f"  largest {result['largest_component']:2d}"
                f"  eff_dim {result['effective_dim']}"
                f"  CC {result['clustering_coefficient']:.3f}"
                f"  δ_max {result['gromov_delta_max']:.3f}"
                f"  geo_d {result['mean_geodesic_d']:.3f}"
            )

    # Save CSV
    save_agnosticity_csv(all_results, output_dir=output_dir)

    # Plot comparison
    plot_j_comparison(all_results, j_names, output_dir=output_dir)
    plot_dimension_curvature(all_results, j_names, output_dir=output_dir)

    # Verdict
    print_agnosticity_verdict(all_results, j_names)

    return all_results


# ---------------------------------------------------------------------------
# 8. Verdict logic
# ---------------------------------------------------------------------------

def print_agnosticity_verdict(all_results, j_names):
    """
    Prints J-agnosticity verdict based on comparison of structured vs random J.
    """
    print("\n=== J-Agnosticity Verdict ===")

    by_j = {}
    for r in all_results:
        jn = r["j_name"]
        if jn not in by_j:
            by_j[jn] = []
        by_j[jn].append(r)

    structured_j = [jn for jn in j_names if jn != "random"]
    random_results = by_j.get("random", [])

    if not random_results:
        print("No random baseline — cannot compute verdict.")
        return

    random_mean_edges = mean(r["edge_count"] for r in random_results)
    random_mean_largest = mean(r["largest_component"] for r in random_results)

    print(f"\nNull model (random J):")
    print(f"  mean edges:           {random_mean_edges:.1f}")
    print(f"  mean largest comp:    {random_mean_largest:.1f}")

    print(f"\nStructured J functions:")
    all_exceed = True

    for jn in structured_j:
        if jn not in by_j:
            continue
        results = by_j[jn]
        me = mean(r["edge_count"] for r in results)
        ml = mean(r["largest_component"] for r in results)
        md = mean(r.get("effective_dim") or 0 for r in results)
        cc = mean(r["clustering_coefficient"] for r in results)

        exceeds = (me > random_mean_edges * 1.2 or ml > random_mean_largest * 1.2)
        flag = "✓" if exceeds else "✗"
        all_exceed = all_exceed and exceeds

        print(
            f"  {flag} {jn:12s}  edges {me:.1f}  largest {ml:.1f}"
            f"  eff_dim {md:.1f}  CC {cc:.3f}"
        )

    print()
    if all_exceed:
        print("VERDICT: J-AGNOSTICITY HOLDS.")
        print("  Nontrivial proto-geometry emerges for all structured J functions.")
        print("  The geometry is a consequence of persistent identity networks,")
        print("  not an artifact of the specific coupling function form.")
    else:
        print("VERDICT: J-AGNOSTICITY PARTIAL OR FAILED.")
        print("  Some J functions do not produce structured geometry.")
        print("  Check coupling_threshold, min_lifetime, or increase steps.")


# ---------------------------------------------------------------------------
# 9. Output helpers
# ---------------------------------------------------------------------------

def save_agnosticity_csv(all_results, output_dir="s8_outputs", filename="s8_agnosticity.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    skip = {"_geometry", "_geodesic", "_history", "_memory_history", "_summary",
            "cc_per_node", "mds_variance_top10", "components", "degree",
            "weighted_degree", "closeness"}

    fieldnames = [k for k in all_results[0].keys() if k not in skip]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow({k: row[k] for k in fieldnames})

    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------

def plot_j_comparison(all_results, j_names, output_dir="s8_outputs"):
    """Bar comparison of key geometry metrics across J functions (mean over seeds)."""
    output_path = ensure_output_dir(output_dir) / "j_agnosticity_comparison.png"

    by_j = {jn: [r for r in all_results if r["j_name"] == jn] for jn in j_names}

    metrics = {
        "edge_count": "Edges",
        "largest_component": "Largest component",
        "mean_geodesic_d": "Mean geodesic distance",
        "clustering_coefficient": "Clustering coefficient",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    colors = ["steelblue", "tomato", "seagreen", "mediumpurple", "orange", "gray"]

    for ax, (metric, label) in zip(axes, metrics.items()):
        vals = [mean(r[metric] for r in by_j[jn]) if jn in by_j and by_j[jn] else 0
                for jn in j_names]
        bars = ax.bar(j_names, vals,
                      color=[colors[i % len(colors)] for i in range(len(j_names))])
        ax.set_title(label, fontsize=10)
        ax.set_xticks(range(len(j_names)))
        ax.set_xticklabels(j_names, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=8)

    fig.suptitle("S8: Proto-geometry metrics by J function (mean over seeds)", fontsize=11)
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print("saved:", output_path)


def plot_dimension_curvature(all_results, j_names, output_dir="s8_outputs"):
    """Effective dimension and Gromov δ across J functions."""
    output_path = ensure_output_dir(output_dir) / "j_dimension_curvature.png"

    by_j = {jn: [r for r in all_results if r["j_name"] == jn] for jn in j_names}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["steelblue", "tomato", "seagreen", "mediumpurple", "orange", "gray"]

    # Effective dimension
    dim_vals = []
    for jn in j_names:
        results = by_j.get(jn, [])
        dims = [r["effective_dim"] for r in results if r["effective_dim"] is not None]
        dim_vals.append(mean(dims) if dims else 0.0)

    ax1.bar(j_names, dim_vals,
            color=[colors[i % len(colors)] for i in range(len(j_names))])
    ax1.set_title("Effective dimension (MDS, 90% variance)", fontsize=10)
    ax1.set_xticks(range(len(j_names)))
    ax1.set_xticklabels(j_names, rotation=35, ha="right", fontsize=8)
    ax1.set_ylabel("dimension", fontsize=9)
    ax1.axhline(1, color="k", linestyle=":", alpha=0.4, label="1D")
    ax1.axhline(2, color="k", linestyle="--", alpha=0.4, label="2D")
    ax1.legend(fontsize=8)

    # Gromov δ
    gromov_vals = []
    for jn in j_names:
        results = by_j.get(jn, [])
        gs = [r["gromov_delta_max"] for r in results if math.isfinite(r.get("gromov_delta_max", float("nan")))]
        gromov_vals.append(mean(gs) if gs else float("nan"))

    ax2.bar(
        [jn for jn, v in zip(j_names, gromov_vals) if not math.isnan(v)],
        [v for v in gromov_vals if not math.isnan(v)],
        color=[colors[i % len(colors)] for i, v in enumerate(gromov_vals) if not math.isnan(v)]
    )
    ax2.set_title("Gromov δ (tree-like ≈ 0, Euclidean > 0)", fontsize=10)
    ax2.set_xticks(range(sum(1 for v in gromov_vals if not math.isnan(v))))
    ax2.set_xticklabels(
        [jn for jn, v in zip(j_names, gromov_vals) if not math.isnan(v)],
        rotation=35, ha="right", fontsize=8
    )
    ax2.set_ylabel("δ_max", fontsize=9)

    fig.suptitle("S8: Dimensionality and curvature by J function", fontsize=11)
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 11. Single detailed run (for debugging / inspection)
# ---------------------------------------------------------------------------

def run_s8_single_detailed(
    j_name="exponential",
    seed=0,
    output_dir="s8_outputs",
    coupling_threshold=0.25,
    min_lifetime=100,
    min_updates=10,
):
    """
    Full single run with verbose output and all plots for one J function.
    """
    print(f"\n=== S8 single detailed run: J={j_name}, seed={seed} ===")

    j_func = J_FAMILY[j_name]

    result = run_one_j(
        j_name, j_func,
        seed=seed,
        output_dir=output_dir,
        coupling_threshold=coupling_threshold,
        min_lifetime=min_lifetime,
        min_updates=min_updates,
        verbose=True,
    )

    s = result["_summary"]

    print(f"\nFinal states:        {result['final_states']}")
    print(f"Stable tracks:       {result['stable_tracks']}")
    print(f"Nodes:               {s['node_count']}")
    print(f"Edges:               {s['edge_count']}")
    print(f"Components:          {s['component_count']}")
    print(f"Largest component:   {s['largest_component']}")
    print(f"Mean geodesic d:     {s['mean_geodesic_d']:.5f}")
    print(f"Max geodesic d:      {s['max_geodesic_d']:.5f}")
    print(f"Clustering coeff:    {s['clustering_coefficient']:.5f}")
    print(f"Gromov δ_max:        {s['gromov_delta_max']:.5f}")
    print(f"Effective dim (MDS): {s['effective_dim']}")
    print(f"Mean K asymmetry:    {s['mean_K_asymmetry']:.5f}")

    if s.get("mds_variance_top10"):
        print(f"MDS cumvar (top5):   {s['mds_variance_top10'][:5]}")

    return result


# ---------------------------------------------------------------------------
# 12. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Quick single run for inspection
    run_s8_single_detailed(
        j_name="exponential",
        seed=0,
        output_dir="s8_outputs",
    )

    # Full J-agnosticity test: all 6 J functions, 5 seeds each
    run_s8_j_agnosticity(
        seeds=(0, 1, 2, 3, 4),
        j_names=["exponential", "step", "cosine", "power", "random", "asymmetric"],
        output_dir="s8_outputs",
        coupling_threshold=0.25,
        min_lifetime=100,
        min_updates=10,
    )


"""
S8 Interpretation Guide
=======================

Key question:
    Does nontrivial proto-geometric structure emerge for ALL structured J
    functions (exponential, step, cosine, power)?

Expected results if CoMath claim holds:
    - exponential/step/cosine/power: edges > 0, largest component > 1,
      finite geodesic distances, effective_dim >= 1
    - random J: near-zero or negligible geometry (null)

If random J produces geometry too:
    The result is noise-driven, not emergent — simulation parameters
    need adjustment (higher min_lifetime, higher coupling_threshold).

Dimension:
    effective_dim = 1: linear chain-like topology
    effective_dim = 2: surface-like
    effective_dim = 3: volume-like
    Euclidean space has exact integer dimension; proto-space may be fractional.

Curvature:
    CC ≈ 0, δ ≈ 0: hyperbolic / tree-like (sparse identity networks)
    CC > 0, δ > 0: positive curvature (dense clusters)

Asymmetry:
    For j_asymmetric, mean_K_asymmetry > 0 confirms directed coupling.
    Full directed geodesics are a future S9 extension.
"""
