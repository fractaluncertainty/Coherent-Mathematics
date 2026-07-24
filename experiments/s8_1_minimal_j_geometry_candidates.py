#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S8.1 — Minimal J-Geometry Candidate Test
========================================

Follow-up to S8 J-ablation.

S8 showed:
    metric_exp produced stable tracks and proto-geometric networks, while
    cosine/mixed/threshold/random couplings preserved populations but collapsed
    the track geometry to a single stable track with no edges.

S8.1 therefore tests a sharper hypothesis:

    Proto-geometry may not require the exact exponential metric J, but it may
    require a local, gradual, difference-sensitive neighborhood order.

Candidate J-families tested here:

    metric_exp                 original S7/S8 control
    exp_alpha_*                exponential local similarity with varied scale
    rational_p*                algebraic monotone decay
    linear_cutoff_*            compact-support local neighborhood
    gaussian_l2_*              squared smooth local neighborhood
    linf_exp_*                 non-Euclidean L_infinity local decay
    anisotropic_exp            unequal sigma/chi relevance

Controls retained from S8:

    cosine_resonance           periodic resonance control
    mixed_resonance            non-monotonic resonance control
    threshold_resonance        hard gate control
    random_projection          deterministic random-feature control

This is a toy-model artifact test, not a proof of FUT/CoMath.
"""

import csv
import math
import random
import heapq
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


class State:
    def __init__(self, state_id, w, sigma, chi, memory=0.0):
        self.id = state_id
        self.w = w
        self.sigma = sigma
        self.chi = chi
        self.memory = memory


def angle_diff(a, b):
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def J_metric_exp(x, y):
    return math.exp(-angle_diff(x.sigma, y.sigma) - angle_diff(x.chi, y.chi))


# ---------------------------------------------------------------------------
# S8.1 candidate family: local, gradual, difference-sensitive J functions
# ---------------------------------------------------------------------------

def d_l1(x, y):
    """Circular L1 distance in the two internal coordinates."""
    return angle_diff(x.sigma, y.sigma) + angle_diff(x.chi, y.chi)


def d_l2(x, y):
    """Circular L2 distance in the two internal coordinates."""
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    return math.sqrt(ds * ds + dc * dc)


def d_linf(x, y):
    """Circular L_infinity distance: non-Euclidean max-coordinate distance."""
    return max(angle_diff(x.sigma, y.sigma), angle_diff(x.chi, y.chi))


def J_exp_alpha(alpha):
    """Exponential monotone local similarity with tunable decay scale."""
    def J(x, y):
        return math.exp(-alpha * d_l1(x, y))
    J.__name__ = f"J_exp_alpha_{alpha}"
    return J


def J_rational(p=1.0, scale=1.0):
    """Algebraic monotone local similarity. Not exponential, but ordered/local."""
    def J(x, y):
        return 1.0 / (1.0 + (d_l1(x, y) / scale) ** p)
    J.__name__ = f"J_rational_p{p}_s{scale}"
    return J


def J_linear_cutoff(lambda_=0.25):
    """Compact-support local similarity: linear decay to zero."""
    def J(x, y):
        return max(0.0, 1.0 - lambda_ * d_l1(x, y))
    J.__name__ = f"J_linear_cutoff_{lambda_}"
    return J


def J_gaussian_l2(beta=1.0):
    """Smooth squared-distance local similarity."""
    def J(x, y):
        d = d_l2(x, y)
        return math.exp(-beta * d * d)
    J.__name__ = f"J_gaussian_l2_{beta}"
    return J


def J_linf_exp(alpha=1.0):
    """Exponential decay using L_infinity distance, not the original L1 form."""
    def J(x, y):
        return math.exp(-alpha * d_linf(x, y))
    J.__name__ = f"J_linf_exp_{alpha}"
    return J


def J_anisotropic_exp(x, y):
    """Local monotone similarity with unequal sigma/chi weights."""
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    return math.exp(-0.65 * ds - 1.35 * dc)


def J_cosine_resonance(x, y):
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    raw = 0.5 * (math.cos(ds) + math.cos(dc))
    return max(0.0, min(1.0, 0.5 + 0.5 * raw))


def J_mixed_resonance(x, y):
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    raw = (
        0.9 * math.cos(2.0 * ds)
        + 0.7 * math.sin(3.0 * dc)
        + 0.6 * math.cos(ds - 2.0 * dc)
        - 0.25 * math.sin(2.0 * ds + dc)
    )
    return sigmoid(raw)


def J_threshold_resonance(x, y):
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    score = (math.cos(2.0 * ds) + math.cos(3.0 * dc) + 0.5 * math.cos(ds - dc)) / 2.5
    if score > 0.45:
        return 0.85
    if score > 0.10:
        return 0.35
    return 0.02


def deterministic_random_feature(value, seed):
    z = math.sin(value * 12.9898 + seed * 78.233) * 43758.5453
    frac = z - math.floor(z)
    return 2.0 * frac - 1.0


def J_random_projection(x, y):
    fx = []
    fy = []
    for seed in range(8):
        vx = math.sin((seed + 1) * x.sigma) + math.cos((seed + 2) * x.chi)
        vy = math.sin((seed + 1) * y.sigma) + math.cos((seed + 2) * y.chi)
        fx.append(deterministic_random_feature(vx, seed))
        fy.append(deterministic_random_feature(vy, seed))
    dot = sum(a * b for a, b in zip(fx, fy)) / len(fx)
    return sigmoid(2.5 * dot)


J_FUNCTIONS = {
    # S7/S8 control
    "metric_exp": J_metric_exp,

    # S8.1 monotone local candidate family
    "exp_alpha_0_5": J_exp_alpha(0.5),
    "exp_alpha_1_0": J_exp_alpha(1.0),
    "exp_alpha_2_0": J_exp_alpha(2.0),
    "rational_p1": J_rational(p=1.0, scale=1.0),
    "rational_p2": J_rational(p=2.0, scale=1.0),
    "linear_cutoff_0_20": J_linear_cutoff(lambda_=0.20),
    "linear_cutoff_0_35": J_linear_cutoff(lambda_=0.35),
    "gaussian_l2_0_5": J_gaussian_l2(beta=0.5),
    "gaussian_l2_1_0": J_gaussian_l2(beta=1.0),
    "linf_exp_1_0": J_linf_exp(alpha=1.0),
    "anisotropic_exp": J_anisotropic_exp,

    # S8 controls that previously failed to generate geometry
    "cosine_resonance": J_cosine_resonance,
    "mixed_resonance": J_mixed_resonance,
    "threshold_resonance": J_threshold_resonance,
    "random_projection": J_random_projection,
}


def get_J(j_name):
    if j_name not in J_FUNCTIONS:
        raise ValueError(f"Unknown J function: {j_name}")
    return J_FUNCTIONS[j_name]


def I(x, y, J_func):
    return J_func(x, y) * math.cos(angle_diff(x.chi, y.chi))


def local_persistence(x, states, J_func):
    others = [y for y in states if y is not x]
    if not others:
        return 0.0
    return x.w * sum(
        J_func(x, y) * max(0.0, I(x, y, J_func))
        for y in others
    ) / len(others)


def persistence_with_memory(x, states, J_func, memory_lambda=0.08):
    return local_persistence(x, states, J_func) + memory_lambda * x.memory


def recurse(x, memory_decay=0.985):
    eta = random.uniform(-0.02, 0.04)
    kappa = random.uniform(0.00, 0.02)
    delta = random.uniform(-0.08, 0.08)
    phi = random.uniform(-0.08, 0.08)
    return State(
        x.id,
        max(0.0, x.w + eta - kappa),
        (x.sigma + delta) % (2 * math.pi),
        (x.chi + phi) % (2 * math.pi),
        memory=max(0.0, x.memory * memory_decay),
    )


def split(x, next_id):
    return State(
        next_id,
        x.w * random.uniform(0.8, 1.05),
        (x.sigma + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        (x.chi + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        memory=x.memory * random.uniform(0.4, 0.8),
    )


def find_clusters(states, J_func, threshold=0.6):
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
                if j not in visited and J_func(states[current], states[j]) > threshold:
                    stack.append(j)
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters


def jaccard_overlap(a, b):
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def assign_clusters_to_tracks(cluster_records, active_tracks, finished_tracks, overlap_threshold, next_track_id_ref):
    assigned_tracks = set()
    new_active_tracks = {}

    for cluster in cluster_records:
        best_track_id = None
        best_overlap = 0.0

        for track_id, track in active_tracks.items():
            if track_id in assigned_tracks:
                continue
            overlap = jaccard_overlap(track["last_ids"], cluster["ids"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_track_id = track_id

        if best_track_id is not None and best_overlap >= overlap_threshold:
            track = active_tracks[best_track_id]
            track["end_step"] = cluster["step"]
            track["last_ids"] = cluster["ids"]
            track["sizes"].append(cluster["size"])
            track["overlaps"].append(best_overlap)
            track["updates"] += 1
            new_active_tracks[best_track_id] = track
            assigned_tracks.add(best_track_id)
        else:
            track_id = next_track_id_ref[0]
            next_track_id_ref[0] += 1
            new_active_tracks[track_id] = {
                "track_id": track_id,
                "start_step": cluster["step"],
                "end_step": cluster["step"],
                "last_ids": cluster["ids"],
                "sizes": [cluster["size"]],
                "overlaps": [],
                "updates": 1,
            }

    for track_id, track in active_tracks.items():
        if track_id not in assigned_tracks and track_id not in new_active_tracks:
            finished_tracks.append(track)

    active_tracks.clear()
    active_tracks.update(new_active_tracks)


def track_lifetime(track, sample_every=5):
    return track["end_step"] - track["start_step"] + sample_every


def apply_track_memory_feedback(states, active_tracks, sample_every=5, memory_gain=0.04, max_memory=1.0, min_track_updates=3):
    id_to_state = {s.id: s for s in states}
    for track in active_tracks.values():
        if track["updates"] < min_track_updates:
            continue
        lifetime = track_lifetime(track, sample_every=sample_every)
        strength = memory_gain * min(1.0, lifetime / 100.0)
        for state_id in track["last_ids"]:
            if state_id in id_to_state:
                state = id_to_state[state_id]
                state.memory = min(max_memory, state.memory + strength)


def run_simulation(
    J_func,
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
):
    next_id = 0
    states = []
    for _ in range(initial_count):
        states.append(
            State(
                next_id,
                random.uniform(0.2, 1.0),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
                memory=0.0,
            )
        )
        next_id += 1

    history = []
    memory_history = []
    active_tracks = {}
    finished_tracks = []
    next_track_id = 0

    for step in range(steps):
        states = [recurse(x, memory_decay=memory_decay) for x in states]
        scored = [(x, persistence_with_memory(x, states, J_func, memory_lambda=memory_lambda)) for x in states]
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
            clusters = find_clusters(states, J_func, threshold=cluster_threshold)
            cluster_records = []
            for cluster in clusters:
                ids = frozenset(states[i].id for i in cluster)
                cluster_records.append({"step": step, "ids": ids, "size": len(ids)})

            assign_clusters_to_tracks(cluster_records, active_tracks, finished_tracks, overlap_threshold, [next_track_id])

            if active_tracks:
                next_track_id = max(active_tracks.keys()) + 1
            elif finished_tracks:
                next_track_id = max(t["track_id"] for t in finished_tracks) + 1
            else:
                next_track_id = 0

            apply_track_memory_feedback(
                states,
                active_tracks,
                sample_every=sample_every,
                memory_gain=memory_gain,
                max_memory=max_memory,
                min_track_updates=min_track_updates,
            )

        if not states:
            break

    finished_tracks.extend(active_tracks.values())
    return states, history, finished_tracks, memory_history


def select_stable_tracks(tracks, sample_every=5, min_lifetime=100, min_updates=10):
    return [
        t for t in tracks
        if track_lifetime(t, sample_every=sample_every) >= min_lifetime and t["updates"] >= min_updates
    ]


def representative_ids(track, max_ids=10):
    return set(list(track["last_ids"])[:max_ids])


def track_coupling(track_a, track_b, id_state, J_func, max_ids=10):
    ids_a = representative_ids(track_a, max_ids=max_ids)
    ids_b = representative_ids(track_b, max_ids=max_ids)
    states_a = [id_state[i] for i in ids_a if i in id_state]
    states_b = [id_state[i] for i in ids_b if i in id_state]
    if not states_a or not states_b:
        return 0.0
    values = []
    for a in states_a:
        for b in states_b:
            if a.id != b.id:
                values.append(J_func(a, b))
    return mean(values) if values else 0.0


def build_track_geometry(stable_tracks, final_states, J_func, coupling_threshold=0.25, max_ids_per_track=10):
    id_state = {s.id: s for s in final_states}
    track_ids = [t["track_id"] for t in stable_tracks]
    edges = []
    for i, track_a in enumerate(stable_tracks):
        for j in range(i + 1, len(stable_tracks)):
            track_b = stable_tracks[j]
            k = track_coupling(track_a, track_b, id_state, J_func, max_ids=max_ids_per_track)
            if k >= coupling_threshold:
                edges.append({
                    "i": i,
                    "j": j,
                    "a": track_ids[i],
                    "b": track_ids[j],
                    "K": k,
                    "d": -math.log(max(k, 1e-12)),
                })
    return {"track_ids": track_ids, "edges": edges}


def adjacency_from_edges(n, edges):
    adjacency = {i: [] for i in range(n)}
    for edge in edges:
        adjacency[edge["i"]].append((edge["j"], edge["d"]))
        adjacency[edge["j"]].append((edge["i"], edge["d"]))
    return adjacency


def dijkstra(adjacency, source):
    distances = {node: math.inf for node in adjacency}
    distances[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for neighbor, weight in adjacency[node]:
            nd = dist + weight
            if nd < distances[neighbor]:
                distances[neighbor] = nd
                heapq.heappush(heap, (nd, neighbor))
    return distances


def geodesic_distances(geometry):
    n = len(geometry["track_ids"])
    adjacency = adjacency_from_edges(n, geometry["edges"])
    finite = []
    for i in range(n):
        dists = dijkstra(adjacency, i)
        for j, d in dists.items():
            if j > i and math.isfinite(d):
                finite.append(d)
    return finite


def network_components(geometry):
    n = len(geometry["track_ids"])
    adjacency = {i: set() for i in range(n)}
    for edge in geometry["edges"]:
        adjacency[edge["i"]].add(edge["j"])
        adjacency[edge["j"]].add(edge["i"])

    visited = set()
    components = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]
        comp = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            for nxt in adjacency[node]:
                if nxt not in visited:
                    stack.append(nxt)
        components.append(comp)
    components.sort(key=len, reverse=True)
    return components


def summarize_geometry(geometry):
    n = len(geometry["track_ids"])
    edges = geometry["edges"]
    comps = network_components(geometry)
    degrees = [0 for _ in range(n)]
    for edge in edges:
        degrees[edge["i"]] += 1
        degrees[edge["j"]] += 1

    edge_K = [e["K"] for e in edges]
    edge_d = [e["d"] for e in edges]
    geo = geodesic_distances(geometry)
    return {
        "node_count": n,
        "edge_count": len(edges),
        "component_count": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "mean_degree": mean(degrees) if degrees else 0.0,
        "max_degree": max(degrees) if degrees else 0,
        "mean_edge_K": mean(edge_K) if edge_K else 0.0,
        "mean_edge_d": mean(edge_d) if edge_d else 0.0,
        "mean_geodesic_d": mean(geo) if geo else 0.0,
        "max_geodesic_d": max(geo) if geo else 0.0,
    }


def run_one(j_name, seed, cluster_threshold=0.6, coupling_threshold=0.25, sample_every=5, min_lifetime=100, min_updates=10):
    random.seed(seed)
    J_func = get_J(j_name)

    states, history, tracks, memory_history = run_simulation(
        J_func,
        cluster_threshold=cluster_threshold,
        sample_every=sample_every,
    )
    stable_tracks = select_stable_tracks(
        tracks,
        sample_every=sample_every,
        min_lifetime=min_lifetime,
        min_updates=min_updates,
    )
    geometry = build_track_geometry(
        stable_tracks,
        states,
        J_func,
        coupling_threshold=coupling_threshold,
    )
    summary = summarize_geometry(geometry)
    final_avg_memory = memory_history[-1][1] if memory_history else 0.0
    final_max_memory = memory_history[-1][2] if memory_history else 0.0
    final_avg_p = history[-1][2] if history else 0.0

    return {
        "j_name": j_name,
        "seed": seed,
        "final_states": len(states),
        "final_avg_p": final_avg_p,
        "track_count": len(tracks),
        "stable_tracks": len(stable_tracks),
        "node_count": summary["node_count"],
        "edge_count": summary["edge_count"],
        "component_count": summary["component_count"],
        "largest_component": summary["largest_component"],
        "mean_degree": summary["mean_degree"],
        "max_degree": summary["max_degree"],
        "mean_edge_K": summary["mean_edge_K"],
        "mean_edge_d": summary["mean_edge_d"],
        "mean_geodesic_d": summary["mean_geodesic_d"],
        "max_geodesic_d": summary["max_geodesic_d"],
        "final_avg_memory": final_avg_memory,
        "final_max_memory": final_max_memory,
    }


def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_records_csv(records, output_dir="s8_outputs", filename="s8_1_j_candidate_results.csv"):
    output_path = ensure_output_dir(output_dir) / filename
    fieldnames = [
        "j_name", "seed", "final_states", "final_avg_p", "track_count",
        "stable_tracks", "node_count", "edge_count", "component_count",
        "largest_component", "mean_degree", "max_degree", "mean_edge_K",
        "mean_edge_d", "mean_geodesic_d", "max_geodesic_d",
        "final_avg_memory", "final_max_memory",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print("saved:", output_path)


def grouped(records):
    groups = {}
    for r in records:
        groups.setdefault(r["j_name"], []).append(r)
    return groups


def print_summary_by_j(records):
    print("\n=== S8.1 summary by J-function ===")
    for j_name, rows in grouped(records).items():
        print(
            "J", j_name,
            "mean_states", round(mean(r["final_states"] for r in rows), 3),
            "mean_stable_tracks", round(mean(r["stable_tracks"] for r in rows), 3),
            "mean_edges", round(mean(r["edge_count"] for r in rows), 3),
            "mean_components", round(mean(r["component_count"] for r in rows), 3),
            "mean_largest", round(mean(r["largest_component"] for r in rows), 3),
            "mean_degree", round(mean(r["mean_degree"] for r in rows), 3),
            "mean_edge_K", round(mean(r["mean_edge_K"] for r in rows), 5),
            "mean_geo", round(mean(r["mean_geodesic_d"] for r in rows), 5),
            "collapse_runs", sum(1 for r in rows if r["final_states"] == 0),
        )


def plot_summary_by_j(records, output_dir="s8_outputs", filename="s8_1_j_candidate_summary.png"):
    output_path = ensure_output_dir(output_dir) / filename
    groups = grouped(records)
    names = list(groups.keys())

    mean_states = [mean(r["final_states"] for r in groups[n]) for n in names]
    mean_tracks = [mean(r["stable_tracks"] for r in groups[n]) for n in names]
    mean_edges = [mean(r["edge_count"] for r in groups[n]) for n in names]
    mean_largest = [mean(r["largest_component"] for r in groups[n]) for n in names]

    x = list(range(len(names)))

    plt.figure(figsize=(12, 6))
    plt.plot(x, mean_states, marker="o", label="mean final states")
    plt.plot(x, mean_tracks, marker="o", label="mean stable tracks")
    plt.plot(x, mean_edges, marker="o", label="mean network edges")
    plt.plot(x, mean_largest, marker="o", label="mean largest component")
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylabel("mean statistic")
    plt.title("S8.1 minimal J-geometry candidate summary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    print("saved:", output_path)


def run_s8_1_ablation(runs=50, j_names=None, output_dir="s8_outputs", cluster_threshold=0.6, coupling_threshold=0.25):
    if j_names is None:
        j_names = list(J_FUNCTIONS.keys())

    print("=== S8.1 minimal J-geometry candidate test ===")
    print("runs per J:", runs)
    print("J-functions:", j_names)

    records = []

    for j_name in j_names:
        print("\n--- J:", j_name, "---")
        for seed in range(runs):
            record = run_one(
                j_name,
                seed,
                cluster_threshold=cluster_threshold,
                coupling_threshold=coupling_threshold,
            )
            records.append(record)
            print(
                "J", j_name,
                "seed", seed,
                "states", record["final_states"],
                "stable_tracks", record["stable_tracks"],
                "edges", record["edge_count"],
                "components", record["component_count"],
                "largest", record["largest_component"],
                "mean_geo", round(record["mean_geodesic_d"], 4),
            )

    save_records_csv(records, output_dir=output_dir)
    print_summary_by_j(records)
    plot_summary_by_j(records, output_dir=output_dir)
    return records


if __name__ == "__main__":
    # For quick tests, reduce runs to 10.
    # For stronger statistics, increase to 100 or more.
    run_s8_1_ablation(
        runs=50,
        j_names=[
            "metric_exp",
            "exp_alpha_0_5",
            "exp_alpha_1_0",
            "exp_alpha_2_0",
            "rational_p1",
            "rational_p2",
            "linear_cutoff_0_20",
            "linear_cutoff_0_35",
            "gaussian_l2_0_5",
            "gaussian_l2_1_0",
            "linf_exp_1_0",
            "anisotropic_exp",
            "cosine_resonance",
            "mixed_resonance",
            "threshold_resonance",
            "random_projection",
        ],
        output_dir="s8_1_outputs",
        cluster_threshold=0.6,
        coupling_threshold=0.25,
    )


"""
How to interpret S8.1
=====================

If only metric_exp produces stable geometry:
    The result is likely tied specifically to the original exponential L1 form.

If other monotone local candidates also produce stable tracks, nontrivial
components, and finite geodesics:
    The necessary structure is probably not the exact formula exp(-d), but a
    broader local, gradual, difference-sensitive neighborhood order.

If resonance/random controls still fail:
    Mere population persistence or resonance is not enough for proto-geometry.

Output files:
    s8_1_outputs/s8_1_j_candidate_results.csv
    s8_1_outputs/s8_1_j_candidate_summary.png
"""
