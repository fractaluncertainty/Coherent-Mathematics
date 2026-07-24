#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S7 — Emergent Proto-Geometry from Coupled Identity Networks
===========================================================

S7 extends S6.

S6:
    Stable identity tracks formed coupled relation networks.

S7:
    The track-track coupling network is interpreted as an emergent
    proto-geometric structure.

Core transition:
----------------

    identity -> relation -> proto-geometry

Track coupling:
---------------

    K(T_i, T_j) = mean connection capability between representative states
                  of two stable identity tracks

Track proto-distance:
---------------------

    d_T(T_i, T_j) = -log(K(T_i, T_j))

Network geodesic distance:
--------------------------

    shortest path distance on the coupled identity network,
    using edge length -log(K).

S7 asks:
--------

    Do persistent identity networks induce nontrivial distance,
    neighborhood, centrality, and proto-geometric structure?

Important:
----------
This is NOT proof of FUT/CoMath.
It is a toy model testing whether primitive recoherence dynamics can generate
a structured proto-geometry from persistent relations.
"""

import csv
import math
import random
import heapq
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. State definition
# ---------------------------------------------------------------------------

class State:
    """
    A simplified continuation profile.

    id:
        Stable numerical identifier for temporal tracking.

    w:
        Continuation weight / internal stability.

    sigma:
        Connection orientation.

    chi:
        Interference phase.

    memory:
        Weak accumulated influence from previous persistent structures.
    """

    def __init__(self, state_id, w, sigma, chi, memory=0.0):
        self.id = state_id
        self.w = w
        self.sigma = sigma
        self.chi = chi
        self.memory = memory


# ---------------------------------------------------------------------------
# 2. Core recoherence functions
# ---------------------------------------------------------------------------

def angle_diff(a, b):
    """Smallest angular distance between two angles on a circle."""
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def J(x, y):
    """Connection capability between two states."""
    return math.exp(
        -angle_diff(x.sigma, y.sigma)
        -angle_diff(x.chi, y.chi)
    )


def I(x, y):
    """Interference between two states."""
    return J(x, y) * math.cos(angle_diff(x.chi, y.chi))


def proto_distance_state(x, y):
    """State-level proto-metric g_f(x,y) = -log(J(x,y))."""
    join = max(J(x, y), 1e-12)
    return -math.log(join)


def local_persistence(x, states):
    """
    Persistence without memory feedback.
    """
    others = [y for y in states if y is not x]

    if not others:
        return 0.0

    return x.w * sum(
        J(x, y) * max(0.0, I(x, y))
        for y in others
    ) / len(others)


def persistence_with_memory(x, states, memory_lambda=0.08):
    """
    Persistence with weak recursive memory feedback.
    """
    return local_persistence(x, states) + memory_lambda * x.memory


# ---------------------------------------------------------------------------
# 3. Primitive dynamics
# ---------------------------------------------------------------------------

def recurse(x, memory_decay=0.985):
    """
    Recursive continuation of one state.
    """
    eta = random.uniform(-0.02, 0.04)
    kappa = random.uniform(0.00, 0.02)
    delta = random.uniform(-0.08, 0.08)
    phi = random.uniform(-0.08, 0.08)

    return State(
        x.id,
        max(0.0, x.w + eta - kappa),
        (x.sigma + delta) % (2 * math.pi),
        (x.chi + phi) % (2 * math.pi),
        memory=max(0.0, x.memory * memory_decay)
    )


def split(x, next_id):
    """
    Creates a similar successor state with a new identity id.
    """
    return State(
        next_id,
        x.w * random.uniform(0.8, 1.05),
        (x.sigma + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        (x.chi + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        memory=x.memory * random.uniform(0.4, 0.8)
    )


# ---------------------------------------------------------------------------
# 4. Cluster detection and identity tracking
# ---------------------------------------------------------------------------

def find_clusters(states, threshold=0.6):
    """
    Finds recoherence clusters as connected components.

    States are linked if J(state_i, state_j) > threshold.
    """
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
                if j not in visited and J(states[current], states[j]) > threshold:
                    stack.append(j)

        clusters.append(cluster)

    clusters.sort(key=len, reverse=True)
    return clusters


def jaccard_overlap(a, b):
    """Jaccard overlap |A ∩ B| / |A ∪ B|."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def assign_clusters_to_tracks(
    cluster_records,
    active_tracks,
    finished_tracks,
    overlap_threshold,
    next_track_id_ref
):
    """
    Assigns current clusters to active identity tracks using maximum overlap.
    """
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
    """Lifetime measured in simulation steps."""
    return track["end_step"] - track["start_step"] + sample_every


def apply_track_memory_feedback(
    states,
    active_tracks,
    sample_every=5,
    memory_gain=0.04,
    max_memory=1.0,
    min_track_updates=3
):
    """
    Adds weak memory to states belonging to sufficiently persistent tracks.
    """
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


# ---------------------------------------------------------------------------
# 5. Main simulation
# ---------------------------------------------------------------------------

def run_simulation_with_self_stabilizing_identity(
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
    verbose=True
):
    """
    Runs the S7 base dynamics.

    Same dynamical core as S5/S6. S7 adds proto-geometric analysis afterward.
    """
    next_id = 0
    states = []

    for _ in range(initial_count):
        states.append(
            State(
                next_id,
                random.uniform(0.2, 1.0),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi),
                memory=0.0
            )
        )
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
            (x, persistence_with_memory(x, states, memory_lambda=memory_lambda))
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
            clusters = find_clusters(states, threshold=cluster_threshold)

            cluster_records = []
            for cluster in clusters:
                ids = frozenset(states[i].id for i in cluster)
                cluster_records.append({
                    "step": step,
                    "ids": ids,
                    "size": len(ids),
                })

            assign_clusters_to_tracks(
                cluster_records,
                active_tracks,
                finished_tracks,
                overlap_threshold,
                next_track_id_ref=[next_track_id]
            )

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
                min_track_updates=min_track_updates
            )

            snapshots.append({
                "step": step,
                "clusters": cluster_records,
                "active_track_ids": list(active_tracks.keys()),
            })

        if verbose and step % 25 == 0:
            print(
                "step", step,
                "states", len(states),
                "avg_p", round(avg_p, 5),
                "avg_memory", round(avg_mem, 5),
                "max_memory", round(max_mem, 5)
            )

        if not states:
            break

    finished_tracks.extend(active_tracks.values())

    return states, history, finished_tracks, snapshots, memory_history


# ---------------------------------------------------------------------------
# 6. Track-network and proto-geometry
# ---------------------------------------------------------------------------

def select_stable_tracks(tracks, sample_every=5, min_lifetime=100, min_updates=10):
    """
    Selects stable tracks for proto-geometric analysis.
    """
    stable = []

    for track in tracks:
        lifetime = track_lifetime(track, sample_every=sample_every)

        if lifetime >= min_lifetime and track["updates"] >= min_updates:
            stable.append(track)

    return stable


def representative_ids(track, max_ids=10):
    """
    Representative final id set of a track.
    """
    ids = list(track["last_ids"])
    return set(ids[:max_ids])


def build_id_state_map(states):
    """Maps final state ids to final State objects."""
    return {s.id: s for s in states}


def track_coupling(track_a, track_b, id_state, max_ids=10):
    """
    Computes K(T_a,T_b), the mean state-level J between representative
    final states of two tracks.
    """
    ids_a = representative_ids(track_a, max_ids=max_ids)
    ids_b = representative_ids(track_b, max_ids=max_ids)

    states_a = [id_state[i] for i in ids_a if i in id_state]
    states_b = [id_state[i] for i in ids_b if i in id_state]

    if not states_a or not states_b:
        return 0.0

    values = []

    for a in states_a:
        for b in states_b:
            if a.id == b.id:
                continue
            values.append(J(a, b))

    return mean(values) if values else 0.0


def build_track_geometry(
    stable_tracks,
    final_states,
    coupling_threshold=0.25,
    max_ids_per_track=10
):
    """
    Builds track coupling matrix, direct proto-distance matrix, and graph edges.

    K_matrix:
        K(T_i,T_j)

    D_matrix:
        direct proto-distance d_T = -log(K)

    Edges:
        K >= coupling_threshold
    """
    id_state = build_id_state_map(final_states)

    track_ids = [t["track_id"] for t in stable_tracks]
    n = len(stable_tracks)

    K_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    D_matrix = [[math.inf for _ in range(n)] for _ in range(n)]
    edges = []

    for i in range(n):
        K_matrix[i][i] = 1.0
        D_matrix[i][i] = 0.0

    for i, track_a in enumerate(stable_tracks):
        for j in range(i + 1, n):
            track_b = stable_tracks[j]
            k = track_coupling(
                track_a,
                track_b,
                id_state,
                max_ids=max_ids_per_track
            )

            K_matrix[i][j] = k
            K_matrix[j][i] = k

            if k > 0:
                d = -math.log(max(k, 1e-12))
                D_matrix[i][j] = d
                D_matrix[j][i] = d

            if k >= coupling_threshold:
                edges.append({
                    "i": i,
                    "j": j,
                    "a": track_ids[i],
                    "b": track_ids[j],
                    "K": k,
                    "d": -math.log(max(k, 1e-12)),
                })

    return {
        "tracks": stable_tracks,
        "track_ids": track_ids,
        "K_matrix": K_matrix,
        "D_matrix": D_matrix,
        "edges": edges,
        "coupling_threshold": coupling_threshold,
    }


def adjacency_from_edges(n, edges):
    """
    Weighted adjacency list from edge list.
    Edge weights are proto-distances d=-log(K).
    """
    adjacency = {i: [] for i in range(n)}

    for edge in edges:
        adjacency[edge["i"]].append((edge["j"], edge["d"]))
        adjacency[edge["j"]].append((edge["i"], edge["d"]))

    return adjacency


def dijkstra(adjacency, source):
    """
    Shortest path distances from one source.
    """
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


def geodesic_distance_matrix(geometry):
    """
    Network geodesic distance matrix using d=-log(K) as edge length.
    """
    n = len(geometry["track_ids"])
    adjacency = adjacency_from_edges(n, geometry["edges"])

    G = [[math.inf for _ in range(n)] for _ in range(n)]

    for i in range(n):
        distances = dijkstra(adjacency, i)

        for j, d in distances.items():
            G[i][j] = d

    return G


def network_components_from_geometry(geometry):
    """
    Connected components of the thresholded track network.
    """
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


def degree_centrality(geometry):
    """
    Weighted and unweighted degree centrality.
    """
    n = len(geometry["track_ids"])

    degree = [0 for _ in range(n)]
    weighted_degree = [0.0 for _ in range(n)]

    for edge in geometry["edges"]:
        i = edge["i"]
        j = edge["j"]
        k = edge["K"]

        degree[i] += 1
        degree[j] += 1
        weighted_degree[i] += k
        weighted_degree[j] += k

    return degree, weighted_degree


def closeness_centrality(geodesic_matrix):
    """
    Closeness centrality on finite geodesic distances.
    """
    centrality = []

    for row in geodesic_matrix:
        finite = [d for d in row if d > 0 and math.isfinite(d)]

        if not finite:
            centrality.append(0.0)
        else:
            centrality.append(len(finite) / sum(finite))

    return centrality


def summarize_geometry(geometry, geodesic_matrix):
    """
    Compact summary of S7 proto-geometry.
    """
    n = len(geometry["track_ids"])
    edges = geometry["edges"]
    components = network_components_from_geometry(geometry)
    degree, weighted_degree = degree_centrality(geometry)
    closeness = closeness_centrality(geodesic_matrix)

    edge_K = [e["K"] for e in edges]
    edge_d = [e["d"] for e in edges]

    finite_geo = []
    for i in range(n):
        for j in range(i + 1, n):
            d = geodesic_matrix[i][j]
            if math.isfinite(d):
                finite_geo.append(d)

    return {
        "node_count": n,
        "edge_count": len(edges),
        "component_count": len(components),
        "largest_component": len(components[0]) if components else 0,
        "mean_degree": mean(degree) if degree else 0.0,
        "max_degree": max(degree) if degree else 0,
        "mean_weighted_degree": mean(weighted_degree) if weighted_degree else 0.0,
        "mean_edge_K": mean(edge_K) if edge_K else 0.0,
        "mean_edge_d": mean(edge_d) if edge_d else 0.0,
        "mean_geodesic_d": mean(finite_geo) if finite_geo else 0.0,
        "max_geodesic_d": max(finite_geo) if finite_geo else 0.0,
        "mean_closeness": mean(closeness) if closeness else 0.0,
        "max_closeness": max(closeness) if closeness else 0.0,
        "components": components,
        "degree": degree,
        "weighted_degree": weighted_degree,
        "closeness": closeness,
    }


# ---------------------------------------------------------------------------
# 7. Output helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_edges_csv(geometry, output_dir="s7_outputs", filename="track_geometry_edges.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["i", "j", "a", "b", "K", "d"])
        writer.writeheader()
        for edge in geometry["edges"]:
            writer.writerow(edge)

    print("saved:", output_path)


def save_node_geometry_csv(
    geometry,
    summary,
    output_dir="s7_outputs",
    filename="track_geometry_nodes.csv"
):
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "index",
        "track_id",
        "degree",
        "weighted_degree",
        "closeness",
        "lifetime",
        "updates",
        "mean_track_size",
        "max_track_size",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, track in enumerate(geometry["tracks"]):
            sizes = track["sizes"]

            writer.writerow({
                "index": i,
                "track_id": geometry["track_ids"][i],
                "degree": summary["degree"][i],
                "weighted_degree": summary["weighted_degree"][i],
                "closeness": summary["closeness"][i],
                "lifetime": track_lifetime(track),
                "updates": track["updates"],
                "mean_track_size": mean(sizes) if sizes else 0.0,
                "max_track_size": max(sizes) if sizes else 0,
            })

    print("saved:", output_path)


def save_s7_batch_csv(records, output_dir="s7_outputs", filename="s7_batch_summary.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "seed",
        "final_states",
        "node_count",
        "edge_count",
        "component_count",
        "largest_component",
        "mean_degree",
        "max_degree",
        "mean_weighted_degree",
        "mean_edge_K",
        "mean_edge_d",
        "mean_geodesic_d",
        "max_geodesic_d",
        "mean_closeness",
        "max_closeness",
        "final_avg_memory",
        "final_max_memory",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in records:
            writer.writerow(row)

    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

def plot_history(history, memory_history, output_dir="s7_outputs", filename="history_memory.png"):
    output_path = ensure_output_dir(output_dir) / filename

    steps = [row[0] for row in history]
    counts = [row[1] for row in history]
    avg_p = [row[2] for row in history]
    avg_mem = [row[1] for row in memory_history]
    max_mem = [row[2] for row in memory_history]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(steps, counts, label="state count")
    ax1.set_xlabel("step")
    ax1.set_ylabel("state count")

    ax2 = ax1.twinx()
    ax2.plot(steps, avg_p, linestyle="--", label="avg persistence")
    ax2.plot(steps, avg_mem, linestyle=":", label="avg memory")
    ax2.plot(steps, max_mem, linestyle="-.", label="max memory")
    ax2.set_ylabel("persistence / memory")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("S7 population, persistence, and memory")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_matrix(matrix, title, colorbar_label, output_dir, filename, replace_inf=True):
    output_path = ensure_output_dir(output_dir) / filename

    data = [row[:] for row in matrix]

    if replace_inf:
        finite = [v for row in data for v in row if math.isfinite(v)]

        replacement = max(finite) if finite else 0.0

        for i in range(len(data)):
            for j in range(len(data[i])):
                if not math.isfinite(data[i][j]):
                    data[i][j] = replacement

    plt.figure(figsize=(8, 7))
    plt.imshow(data, interpolation="nearest")
    plt.colorbar(label=colorbar_label)
    plt.title(title)
    plt.xlabel("track index")
    plt.ylabel("track index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_degree_closeness(summary, output_dir="s7_outputs", filename="degree_closeness.png"):
    output_path = ensure_output_dir(output_dir) / filename

    degree = summary["degree"]
    closeness = summary["closeness"]

    plt.figure(figsize=(8, 5))
    plt.scatter(degree, closeness, alpha=0.8)
    plt.xlabel("degree")
    plt.ylabel("closeness")
    plt.title("S7 degree vs closeness centrality")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_components(summary, output_dir="s7_outputs", filename="components.png"):
    output_path = ensure_output_dir(output_dir) / filename

    sizes = [len(c) for c in summary["components"]]

    plt.figure(figsize=(8, 5))
    plt.bar([str(i) for i in range(len(sizes))], sizes)
    plt.xlabel("component index")
    plt.ylabel("component size")
    plt.title("S7 proto-geometric component sizes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_s7_batch_summary(records, output_dir="s7_outputs", filename="s7_batch_summary.png"):
    output_path = ensure_output_dir(output_dir) / filename

    seeds = [r["seed"] for r in records]
    edges = [r["edge_count"] for r in records]
    largest = [r["largest_component"] for r in records]
    mean_geo = [r["mean_geodesic_d"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(seeds, edges, marker="o", label="edges")
    ax1.plot(seeds, largest, marker="o", label="largest component")
    ax1.set_xlabel("seed")
    ax1.set_ylabel("network size")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(seeds, mean_geo, linestyle="--", label="mean geodesic distance")
    ax2.set_ylabel("mean geodesic distance")
    ax2.legend(loc="upper right")

    plt.title("S7 proto-geometry across seeds")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 9. Run helpers
# ---------------------------------------------------------------------------

def run_s7_single(
    seed=0,
    output_dir="s7_outputs",
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    memory_lambda=0.08,
    memory_gain=0.04,
    memory_decay=0.985,
    min_lifetime=100,
    min_updates=10,
    coupling_threshold=0.25
):
    random.seed(seed)

    print("=== Single S7 run ===")
    print("seed:", seed)
    print("coupling_threshold:", coupling_threshold)

    states, history, tracks, snapshots, memory_history = run_simulation_with_self_stabilizing_identity(
        cluster_threshold=cluster_threshold,
        sample_every=sample_every,
        overlap_threshold=overlap_threshold,
        memory_lambda=memory_lambda,
        memory_gain=memory_gain,
        memory_decay=memory_decay,
        verbose=True
    )

    stable_tracks = select_stable_tracks(
        tracks,
        sample_every=sample_every,
        min_lifetime=min_lifetime,
        min_updates=min_updates
    )

    geometry = build_track_geometry(
        stable_tracks,
        states,
        coupling_threshold=coupling_threshold
    )

    geodesic_matrix = geodesic_distance_matrix(geometry)
    summary = summarize_geometry(geometry, geodesic_matrix)

    final_avg_memory = memory_history[-1][1] if memory_history else 0.0
    final_max_memory = memory_history[-1][2] if memory_history else 0.0

    print("\nFinal states:", len(states))
    print("Stable tracks:", len(stable_tracks))
    print("Geometry nodes:", summary["node_count"])
    print("Geometry edges:", summary["edge_count"])
    print("Component count:", summary["component_count"])
    print("Largest component:", summary["largest_component"])
    print("Mean degree:", round(summary["mean_degree"], 3))
    print("Max degree:", summary["max_degree"])
    print("Mean weighted degree:", round(summary["mean_weighted_degree"], 5))
    print("Mean edge K:", round(summary["mean_edge_K"], 5))
    print("Mean direct edge distance:", round(summary["mean_edge_d"], 5))
    print("Mean geodesic distance:", round(summary["mean_geodesic_d"], 5))
    print("Max geodesic distance:", round(summary["max_geodesic_d"], 5))
    print("Mean closeness:", round(summary["mean_closeness"], 5))
    print("Max closeness:", round(summary["max_closeness"], 5))
    print("Final avg memory:", round(final_avg_memory, 5))
    print("Final max memory:", round(final_max_memory, 5))

    print("\nTop component sizes:", [len(c) for c in summary["components"][:10]])

    save_edges_csv(geometry, output_dir=output_dir)
    save_node_geometry_csv(geometry, summary, output_dir=output_dir)

    plot_history(history, memory_history, output_dir=output_dir)
    plot_matrix(
        geometry["K_matrix"],
        "S7 track coupling matrix K",
        "K",
        output_dir,
        "track_coupling_matrix.png",
        replace_inf=False
    )
    plot_matrix(
        geometry["D_matrix"],
        "S7 direct track proto-distance d_T=-log(K)",
        "d_T",
        output_dir,
        "track_direct_distance_matrix.png",
        replace_inf=True
    )
    plot_matrix(
        geodesic_matrix,
        "S7 geodesic distance matrix",
        "geodesic distance",
        output_dir,
        "track_geodesic_distance_matrix.png",
        replace_inf=True
    )
    plot_degree_closeness(summary, output_dir=output_dir)
    plot_components(summary, output_dir=output_dir)

    return states, tracks, stable_tracks, geometry, geodesic_matrix, summary


def run_s7_batch(
    runs=50,
    output_dir="s7_outputs",
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    memory_lambda=0.08,
    memory_gain=0.04,
    memory_decay=0.985,
    min_lifetime=100,
    min_updates=10,
    coupling_threshold=0.25
):
    print("\n=== S7 multi-seed proto-geometry analysis ===")
    print("runs:", runs)

    records = []

    for seed in range(runs):
        random.seed(seed)

        states, history, tracks, snapshots, memory_history = run_simulation_with_self_stabilizing_identity(
            cluster_threshold=cluster_threshold,
            sample_every=sample_every,
            overlap_threshold=overlap_threshold,
            memory_lambda=memory_lambda,
            memory_gain=memory_gain,
            memory_decay=memory_decay,
            verbose=False
        )

        stable_tracks = select_stable_tracks(
            tracks,
            sample_every=sample_every,
            min_lifetime=min_lifetime,
            min_updates=min_updates
        )

        geometry = build_track_geometry(
            stable_tracks,
            states,
            coupling_threshold=coupling_threshold
        )

        geodesic_matrix = geodesic_distance_matrix(geometry)
        summary = summarize_geometry(geometry, geodesic_matrix)

        final_avg_memory = memory_history[-1][1] if memory_history else 0.0
        final_max_memory = memory_history[-1][2] if memory_history else 0.0

        record = {
            "seed": seed,
            "final_states": len(states),
            "node_count": summary["node_count"],
            "edge_count": summary["edge_count"],
            "component_count": summary["component_count"],
            "largest_component": summary["largest_component"],
            "mean_degree": summary["mean_degree"],
            "max_degree": summary["max_degree"],
            "mean_weighted_degree": summary["mean_weighted_degree"],
            "mean_edge_K": summary["mean_edge_K"],
            "mean_edge_d": summary["mean_edge_d"],
            "mean_geodesic_d": summary["mean_geodesic_d"],
            "max_geodesic_d": summary["max_geodesic_d"],
            "mean_closeness": summary["mean_closeness"],
            "max_closeness": summary["max_closeness"],
            "final_avg_memory": final_avg_memory,
            "final_max_memory": final_max_memory,
        }

        records.append(record)

        print(
            "seed", seed,
            "states", record["final_states"],
            "nodes", record["node_count"],
            "edges", record["edge_count"],
            "components", record["component_count"],
            "largest", record["largest_component"],
            "mean_geo", round(record["mean_geodesic_d"], 4),
            "mean_close", round(record["mean_closeness"], 4)
        )

    save_s7_batch_csv(records, output_dir=output_dir)
    plot_s7_batch_summary(records, output_dir=output_dir)

    print("\n=== S7 batch summary ===")
    print("mean final states:", round(mean(r["final_states"] for r in records), 3))
    print("mean nodes:", round(mean(r["node_count"] for r in records), 3))
    print("mean edges:", round(mean(r["edge_count"] for r in records), 3))
    print("mean component count:", round(mean(r["component_count"] for r in records), 3))
    print("mean largest component:", round(mean(r["largest_component"] for r in records), 3))
    print("mean degree:", round(mean(r["mean_degree"] for r in records), 3))
    print("mean edge K:", round(mean(r["mean_edge_K"] for r in records), 6))
    print("mean edge distance:", round(mean(r["mean_edge_d"] for r in records), 6))
    print("mean geodesic distance:", round(mean(r["mean_geodesic_d"] for r in records), 6))
    print("mean closeness:", round(mean(r["mean_closeness"] for r in records), 6))

    return records


if __name__ == "__main__":
    run_s7_single(
        seed=0,
        output_dir="s7_outputs",
        cluster_threshold=0.6,
        sample_every=5,
        overlap_threshold=0.30,
        memory_lambda=0.08,
        memory_gain=0.04,
        memory_decay=0.985,
        min_lifetime=100,
        min_updates=10,
        coupling_threshold=0.25
    )

    run_s7_batch(
        runs=50,
        output_dir="s7_outputs",
        cluster_threshold=0.6,
        sample_every=5,
        overlap_threshold=0.30,
        memory_lambda=0.08,
        memory_gain=0.04,
        memory_decay=0.985,
        min_lifetime=100,
        min_updates=10,
        coupling_threshold=0.25
    )


"""
How to interpret S7
===================

S6:
    stable identities formed relation networks.

S7:
    those relation networks induce proto-geometric structure.

Important outputs:
------------------
track_coupling_matrix.png:
    Pairwise K(T_i,T_j). Bright/high values mean strong relation.

track_direct_distance_matrix.png:
    Direct proto-distance d_T=-log(K).

track_geodesic_distance_matrix.png:
    Shortest path distances on the identity network.

degree_closeness.png:
    Centrality structure of the emergent identity geometry.

components.png:
    Mesoscopic connected regions of the proto-geometry.

Interpretation:
---------------
If the geometry has nontrivial components, finite geodesic paths,
central nodes, and structured distance matrices, then the model has generated
a primitive relational geometry among persistent identities.

This is a toy-model step toward:

    identity -> relation -> topology -> proto-geometry
"""
