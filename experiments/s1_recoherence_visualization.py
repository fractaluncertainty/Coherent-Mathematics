#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1 — Primitive Recoherence Simulation with Visualization
=======================================================

This program extends the S0 primitive recoherence simulation.

It tests whether stable structure can emerge from:
    - continuation,
    - connection capability,
    - interference,
    - selection,
    - and cluster formation.

This version adds visual output:
    1. Scatter plot of surviving states in the (sigma, chi) plane.
    2. Cluster-colored scatter plot.
    3. Proto-distance matrix heatmap using g_f(x,y) = -log(J(x,y)).

Important:
----------
This is NOT proof of FUT/CoMath.
It is a toy model / primitive experimental system.

The purpose is to test whether recoherence selection can generate:
    - stable populations,
    - critical transitions,
    - clusters,
    - and proto-geometric relations.
"""

import math
import random
from statistics import mean
from pathlib import Path

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. State definition
# ---------------------------------------------------------------------------

class State:
    """
    A simplified continuation profile.

    w:
        Continuation weight / internal stability.

    sigma:
        Connection orientation.

    chi:
        Interference phase.

    sigma and chi are represented as angles only for simulation convenience.
    """

    def __init__(self, w, sigma, chi):
        self.w = w
        self.sigma = sigma
        self.chi = chi


# ---------------------------------------------------------------------------
# 2. Core mathematical helpers
# ---------------------------------------------------------------------------

def angle_diff(a, b):
    """Smallest angular distance between two angles on a circle."""
    d = abs(a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d)


def J(x, y):
    """
    Connection capability.

    J near 1:
        strong connection / high recoherence compatibility.

    J near 0:
        weak connection.
    """
    return math.exp(
        -angle_diff(x.sigma, y.sigma)
        -angle_diff(x.chi, y.chi)
    )


def I(x, y):
    """
    Interference.

    I > 0:
        constructive recoherence.

    I < 0:
        destructive recoherence.
    """
    return J(x, y) * math.cos(angle_diff(x.chi, y.chi))


def proto_distance(x, y):
    """
    Proto-metric distance induced by connection capability.

        g_f(x,y) = -log(J(x,y))

    Interpretation:
        high J -> small distance
        low J  -> large distance

    This models the idea that nearness can emerge from connection capability
    rather than being assumed as spatial distance.
    """
    join = max(J(x, y), 1e-12)
    return -math.log(join)


def persistence(x, states):
    """
    Persistence of a state relative to all other states.

    A state survives if it maintains enough constructive recoherent connection
    with other states.
    """
    others = [y for y in states if y is not x]

    if not others:
        return 0.0

    return x.w * sum(
        J(x, y) * max(0.0, I(x, y))
        for y in others
    ) / len(others)


# ---------------------------------------------------------------------------
# 3. Dynamics
# ---------------------------------------------------------------------------

def recurse(x):
    """
    Recursive continuation of one state.

    A state does not remain perfectly identical to itself. It continues through
    small variation.
    """
    eta = random.uniform(-0.02, 0.04)
    kappa = random.uniform(0.00, 0.02)
    delta = random.uniform(-0.08, 0.08)
    phi = random.uniform(-0.08, 0.08)

    return State(
        max(0.0, x.w + eta - kappa),
        (x.sigma + delta) % (2 * math.pi),
        (x.chi + phi) % (2 * math.pi)
    )


def split(x):
    """
    Creates a similar successor state.

    The successor is neither identical nor completely disconnected.
    """
    return State(
        x.w * random.uniform(0.8, 1.05),
        (x.sigma + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        (x.chi + random.uniform(-0.05, 0.05)) % (2 * math.pi)
    )


def run_simulation(
    initial_count=40,
    steps=300,
    delta_min=0.001,
    delta_split=0.08,
    max_states=120,
    verbose=True
):
    """
    Runs the S1 recoherence simulation.

    Loop:
        1. Recurse all states.
        2. Compute persistence.
        3. Remove weak states.
        4. Let strong states generate successors.
        5. Store statistics.
    """
    states = [
        State(
            random.uniform(0.2, 1.0),
            random.uniform(0, 2 * math.pi),
            random.uniform(0, 2 * math.pi)
        )
        for _ in range(initial_count)
    ]

    history = []

    for step in range(steps):
        states = [recurse(x) for x in states]
        scored = [(x, persistence(x, states)) for x in states]

        survivors = [x for x, p in scored if p >= delta_min]

        offspring = []
        for x, p in scored:
            if p >= delta_split and len(survivors) + len(offspring) < max_states:
                offspring.append(split(x))

        states = survivors + offspring

        avg_p = sum(p for _, p in scored) / len(scored) if scored else 0.0
        history.append((step, len(states), avg_p))

        if verbose and step % 25 == 0:
            print("step", step, "states", len(states), "avg_p", round(avg_p, 5))

        if not states:
            break

    return states, history


# ---------------------------------------------------------------------------
# 4. Cluster and metric analysis
# ---------------------------------------------------------------------------

def count_strong_joins(states, threshold=0.15):
    """Counts pairs of states with J above threshold."""
    count = 0

    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue
            if J(a, b) > threshold:
                count += 1

    return count


def find_clusters(states, threshold=0.15):
    """
    Finds recoherence clusters.

    A cluster is a connected component in the graph where states are linked
    if their connection capability J is above the threshold.
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


def cluster_stats(states, clusters):
    """
    Computes compact statistics for each cluster.

    For each cluster:
        size
        average internal connection J
        average internal proto-distance g_f
    """
    stats = []

    for cluster in clusters:
        if len(cluster) < 2:
            stats.append({
                "size": len(cluster),
                "avg_internal_J": 0.0,
                "avg_internal_g": 0.0,
            })
            continue

        joins = []
        distances = []

        for a_idx, i in enumerate(cluster):
            for j in cluster[a_idx + 1:]:
                joins.append(J(states[i], states[j]))
                distances.append(proto_distance(states[i], states[j]))

        stats.append({
            "size": len(cluster),
            "avg_internal_J": mean(joins) if joins else 0.0,
            "avg_internal_g": mean(distances) if distances else 0.0,
        })

    return stats


def average_proto_distance(states):
    """Average proto-distance over all state pairs."""
    if len(states) < 2:
        return 0.0

    distances = []

    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue
            distances.append(proto_distance(a, b))

    return mean(distances) if distances else 0.0


def print_cluster_summary(states, threshold=0.15, max_clusters=10):
    """Prints compact cluster and proto-metric summary."""
    clusters = find_clusters(states, threshold=threshold)
    stats = cluster_stats(states, clusters)

    print("\n=== Cluster summary ===")
    print("threshold:", threshold)
    print("cluster count:", len(clusters))
    print("largest cluster size:", len(clusters[0]) if clusters else 0)
    print("average proto-distance:", round(average_proto_distance(states), 5))

    print("\nTop clusters:")
    for idx, stat in enumerate(stats[:max_clusters]):
        print(
            "cluster", idx,
            "size", stat["size"],
            "avg_J", round(stat["avg_internal_J"], 5),
            "avg_g", round(stat["avg_internal_g"], 5)
        )

    return clusters, stats


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_states(states, output_dir="s1_outputs", filename="states_scatter.png"):
    """
    Plots surviving states in the (sigma, chi) plane.

    This shows whether the final population looks like:
        - a random cloud,
        - compact regions,
        - filaments,
        - or clustered structures.
    """
    output_path = ensure_output_dir(output_dir) / filename

    sigmas = [s.sigma for s in states]
    chis = [s.chi for s in states]
    weights = [20 + 80 * s.w for s in states]

    plt.figure(figsize=(8, 6))
    plt.scatter(sigmas, chis, s=weights, alpha=0.75)
    plt.xlabel("sigma — connection orientation")
    plt.ylabel("chi — interference phase")
    plt.title("S1 surviving states in (sigma, chi)")
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_clusters(states, clusters, output_dir="s1_outputs", filename="clusters_scatter.png"):
    """
    Plots states colored by detected recoherence cluster.
    """
    output_path = ensure_output_dir(output_dir) / filename

    cluster_id_by_state = {}
    for cluster_id, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_id_by_state[idx] = cluster_id

    xs = [s.sigma for s in states]
    ys = [s.chi for s in states]
    colors = [cluster_id_by_state.get(i, -1) for i in range(len(states))]
    sizes = [20 + 80 * s.w for s in states]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c=colors, s=sizes, alpha=0.8)
    plt.xlabel("sigma — connection orientation")
    plt.ylabel("chi — interference phase")
    plt.title("S1 recoherence clusters")
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_distance_matrix(states, output_dir="s1_outputs", filename="proto_distance_matrix.png"):
    """
    Plots the proto-distance matrix g_f(x,y) = -log(J(x,y)).

    Darker/lower values indicate states that are closer in the emergent
    recoherence metric.
    """
    output_path = ensure_output_dir(output_dir) / filename

    n = len(states)
    matrix = [
        [proto_distance(states[i], states[j]) for j in range(n)]
        for i in range(n)
    ]

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar(label="g_f = -log(J)")
    plt.title("S1 proto-distance matrix")
    plt.xlabel("state index")
    plt.ylabel("state index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_history(history, output_dir="s1_outputs", filename="history.png"):
    """
    Plots population size and average persistence over time.

    This helps identify:
        - collapse,
        - stable plateaus,
        - sudden recoherence transitions,
        - population saturation.
    """
    output_path = ensure_output_dir(output_dir) / filename

    steps = [row[0] for row in history]
    counts = [row[1] for row in history]
    avg_p = [row[2] for row in history]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(steps, counts)
    ax1.set_xlabel("step")
    ax1.set_ylabel("state count")

    ax2 = ax1.twinx()
    ax2.plot(steps, avg_p, linestyle="--")
    ax2.set_ylabel("average persistence")

    plt.title("S1 population and persistence history")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def create_visualizations(states, history, clusters, output_dir="s1_outputs"):
    """Creates all visualization files."""
    plot_states(states, output_dir=output_dir)
    plot_clusters(states, clusters, output_dir=output_dir)
    plot_distance_matrix(states, output_dir=output_dir)
    plot_history(history, output_dir=output_dir)


# ---------------------------------------------------------------------------
# 6. Running and batch tests
# ---------------------------------------------------------------------------

def run_single_demo(seed=0, output_dir="s1_outputs"):
    """Runs one detailed demonstration simulation and creates plots."""
    random.seed(seed)

    print("=== Single S1 run ===")
    print("seed:", seed)

    states, history = run_simulation(verbose=True)

    final_avg_p = history[-1][2] if history else 0.0
    strong_joins = count_strong_joins(states)

    print("\nFinal states:", len(states))
    print("Final avg_p:", round(final_avg_p, 6))
    print("Strong joins:", strong_joins)

    print("\nLast 10 history entries:")
    for row in history[-10:]:
        print(row)

    if states:
        print("\nSample surviving states:")
        for s in states[:10]:
            print(
                "w=", round(s.w, 4),
                "sigma=", round(s.sigma, 4),
                "chi=", round(s.chi, 4)
            )

    clusters, stats = print_cluster_summary(states, threshold=0.15)

    create_visualizations(states, history, clusters, output_dir=output_dir)

    return states, history, clusters, stats


def run_batch(runs=20, threshold=0.15):
    """
    Batch robustness test over many random seeds.

    Measures:
        final number of states
        final average persistence
        strong joins
        cluster count
        largest cluster size
        average proto-distance
    """
    print("\n=== Batch robustness test ===")
    print("runs:", runs)

    results = []

    for seed in range(runs):
        random.seed(seed)
        states, history = run_simulation(verbose=False)

        clusters = find_clusters(states, threshold=threshold)

        result = {
            "seed": seed,
            "final_states": len(states),
            "final_avg_p": history[-1][2] if history else 0.0,
            "strong_joins": count_strong_joins(states, threshold=threshold),
            "cluster_count": len(clusters),
            "largest_cluster": len(clusters[0]) if clusters else 0,
            "avg_proto_distance": average_proto_distance(states),
        }

        results.append(result)

        print(
            "seed", seed,
            "final_states", result["final_states"],
            "avg_p", round(result["final_avg_p"], 5),
            "strong_joins", result["strong_joins"],
            "clusters", result["cluster_count"],
            "largest", result["largest_cluster"],
            "avg_g", round(result["avg_proto_distance"], 5)
        )

    print("\n=== Batch summary ===")
    print("mean final states:", round(mean(r["final_states"] for r in results), 3))
    print("mean final avg_p:", round(mean(r["final_avg_p"] for r in results), 6))
    print("mean strong joins:", round(mean(r["strong_joins"] for r in results), 3))
    print("mean cluster count:", round(mean(r["cluster_count"] for r in results), 3))
    print("mean largest cluster:", round(mean(r["largest_cluster"] for r in results), 3))
    print("mean avg proto-distance:", round(mean(r["avg_proto_distance"] for r in results), 6))
    print("collapsed runs:", sum(1 for r in results if r["final_states"] == 0), "of", runs)

    return results


if __name__ == "__main__":
    # One detailed run with visual output.
    run_single_demo(seed=0, output_dir="s1_outputs")

    # Robustness test.
    run_batch(runs=20)


"""
How to interpret the visual outputs
===================================

1. states_scatter.png
   Shows the final surviving states in the (sigma, chi) plane.
   Compact regions suggest emergent clustering.

2. clusters_scatter.png
   Same states, colored by detected cluster.

3. proto_distance_matrix.png
   Shows g_f(x,y) = -log(J(x,y)).
   Small values indicate strong recoherence nearness.

4. history.png
   Shows state count and average persistence over time.
   Sudden jumps may indicate recoherence cascades or percolation-like transitions.

Scientific caution:
-------------------
This is not proof of FUT/CoMath. It is a toy model testing whether primitive
recoherence selection can produce stable structure and proto-geometric relations.
"""
