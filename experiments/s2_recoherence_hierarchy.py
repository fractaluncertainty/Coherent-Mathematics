#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S2 — Primitive Recoherence Simulation with Hierarchical Threshold Sweep
=====================================================================

This program extends S1.

It tests whether stable structure can emerge from:
    - continuation,
    - connection capability,
    - interference,
    - selection,
    - cluster formation,
    - and hierarchical cluster splitting under stronger connection thresholds.

Core idea:
----------
We start with random states. States change slightly. States that connect well
with others survive more easily. Highly persistent states may generate similar
successors.

The key new S2 question:
------------------------
Does the final recoherence network have hierarchy?

Instead of using only one cluster threshold, we test multiple thresholds:

    J > 0.15
    J > 0.25
    J > 0.40
    J > 0.60
    J > 0.80

Interpretation:
---------------
low threshold:
    global weak connection

medium threshold:
    recoherence families

high threshold:
    strong attractor cores

very high threshold:
    near-identical continuation doublets or micro-clusters

If the cluster structure breaks apart gradually as the threshold increases,
this suggests hierarchical emergent structure rather than a featureless cloud.

Scientific caution:
-------------------
This is NOT proof of FUT/CoMath.
It is a toy model testing whether primitive recoherence selection can generate
stable, proto-geometric, and hierarchical structure.
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
    Runs the S2 recoherence simulation.

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


def threshold_sweep(states, thresholds=(0.15, 0.25, 0.40, 0.60, 0.80)):
    """
    Tests cluster structure across multiple connection thresholds.

    This is the central S2 analysis.

    If the system has hierarchy, then increasing the threshold should break
    one global cluster into smaller recoherence families and attractor cores.
    """
    print("\n=== Threshold sweep ===")

    sweep = []

    for threshold in thresholds:
        clusters = find_clusters(states, threshold=threshold)
        sizes = sorted([len(c) for c in clusters], reverse=True)
        strong_joins = count_strong_joins(states, threshold=threshold)

        row = {
            "threshold": threshold,
            "cluster_count": len(clusters),
            "largest_cluster": sizes[0] if sizes else 0,
            "top_sizes": sizes[:10],
            "strong_joins": strong_joins,
        }

        sweep.append(row)

        print(
            "threshold", threshold,
            "cluster_count", row["cluster_count"],
            "largest", row["largest_cluster"],
            "strong_joins", row["strong_joins"],
            "top_sizes", row["top_sizes"]
        )

    return sweep


# ---------------------------------------------------------------------------
# 5. Visualization
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_states(states, output_dir="s2_outputs", filename="states_scatter.png"):
    """Plots surviving states in the (sigma, chi) plane."""
    output_path = ensure_output_dir(output_dir) / filename

    sigmas = [s.sigma for s in states]
    chis = [s.chi for s in states]
    weights = [20 + 80 * s.w for s in states]

    plt.figure(figsize=(8, 6))
    plt.scatter(sigmas, chis, s=weights, alpha=0.75)
    plt.xlabel("sigma — connection orientation")
    plt.ylabel("chi — interference phase")
    plt.title("S2 surviving states in (sigma, chi)")
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_clusters(states, clusters, output_dir="s2_outputs", filename="clusters_scatter.png"):
    """Plots states colored by detected recoherence cluster."""
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
    plt.title("S2 recoherence clusters")
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_distance_matrix(states, output_dir="s2_outputs", filename="proto_distance_matrix.png"):
    """Plots the proto-distance matrix g_f(x,y) = -log(J(x,y))."""
    output_path = ensure_output_dir(output_dir) / filename

    n = len(states)
    matrix = [
        [proto_distance(states[i], states[j]) for j in range(n)]
        for i in range(n)
    ]

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, interpolation="nearest")
    plt.colorbar(label="g_f = -log(J)")
    plt.title("S2 proto-distance matrix")
    plt.xlabel("state index")
    plt.ylabel("state index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_history(history, output_dir="s2_outputs", filename="history.png"):
    """Plots population size and average persistence over time."""
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

    plt.title("S2 population and persistence history")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_threshold_sweep(sweep, output_dir="s2_outputs", filename="threshold_sweep.png"):
    """Plots how cluster structure changes with threshold."""
    output_path = ensure_output_dir(output_dir) / filename

    thresholds = [row["threshold"] for row in sweep]
    cluster_counts = [row["cluster_count"] for row in sweep]
    largest = [row["largest_cluster"] for row in sweep]
    strong_joins = [row["strong_joins"] for row in sweep]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(thresholds, cluster_counts, marker="o", label="cluster count")
    ax1.plot(thresholds, largest, marker="o", label="largest cluster")
    ax1.set_xlabel("J threshold")
    ax1.set_ylabel("clusters / largest cluster size")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, strong_joins, marker="x", linestyle="--", label="strong joins")
    ax2.set_ylabel("strong joins")
    ax2.legend(loc="upper right")

    plt.title("S2 hierarchical threshold sweep")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def create_visualizations(states, history, clusters, sweep, output_dir="s2_outputs"):
    """Creates all visualization files."""
    plot_states(states, output_dir=output_dir)
    plot_clusters(states, clusters, output_dir=output_dir)
    plot_distance_matrix(states, output_dir=output_dir)
    plot_history(history, output_dir=output_dir)
    plot_threshold_sweep(sweep, output_dir=output_dir)


# ---------------------------------------------------------------------------
# 6. Running and batch tests
# ---------------------------------------------------------------------------

def run_single_demo(seed=0, output_dir="s2_outputs"):
    """Runs one detailed demonstration simulation and creates plots."""
    random.seed(seed)

    print("=== Single S2 run ===")
    print("seed:", seed)

    states, history = run_simulation(verbose=True)

    final_avg_p = history[-1][2] if history else 0.0
    strong_joins = count_strong_joins(states)

    print("\nFinal states:", len(states))
    print("Final avg_p:", round(final_avg_p, 6))
    print("Strong joins at threshold 0.15:", strong_joins)

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
    sweep = threshold_sweep(states)

    create_visualizations(states, history, clusters, sweep, output_dir=output_dir)

    return states, history, clusters, stats, sweep


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
    run_single_demo(seed=0, output_dir="s2_outputs")

    # Robustness test.
    run_batch(runs=20)


"""
How to interpret S2 outputs
===========================

history.png:
    Population count and average persistence over time.
    Sudden jumps may indicate recoherence cascades.

states_scatter.png:
    Final surviving states in the (sigma, chi) plane.

clusters_scatter.png:
    Cluster-colored final states.

proto_distance_matrix.png:
    Matrix of g_f(x,y) = -log(J(x,y)).
    Small values indicate emergent recoherence nearness.

threshold_sweep.png:
    Shows whether the cluster structure has hierarchy.

Key S2 interpretation:
----------------------
If increasing the threshold breaks one global cluster into families, cores,
and small micro-clusters, then the system has hierarchical recoherence
structure rather than just one featureless connected component.
"""
