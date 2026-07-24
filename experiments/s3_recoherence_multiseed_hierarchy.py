#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S3 — Primitive Recoherence Simulation with Multi-Seed Hierarchy Statistics
=========================================================================

This program extends S2.

S2 showed that one run can produce a hierarchical recoherence structure:
    low J-threshold  -> global cluster
    medium threshold -> recoherence families
    high threshold   -> attractor cores / micro-clusters

S3 asks the next scientific question:

    Is this hierarchy robust across many random seeds?

This version includes:
----------------------
1. S0/S1/S2 simulation core.
2. Single-run visualization.
3. Threshold sweep for one seed.
4. Multi-seed hierarchy statistics.
5. CSV export of hierarchy results.
6. Plots of mean hierarchy behavior across seeds.

Scientific caution:
-------------------
This is NOT proof of FUT/CoMath.
It is a toy model testing whether primitive recoherence selection can generate
stable, proto-geometric, and hierarchical structure robustly.
"""

import csv
import math
import random
from pathlib import Path
from statistics import mean

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
# 2. Core recoherence functions
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
# 3. Primitive dynamics
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
    Runs the recoherence simulation.

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


def threshold_sweep(states, thresholds=(0.15, 0.25, 0.40, 0.60, 0.80)):
    """
    Tests cluster structure across multiple connection thresholds.

    If hierarchy exists, increasing the threshold should break one global
    cluster into smaller recoherence families and attractor cores.
    """
    sweep = []

    for threshold in thresholds:
        clusters = find_clusters(states, threshold=threshold)
        sizes = sorted([len(c) for c in clusters], reverse=True)

        isolated = sum(1 for size in sizes if size == 1)

        row = {
            "threshold": threshold,
            "cluster_count": len(clusters),
            "largest_cluster": sizes[0] if sizes else 0,
            "mean_cluster_size": mean(sizes) if sizes else 0.0,
            "isolated_states": isolated,
            "strong_joins": count_strong_joins(states, threshold=threshold),
            "top_sizes": sizes[:10],
        }

        sweep.append(row)

    return sweep


def print_threshold_sweep(sweep):
    """Prints threshold sweep in readable form."""
    print("\n=== Threshold sweep ===")

    for row in sweep:
        print(
            "threshold", row["threshold"],
            "cluster_count", row["cluster_count"],
            "largest", row["largest_cluster"],
            "mean_size", round(row["mean_cluster_size"], 3),
            "isolated", row["isolated_states"],
            "strong_joins", row["strong_joins"],
            "top_sizes", row["top_sizes"]
        )


# ---------------------------------------------------------------------------
# 5. Output helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_hierarchy_csv(records, output_dir="s3_outputs", filename="hierarchy_results.csv"):
    """Saves hierarchy records to CSV."""
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "seed",
        "final_states",
        "final_avg_p",
        "avg_proto_distance",
        "threshold",
        "cluster_count",
        "largest_cluster",
        "mean_cluster_size",
        "isolated_states",
        "strong_joins",
        "top_sizes",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in records:
            row_copy = dict(row)
            row_copy["top_sizes"] = " ".join(str(x) for x in row_copy["top_sizes"])
            writer.writerow(row_copy)

    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------

def plot_states(states, output_dir="s3_outputs", filename="states_scatter.png"):
    """Plots surviving states in the (sigma, chi) plane."""
    output_path = ensure_output_dir(output_dir) / filename

    sigmas = [s.sigma for s in states]
    chis = [s.chi for s in states]
    weights = [20 + 80 * s.w for s in states]

    plt.figure(figsize=(8, 6))
    plt.scatter(sigmas, chis, s=weights, alpha=0.75)
    plt.xlabel("sigma — connection orientation")
    plt.ylabel("chi — interference phase")
    plt.title("S3 surviving states in (sigma, chi)")
    plt.xlim(0, 2 * math.pi)
    plt.ylim(0, 2 * math.pi)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_distance_matrix(states, output_dir="s3_outputs", filename="proto_distance_matrix.png"):
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
    plt.title("S3 proto-distance matrix")
    plt.xlabel("state index")
    plt.ylabel("state index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_history(history, output_dir="s3_outputs", filename="history.png"):
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

    plt.title("S3 population and persistence history")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_single_threshold_sweep(sweep, output_dir="s3_outputs", filename="single_threshold_sweep.png"):
    """Plots threshold sweep for one seed."""
    output_path = ensure_output_dir(output_dir) / filename

    thresholds = [row["threshold"] for row in sweep]
    cluster_counts = [row["cluster_count"] for row in sweep]
    largest = [row["largest_cluster"] for row in sweep]
    isolated = [row["isolated_states"] for row in sweep]
    strong_joins = [row["strong_joins"] for row in sweep]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(thresholds, cluster_counts, marker="o", label="cluster count")
    ax1.plot(thresholds, largest, marker="o", label="largest cluster")
    ax1.plot(thresholds, isolated, marker="o", label="isolated states")
    ax1.set_xlabel("J threshold")
    ax1.set_ylabel("cluster statistics")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, strong_joins, marker="x", linestyle="--", label="strong joins")
    ax2.set_ylabel("strong joins")
    ax2.legend(loc="upper right")

    plt.title("S3 single-seed hierarchical threshold sweep")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_mean_hierarchy(records, output_dir="s3_outputs", filename="mean_hierarchy.png"):
    """
    Plots mean hierarchy behavior across seeds.

    Shows mean cluster count, mean largest cluster, mean isolated states,
    and mean strong joins for each threshold.
    """
    output_path = ensure_output_dir(output_dir) / filename

    thresholds = sorted(set(row["threshold"] for row in records))

    mean_cluster_counts = []
    mean_largest = []
    mean_isolated = []
    mean_joins = []

    for threshold in thresholds:
        subset = [r for r in records if r["threshold"] == threshold]
        mean_cluster_counts.append(mean(r["cluster_count"] for r in subset))
        mean_largest.append(mean(r["largest_cluster"] for r in subset))
        mean_isolated.append(mean(r["isolated_states"] for r in subset))
        mean_joins.append(mean(r["strong_joins"] for r in subset))

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(thresholds, mean_cluster_counts, marker="o", label="mean cluster count")
    ax1.plot(thresholds, mean_largest, marker="o", label="mean largest cluster")
    ax1.plot(thresholds, mean_isolated, marker="o", label="mean isolated states")
    ax1.set_xlabel("J threshold")
    ax1.set_ylabel("mean cluster statistics")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(thresholds, mean_joins, marker="x", linestyle="--", label="mean strong joins")
    ax2.set_ylabel("mean strong joins")
    ax2.legend(loc="upper right")

    plt.title("S3 mean hierarchy across seeds")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 7. Single run and multi-seed hierarchy analysis
# ---------------------------------------------------------------------------

def run_single_demo(seed=0, output_dir="s3_outputs"):
    """Runs one detailed demonstration simulation and creates plots."""
    random.seed(seed)

    print("=== Single S3 run ===")
    print("seed:", seed)

    states, history = run_simulation(verbose=True)

    final_avg_p = history[-1][2] if history else 0.0
    avg_g = average_proto_distance(states)
    sweep = threshold_sweep(states)

    print("\nFinal states:", len(states))
    print("Final avg_p:", round(final_avg_p, 6))
    print("Average proto-distance:", round(avg_g, 6))

    print("\nLast 10 history entries:")
    for row in history[-10:]:
        print(row)

    print_threshold_sweep(sweep)

    plot_states(states, output_dir=output_dir)
    plot_distance_matrix(states, output_dir=output_dir)
    plot_history(history, output_dir=output_dir)
    plot_single_threshold_sweep(sweep, output_dir=output_dir)

    return states, history, sweep


def run_hierarchy_batch(
    runs=50,
    thresholds=(0.15, 0.25, 0.40, 0.60, 0.80),
    output_dir="s3_outputs"
):
    """
    Runs many simulations and performs threshold hierarchy analysis for each.

    This is the central S3 analysis.
    """
    print("\n=== S3 multi-seed hierarchy analysis ===")
    print("runs:", runs)
    print("thresholds:", thresholds)

    records = []

    for seed in range(runs):
        random.seed(seed)
        states, history = run_simulation(verbose=False)

        final_states = len(states)
        final_avg_p = history[-1][2] if history else 0.0
        avg_g = average_proto_distance(states)

        sweep = threshold_sweep(states, thresholds=thresholds)

        print(
            "seed", seed,
            "final_states", final_states,
            "avg_p", round(final_avg_p, 5),
            "avg_g", round(avg_g, 5)
        )

        for row in sweep:
            records.append({
                "seed": seed,
                "final_states": final_states,
                "final_avg_p": final_avg_p,
                "avg_proto_distance": avg_g,
                "threshold": row["threshold"],
                "cluster_count": row["cluster_count"],
                "largest_cluster": row["largest_cluster"],
                "mean_cluster_size": row["mean_cluster_size"],
                "isolated_states": row["isolated_states"],
                "strong_joins": row["strong_joins"],
                "top_sizes": row["top_sizes"],
            })

    save_hierarchy_csv(records, output_dir=output_dir)
    plot_mean_hierarchy(records, output_dir=output_dir)

    print("\n=== S3 summary by threshold ===")

    for threshold in thresholds:
        subset = [r for r in records if r["threshold"] == threshold]

        print(
            "threshold", threshold,
            "mean_cluster_count", round(mean(r["cluster_count"] for r in subset), 3),
            "mean_largest", round(mean(r["largest_cluster"] for r in subset), 3),
            "mean_isolated", round(mean(r["isolated_states"] for r in subset), 3),
            "mean_strong_joins", round(mean(r["strong_joins"] for r in subset), 3),
        )

    collapsed = len({
        r["seed"] for r in records
        if r["final_states"] == 0
    })

    print("collapsed runs:", collapsed, "of", runs)

    return records


if __name__ == "__main__":
    # One detailed run with visual output.
    run_single_demo(seed=0, output_dir="s3_outputs")

    # Multi-seed hierarchy statistics.
    # Increase runs to 100 or 200 later if needed.
    run_hierarchy_batch(runs=50, output_dir="s3_outputs")


"""
How to interpret S3 outputs
===========================

single_threshold_sweep.png:
    Shows hierarchy for one seed.

mean_hierarchy.png:
    Shows hierarchy averaged over many seeds.

hierarchy_results.csv:
    Raw data for all seeds and thresholds.

Main question:
--------------
Does the system show a reproducible pattern?

Expected hierarchical pattern:
    low threshold:
        one or few large global clusters

    medium threshold:
        several recoherence families

    high threshold:
        many smaller attractor cores

    very high threshold:
        micro-clusters and isolated states

If this pattern holds across many seeds, the toy model supports the claim that
primitive recoherence selection can generate multi-scale proto-geometric
hierarchy.
"""
