#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S0 — Primitive Recoherence Simulation with Cluster and Proto-Metric Analysis
============================================================================

This program is a small experimental simulation prototype inspired by FUT / CoMath.

It does NOT directly simulate real physics. It tests a more primitive question:

    Can stable structure emerge from continuation, connection capability,
    interference, and selection?

Core idea:
----------
We start with random states. States change slightly. States that connect well
with others survive more easily. Highly persistent states may generate similar
successors.

This models the statement:

    "Coherence is not assumed. Coherence is what survives."

This version includes:
----------------------
1. Single-run simulation.
2. Batch robustness test over multiple random seeds.
3. Strong-join counting.
4. Proto-metric definition: g_f(x,y) = -log(J(x,y)).
5. Automatic cluster detection.
6. Compact cluster summaries.
"""

import math
import random
from statistics import mean


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
    This is a representation, not a claim that FUT/CoMath is fundamentally
    based on classical angular geometry.
    """

    def __init__(self, w, sigma, chi):
        self.w = w
        self.sigma = sigma
        self.chi = chi


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

    This is important because it models the idea that "nearness" can emerge
    from connection capability instead of being assumed as spatial distance.
    """
    join = max(J(x, y), 1e-12)
    return -math.log(join)


def persistence(x, states):
    """
    Persistence of a state relative to all other states.

    A state survives if it can maintain enough constructive recoherent
    connection with other states.
    """
    others = [y for y in states if y is not x]

    if not others:
        return 0.0

    return x.w * sum(
        J(x, y) * max(0.0, I(x, y))
        for y in others
    ) / len(others)


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
    Runs the S0 simulation.

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


def run_single_demo(seed=0):
    """Runs one detailed demonstration simulation."""
    random.seed(seed)

    print("=== Single S0 run ===")
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

    print_cluster_summary(states, threshold=0.15)
    return states, history


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
    # One detailed run.
    run_single_demo(seed=0)

    # Robustness test.
    run_batch(runs=20)


"""
How to interpret the results
============================

Final states > 0:
    The system did not completely collapse.

Final states remains moderate:
    The system did not explode without limit.

Strong joins > 0:
    Surviving states form connection relations.

Large cluster:
    A recoherence family emerged.

Small average internal g_f:
    Members of a cluster are close in the emergent proto-metric.

Important caution:
    This is not proof of FUT/CoMath.
    It is an experimental toy model testing whether primitive recoherence
    selection can generate stable structure and proto-geometric relations.
"""
