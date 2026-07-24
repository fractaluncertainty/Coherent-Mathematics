#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S6 — Primitive Recoherence Simulation with Coupled Identity Networks
===================================================================

S6 extends S5.

S5:
    Persistent identity tracks weakly stabilize their own future continuation.

S6:
    Persistent identity tracks can also form stable relations to each other.

This is the next conceptual step:

    self-stabilizing identities -> coupled identity networks

In FUT/CoMath terms, this is a toy-model step toward:

    persistence -> identity -> relation -> structured network

Core S6 question:
-----------------
Can long-lived identity tracks form stable coupling networks?

If yes, this suggests that relations need not be assumed as primitive objects.
They can emerge from mutual recoherence compatibility between persistent
identity lines.

Important:
----------
This is NOT proof of FUT/CoMath.
It is a toy model testing whether primitive recoherence dynamics can generate
self-stabilizing identities and then stable relations among those identities.
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


def proto_distance(x, y):
    """Proto-metric distance g_f(x,y) = -log(J(x,y))."""
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
# 5. Main S6 simulation
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
    Runs S6 base dynamics.

    This is equivalent to S5 during simulation.
    S6 then adds track-track network analysis after tracks are built.
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
# 6. Track-network analysis
# ---------------------------------------------------------------------------

def select_stable_tracks(tracks, sample_every=5, min_lifetime=100, min_updates=10):
    """
    Selects stable tracks for network analysis.

    These are the identity lines that lived long enough to be interpreted as
    persistent recoherence identities.
    """
    stable = []

    for track in tracks:
        lifetime = track_lifetime(track, sample_every=sample_every)

        if lifetime >= min_lifetime and track["updates"] >= min_updates:
            stable.append(track)

    return stable


def representative_ids(track, max_ids=10):
    """
    Returns a representative set of state ids for a track.

    Currently we use the final id set of the track.
    Large tracks are truncated to keep coupling calculation simple.
    """
    ids = list(track["last_ids"])
    return set(ids[:max_ids])


def build_id_state_map(states):
    """Maps state ids to final State objects."""
    return {s.id: s for s in states}


def track_coupling(track_a, track_b, id_state, max_ids=10):
    """
    Computes coupling K(T_a,T_b) between two tracks.

    K is the mean J between representative final states of both tracks.

    If a track's final ids are no longer present in the final state set,
    it contributes no coupling.
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


def build_track_network(
    stable_tracks,
    final_states,
    coupling_threshold=0.25,
    max_ids_per_track=10
):
    """
    Builds a track-track network.

    Nodes:
        stable identity tracks

    Edges:
        K(T_i,T_j) >= coupling_threshold
    """
    id_state = build_id_state_map(final_states)

    nodes = [t["track_id"] for t in stable_tracks]
    track_by_id = {t["track_id"]: t for t in stable_tracks}

    edges = []
    coupling_values = []

    for i, track_a in enumerate(stable_tracks):
        for track_b in stable_tracks[i + 1:]:
            k = track_coupling(
                track_a,
                track_b,
                id_state,
                max_ids=max_ids_per_track
            )

            if k > 0:
                coupling_values.append(k)

            if k >= coupling_threshold:
                edges.append({
                    "a": track_a["track_id"],
                    "b": track_b["track_id"],
                    "K": k,
                })

    return {
        "nodes": nodes,
        "edges": edges,
        "track_by_id": track_by_id,
        "coupling_values": coupling_values,
    }


def network_components(nodes, edges):
    """
    Connected components of the track network.
    """
    adjacency = {node: set() for node in nodes}

    for edge in edges:
        adjacency[edge["a"]].add(edge["b"])
        adjacency[edge["b"]].add(edge["a"])

    visited = set()
    components = []

    for node in nodes:
        if node in visited:
            continue

        stack = [node]
        comp = []

        while stack:
            current = stack.pop()

            if current in visited:
                continue

            visited.add(current)
            comp.append(current)

            for nxt in adjacency[current]:
                if nxt not in visited:
                    stack.append(nxt)

        components.append(comp)

    components.sort(key=len, reverse=True)
    return components


def summarize_track_network(network):
    """
    Compact summary of the identity coupling network.
    """
    nodes = network["nodes"]
    edges = network["edges"]
    components = network_components(nodes, edges)

    degrees = {node: 0 for node in nodes}
    for edge in edges:
        degrees[edge["a"]] += 1
        degrees[edge["b"]] += 1

    coupling_values = [edge["K"] for edge in edges]

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(components),
        "largest_component": len(components[0]) if components else 0,
        "mean_degree": mean(degrees.values()) if degrees else 0.0,
        "max_degree": max(degrees.values()) if degrees else 0,
        "mean_edge_K": mean(coupling_values) if coupling_values else 0.0,
        "components": components,
        "degrees": degrees,
    }


# ---------------------------------------------------------------------------
# 7. Output helpers
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_network_edges_csv(network, output_dir="s6_outputs", filename="track_network_edges.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["a", "b", "K"])
        writer.writeheader()
        for edge in network["edges"]:
            writer.writerow(edge)

    print("saved:", output_path)


def save_s6_batch_csv(records, output_dir="s6_outputs", filename="s6_batch_summary.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "seed",
        "final_states",
        "stable_tracks",
        "network_nodes",
        "network_edges",
        "component_count",
        "largest_component",
        "mean_degree",
        "max_degree",
        "mean_edge_K",
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

def plot_history(history, memory_history, output_dir="s6_outputs", filename="history_memory.png"):
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

    plt.title("S6 population, persistence, and memory")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_network_degree_histogram(summary, output_dir="s6_outputs", filename="network_degrees.png"):
    output_path = ensure_output_dir(output_dir) / filename

    degrees = list(summary["degrees"].values())

    plt.figure(figsize=(8, 5))
    plt.hist(degrees, bins=20)
    plt.xlabel("track-network degree")
    plt.ylabel("count")
    plt.title("S6 identity-network degree distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_network_components(summary, output_dir="s6_outputs", filename="network_components.png"):
    output_path = ensure_output_dir(output_dir) / filename

    sizes = [len(c) for c in summary["components"]]

    plt.figure(figsize=(8, 5))
    plt.bar([str(i) for i in range(len(sizes))], sizes)
    plt.xlabel("component index")
    plt.ylabel("component size")
    plt.title("S6 track-network component sizes")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_s6_batch_summary(records, output_dir="s6_outputs", filename="s6_batch_summary.png"):
    output_path = ensure_output_dir(output_dir) / filename

    seeds = [r["seed"] for r in records]
    edges = [r["network_edges"] for r in records]
    largest = [r["largest_component"] for r in records]
    final_states = [r["final_states"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(seeds, edges, marker="o", label="network edges")
    ax1.plot(seeds, largest, marker="o", label="largest component")
    ax1.set_xlabel("seed")
    ax1.set_ylabel("network statistics")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(seeds, final_states, linestyle="--", label="final states")
    ax2.set_ylabel("final states")
    ax2.legend(loc="upper right")

    plt.title("S6 coupled identity networks across seeds")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


# ---------------------------------------------------------------------------
# 9. Run helpers
# ---------------------------------------------------------------------------

def run_s6_single(
    seed=0,
    output_dir="s6_outputs",
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

    print("=== Single S6 run ===")
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

    network = build_track_network(
        stable_tracks,
        states,
        coupling_threshold=coupling_threshold
    )

    summary = summarize_track_network(network)

    final_avg_memory = memory_history[-1][1] if memory_history else 0.0
    final_max_memory = memory_history[-1][2] if memory_history else 0.0

    print("\nFinal states:", len(states))
    print("Stable tracks:", len(stable_tracks))
    print("Network nodes:", summary["node_count"])
    print("Network edges:", summary["edge_count"])
    print("Component count:", summary["component_count"])
    print("Largest component:", summary["largest_component"])
    print("Mean degree:", round(summary["mean_degree"], 3))
    print("Max degree:", summary["max_degree"])
    print("Mean edge K:", round(summary["mean_edge_K"], 5))
    print("Final avg memory:", round(final_avg_memory, 5))
    print("Final max memory:", round(final_max_memory, 5))

    print("\nTop component sizes:", [len(c) for c in summary["components"][:10]])

    save_network_edges_csv(network, output_dir=output_dir)
    plot_history(history, memory_history, output_dir=output_dir)
    plot_network_degree_histogram(summary, output_dir=output_dir)
    plot_network_components(summary, output_dir=output_dir)

    return states, tracks, stable_tracks, network, summary


def run_s6_batch(
    runs=50,
    output_dir="s6_outputs",
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
    print("\n=== S6 multi-seed coupled identity-network analysis ===")
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

        network = build_track_network(
            stable_tracks,
            states,
            coupling_threshold=coupling_threshold
        )

        summary = summarize_track_network(network)

        final_avg_memory = memory_history[-1][1] if memory_history else 0.0
        final_max_memory = memory_history[-1][2] if memory_history else 0.0

        record = {
            "seed": seed,
            "final_states": len(states),
            "stable_tracks": len(stable_tracks),
            "network_nodes": summary["node_count"],
            "network_edges": summary["edge_count"],
            "component_count": summary["component_count"],
            "largest_component": summary["largest_component"],
            "mean_degree": summary["mean_degree"],
            "max_degree": summary["max_degree"],
            "mean_edge_K": summary["mean_edge_K"],
            "final_avg_memory": final_avg_memory,
            "final_max_memory": final_max_memory,
        }

        records.append(record)

        print(
            "seed", seed,
            "final_states", record["final_states"],
            "stable_tracks", record["stable_tracks"],
            "edges", record["network_edges"],
            "components", record["component_count"],
            "largest", record["largest_component"],
            "mean_degree", round(record["mean_degree"], 3),
            "mean_K", round(record["mean_edge_K"], 4)
        )

    save_s6_batch_csv(records, output_dir=output_dir)
    plot_s6_batch_summary(records, output_dir=output_dir)

    print("\n=== S6 batch summary ===")
    print("mean final states:", round(mean(r["final_states"] for r in records), 3))
    print("mean stable tracks:", round(mean(r["stable_tracks"] for r in records), 3))
    print("mean network edges:", round(mean(r["network_edges"] for r in records), 3))
    print("mean component count:", round(mean(r["component_count"] for r in records), 3))
    print("mean largest component:", round(mean(r["largest_component"] for r in records), 3))
    print("mean degree:", round(mean(r["mean_degree"] for r in records), 3))
    print("mean edge K:", round(mean(r["mean_edge_K"] for r in records), 6))

    return records


if __name__ == "__main__":
    run_s6_single(
        seed=0,
        output_dir="s6_outputs",
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

    run_s6_batch(
        runs=50,
        output_dir="s6_outputs",
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
How to interpret S6
===================

S5:
    identities became weakly self-stabilizing.

S6:
    stable identities can form networks with each other.

Important outputs:
------------------
track_network_edges.csv:
    List of coupled identity-track pairs.

network_degrees.png:
    Degree distribution of the identity network.

network_components.png:
    Sizes of connected components in the identity network.

s6_batch_summary.csv:
    Multi-seed statistics of coupled identity networks.

Interpretation:
---------------
If stable tracks form nontrivial connected components, then the model has
generated relations between identities.

This is a toy-model step from:

    identity -> relation -> structure

In FUT/CoMath language:
    relation is not assumed;
    relation emerges as stable mutual fortsetzbarkeit / recoherence coupling.
"""
