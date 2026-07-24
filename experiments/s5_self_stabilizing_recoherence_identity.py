#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S5 — Primitive Recoherence Simulation with Recursive Self-Stabilizing Identity
============================================================================

S5 extends S4.

S4 detected temporally persistent identity tracks.
However, those tracks were only observed after the fact.
They did not influence the future dynamics.

S5 adds the next conceptual step:

    persistent identity tracks weakly stabilize future states.

This models a first primitive form of recursive self-return:

    past persistence -> weakly affects future persistence

In FUT/CoMath language, this is a toy-model step toward:

    rho_f = incomplete recursive self-return

Important:
----------
This is NOT proof of FUT/CoMath.
It is a toy model testing whether primitive recoherence selection plus weak
memory feedback can generate more stable temporal identity structures.

Core S5 question:
-----------------
Can long-lived recoherence tracks actively stabilize their own continuation
without collapsing the system into rigid repetition?

If yes, this would be an algorithmic first step toward:

    persistence -> identity -> recursive identity -> object-like structure
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
        This is the S5 addition.
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
    Local persistence without memory feedback.

    This is the S0-S4 persistence term.
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
    S5 persistence.

    P_total = P_local + memory_lambda * memory

    The memory term is intentionally weak.

    If memory_lambda is too large, the system may become too rigid.
    If memory_lambda is too small, S5 behaves almost like S4.
    """
    return local_persistence(x, states) + memory_lambda * x.memory


# ---------------------------------------------------------------------------
# 3. Primitive dynamics
# ---------------------------------------------------------------------------

def recurse(x, memory_decay=0.985):
    """
    Recursive continuation of one state.

    The state keeps its id under ordinary continuation.

    memory_decay:
        Memory is not perfect.
        It decays slightly each step.
        This models incomplete recursive self-return rather than rigid identity.
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

    The successor inherits some memory, but not all.
    This prevents perfect copying.
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
    if not a and not b:
        return 0.0

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

    If no sufficient overlap exists, a new track is created.
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


# ---------------------------------------------------------------------------
# 5. Memory feedback
# ---------------------------------------------------------------------------

def apply_track_memory_feedback(
    states,
    active_tracks,
    sample_every=5,
    memory_gain=0.04,
    max_memory=1.0,
    min_track_updates=3
):
    """
    Adds weak memory to states that belong to sufficiently persistent tracks.

    This is the central S5 step.

    A track that has survived several updates becomes a weak stabilizing
    source for its member states.

    This models:
        past persistence -> weak future stabilization

    min_track_updates:
        A track must persist for a few samples before it stabilizes anything.

    memory_gain:
        How strongly persistent identity feeds back into states.

    max_memory:
        Memory is capped to avoid runaway self-locking.
    """

    id_to_state = {s.id: s for s in states}

    for track in active_tracks.values():
        if track["updates"] < min_track_updates:
            continue

        lifetime = track_lifetime(track, sample_every=sample_every)

        # Longer-lived tracks produce slightly stronger memory,
        # but with a saturating factor.
        strength = memory_gain * min(1.0, lifetime / 100.0)

        for state_id in track["last_ids"]:
            if state_id in id_to_state:
                state = id_to_state[state_id]
                state.memory = min(max_memory, state.memory + strength)


# ---------------------------------------------------------------------------
# 6. Simulation with S5 feedback
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
    Runs S5.

    Difference from S4:
        persistence includes memory
        and persistent tracks add memory back to their member states.

    Returns:
        final states
        history
        tracks
        snapshots
        memory history
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
# 7. Summary helpers
# ---------------------------------------------------------------------------

def summarize_tracks(tracks, sample_every=5, min_lifetime=10):
    if not tracks:
        return {
            "track_count": 0,
            "mean_lifetime": 0.0,
            "max_lifetime": 0,
            "stable_track_count": 0,
            "mean_size": 0.0,
        }

    lifetimes = [track_lifetime(t, sample_every=sample_every) for t in tracks]
    mean_sizes = [mean(t["sizes"]) for t in tracks if t["sizes"]]

    return {
        "track_count": len(tracks),
        "mean_lifetime": mean(lifetimes),
        "max_lifetime": max(lifetimes),
        "stable_track_count": sum(1 for l in lifetimes if l >= min_lifetime),
        "mean_size": mean(mean_sizes) if mean_sizes else 0.0,
    }


def average_proto_distance(states):
    if len(states) < 2:
        return 0.0

    distances = []

    for i, a in enumerate(states):
        for j, b in enumerate(states):
            if j <= i:
                continue
            distances.append(proto_distance(a, b))

    return mean(distances) if distances else 0.0


# ---------------------------------------------------------------------------
# 8. Output and plotting
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tracks_csv(tracks, output_dir="s5_outputs", filename="identity_tracks.csv", sample_every=5):
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "track_id",
        "start_step",
        "end_step",
        "lifetime",
        "updates",
        "mean_size",
        "max_size",
        "mean_overlap",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for track in tracks:
            sizes = track["sizes"]
            overlaps = track["overlaps"]

            writer.writerow({
                "track_id": track["track_id"],
                "start_step": track["start_step"],
                "end_step": track["end_step"],
                "lifetime": track_lifetime(track, sample_every=sample_every),
                "updates": track["updates"],
                "mean_size": mean(sizes) if sizes else 0.0,
                "max_size": max(sizes) if sizes else 0,
                "mean_overlap": mean(overlaps) if overlaps else 0.0,
            })

    print("saved:", output_path)


def save_s5_batch_csv(records, output_dir="s5_outputs", filename="s5_batch_summary.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "seed",
        "final_states",
        "avg_proto_distance",
        "track_count",
        "mean_lifetime",
        "max_lifetime",
        "stable_track_count",
        "mean_track_size",
        "final_avg_memory",
        "final_max_memory",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in records:
            writer.writerow(row)

    print("saved:", output_path)


def plot_history(history, memory_history, output_dir="s5_outputs", filename="history_memory.png"):
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

    plt.title("S5 population, persistence, and memory")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_track_lifetimes(tracks, output_dir="s5_outputs", filename="track_lifetimes.png", sample_every=5):
    output_path = ensure_output_dir(output_dir) / filename

    lifetimes = [track_lifetime(t, sample_every=sample_every) for t in tracks]

    plt.figure(figsize=(8, 5))
    plt.hist(lifetimes, bins=20)
    plt.xlabel("track lifetime")
    plt.ylabel("count")
    plt.title("S5 identity track lifetime distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_s5_batch_summary(records, output_dir="s5_outputs", filename="s5_batch_summary.png"):
    output_path = ensure_output_dir(output_dir) / filename

    seeds = [r["seed"] for r in records]
    max_lifetimes = [r["max_lifetime"] for r in records]
    stable_tracks = [r["stable_track_count"] for r in records]
    final_states = [r["final_states"] for r in records]
    final_avg_memory = [r["final_avg_memory"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(seeds, max_lifetimes, marker="o", label="max lifetime")
    ax1.plot(seeds, stable_tracks, marker="o", label="stable tracks")
    ax1.set_xlabel("seed")
    ax1.set_ylabel("identity statistics")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(seeds, final_states, linestyle="--", label="final states")
    ax2.plot(seeds, final_avg_memory, linestyle=":", label="final avg memory")
    ax2.set_ylabel("final states / memory")
    ax2.legend(loc="upper right")

    plt.title("S5 self-stabilizing identity across seeds")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def print_top_tracks(tracks, sample_every=5, top_n=10):
    sorted_tracks = sorted(
        tracks,
        key=lambda t: track_lifetime(t, sample_every=sample_every),
        reverse=True
    )[:top_n]

    print("\n=== Longest identity tracks ===")

    for track in sorted_tracks:
        sizes = track["sizes"]
        overlaps = track["overlaps"]

        print(
            "track", track["track_id"],
            "start", track["start_step"],
            "end", track["end_step"],
            "lifetime", track_lifetime(track, sample_every=sample_every),
            "updates", track["updates"],
            "mean_size", round(mean(sizes), 3) if sizes else 0.0,
            "max_size", max(sizes) if sizes else 0,
            "mean_overlap", round(mean(overlaps), 3) if overlaps else 0.0
        )


# ---------------------------------------------------------------------------
# 9. Run helpers
# ---------------------------------------------------------------------------

def run_s5_single(
    seed=0,
    output_dir="s5_outputs",
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    memory_lambda=0.08,
    memory_gain=0.04,
    memory_decay=0.985
):
    random.seed(seed)

    print("=== Single S5 run ===")
    print("seed:", seed)
    print("cluster_threshold:", cluster_threshold)
    print("sample_every:", sample_every)
    print("overlap_threshold:", overlap_threshold)
    print("memory_lambda:", memory_lambda)
    print("memory_gain:", memory_gain)
    print("memory_decay:", memory_decay)

    states, history, tracks, snapshots, memory_history = run_simulation_with_self_stabilizing_identity(
        cluster_threshold=cluster_threshold,
        sample_every=sample_every,
        overlap_threshold=overlap_threshold,
        memory_lambda=memory_lambda,
        memory_gain=memory_gain,
        memory_decay=memory_decay,
        verbose=True
    )

    summary = summarize_tracks(tracks, sample_every=sample_every)

    final_avg_memory = memory_history[-1][1] if memory_history else 0.0
    final_max_memory = memory_history[-1][2] if memory_history else 0.0

    print("\nFinal states:", len(states))
    print("Average proto-distance:", round(average_proto_distance(states), 6))
    print("Track count:", summary["track_count"])
    print("Mean lifetime:", round(summary["mean_lifetime"], 3))
    print("Max lifetime:", summary["max_lifetime"])
    print("Stable track count:", summary["stable_track_count"])
    print("Mean track size:", round(summary["mean_size"], 3))
    print("Final avg memory:", round(final_avg_memory, 6))
    print("Final max memory:", round(final_max_memory, 6))

    print_top_tracks(tracks, sample_every=sample_every)

    save_tracks_csv(tracks, output_dir=output_dir, sample_every=sample_every)
    plot_history(history, memory_history, output_dir=output_dir)
    plot_track_lifetimes(tracks, output_dir=output_dir, sample_every=sample_every)

    return states, history, tracks, snapshots, memory_history


def run_s5_batch(
    runs=50,
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    memory_lambda=0.08,
    memory_gain=0.04,
    memory_decay=0.985,
    output_dir="s5_outputs"
):
    print("\n=== S5 multi-seed self-stabilizing identity analysis ===")
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

        summary = summarize_tracks(tracks, sample_every=sample_every)

        final_avg_memory = memory_history[-1][1] if memory_history else 0.0
        final_max_memory = memory_history[-1][2] if memory_history else 0.0

        record = {
            "seed": seed,
            "final_states": len(states),
            "avg_proto_distance": average_proto_distance(states),
            "track_count": summary["track_count"],
            "mean_lifetime": summary["mean_lifetime"],
            "max_lifetime": summary["max_lifetime"],
            "stable_track_count": summary["stable_track_count"],
            "mean_track_size": summary["mean_size"],
            "final_avg_memory": final_avg_memory,
            "final_max_memory": final_max_memory,
        }

        records.append(record)

        print(
            "seed", seed,
            "final_states", record["final_states"],
            "avg_g", round(record["avg_proto_distance"], 5),
            "tracks", record["track_count"],
            "mean_lifetime", round(record["mean_lifetime"], 3),
            "max_lifetime", record["max_lifetime"],
            "stable_tracks", record["stable_track_count"],
            "mean_size", round(record["mean_track_size"], 3),
            "avg_memory", round(record["final_avg_memory"], 5),
            "max_memory", round(record["final_max_memory"], 5)
        )

    save_s5_batch_csv(records, output_dir=output_dir)
    plot_s5_batch_summary(records, output_dir=output_dir)

    print("\n=== S5 batch summary ===")
    print("mean final states:", round(mean(r["final_states"] for r in records), 3))
    print("mean avg proto-distance:", round(mean(r["avg_proto_distance"] for r in records), 6))
    print("mean track count:", round(mean(r["track_count"] for r in records), 3))
    print("mean lifetime:", round(mean(r["mean_lifetime"] for r in records), 3))
    print("mean max lifetime:", round(mean(r["max_lifetime"] for r in records), 3))
    print("mean stable tracks:", round(mean(r["stable_track_count"] for r in records), 3))
    print("mean track size:", round(mean(r["mean_track_size"] for r in records), 3))
    print("mean final avg memory:", round(mean(r["final_avg_memory"] for r in records), 6))
    print("mean final max memory:", round(mean(r["final_max_memory"] for r in records), 6))

    return records


if __name__ == "__main__":
    run_s5_single(
        seed=0,
        output_dir="s5_outputs",
        cluster_threshold=0.6,
        sample_every=5,
        overlap_threshold=0.30,
        memory_lambda=0.08,
        memory_gain=0.04,
        memory_decay=0.985
    )

    run_s5_batch(
        runs=50,
        cluster_threshold=0.6,
        sample_every=5,
        overlap_threshold=0.30,
        memory_lambda=0.08,
        memory_gain=0.04,
        memory_decay=0.985,
        output_dir="s5_outputs"
    )


"""
How to interpret S5
===================

S4:
    persistent tracks were detected.

S5:
    persistent tracks weakly feed back into the survival of their states.

This tests whether identity can become self-stabilizing.

Important signs:
----------------
1. max lifetime stays high
2. stable track count increases or remains robust
3. memory grows but does not immediately saturate everything
4. the system avoids both collapse and total rigid locking

If successful, S5 is the first toy-model step toward:

    persistence -> identity -> recursive self-stabilization

In FUT/CoMath language:
    an object is not a thing,
    but a self-stabilizing continuation line of incomplete recursive closure.
"""
