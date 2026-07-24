#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4 — Primitive Recoherence Simulation with Temporal Cluster Identity
===================================================================

This program extends S3.

S3 showed statistically robust hierarchy across many random seeds.
S4 asks the next deeper question:

    Do clusters persist through time as recognizable identities?

This is important because in FUT/CoMath terms:

    persistence -> identity -> object-like structure

The program tracks clusters across time using overlap.

Important:
----------
This is NOT proof of FUT/CoMath.
It is a toy model testing whether primitive recoherence selection can generate
temporally persistent recoherence classes.

Core S4 idea:
-------------
At selected time steps, we detect clusters.

Then we compare clusters from the current step with active clusters from
the previous sampled step.

If overlap is high enough, the cluster is treated as the continuation of
the same identity line.

This produces:
    - cluster lifetimes
    - identity tracks
    - stable persistence classes
    - lifetime distributions
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
        Stable numerical identifier.
        This is needed for temporal overlap tracking.

    w:
        Continuation weight / internal stability.

    sigma:
        Connection orientation.

    chi:
        Interference phase.
    """

    def __init__(self, state_id, w, sigma, chi):
        self.id = state_id
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
    """Connection capability between two states."""
    return math.exp(
        -angle_diff(x.sigma, y.sigma)
        -angle_diff(x.chi, y.chi)
    )


def I(x, y):
    """Interference between two states."""
    return J(x, y) * math.cos(angle_diff(x.chi, y.chi))


def proto_distance(x, y):
    """
    Proto-metric distance:

        g_f(x,y) = -log(J(x,y))
    """
    join = max(J(x, y), 1e-12)
    return -math.log(join)


def persistence(x, states):
    """
    Persistence of a state relative to all other states.
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

    The state keeps the same identity id across ordinary recursive update.
    """
    eta = random.uniform(-0.02, 0.04)
    kappa = random.uniform(0.00, 0.02)
    delta = random.uniform(-0.08, 0.08)
    phi = random.uniform(-0.08, 0.08)

    return State(
        x.id,
        max(0.0, x.w + eta - kappa),
        (x.sigma + delta) % (2 * math.pi),
        (x.chi + phi) % (2 * math.pi)
    )


def split(x, next_id):
    """
    Creates a similar successor state with a new identity id.

    The successor is not identical, but remains close to the parent.
    """
    return State(
        next_id,
        x.w * random.uniform(0.8, 1.05),
        (x.sigma + random.uniform(-0.05, 0.05)) % (2 * math.pi),
        (x.chi + random.uniform(-0.05, 0.05)) % (2 * math.pi)
    )


def run_simulation_with_identity_tracking(
    initial_count=40,
    steps=300,
    delta_min=0.001,
    delta_split=0.08,
    max_states=120,
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    verbose=True
):
    """
    Runs the S4 simulation and tracks cluster identity over time.

    cluster_threshold:
        J-threshold used to form clusters at sampled time steps.
        Higher values track stronger attractor cores.

    sample_every:
        Cluster tracking is performed every N steps.

    overlap_threshold:
        Minimum Jaccard overlap needed to continue an identity track.

    Returns:
        final states
        history
        identity tracks
        cluster snapshots
    """

    next_id = 0

    states = []
    for _ in range(initial_count):
        states.append(
            State(
                next_id,
                random.uniform(0.2, 1.0),
                random.uniform(0, 2 * math.pi),
                random.uniform(0, 2 * math.pi)
            )
        )
        next_id += 1

    history = []
    snapshots = []

    active_tracks = {}
    finished_tracks = []
    next_track_id = 0

    previous_clusters = []

    for step in range(steps):
        states = [recurse(x) for x in states]

        scored = [(x, persistence(x, states)) for x in states]
        survivors = [x for x, p in scored if p >= delta_min]

        offspring = []
        for x, p in scored:
            if p >= delta_split and len(survivors) + len(offspring) < max_states:
                offspring.append(split(x, next_id))
                next_id += 1

        states = survivors + offspring

        avg_p = sum(p for _, p in scored) / len(scored) if scored else 0.0
        history.append((step, len(states), avg_p))

        if verbose and step % 25 == 0:
            print("step", step, "states", len(states), "avg_p", round(avg_p, 5))

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

            # update next_track_id after possible new tracks
            if active_tracks:
                next_track_id = max(active_tracks.keys()) + 1
            elif finished_tracks:
                next_track_id = max(t["track_id"] for t in finished_tracks) + 1
            else:
                next_track_id = 0

            snapshots.append({
                "step": step,
                "clusters": cluster_records,
            })

            previous_clusters = cluster_records

        if not states:
            break

    # Close all active tracks at the end
    finished_tracks.extend(active_tracks.values())

    return states, history, finished_tracks, snapshots


# ---------------------------------------------------------------------------
# 4. Cluster detection and identity tracking
# ---------------------------------------------------------------------------

def find_clusters(states, threshold=0.6):
    """
    Finds recoherence clusters as connected components in a graph.

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
    """
    Jaccard overlap of two id sets.

        |A ∩ B| / |A ∪ B|
    """
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
    Assigns current clusters to existing identity tracks by maximum overlap.

    A track represents a temporally persistent cluster identity.

    If a current cluster overlaps strongly enough with an active track,
    it continues that track.

    Otherwise a new track is created.
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

    # Tracks that were not continued are finished
    for track_id, track in active_tracks.items():
        if track_id not in assigned_tracks and track_id not in new_active_tracks:
            finished_tracks.append(track)

    active_tracks.clear()
    active_tracks.update(new_active_tracks)


def track_lifetime(track, sample_every=5):
    """
    Lifetime measured in simulation steps.
    """
    return track["end_step"] - track["start_step"] + sample_every


def summarize_tracks(tracks, sample_every=5, min_lifetime=10):
    """
    Computes summary statistics for identity tracks.
    """
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


# ---------------------------------------------------------------------------
# 5. Output and plotting
# ---------------------------------------------------------------------------

def ensure_output_dir(output_dir):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_tracks_csv(tracks, output_dir="s4_outputs", filename="identity_tracks.csv", sample_every=5):
    """
    Saves identity track data to CSV.
    """
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


def plot_history(history, output_dir="s4_outputs", filename="history.png"):
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

    plt.title("S4 population and persistence history")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_track_lifetimes(tracks, output_dir="s4_outputs", filename="track_lifetimes.png", sample_every=5):
    output_path = ensure_output_dir(output_dir) / filename

    lifetimes = [track_lifetime(t, sample_every=sample_every) for t in tracks]

    plt.figure(figsize=(8, 5))
    plt.hist(lifetimes, bins=20)
    plt.xlabel("track lifetime")
    plt.ylabel("count")
    plt.title("S4 identity track lifetime distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


def plot_longest_tracks(tracks, output_dir="s4_outputs", filename="longest_tracks.png", sample_every=5, top_n=10):
    output_path = ensure_output_dir(output_dir) / filename

    sorted_tracks = sorted(
        tracks,
        key=lambda t: track_lifetime(t, sample_every=sample_every),
        reverse=True
    )[:top_n]

    labels = [str(t["track_id"]) for t in sorted_tracks]
    lifetimes = [track_lifetime(t, sample_every=sample_every) for t in sorted_tracks]
    mean_sizes = [mean(t["sizes"]) if t["sizes"] else 0 for t in sorted_tracks]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.bar(labels, lifetimes)
    ax1.set_xlabel("track id")
    ax1.set_ylabel("lifetime")

    ax2 = ax1.twinx()
    ax2.plot(labels, mean_sizes, marker="o", linestyle="--")
    ax2.set_ylabel("mean cluster size")

    plt.title("S4 longest identity tracks")
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
# 6. Batch tracking statistics
# ---------------------------------------------------------------------------

def run_s4_single(
    seed=0,
    output_dir="s4_outputs",
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30
):
    random.seed(seed)

    print("=== Single S4 run ===")
    print("seed:", seed)
    print("cluster_threshold:", cluster_threshold)
    print("sample_every:", sample_every)
    print("overlap_threshold:", overlap_threshold)

    states, history, tracks, snapshots = run_simulation_with_identity_tracking(
        cluster_threshold=cluster_threshold,
        sample_every=sample_every,
        overlap_threshold=overlap_threshold,
        verbose=True
    )

    summary = summarize_tracks(tracks, sample_every=sample_every)

    print("\nFinal states:", len(states))
    print("Track count:", summary["track_count"])
    print("Mean lifetime:", round(summary["mean_lifetime"], 3))
    print("Max lifetime:", summary["max_lifetime"])
    print("Stable track count:", summary["stable_track_count"])
    print("Mean track size:", round(summary["mean_size"], 3))

    print_top_tracks(tracks, sample_every=sample_every)

    save_tracks_csv(tracks, output_dir=output_dir, sample_every=sample_every)
    plot_history(history, output_dir=output_dir)
    plot_track_lifetimes(tracks, output_dir=output_dir, sample_every=sample_every)
    plot_longest_tracks(tracks, output_dir=output_dir, sample_every=sample_every)

    return states, history, tracks, snapshots


def run_s4_batch(
    runs=50,
    cluster_threshold=0.6,
    sample_every=5,
    overlap_threshold=0.30,
    output_dir="s4_outputs"
):
    print("\n=== S4 multi-seed temporal identity analysis ===")
    print("runs:", runs)

    records = []

    for seed in range(runs):
        random.seed(seed)

        states, history, tracks, snapshots = run_simulation_with_identity_tracking(
            cluster_threshold=cluster_threshold,
            sample_every=sample_every,
            overlap_threshold=overlap_threshold,
            verbose=False
        )

        summary = summarize_tracks(tracks, sample_every=sample_every)

        record = {
            "seed": seed,
            "final_states": len(states),
            "track_count": summary["track_count"],
            "mean_lifetime": summary["mean_lifetime"],
            "max_lifetime": summary["max_lifetime"],
            "stable_track_count": summary["stable_track_count"],
            "mean_track_size": summary["mean_size"],
        }

        records.append(record)

        print(
            "seed", seed,
            "final_states", record["final_states"],
            "tracks", record["track_count"],
            "mean_lifetime", round(record["mean_lifetime"], 3),
            "max_lifetime", record["max_lifetime"],
            "stable_tracks", record["stable_track_count"],
            "mean_size", round(record["mean_track_size"], 3)
        )

    save_s4_batch_csv(records, output_dir=output_dir)
    plot_s4_batch_summary(records, output_dir=output_dir)

    print("\n=== S4 batch summary ===")
    print("mean final states:", round(mean(r["final_states"] for r in records), 3))
    print("mean track count:", round(mean(r["track_count"] for r in records), 3))
    print("mean lifetime:", round(mean(r["mean_lifetime"] for r in records), 3))
    print("mean max lifetime:", round(mean(r["max_lifetime"] for r in records), 3))
    print("mean stable tracks:", round(mean(r["stable_track_count"] for r in records), 3))
    print("mean track size:", round(mean(r["mean_track_size"] for r in records), 3))

    return records


def save_s4_batch_csv(records, output_dir="s4_outputs", filename="s4_batch_summary.csv"):
    output_path = ensure_output_dir(output_dir) / filename

    fieldnames = [
        "seed",
        "final_states",
        "track_count",
        "mean_lifetime",
        "max_lifetime",
        "stable_track_count",
        "mean_track_size",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print("saved:", output_path)


def plot_s4_batch_summary(records, output_dir="s4_outputs", filename="s4_batch_summary.png"):
    output_path = ensure_output_dir(output_dir) / filename

    seeds = [r["seed"] for r in records]
    max_lifetimes = [r["max_lifetime"] for r in records]
    stable_tracks = [r["stable_track_count"] for r in records]
    final_states = [r["final_states"] for r in records]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(seeds, max_lifetimes, marker="o", label="max lifetime")
    ax1.plot(seeds, stable_tracks, marker="o", label="stable tracks")
    ax1.set_xlabel("seed")
    ax1.set_ylabel("identity statistics")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.plot(seeds, final_states, linestyle="--", label="final states")
    ax2.set_ylabel("final states")
    ax2.legend(loc="upper right")

    plt.title("S4 temporal identity robustness across seeds")
    fig.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()

    print("saved:", output_path)


if __name__ == "__main__":
    # Single detailed run.
    run_s4_single(
        seed=0,
        output_dir="s4_outputs",
        cluster_threshold=0.6,
        sample_every=5,
        overlap_threshold=0.30
    )

    # Multi-seed temporal identity test.
    run_s4_batch(
        runs=50,
        cluster_threshold=0.6,
        sample_every=5,
        overlap_threshold=0.30,
        output_dir="s4_outputs"
    )


"""
How to interpret S4
===================

If long-lived identity tracks appear, then the model produces something
stronger than final-state clusters.

It produces temporally persistent recoherence classes.

This is the first toy-model step toward:

    persistence -> identity -> object-like structure

Important parameters:
---------------------
cluster_threshold:
    Higher values track stronger cluster cores.

sample_every:
    Smaller values track more frequently.

overlap_threshold:
    Higher values require stricter identity continuity.

Scientific caution:
-------------------
This remains a toy model. It does not prove FUT/CoMath.
It tests whether the conceptual chain "recoherence -> persistence -> identity"
can be represented algorithmically.
"""
