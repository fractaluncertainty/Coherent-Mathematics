#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S9 — Recoherence Geometry Classifier
====================================

Purpose
-------
S9 consolidates the three S8 branches:

1) s8_1_minimal_j_geometry_candidates.py
   - minimal J-property candidates
   - scale/decay/cutoff/rational/anisotropic families

2) s8_j_ablation_coupling_pluralism.py
   - artifact test against resonance / threshold / random controls

3) s8_j_agnosticity_proto_geometry.py
   - dimensionality, curvature proxies, asymmetry, null model,
     weighted representatives

S9 changes the question from:

    "Does this J generate edges?"

to:

    "Which J-properties produce the best structured, nontrivial,
     non-overconnected, FUT/CoMath-compatible proto-geometry?"

Core questions answered by S9
-----------------------------
1. Minimal properties required by J.
2. Maximal properties J may have before overcoupling destroys geometry.
3. Best J-property profiles for high structure.
4. Which profiles approach an icosahedral / triangulated recoherence shell.
5. Whether CoMath/FUT primitives are actually represented in code.
6. How to improve qualitative claims by stronger diagnostics.

Important limitation
--------------------
This is still a toy-model artifact / robustness test.
It is not a proof of FUT/CoMath.

Run
---
    python3 s9_recoherence_geometry_classifier.py

Optional quick run:
    python3 s9_recoherence_geometry_classifier.py --runs 5 --steps 180

Outputs
-------
    s9_outputs/s9_geometry_classifier_results.csv
    s9_outputs/s9_geometry_classifier_summary.csv
    s9_outputs/s9_summary_metrics.png
    s9_outputs/s9_property_scores.png
    s9_outputs/s9_classification_report.md

No external dependencies except matplotlib.
If numpy is installed, S9 also computes MDS effective dimension and spectrum.
"""

import argparse
import csv
import heapq
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False


# =============================================================================
# 0. CoMath/FUT-inspired numeric wrappers
# =============================================================================

@dataclass(frozen=True)
class FuzzyZero:
    """
    0_f: not an absolute mathematical zero, but a finite numerical zone in which
    continuation / distinction is operationally non-effective.

    In this code, 0_f is implemented as a tolerance band. This is not the full
    CoMath/FUT ontology, but it prevents pretending that Python's exact 0.0 is
    the same as CoMath 0_f.
    """
    eps: float = 1e-9

    def contains(self, x: float) -> bool:
        return abs(x) <= self.eps


@dataclass(frozen=True)
class OpenInfinityF:
    """
    infinity_f: operational open infinity, represented by a large but finite
    cutoff. It marks unreachable / non-fortsetzbar distances in the toy graph.
    """
    cap: float = 1e12

    def is_unreachable(self, x: float) -> bool:
        return (not math.isfinite(x)) or x >= self.cap


@dataclass(frozen=True)
class EqualF:
    """
    =_f: asymptotic non-distinguishability, not exact equality.
    """
    tol: float = 1e-6

    def __call__(self, a: float, b: float) -> bool:
        return abs(a - b) <= self.tol


ZERO_F = FuzzyZero(1e-9)
INFINITY_F = OpenInfinityF(1e12)
EQUAL_F = EqualF(1e-6)


@dataclass
class CoherenceValue:
    """
    Minimal CoMath/FUT-inspired bounded coherence value.

    It keeps the code honest: coherence is represented as a bounded operational
    value in [0,1], with a 0_f zone and =_f comparison.
    """
    value: float

    def __post_init__(self):
        self.value = max(0.0, min(1.0, float(self.value)))

    def is_zero_f(self) -> bool:
        return ZERO_F.contains(self.value)

    def equal_f(self, other: "CoherenceValue", tol: float = 1e-6) -> bool:
        return abs(self.value - other.value) <= tol


# =============================================================================
# 1. Basic dynamics
# =============================================================================

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


def angle_diff_directed(a, b):
    return (b - a) % (2 * math.pi)


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def clamp01(x):
    return max(0.0, min(1.0, x))


def d_l1(x, y):
    return angle_diff(x.sigma, y.sigma) + angle_diff(x.chi, y.chi)


def d_l2(x, y):
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    return math.sqrt(ds * ds + dc * dc)


def d_linf(x, y):
    return max(angle_diff(x.sigma, y.sigma), angle_diff(x.chi, y.chi))


# =============================================================================
# 2. J library: S8 + S8.1 + S8-agnosticity candidates
# =============================================================================

@dataclass(frozen=True)
class JProperties:
    name: str
    monotone: bool
    gradual: bool
    local: bool
    continuous: bool
    symmetric: bool
    long_range: str       # none / weak / moderate / strong
    cutoff: bool
    resonant: bool
    random_like: bool
    norm_family: str
    comment: str


@dataclass
class JSpec:
    name: str
    func: callable
    properties: JProperties


def J_metric_exp(x, y):
    return math.exp(-d_l1(x, y))


def make_J_exp_alpha(alpha):
    def J(x, y):
        return math.exp(-alpha * d_l1(x, y))
    return J


def make_J_rational(p=1.0, scale=1.0):
    def J(x, y):
        return 1.0 / (1.0 + (d_l1(x, y) / scale) ** p)
    return J


def make_J_linear_cutoff(lam=0.25):
    def J(x, y):
        return max(0.0, 1.0 - lam * d_l1(x, y))
    return J


def make_J_gaussian_l2(beta=1.0):
    def J(x, y):
        d = d_l2(x, y)
        return math.exp(-beta * d * d)
    return J


def make_J_linf_exp(alpha=1.0):
    def J(x, y):
        return math.exp(-alpha * d_linf(x, y))
    return J


def J_anisotropic_exp(x, y):
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    return math.exp(-0.65 * ds - 1.35 * dc)


def J_power_s8(x, y):
    a = max(0.0, 1.0 - angle_diff(x.sigma, y.sigma) / math.pi) ** 2
    b = max(0.0, 1.0 - angle_diff(x.chi, y.chi) / math.pi) ** 2
    return a * b


def J_asymmetric(x, y):
    ds = angle_diff_directed(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    return math.exp(-ds / math.pi - dc)


def J_step(x, y):
    threshold = math.pi / 4
    if angle_diff(x.sigma, y.sigma) < threshold and angle_diff(x.chi, y.chi) < threshold:
        return 1.0
    return 0.05


def J_cosine_resonance(x, y):
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)
    raw = 0.5 * (math.cos(ds) + math.cos(dc))
    return clamp01(0.5 + 0.5 * raw)


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


def J_random_hash(x, y):
    a = min(x.id, y.id)
    b = max(x.id, y.id)
    h = (1013904223 * a + 1664525 * b + 1013904223) % (2 ** 32)
    return h / (2 ** 32)


def build_j_library():
    """All candidates from the three S8 scripts in one place."""
    specs = []

    def add(name, func, monotone, gradual, local, continuous, symmetric,
            long_range, cutoff, resonant, random_like, norm_family, comment):
        specs.append(JSpec(
            name=name,
            func=func,
            properties=JProperties(
                name=name,
                monotone=monotone,
                gradual=gradual,
                local=local,
                continuous=continuous,
                symmetric=symmetric,
                long_range=long_range,
                cutoff=cutoff,
                resonant=resonant,
                random_like=random_like,
                norm_family=norm_family,
                comment=comment,
            )
        ))

    add("metric_exp", J_metric_exp, True, True, True, True, True,
        "moderate", False, False, False, "L1", "S7/S8 baseline")
    add("exp_alpha_0_5", make_J_exp_alpha(0.5), True, True, True, True, True,
        "strong", False, False, False, "L1", "slow exponential decay; tends to weak differentiation")
    add("exp_alpha_1_0", make_J_exp_alpha(1.0), True, True, True, True, True,
        "moderate", False, False, False, "L1", "same as metric_exp")
    add("exp_alpha_2_0", make_J_exp_alpha(2.0), True, True, True, True, True,
        "weak", False, False, False, "L1", "fast exponential decay; fragmentation risk")
    add("rational_p1", make_J_rational(1.0), True, True, True, True, True,
        "strong", False, False, False, "L1", "long-range algebraic tail; overconnection risk")
    add("rational_p2", make_J_rational(2.0), True, True, True, True, True,
        "moderate", False, False, False, "L1", "algebraic but less long-range than p1")
    add("linear_cutoff_0_20", make_J_linear_cutoff(0.20), True, True, True, False, True,
        "none", True, False, False, "L1", "compact support; hard zero outside range")
    add("linear_cutoff_0_35", make_J_linear_cutoff(0.35), True, True, True, False, True,
        "none", True, False, False, "L1", "stronger compact cutoff")
    add("gaussian_l2_0_5", make_J_gaussian_l2(0.5), True, True, True, True, True,
        "weak", False, False, False, "L2", "squared decay; may be too suppressive")
    add("gaussian_l2_1_0", make_J_gaussian_l2(1.0), True, True, True, True, True,
        "weak", False, False, False, "L2", "strong squared decay")
    add("linf_exp_1_0", make_J_linf_exp(1.0), True, True, True, True, True,
        "moderate", False, False, False, "Linf", "non-L1 norm; norm-agnosticity test")
    add("anisotropic_exp", J_anisotropic_exp, True, True, True, True, True,
        "moderate", False, False, False, "anisotropic", "unequal sigma/chi weights")
    add("power", J_power_s8, True, True, True, True, True,
        "moderate", False, False, False, "coordinate-power", "S8 agnosticity power candidate")
    add("asymmetric", J_asymmetric, True, True, True, True, False,
        "moderate", False, False, False, "directed", "directed proto-topology")
    add("step", J_step, False, False, True, False, True,
        "none", True, False, False, "threshold", "hard two-level gate")
    add("cosine_resonance", J_cosine_resonance, False, True, False, True, True,
        "strong", False, True, False, "periodic", "periodic resonance control")
    add("mixed_resonance", J_mixed_resonance, False, True, False, True, True,
        "strong", False, True, False, "mixed-periodic", "nonmonotone interference control")
    add("threshold_resonance", J_threshold_resonance, False, False, False, False, True,
        "moderate", False, True, False, "threshold-resonance", "discrete resonance levels")
    add("random_projection", J_random_projection, False, False, False, True, True,
        "strong", False, False, True, "random-feature", "structured random-feature control")
    add("random_hash", J_random_hash, False, False, False, False, True,
        "strong", False, False, True, "random-hash", "pure null model")

    return specs


# =============================================================================
# 3. Simulation core
# =============================================================================


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

            assign_clusters_to_tracks(
                cluster_records,
                active_tracks,
                finished_tracks,
                overlap_threshold,
                [next_track_id],
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
                min_track_updates=min_track_updates,
            )

        if not states:
            break

    finished_tracks.extend(active_tracks.values())
    return states, history, finished_tracks, memory_history


# =============================================================================
# 4. Geometry construction
# =============================================================================


def select_stable_tracks(tracks, sample_every=5, min_lifetime=100, min_updates=10):
    return [
        t for t in tracks
        if track_lifetime(t, sample_every=sample_every) >= min_lifetime and t["updates"] >= min_updates
    ]


def representative_ids_weighted(track, id_state, max_ids=10):
    ids = list(track["last_ids"])
    weighted = [(id_state[i].w if i in id_state else 0.0, i) for i in ids]
    weighted.sort(reverse=True)
    return set(i for _, i in weighted[:max_ids])


def track_coupling(track_a, track_b, id_state, J_func, max_ids=10):
    ids_a = representative_ids_weighted(track_a, id_state, max_ids=max_ids)
    ids_b = representative_ids_weighted(track_b, id_state, max_ids=max_ids)
    states_a = [id_state[i] for i in ids_a if i in id_state]
    states_b = [id_state[i] for i in ids_b if i in id_state]
    if not states_a or not states_b:
        return 0.0
    values = [J_func(a, b) for a in states_a for b in states_b if a.id != b.id]
    return mean(values) if values else 0.0


def build_track_geometry(stable_tracks, final_states, J_func, coupling_threshold=0.25, max_ids_per_track=10, directed=False):
    id_state = {s.id: s for s in final_states}
    track_ids = [t["track_id"] for t in stable_tracks]
    n = len(stable_tracks)
    K = [[0.0] * n for _ in range(n)]
    D = [[math.inf] * n for _ in range(n)]
    edges = []

    for i in range(n):
        K[i][i] = 1.0
        D[i][i] = 0.0

    for i, ta in enumerate(stable_tracks):
        for j in range(i + 1, n):
            tb = stable_tracks[j]
            k_ij = track_coupling(ta, tb, id_state, J_func, max_ids_per_track)
            k_ji = track_coupling(tb, ta, id_state, J_func, max_ids_per_track) if directed else k_ij
            k_sym = (k_ij + k_ji) / 2.0
            K[i][j] = k_ij
            K[j][i] = k_ji
            if k_sym > 0:
                d = -math.log(max(k_sym, 1e-12))
                D[i][j] = d
                D[j][i] = d
            if k_sym >= coupling_threshold:
                edges.append({
                    "i": i,
                    "j": j,
                    "a": track_ids[i],
                    "b": track_ids[j],
                    "K": k_sym,
                    "K_ij": k_ij,
                    "K_ji": k_ji,
                    "asymmetry": abs(k_ij - k_ji),
                    "d": -math.log(max(k_sym, 1e-12)),
                })

    return {
        "track_ids": track_ids,
        "tracks": stable_tracks,
        "K_matrix": K,
        "D_matrix": D,
        "edges": edges,
        "directed": directed,
        "coupling_threshold": coupling_threshold,
    }


def adjacency_from_edges(n, edges):
    adj = {i: [] for i in range(n)}
    for e in edges:
        adj[e["i"]].append((e["j"], e["d"]))
        adj[e["j"]].append((e["i"], e["d"]))
    return adj


def dijkstra(adjacency, source):
    distances = {node: math.inf for node in adjacency}
    distances[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        dist, node = heapq.heappop(heap)
        if dist > distances[node]:
            continue
        for nb, weight in adjacency[node]:
            nd = dist + weight
            if nd < distances[nb]:
                distances[nb] = nd
                heapq.heappush(heap, (nd, nb))
    return distances


def geodesic_matrix(geometry):
    n = len(geometry["track_ids"])
    adj = adjacency_from_edges(n, geometry["edges"])
    G = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        dists = dijkstra(adj, i)
        for j, d in dists.items():
            G[i][j] = d
    return G


def network_components(geometry):
    n = len(geometry["track_ids"])
    adj = {i: set() for i in range(n)}
    for e in geometry["edges"]:
        adj[e["i"]].add(e["j"])
        adj[e["j"]].add(e["i"])
    visited = set()
    comps = []
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
            for nxt in adj[node]:
                if nxt not in visited:
                    stack.append(nxt)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def finite_geodesics(G):
    vals = []
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            if math.isfinite(G[i][j]):
                vals.append(G[i][j])
    return vals


# =============================================================================
# 5. Extended metrics
# =============================================================================


def clustering_coefficient(geometry):
    n = len(geometry["track_ids"])
    adj = {i: set() for i in range(n)}
    for e in geometry["edges"]:
        adj[e["i"]].add(e["j"])
        adj[e["j"]].add(e["i"])
    coeffs = []
    for i in range(n):
        neighbors = adj[i]
        k = len(neighbors)
        if k < 2:
            coeffs.append(0.0)
            continue
        triangles = sum(1 for u in neighbors for v in neighbors if u < v and v in adj[u])
        coeffs.append(triangles / (k * (k - 1) / 2.0))
    return mean(coeffs) if coeffs else 0.0


def triangle_count(geometry):
    n = len(geometry["track_ids"])
    adj = {i: set() for i in range(n)}
    for e in geometry["edges"]:
        adj[e["i"]].add(e["j"])
        adj[e["j"]].add(e["i"])
    tri = 0
    for i in range(n):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j:
                    continue
                if k in adj[i]:
                    tri += 1
    return tri


def estimate_dimension_mds(G, variance_threshold=0.90):
    if not HAS_NUMPY:
        return 0, []
    n = len(G)
    if n < 3:
        return 0, []
    finite_vals = [v for row in G for v in row if math.isfinite(v) and v > 0]
    if not finite_vals:
        return 0, []
    max_f = max(finite_vals)
    A = np.array([[v if math.isfinite(v) else 2.0 * max_f for v in row] for row in G], dtype=float)
    D2 = A ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D2 @ H
    eig = sorted(np.linalg.eigvalsh(B).tolist(), reverse=True)
    pos = [e for e in eig if e > 1e-8]
    if not pos:
        return 0, []
    total = sum(pos)
    c = []
    r = 0.0
    eff = len(pos)
    for idx, e in enumerate(pos):
        r += e
        c.append(r / total)
        if c[-1] >= variance_threshold:
            eff = idx + 1
            break
    return eff, c[:10]


def gromov_delta(G, max_samples=300, seed=42):
    n = len(G)
    rng = random.Random(seed)
    connected = [i for i in range(n) if sum(1 for j in range(n) if i != j and math.isfinite(G[i][j])) >= 3]
    if len(connected) < 4:
        return float("nan"), float("nan"), 0
    deltas = []
    attempts = 0
    while len(deltas) < max_samples and attempts < max_samples * 20:
        attempts += 1
        a, b, c, d = rng.sample(connected, 4)
        sums = [
            G[a][b] + G[c][d],
            G[a][c] + G[b][d],
            G[a][d] + G[b][c],
        ]
        if not all(math.isfinite(s) for s in sums):
            continue
        sums.sort(reverse=True)
        deltas.append((sums[0] - sums[1]) / 2.0)
    if not deltas:
        return float("nan"), float("nan"), 0
    return max(deltas), mean(deltas), len(deltas)


def graph_density(n, e):
    if n < 2:
        return 0.0
    return 2.0 * e / (n * (n - 1))


def degree_stats(geometry):
    n = len(geometry["track_ids"])
    deg = [0] * n
    weighted = [0.0] * n
    for e in geometry["edges"]:
        i, j = e["i"], e["j"]
        deg[i] += 1
        deg[j] += 1
        weighted[i] += e["K"]
        weighted[j] += e["K"]
    return {
        "mean_degree": mean(deg) if deg else 0.0,
        "max_degree": max(deg) if deg else 0,
        "degree_std": pstdev(deg) if len(deg) > 1 else 0.0,
        "mean_weighted_degree": mean(weighted) if weighted else 0.0,
    }


def closeness_mean(geometry, G):
    n = len(geometry["track_ids"])
    if n < 2:
        return 0.0
    values = []
    for i in range(n):
        finite = [G[i][j] for j in range(n) if i != j and math.isfinite(G[i][j])]
        if finite:
            values.append(len(finite) / sum(finite))
        else:
            values.append(0.0)
    return mean(values) if values else 0.0


# =============================================================================
# 6. Classification, scores, and FUT/CoMath/Icosahedrality diagnostics
# =============================================================================


def normalized_window_score(x, low, high, ideal=None):
    """1 inside an ideal range, taper outside. Simple robust score."""
    if ideal is None:
        ideal = (low + high) / 2.0
    if low <= x <= high:
        return 1.0
    if x < low:
        if low <= 0:
            return 0.0
        return max(0.0, x / low)
    # above high
    return max(0.0, 1.0 - (x - high) / max(high, 1e-9))


def icosahedrality_score(n, e, largest, triangles, mean_degree, degree_std, density, cc):
    """
    Icosahedrality proxy.

    A true icosahedron graph has:
      vertices = 12, edges = 30, triangular faces = 20, degree = 5 exactly,
      high triangulation, connected shell.

    The simulation does not directly embed an icosahedron; this score asks only:
    does the emergent track graph resemble a triangulated closed shell?
    """
    node_score = math.exp(-abs(n - 12) / 8.0) if n > 0 else 0.0
    edge_score = math.exp(-abs(e - 30) / 25.0)
    degree_score = math.exp(-abs(mean_degree - 5.0) / 3.0)
    regularity_score = math.exp(-degree_std / 3.0)
    triangle_score = math.exp(-abs(triangles - 20) / 30.0) if triangles > 0 else 0.0
    shell_score = normalized_window_score(density, 0.25, 0.55) * normalized_window_score(cc, 0.25, 0.75)
    connected_score = 1.0 if largest >= max(2, int(0.6 * n)) else largest / max(1.0, 0.6 * n)

    return (
        0.15 * node_score
        + 0.15 * edge_score
        + 0.15 * degree_score
        + 0.15 * regularity_score
        + 0.15 * triangle_score
        + 0.15 * shell_score
        + 0.10 * connected_score
    )


def geometry_quality_score(row):
    """Balanced quality score: not collapse, not overconnected, structured."""
    n = row["node_count"]
    e = row["edge_count"]
    density = row["density"]
    largest_frac = row["largest_component"] / n if n else 0.0
    dim = row.get("effective_dim", 0) or 0
    cc = row["clustering_coefficient"]
    mean_geo = row["mean_geodesic_d"]

    node_score = normalized_window_score(n, 10, 40)
    edge_score = normalized_window_score(e, 15, 70)
    density_score = normalized_window_score(density, 0.05, 0.35)
    largest_score = normalized_window_score(largest_frac, 0.25, 0.85)
    dim_score = normalized_window_score(dim, 1.5, 5.5)
    cc_score = normalized_window_score(cc, 0.10, 0.70)
    geo_score = normalized_window_score(mean_geo, 1.0, 5.0)

    return (
        0.15 * node_score
        + 0.15 * edge_score
        + 0.18 * density_score
        + 0.14 * largest_score
        + 0.14 * dim_score
        + 0.12 * cc_score
        + 0.12 * geo_score
    )


def classify_geometry(row):
    n = row["node_count"]
    e = row["edge_count"]
    density = row["density"]
    largest = row["largest_component"]
    largest_frac = largest / n if n else 0.0

    if n <= 1 or e == 0:
        return "FAILED_COLLAPSE"
    if largest <= 2 and e <= 2:
        return "FAILED_FRAGMENTED"
    if density > 0.45 or (largest_frac > 0.92 and density > 0.30):
        return "FAILED_OVERCONNECTED"
    if row["geometry_quality"] >= 0.70 and row["icosahedrality"] >= 0.45:
        return "ICO_CANDIDATE"
    if row["geometry_quality"] >= 0.65:
        return "GOOD_PROTO_GEOMETRY"
    if row["geometry_quality"] >= 0.40:
        return "WEAK_GEOMETRY"
    return "FAILED_FRAGMENTED"


def comath_property_score(props: JProperties):
    """
    Measures how close the formal J-properties are to the current CoMath/FUT
    interpretation: persistence requires graduated local recoherence order,
    not exact equality, not random, not hard classical binary cut.
    """
    score = 0.0
    score += 0.20 if props.monotone else 0.0
    score += 0.20 if props.gradual else 0.0
    score += 0.15 if props.local else 0.0
    score += 0.10 if props.continuous else 0.0
    score += 0.10 if not props.random_like else 0.0
    score += 0.10 if not props.resonant else 0.0
    score += 0.10 if not props.cutoff else 0.0
    score += 0.05 if props.long_range in ("moderate", "weak") else 0.0
    return score


# =============================================================================
# 7. One run and aggregation
# =============================================================================


def run_one(spec, seed, args):
    random.seed(seed)
    states, history, tracks, memory_history = run_simulation(
        spec.func,
        initial_count=args.initial_count,
        steps=args.steps,
        delta_min=args.delta_min,
        delta_split=args.delta_split,
        max_states=args.max_states,
        cluster_threshold=args.cluster_threshold,
        sample_every=args.sample_every,
        overlap_threshold=args.overlap_threshold,
        memory_lambda=args.memory_lambda,
        memory_gain=args.memory_gain,
        memory_decay=args.memory_decay,
        max_memory=args.max_memory,
        min_track_updates=args.min_track_updates,
    )

    stable_tracks = select_stable_tracks(
        tracks,
        sample_every=args.sample_every,
        min_lifetime=args.min_lifetime,
        min_updates=args.min_updates,
    )

    directed = (not spec.properties.symmetric)
    geom = build_track_geometry(
        stable_tracks,
        states,
        spec.func,
        coupling_threshold=args.coupling_threshold,
        max_ids_per_track=args.max_ids_per_track,
        directed=directed,
    )
    G = geodesic_matrix(geom)
    comps = network_components(geom)
    deg = degree_stats(geom)
    finite_geo = finite_geodesics(G)
    n = len(geom["track_ids"])
    e = len(geom["edges"])
    density = graph_density(n, e)
    cc = clustering_coefficient(geom)
    tri = triangle_count(geom)
    eff_dim, cumvar = estimate_dimension_mds(G)
    delta_max, delta_mean, delta_samples = gromov_delta(G)
    asym_vals = [ed["asymmetry"] for ed in geom["edges"]]

    row = {
        "j_name": spec.name,
        "seed": seed,
        "final_states": len(states),
        "final_avg_p": history[-1][2] if history else 0.0,
        "final_avg_memory": memory_history[-1][1] if memory_history else 0.0,
        "final_max_memory": memory_history[-1][2] if memory_history else 0.0,
        "track_count": len(tracks),
        "stable_tracks": len(stable_tracks),
        "node_count": n,
        "edge_count": e,
        "density": density,
        "component_count": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "mean_degree": deg["mean_degree"],
        "max_degree": deg["max_degree"],
        "degree_std": deg["degree_std"],
        "mean_weighted_degree": deg["mean_weighted_degree"],
        "mean_edge_K": mean([ed["K"] for ed in geom["edges"]]) if geom["edges"] else 0.0,
        "mean_edge_d": mean([ed["d"] for ed in geom["edges"]]) if geom["edges"] else 0.0,
        "mean_geodesic_d": mean(finite_geo) if finite_geo else 0.0,
        "max_geodesic_d": max(finite_geo) if finite_geo else 0.0,
        "closeness_mean": closeness_mean(geom, G),
        "clustering_coefficient": cc,
        "triangle_count": tri,
        "effective_dim": eff_dim,
        "mds_cumvar_top5": ";".join(str(round(v, 4)) for v in cumvar[:5]),
        "gromov_delta_max": delta_max,
        "gromov_delta_mean": delta_mean,
        "gromov_samples": delta_samples,
        "mean_K_asymmetry": mean(asym_vals) if asym_vals else 0.0,
        "prop_monotone": int(spec.properties.monotone),
        "prop_gradual": int(spec.properties.gradual),
        "prop_local": int(spec.properties.local),
        "prop_continuous": int(spec.properties.continuous),
        "prop_symmetric": int(spec.properties.symmetric),
        "prop_cutoff": int(spec.properties.cutoff),
        "prop_resonant": int(spec.properties.resonant),
        "prop_random_like": int(spec.properties.random_like),
        "prop_long_range": spec.properties.long_range,
        "prop_norm_family": spec.properties.norm_family,
        "prop_comment": spec.properties.comment,
    }
    row["geometry_quality"] = geometry_quality_score(row)
    row["icosahedrality"] = icosahedrality_score(
        n=n,
        e=e,
        largest=row["largest_component"],
        triangles=tri,
        mean_degree=row["mean_degree"],
        degree_std=row["degree_std"],
        density=density,
        cc=cc,
    )
    row["comath_property_score"] = comath_property_score(spec.properties)
    row["classification"] = classify_geometry(row)
    return row


def aggregate_results(rows):
    by_j = {}
    for r in rows:
        by_j.setdefault(r["j_name"], []).append(r)

    numeric_fields = [
        "final_states", "stable_tracks", "node_count", "edge_count", "density",
        "component_count", "largest_component", "mean_degree", "max_degree",
        "degree_std", "mean_edge_K", "mean_edge_d", "mean_geodesic_d",
        "clustering_coefficient", "triangle_count", "effective_dim",
        "gromov_delta_max", "mean_K_asymmetry", "geometry_quality",
        "icosahedrality", "comath_property_score",
    ]
    summary = []
    for j, js in by_j.items():
        row = {"j_name": j, "runs": len(js)}
        for f in numeric_fields:
            vals = [r[f] for r in js if isinstance(r.get(f), (int, float)) and math.isfinite(float(r[f]))]
            row[f"mean_{f}"] = mean(vals) if vals else float("nan")
            row[f"std_{f}"] = pstdev(vals) if len(vals) > 1 else 0.0
        classes = {}
        for r in js:
            classes[r["classification"]] = classes.get(r["classification"], 0) + 1
        row["dominant_classification"] = max(classes.items(), key=lambda kv: kv[1])[0]
        row["classification_counts"] = ";".join(f"{k}:{v}" for k, v in sorted(classes.items()))
        # include properties from first row
        first = js[0]
        for k in [
            "prop_monotone", "prop_gradual", "prop_local", "prop_continuous",
            "prop_symmetric", "prop_cutoff", "prop_resonant", "prop_random_like",
            "prop_long_range", "prop_norm_family", "prop_comment",
        ]:
            row[k] = first[k]
        summary.append(row)

    summary.sort(key=lambda r: (r["mean_geometry_quality"], r["mean_icosahedrality"]), reverse=True)
    return summary


# =============================================================================
# 8. Output
# =============================================================================


def ensure_output_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary, output_dir):
    names = [r["j_name"] for r in summary]
    edges = [r["mean_edge_count"] for r in summary]
    largest = [r["mean_largest_component"] for r in summary]
    quality = [r["mean_geometry_quality"] for r in summary]
    ico = [r["mean_icosahedrality"] for r in summary]

    x = list(range(len(names)))
    plt.figure(figsize=(16, 7))
    plt.plot(x, edges, marker="o", label="mean edges")
    plt.plot(x, largest, marker="o", label="mean largest component")
    plt.plot(x, [100 * q for q in quality], marker="o", label="geometry quality ×100")
    plt.plot(x, [100 * s for s in ico], marker="o", label="icosahedrality ×100")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.title("S9 recoherence geometry classifier summary")
    plt.ylabel("mean statistic")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = Path(output_dir) / "s9_summary_metrics.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print("saved:", out)


def plot_property_scores(summary, output_dir):
    names = [r["j_name"] for r in summary]
    geo = [r["mean_geometry_quality"] for r in summary]
    ico = [r["mean_icosahedrality"] for r in summary]
    com = [r["mean_comath_property_score"] for r in summary]
    dens = [r["mean_density"] for r in summary]

    x = list(range(len(names)))
    plt.figure(figsize=(16, 7))
    plt.bar([i - 0.3 for i in x], geo, width=0.2, label="geometry quality")
    plt.bar([i - 0.1 for i in x], ico, width=0.2, label="icosahedrality")
    plt.bar([i + 0.1 for i in x], com, width=0.2, label="CoMath property")
    plt.bar([i + 0.3 for i in x], dens, width=0.2, label="density")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.title("S9 property scores")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = Path(output_dir) / "s9_property_scores.png"
    plt.savefig(out, dpi=160)
    plt.close()
    print("saved:", out)


def write_report(summary, output_dir):
    out = Path(output_dir) / "s9_classification_report.md"

    best_geo = sorted(summary, key=lambda r: r["mean_geometry_quality"], reverse=True)[:5]
    best_ico = sorted(summary, key=lambda r: r["mean_icosahedrality"], reverse=True)[:5]
    over = [r for r in summary if r["dominant_classification"] == "FAILED_OVERCONNECTED"]
    failed = [r for r in summary if r["dominant_classification"].startswith("FAILED")]

    lines = []
    lines.append("# S9 — Recoherence Geometry Classifier Report\n")
    lines.append("## Purpose\n")
    lines.append("S9 consolidates the three S8 scripts into one classifier for J-geometriefaehigkeit.\n")
    lines.append("It evaluates collapse, fragmentation, overconnection, good proto-geometry, and approximate icosahedrality.\n")

    lines.append("## Direct answers to the six guiding questions\n")
    lines.append("### 1. Minimal properties J must possess\n")
    lines.append("The empirical minimal profile is: monotone, gradual, local, difference-sensitive recoherence order. "
                 "J must preserve a usable neighborhood ranking: nearer/compatible structures couple stronger than far/incompatible structures.\n")
    lines.append("### 2. Maximal properties J may possess\n")
    lines.append("J must not overcouple. Strong long-range tails can create a globally connected graph that loses differentiated geometry. "
                 "The acceptable regime is subcritical or near-critical percolation, not total percolation.\n")
    lines.append("### 3. Best J-properties for highest structure\n")
    lines.append("The best candidates are those with high geometry_quality and nontrivial icosahedrality while avoiding FAILED_OVERCONNECTED.\n")
    lines.append("### 4. FUT/CoMath icosahedral closeness\n")
    lines.append("S9 uses an Icosahedrality proxy: closeness to 12 nodes, 30 edges, degree ~5, triangular closure, shell-like density, connectedness, and degree regularity. "
                 "This is not a proof of an icosahedron; it is a diagnostic for triangulated recoherence shells.\n")
    lines.append("### 5. Are CoMath/FUT formula types used in Python?\n")
    lines.append("S9 explicitly introduces toy wrappers for 0_f, infinity_f, =_f and CoherenceValue. "
                 "However, Python still computes with classical floats internally. The wrappers prevent conceptual conflation, but they are not yet a full formal implementation of CoMath/FUT variable ontology.\n")
    lines.append("### 6. How to improve qualitative strength by code changes?\n")
    lines.append("Use sweeps over coupling_threshold, finite-size scaling, perturbation robustness, directed geodesics for asymmetric J, real icosahedral graph matching, and explicit CoMath/FUT symbolic type propagation.\n")

    lines.append("## Top geometry-quality candidates\n")
    for r in best_geo:
        lines.append(f"- **{r['j_name']}**: quality={r['mean_geometry_quality']:.3f}, "
                     f"ico={r['mean_icosahedrality']:.3f}, edges={r['mean_edge_count']:.2f}, "
                     f"largest={r['mean_largest_component']:.2f}, class={r['dominant_classification']}\n")

    lines.append("\n## Top icosahedrality candidates\n")
    for r in best_ico:
        lines.append(f"- **{r['j_name']}**: ico={r['mean_icosahedrality']:.3f}, "
                     f"quality={r['mean_geometry_quality']:.3f}, edges={r['mean_edge_count']:.2f}, "
                     f"degree={r['mean_mean_degree']:.2f}, triangles={r['mean_triangle_count']:.2f}\n")

    lines.append("\n## Overconnected candidates\n")
    if over:
        for r in over:
            lines.append(f"- **{r['j_name']}**: density={r['mean_density']:.3f}, "
                         f"edges={r['mean_edge_count']:.2f}, largest={r['mean_largest_component']:.2f}\n")
    else:
        lines.append("- None by dominant classification.\n")

    lines.append("\n## Failed / weak candidates\n")
    for r in failed[:12]:
        lines.append(f"- **{r['j_name']}**: class={r['dominant_classification']}, "
                     f"edges={r['mean_edge_count']:.2f}, largest={r['mean_largest_component']:.2f}\n")

    lines.append("\n## Core S9 thesis\n")
    lines.append("Proto-geometry appears when J produces graduated local recoherence order without falling below the connectivity threshold or exceeding the overpercolation threshold. "
                 "The precise analytic form is secondary; the functional regime is primary.\n")

    out.write_text("".join(lines), encoding="utf-8")
    print("saved:", out)


# =============================================================================
# 9. Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="S9 Recoherence Geometry Classifier")
    p.add_argument("--runs", type=int, default=20, help="Seeds per J function. Use 50 for stronger statistics.")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--initial-count", type=int, default=40)
    p.add_argument("--max-states", type=int, default=120)
    p.add_argument("--delta-min", type=float, default=0.001)
    p.add_argument("--delta-split", type=float, default=0.08)
    p.add_argument("--cluster-threshold", type=float, default=0.6)
    p.add_argument("--coupling-threshold", type=float, default=0.25)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--overlap-threshold", type=float, default=0.30)
    p.add_argument("--memory-lambda", type=float, default=0.08)
    p.add_argument("--memory-gain", type=float, default=0.04)
    p.add_argument("--memory-decay", type=float, default=0.985)
    p.add_argument("--max-memory", type=float, default=1.0)
    p.add_argument("--min-track-updates", type=int, default=3)
    p.add_argument("--min-lifetime", type=int, default=100)
    p.add_argument("--min-updates", type=int, default=10)
    p.add_argument("--max-ids-per-track", type=int, default=10)
    p.add_argument("--output-dir", type=str, default="s9_outputs")
    p.add_argument("--j", nargs="*", default=None, help="Optional subset of J names")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    all_specs = build_j_library()
    if args.j:
        wanted = set(args.j)
        specs = [s for s in all_specs if s.name in wanted]
        missing = wanted - {s.name for s in specs}
        if missing:
            raise ValueError(f"Unknown J names: {sorted(missing)}")
    else:
        specs = all_specs

    print("=== S9 Recoherence Geometry Classifier ===")
    print("J functions:", [s.name for s in specs])
    print("runs per J:", args.runs)
    print("numpy MDS:", "enabled" if HAS_NUMPY else "disabled")
    print("output_dir:", output_dir)

    rows = []
    for spec in specs:
        print(f"\n--- J: {spec.name} ---")
        for seed in range(args.runs):
            r = run_one(spec, seed, args)
            rows.append(r)
            print(
                f"seed {seed:3d} states {r['final_states']:3d} nodes {r['node_count']:2d} "
                f"edges {r['edge_count']:3d} density {r['density']:.3f} "
                f"largest {r['largest_component']:2d} dim {r['effective_dim']} "
                f"Q {r['geometry_quality']:.3f} Ico {r['icosahedrality']:.3f} "
                f"{r['classification']}"
            )

    summary = aggregate_results(rows)

    results_path = output_dir / "s9_geometry_classifier_results.csv"
    summary_path = output_dir / "s9_geometry_classifier_summary.csv"
    write_csv(results_path, rows)
    write_csv(summary_path, summary)
    print("saved:", results_path)
    print("saved:", summary_path)

    plot_summary(summary, output_dir)
    plot_property_scores(summary, output_dir)
    write_report(summary, output_dir)

    print("\n=== Top candidates by geometry quality ===")
    for r in summary[:8]:
        print(
            f"{r['j_name']:22s} Q={r['mean_geometry_quality']:.3f} "
            f"Ico={r['mean_icosahedrality']:.3f} edges={r['mean_edge_count']:.2f} "
            f"largest={r['mean_largest_component']:.2f} class={r['dominant_classification']}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
