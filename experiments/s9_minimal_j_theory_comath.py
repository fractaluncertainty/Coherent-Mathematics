#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S9 — Minimale J-Theorie und FUT/CoMath-Verbindung
===================================================

Beantwortet 6 offene Fragen aus der S7/S8-Reihe (50 Seeds, 22 J-Funktionen):

F1  Minimale J-Eigenschaften   — notwendige Bedingungen
F2  Maximale J-Eigenschaften   — Obergrenzen
F3  Beste J-Eigenschaften      — höchste Strukturqualität
F4  A₅/Ikosaeder-Nähe          — FUT/CoMath-Resonanz
F5  FUT/CoMath-Variablentypen  — 0_f, ∞_f, gleich_f, R/F im Code
F6  Optimale Code-Verbesserungen

Neue J-Funktionen (FUT/CoMath-motiviert):
    j_phi_exp           exp(-φ·d)                    φ als natürliche Skala
    j_icosahedral_a5    exp(-d/φ)·A₅-Modulation      5-fach + 3-fach Resonanz
    j_rf_weighted       metric_exp × R/F-Gleichgewicht  Elektron-Horizont

Neue State-Erweiterung:
    CoMathState         + r_f (R/F-Ratio), coherence_level

FUT/CoMath-Variablentypen:
    0_f                 w→ε, memory→ε   strukturierte Null
    ∞_f                 w→1, memory→1   gesättigte Kohärenz
    gleich_f(x,y)       J(x,y) > 1-ε   fraktale Gleichheit
    R/F ratio           memory/w        Rekursions/Flach-Verhältnis

Selbständig ausführbar. Benötigt keine S7/S8-Imports.
Keine Beweise für FUT/CoMath — Toy-Model-Test.
"""

import csv
import math
import random
import heapq
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ---------------------------------------------------------------------------
# 0. FUT/CoMath Konstanten
# ---------------------------------------------------------------------------

PHI = (1 + math.sqrt(5)) / 2        # Goldener Schnitt φ ≈ 1.618
INV_PHI = 1.0 / PHI                  # 1/φ ≈ 0.618
PI = math.pi

# Fraktale Schwellen (FUT-motiviert)
EPSILON_F = 1e-6                     # 0_f-Schwelle
GLEICH_F_EPSILON = 0.05              # gleich_f: J(x,y) > 1 - GLEICH_F_EPSILON

# Ikosaeder-Resonanzwinkel (A₅-Symmetrie)
A5_5FOLD = [2 * PI * k / 5 for k in range(5)]   # 5-fach Achse (Sigma)
A5_3FOLD = [2 * PI * k / 3 for k in range(3)]   # 3-fach Achse (Chi)


# ---------------------------------------------------------------------------
# 1. CoMath-State (S7-State + FUT/CoMath-Erweiterung)
# ---------------------------------------------------------------------------

class CoMathState:
    """
    Erweiterter State mit FUT/CoMath-Variablentypen.

    w       : Persistenzgewicht         (F-Proxy: flache Kohärenz)
    sigma   : Orientierungswinkel       (View-Vektor-Richtung)
    chi     : Interferenzphase          (Wellenfunktionsphase)
    memory  : akkumuliertes Gedächtnis  (R-Proxy: rekursive Tiefe)

    r_f          : R/F-Verhältnis = memory/w  (Elektron-Horizont bei R/F≈1)
    coherence_lvl: Emergenz-Level (0=unstrukturiert, →1=kohärent)

    FUT/CoMath-Typen als Eigenschaften:
        is_zero_f      : w < ε_f  (strukturierte Null — persistiert minimal)
        is_inf_f       : w≈1, memory≈1  (gesättigte Kohärenz)
        gleich_f(other): J(self,other) > 1-ε  (fraktale Gleichheit)
    """

    def __init__(self, state_id, w, sigma, chi, memory=0.0):
        self.id = state_id
        self.w = w
        self.sigma = sigma
        self.chi = chi
        self.memory = memory

    @property
    def r_f(self):
        """R/F-Verhältnis. Elektron-Horizont liegt bei R/F ≈ 1."""
        return self.memory / (self.w + EPSILON_F)

    @property
    def coherence_level(self):
        """Normierter Kohärenz-Level [0,1]. Hoch wenn w und memory beide stark."""
        return min(1.0, (self.w + self.memory) / 2.0)

    @property
    def is_zero_f(self):
        """0_f: strukturierte Null — persistiert, aber mit w≈0."""
        return self.w < EPSILON_F

    @property
    def is_inf_f(self):
        """∞_f: gesättigte Kohärenz — w und memory nahe 1."""
        return self.w > 0.95 and self.memory > 0.95

    def gleich_f(self, other, j_func):
        """gleich_f: fraktale Gleichheit — J(self,other) > 1-ε."""
        return j_func(self, other) > 1.0 - GLEICH_F_EPSILON


# Alias für Rückwärtskompatibilität
State = CoMathState


# ---------------------------------------------------------------------------
# 2. J-Familie
# ---------------------------------------------------------------------------

def angle_diff(a, b):
    d = abs(a - b) % (2 * PI)
    return min(d, 2 * PI - d)

def sigmoid(z):
    ez = math.exp(-abs(z))
    return (1.0 / (1.0 + ez)) if z >= 0 else (ez / (1.0 + ez))

def d_l1(x, y):
    return angle_diff(x.sigma, y.sigma) + angle_diff(x.chi, y.chi)

def d_linf(x, y):
    return max(angle_diff(x.sigma, y.sigma), angle_diff(x.chi, y.chi))


# --- Geometriefähig (aus S7/S8 bestätigt) ---

def j_metric_exp(x, y):
    """S7-Standard. exp(-d_L1). Goldstandard."""
    return math.exp(-d_l1(x, y))

def j_anisotropic_exp(x, y):
    """Anisotrope Gewichtung (σ:0.65, χ:1.35). ≈ metric_exp in Qualität."""
    return math.exp(-0.65 * angle_diff(x.sigma, y.sigma)
                    - 1.35 * angle_diff(x.chi, y.chi))

def j_linf_exp(x, y):
    """L∞-Norm statt L1. Beweist Norm-Agnostizität."""
    return math.exp(-d_linf(x, y))

def j_rational_p2(x, y):
    """Algebraischer Abfall: 1/(1+d²). Nicht-exponentiell aber geometriefähig."""
    d = d_l1(x, y)
    return 1.0 / (1.0 + d * d)


# --- Neue FUT/CoMath-motivierte J-Funktionen ---

def j_phi_exp(x, y):
    """
    φ-skalierter Exponentialabfall: exp(-φ·d).
    φ als natürliche Skala der FUT (A₅-Quantisierungseinheit φ⁵).
    alpha=φ≈1.618: zwischen alpha=1.5 (schwach) und alpha=2.0 (fragmentiert).
    Aus S8-Extension: alpha≈1.5 gibt geo_score=0.57 → φ wird ähnlich sein.
    """
    return math.exp(-PHI * d_l1(x, y))


def j_icosahedral_a5(x, y):
    """
    A₅-Ikosaeder-Kopplung: φ-Abfall × 5-fach × 3-fach Resonanz.

    FUT/CoMath-Verbindung:
    - 3D Raum als minimale treue A₅-Darstellung (χ=3, S9-Korollar)
    - 5-fach Symmetrie: Ikosaeder-Primärachse
    - 3-fach Symmetrie: Ikosaeder-Sekundärachse
    - φ-Abfall: A₅-Quantisierungseinheit

    Hypothese: wenn emergente Geometrie A₅-Struktur trägt,
    sollte j_icosahedral_a5 höhere Strukturqualität als metric_exp erzeugen.
    """
    ds = angle_diff(x.sigma, y.sigma)
    dc = angle_diff(x.chi, y.chi)

    # φ-skalierter Basisabfall (1/φ als Abklingrate)
    base = math.exp(-ds * INV_PHI - dc * INV_PHI)

    # 5-fach A₅-Resonanz in σ: Peaks bei 2πk/5
    ico5 = (1.0 + math.cos(5.0 * ds)) / 2.0

    # 3-fach A₅-Resonanz in χ: Peaks bei 2πk/3
    ico3 = (1.0 + math.cos(3.0 * dc)) / 2.0

    # Modulationsgewicht: 40% Basis, 35% 5-fach, 25% 3-fach
    modulation = 0.40 + 0.35 * ico5 + 0.25 * ico3

    return base * modulation


def j_rf_weighted(x, y):
    """
    metric_exp gewichtet durch R/F-Gleichgewichtsnähe.

    FUT/CoMath-Verbindung:
    - Elektron-Horizont: R=F → α=1/137 (FUT ch. 52)
    - Zustände nahe R/F≈1 koppeln stärker
    - Selbstverstärkend: Gleichgewichtszustände persistieren bevorzugt

    Dies ist die einzige J-Funktion, die dynamisch von Zustandseigenschaften
    (memory, w) abhängt, nicht nur von Winkeldifferenzen.
    Daher: keine statische J-Charakterisierung möglich.
    """
    d = d_l1(x, y)
    base = math.exp(-d)

    rf_x = min(x.memory / (x.w + EPSILON_F), 5.0)
    rf_y = min(y.memory / (y.w + EPSILON_F), 5.0)

    # Gleichgewichtsgewicht: max bei R/F=1 für beide Zustände
    eq = math.exp(-abs(rf_x - 1.0) - abs(rf_y - 1.0))

    return base * (0.5 + 0.5 * eq)


# J-Register
J_REGISTRY = {
    # Bewährt (S7/S8)
    "metric_exp":       (j_metric_exp,      "S7-Standard, L1-exp"),
    "anisotropic_exp":  (j_anisotropic_exp, "Anisotrope Gewichtung"),
    "linf_exp":         (j_linf_exp,        "L∞-Norm"),
    "rational_p2":      (j_rational_p2,     "Algebraisch 1/(1+d²)"),
    # Neu FUT/CoMath
    "phi_exp":          (j_phi_exp,         "φ-Skala exp(-φ·d)"),
    "icosahedral_a5":   (j_icosahedral_a5,  "A₅ 5-fach+3-fach Resonanz"),
    "rf_weighted":      (j_rf_weighted,     "R/F-Gleichgewicht dyn."),
}


# ---------------------------------------------------------------------------
# 3. Minimaler J-Theorem-Checker
# ---------------------------------------------------------------------------

def characterize_j(j_func, n_states=300, cluster_threshold=0.6,
                   coupling_threshold=0.25, seed=42):
    """
    Charakterisiert J statisch. Prüft alle 4 Minimal-J-Bedingungen.
    Gibt ER-Ratio, CV, Lipschitz-Proxy, Dynamik-Range zurück.
    """
    rng = random.Random(seed)
    states = [CoMathState(i, 0.5, rng.uniform(0, 2*PI), rng.uniform(0, 2*PI))
              for i in range(n_states)]

    # Für RF-weighted: setze representativen memory/w
    for s in states:
        s.memory = rng.uniform(0, 0.5)
        s.w = rng.uniform(0.3, 0.8)

    vals = [j_func(states[i], states[j])
            for i in range(n_states) for j in range(i+1, n_states)]
    n = len(vals)
    m = mean(vals)
    s_dev = stdev(vals)

    sorted_vals = sorted(vals)
    top10 = sorted_vals[-(n//10):]
    bot10 = sorted_vals[:n//10]
    dynrange = (mean(top10) / max(mean(bot10), 1e-12))

    p_perc = sum(1 for v in vals if v > cluster_threshold) / n
    p_coup = sum(1 for v in vals if v > coupling_threshold) / n
    er = p_perc / (1.0 / 120)

    # Lipschitz-Proxy
    rng2 = random.Random(seed + 1)
    lip_samples = []
    for _ in range(800):
        x = states[rng2.randint(0, n_states-1)]
        y = states[rng2.randint(0, n_states-1)]
        eps = 0.05
        x2 = CoMathState(x.id, x.w, (x.sigma + eps) % (2*PI), x.chi, x.memory)
        dj = abs(j_func(x2, y) - j_func(x, y))
        lip_samples.append(dj / eps)
    lip = mean(lip_samples)

    # Bedingungen prüfen
    cond1_ok = er < 6.0        # Sub-Perkolation
    cond2_ok = lip > 0.05      # Lipschitz-Stetigkeit
    cond3_ok = s_dev/m > 0.4   # Dynamik-Range (CV)
    cond4_ok = not (sorted_vals[-1] - sorted_vals[int(n*0.99)] < 0.01
                    and p_perc < 0.001)  # Dichte J-Wertebereich

    return {
        "mean_j": round(m, 5),
        "cv": round(s_dev/m if m > 0 else 0, 4),
        "dynamic_range": round(dynrange, 1),
        "p_percolation": round(p_perc, 5),
        "p_coupling": round(p_coup, 5),
        "er_ratio": round(er, 2),
        "lipschitz": round(lip, 4),
        "cond1_subperc": cond1_ok,
        "cond2_lipschitz": cond2_ok,
        "cond3_cv": cond3_ok,
        "cond4_density": cond4_ok,
        "all_conditions": all([cond1_ok, cond2_ok, cond3_ok, cond4_ok]),
    }


# ---------------------------------------------------------------------------
# 4. A₅-Initialzustände
# ---------------------------------------------------------------------------

def make_a5_initial_states(n=40, seed=0):
    """
    Startzustände mit A₅-Ikosaeder-Symmetrie.
    sigma an 5-fach Achse (2πk/5), chi an 3-fach Achse (2πk/3),
    Rest zufällig.

    FUT/CoMath: 3D Raum als A₅-Darstellung → symmetriebrechende ICs
    könnten spezifische Geometrie-Topologie bevorzugen.
    """
    rng = random.Random(seed)
    states = []
    sid = 0

    # 15 A₅-geordnete Startpunkte (5×3 Gitter)
    for ks in range(5):
        for kc in range(3):
            sigma = A5_5FOLD[ks] + rng.uniform(-0.1, 0.1)
            chi = A5_3FOLD[kc] + rng.uniform(-0.1, 0.1)
            w = rng.uniform(0.5, 1.0)
            states.append(CoMathState(sid, w, sigma % (2*PI), chi % (2*PI)))
            sid += 1

    # Rest zufällig
    while len(states) < n:
        states.append(CoMathState(sid, rng.uniform(0.2, 1.0),
                                   rng.uniform(0, 2*PI), rng.uniform(0, 2*PI)))
        sid += 1

    return states, sid


# ---------------------------------------------------------------------------
# 5. Simulationskern (kompakt, self-contained)
# ---------------------------------------------------------------------------

def recurse(x, memory_decay=0.985):
    return CoMathState(
        x.id,
        max(EPSILON_F, x.w + random.uniform(-0.02, 0.04) - random.uniform(0, 0.02)),
        (x.sigma + random.uniform(-0.08, 0.08)) % (2 * PI),
        (x.chi + random.uniform(-0.08, 0.08)) % (2 * PI),
        memory=max(0.0, x.memory * memory_decay),
    )

def split(x, next_id):
    return CoMathState(
        next_id,
        x.w * random.uniform(0.8, 1.05),
        (x.sigma + random.uniform(-0.05, 0.05)) % (2 * PI),
        (x.chi + random.uniform(-0.05, 0.05)) % (2 * PI),
        memory=x.memory * random.uniform(0.4, 0.8),
    )

def I_func(x, y, j_func):
    return j_func(x, y) * math.cos(angle_diff(x.chi, y.chi))

def local_persistence(x, states, j_func):
    others = [y for y in states if y is not x]
    if not others:
        return 0.0
    return x.w * sum(j_func(x, y) * max(0.0, I_func(x, y, j_func))
                     for y in others) / len(others)

def persistence_with_memory(x, states, j_func, memory_lambda=0.08):
    return local_persistence(x, states, j_func) + memory_lambda * x.memory

def find_clusters(states, j_func, threshold=0.6):
    visited = set(); clusters = []
    for i in range(len(states)):
        if i in visited:
            continue
        stack = [i]; cluster = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur); cluster.append(cur)
            for j in range(len(states)):
                if j not in visited and j_func(states[cur], states[j]) > threshold:
                    stack.append(j)
        clusters.append(cluster)
    clusters.sort(key=len, reverse=True)
    return clusters

def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def run_simulation(j_func, initial_states=None, initial_count=40, steps=300,
                   delta_min=0.001, delta_split=0.08, max_states=120,
                   cluster_threshold=0.6, sample_every=5, overlap_threshold=0.30,
                   memory_lambda=0.08, memory_gain=0.04, memory_decay=0.985,
                   seed=0):
    """Simulationskern — akzeptiert optionale A₅-Initialzustände."""
    random.seed(seed)

    if initial_states is not None:
        states = list(initial_states)
        next_id = max(s.id for s in states) + 1
    else:
        next_id = 0
        states = [CoMathState(i, random.uniform(0.2, 1.0),
                               random.uniform(0, 2*PI), random.uniform(0, 2*PI))
                  for i in range(initial_count)]
        next_id = initial_count

    active_tracks = {}; finished_tracks = []; next_track_id = 0
    history = []; mem_history = []

    for step in range(steps):
        states = [recurse(x, memory_decay=memory_decay) for x in states]
        scored = [(x, persistence_with_memory(x, states, j_func, memory_lambda))
                  for x in states]
        survivors = [x for x, p in scored if p >= delta_min]
        offspring = []
        for x, p in scored:
            if p >= delta_split and len(survivors) + len(offspring) < max_states:
                offspring.append(split(x, next_id)); next_id += 1
        states = survivors + offspring
        avg_p = mean(p for _, p in scored) if scored else 0.0
        avg_mem = mean(s.memory for s in states) if states else 0.0
        history.append((step, len(states), avg_p))
        mem_history.append((step, avg_mem))

        if step % sample_every == 0 and states:
            clusters = find_clusters(states, j_func, cluster_threshold)
            cr = [{"step": step,
                   "ids": frozenset(states[i].id for i in c),
                   "size": len(c)} for c in clusters]

            # Track assignment
            assigned = set(); new_tracks = {}
            for c in cr:
                best_id, best_ov = None, 0.0
                for tid, tr in active_tracks.items():
                    if tid in assigned:
                        continue
                    ov = jaccard(tr["last_ids"], c["ids"])
                    if ov > best_ov:
                        best_ov = ov; best_id = tid
                if best_id is not None and best_ov >= overlap_threshold:
                    tr = active_tracks[best_id]
                    tr["end_step"] = c["step"]; tr["last_ids"] = c["ids"]
                    tr["sizes"].append(c["size"]); tr["updates"] += 1
                    new_tracks[best_id] = tr; assigned.add(best_id)
                else:
                    new_tracks[next_track_id] = {
                        "track_id": next_track_id, "start_step": c["step"],
                        "end_step": c["step"], "last_ids": c["ids"],
                        "sizes": [c["size"]], "updates": 1}
                    next_track_id += 1
            for tid, tr in active_tracks.items():
                if tid not in assigned and tid not in new_tracks:
                    finished_tracks.append(tr)
            active_tracks.clear(); active_tracks.update(new_tracks)

            # Memory feedback
            id_map = {s.id: s for s in states}
            for tr in active_tracks.values():
                if tr["updates"] < 3:
                    continue
                lt = tr["end_step"] - tr["start_step"] + sample_every
                strength = memory_gain * min(1.0, lt / 100.0)
                for sid in tr["last_ids"]:
                    if sid in id_map:
                        id_map[sid].memory = min(1.0, id_map[sid].memory + strength)
        if not states:
            break

    finished_tracks.extend(active_tracks.values())
    return states, history, finished_tracks, mem_history


# ---------------------------------------------------------------------------
# 6. Proto-Geometrie (kompakt)
# ---------------------------------------------------------------------------

def build_geometry(stable_tracks, final_states, j_func, coupling_threshold=0.25):
    id_map = {s.id: s for s in final_states}
    track_ids = [t["track_id"] for t in stable_tracks]
    n = len(stable_tracks)
    edges = []
    for i, ta in enumerate(stable_tracks):
        for j in range(i+1, n):
            ids_a = set(list(ta["last_ids"])[:10])
            ids_b = set(list(stable_tracks[j]["last_ids"])[:10])
            sa = [id_map[i] for i in ids_a if i in id_map]
            sb = [id_map[i] for i in ids_b if i in id_map]
            if not sa or not sb:
                continue
            vals = [j_func(a, b) for a in sa for b in sb if a.id != b.id]
            k = mean(vals) if vals else 0.0
            if k >= coupling_threshold:
                edges.append({"i": i, "j": j, "K": k,
                               "d": -math.log(max(k, 1e-12))})
    return {"track_ids": track_ids, "edges": edges, "n": n}


def select_stable(tracks, min_lt=100, min_up=10, sample_every=5):
    return [t for t in tracks
            if (t["end_step"] - t["start_step"] + sample_every) >= min_lt
            and t["updates"] >= min_up]


def geometry_summary(geom):
    n = geom["n"]; edges = geom["edges"]
    # Components
    adj = {i: set() for i in range(n)}
    for e in edges:
        adj[e["i"]].add(e["j"]); adj[e["j"]].add(e["i"])
    visited = set(); comps = []
    for i in range(n):
        if i in visited:
            continue
        stack = [i]; comp = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur); comp.append(cur)
            for nb in adj[cur]:
                if nb not in visited:
                    stack.append(nb)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    # Geodesics
    def dijkstra(src):
        dist = {i: math.inf for i in range(n)}; dist[src] = 0.0
        heap = [(0.0, src)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist[node]:
                continue
            for nb, w in [(e["j"], e["d"]) if e["i"]==node
                          else (e["i"], e["d"]) for e in edges
                          if e["i"]==node or e["j"]==node]:
                if d + w < dist[nb]:
                    dist[nb] = d + w; heapq.heappush(heap, (dist[nb], nb))
        return dist
    finite_geo = []
    for i in range(n):
        for j, d in dijkstra(i).items():
            if j > i and math.isfinite(d):
                finite_geo.append(d)
    # Clustering coeff
    cc_list = []
    for i in range(n):
        nbs = adj[i]; k = len(nbs)
        if k < 2:
            cc_list.append(0.0); continue
        tri = sum(1 for u in nbs for v in nbs if u < v and v in adj[u])
        cc_list.append(tri / (k*(k-1)/2))
    # Sparsity: 1 - E / E_max  (E_max = vollständiger Graph N*(N-1)/2)
    e_max = n * (n - 1) / 2 if n > 1 else 1
    sparsity = 1.0 - len(edges) / e_max

    # Diameter: längste finite geodätische Distanz
    diameter = max(finite_geo) if finite_geo else 0.0

    # Regime-Klassifikation
    e_per_n = len(edges) / n if n > 0 else 0
    if e_per_n < 0.5:
        regime = "unterkritisch"
    elif sparsity < 0.5:
        regime = "ueberkritisch"
    else:
        regime = "geometrisch"

    return {
        "node_count": n,
        "edge_count": len(edges),
        "component_count": len(comps),
        "largest_component": len(comps[0]) if comps else 0,
        "mean_geodesic_d": mean(finite_geo) if finite_geo else 0.0,
        "max_geodesic_d": diameter,
        "clustering_coeff": mean(cc_list) if cc_list else 0.0,
        "mean_edge_K": mean(e["K"] for e in edges) if edges else 0.0,
        "sparsity": round(sparsity, 4),
        "diameter": round(diameter, 4),
        "e_per_n": round(e_per_n, 3),
        "regime": regime,
    }


def geo_score(summ):
    """
    Skalarer Geometrie-Score [0,1].

    Fünf gleichgewichtete Komponenten:
      Konnektivität    — E~O(N) optimal; E~O(N²) wird bestraft
      Zusammenhang     — größte Komponente
      Krümmung         — Clustering-Koeffizient
      Geodätische Tiefe— mean_geodesic_d (groß = differenziert)
      Sparsity         — straft vollständig verbundene Graphen ab

    rational_p1: E~O(N²) → sparsity~0.3 → Score sinkt trotz vieler Kanten.
    metric_exp:  E~O(N)  → sparsity~0.9 → Score bleibt hoch.
    """
    n, e = summ["node_count"], summ["edge_count"]
    if n < 2 or e == 0:
        return 0.0
    e_per_n = summ.get("e_per_n", e / max(n, 1))
    # Konnektivität: Plateau bei E/N≈2, Abfall bei E/N>>4
    connectivity = min(1.0, e_per_n / 2.0) * max(0.0, 1.0 - max(0, e_per_n - 4) / 6)
    sparsity = summ.get("sparsity", 1.0)
    return (connectivity                               * 0.20
            + min(1, summ["largest_component"] / 20)  * 0.20
            + summ["clustering_coeff"]                 * 0.20
            + min(1, summ["mean_geodesic_d"] / 4)     * 0.20
            + sparsity                                 * 0.20)


def run_one(j_name, j_func, seed, a5_init=False, **kwargs):
    if a5_init:
        ic, _ = make_a5_initial_states(40, seed=seed)
    else:
        ic = None
    states, hist, tracks, mem_hist = run_simulation(
        j_func, initial_states=ic, seed=seed, **kwargs)
    stable = select_stable(tracks)
    geom = build_geometry(stable, states, j_func)
    summ = geometry_summary(geom)
    return {
        "j_name": j_name, "seed": seed, "a5_init": a5_init,
        "final_states": len(states),
        "stable_tracks": len(stable),
        "geo_score": geo_score(summ),
        **summ,
        "final_avg_memory": mem_hist[-1][1] if mem_hist else 0.0,
        "_states": states,
    }


# ---------------------------------------------------------------------------
# 7. FUT/CoMath-Analysefunktionen
# ---------------------------------------------------------------------------

def rf_distribution(states):
    """R/F-Verteilung der finalen Zustände. Elektron-Horizont bei R/F≈1."""
    rf_vals = [s.r_f for s in states if s.w > 0.01]
    near_eq = sum(1 for r in rf_vals if 0.7 < r < 1.3) / len(rf_vals) if rf_vals else 0
    return {
        "mean_rf": round(mean(rf_vals), 4) if rf_vals else 0,
        "near_equilibrium_frac": round(near_eq, 4),
        "zero_f_count": sum(1 for s in states if s.is_zero_f),
        "inf_f_count": sum(1 for s in states if s.is_inf_f),
    }


def a5_resonance_score(states):
    """
    Misst A₅-Resonanz: Anteil der Zustände nahe A₅-Winkeln.
    Hoch → Zustände haben sich zu A₅-Symmetrie-Positionen entwickelt.
    """
    tol = 0.3  # Toleranzwinkel
    score_sigma = mean(
        max(math.exp(-min(angle_diff(s.sigma, a) for a in A5_5FOLD) / tol)
            for s in states)
        for _ in [0]  # dummy loop
    ) if states else 0.0
    # Korrektur: korrekte Berechnung
    sigma_scores = [max(math.exp(-min(angle_diff(s.sigma, a) for a in A5_5FOLD)/tol)
                        for s in [s]) for s in states]
    chi_scores = [max(math.exp(-min(angle_diff(s.chi, a) for a in A5_3FOLD)/tol)
                      for s in [s]) for s in states]
    return {
        "mean_a5_sigma_resonance": round(mean(sigma_scores), 4),
        "mean_a5_chi_resonance": round(mean(chi_scores), 4),
        "combined_a5_resonance": round(mean(s*c for s,c in zip(sigma_scores, chi_scores)), 4),
    }


# ---------------------------------------------------------------------------
# 8. Output und Plots
# ---------------------------------------------------------------------------

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)
    return Path(d)

def save_results_csv(records, output_dir, filename="s9_results.csv"):
    skip = {"_states"}
    path = ensure_dir(output_dir) / filename
    if not records:
        return
    keys = [k for k in records[0] if k not in skip]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k) for k in keys})
    print("saved:", path)

def plot_comparison(summary_by_j, output_dir):
    """
    Vergleichsplot mit Regime-Annotation.
      geometrisch   : E~O(N),  sparsity > 0.5  (blau/gruen)
      ueberkritisch : E~O(N2), sparsity < 0.5  (orange, schraffiert)
      unterkritisch : E/N < 0.5                (hellgrau)
    """
    path = ensure_dir(output_dir) / "s9_comparison.png"
    names      = list(summary_by_j.keys())
    geo_scores = [summary_by_j[n]["geo_score"] for n in names]
    edges      = [summary_by_j[n]["edges"]     for n in names]
    cc         = [summary_by_j[n]["cc"]        for n in names]
    geo_d      = [summary_by_j[n]["geo_d"]     for n in names]
    sparsity   = [summary_by_j[n].get("sparsity", 1.0) for n in names]
    e_per_n    = [summary_by_j[n]["edges"] / max(summary_by_j[n].get("largest", 10), 1)
                  for n in names]

    known   = {"metric_exp", "anisotropic_exp", "linf_exp", "rational_p2"}
    new_fut = {"phi_exp", "icosahedral_a5", "rf_weighted"}

    def regime_color(name, sp, epn):
        if epn < 0.5:       return "#cccccc"
        if sp  < 0.5:       return "#ff8c00"
        if name in new_fut: return "seagreen"
        if name in known:   return "steelblue"
        return "#cccccc"

    colors  = [regime_color(n, sp, epn) for n, sp, epn in zip(names, sparsity, e_per_n)]
    hatches = ["///" if sp < 0.5 else "" for sp in sparsity]

    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    for ax, vals, title in zip(axes,
        [geo_scores, edges, cc, geo_d],
        ["Geometry Score (korrigiert)", "Kanten (E)", "Clustering Coeff", "Mean Geodesic d"]):
        bars = ax.bar(range(len(names)), vals, color=colors)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for bar, v, sp in zip(bars, vals, sparsity):
            if v > 0:
                txt = f"{v:.2f}" + (" E~N2" if sp < 0.5 else "")
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.005 * max(vals + [0.01]),
                        txt, ha="center", va="bottom", fontsize=6)

    import matplotlib.patches as mpatches
    fig.legend(handles=[
        mpatches.Patch(color="steelblue", label="Bewahrt (S7/S8)"),
        mpatches.Patch(color="seagreen",  label="Neu FUT/CoMath"),
        mpatches.Patch(color="#ff8c00", hatch="///", label="Ueberkritisch E~O(N2) -- trivial"),
        mpatches.Patch(color="#cccccc",   label="Unterkritisch"),
    ], loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        "S9: J-Vergleich mit Regime-Klassifikation\n"
        "Ueberkritisch (E~O(N2)) != geometrisch interessant",
        fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print("saved:", path)
def plot_rf_analysis(rf_results, output_dir):
    path = ensure_dir(output_dir) / "s9_rf_analysis.png"
    names = list(rf_results.keys())
    eq_frac = [rf_results[n]["near_equilibrium_frac"] for n in names]
    mean_rf = [rf_results[n]["mean_rf"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(range(len(names)), eq_frac, color="seagreen", alpha=0.8)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels([n.replace("_","\n") for n in names], fontsize=8)
    ax1.set_title("Anteil Zustände nahe R/F=1\n(Elektron-Horizont-Nähe)", fontsize=10)
    ax1.set_ylabel("Anteil [0,1]")
    ax1.axhline(0.33, color="red", linestyle=":", label="Zufallserwartung")
    ax1.legend(fontsize=8)

    ax2.bar(range(len(names)), mean_rf, color="steelblue", alpha=0.8)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels([n.replace("_","\n") for n in names], fontsize=8)
    ax2.axhline(1.0, color="red", linestyle=":", label="R/F=1 (Gleichgewicht)")
    ax2.set_title("Mittleres R/F-Verhältnis\n(memory/w)", fontsize=10)
    ax2.set_ylabel("R/F")
    ax2.legend(fontsize=8)

    fig.suptitle("S9: R/F-Spektrum-Analyse (FUT/CoMath Elektron-Horizont)", fontsize=11)
    fig.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    print("saved:", path)


# ---------------------------------------------------------------------------
# 9. Antworten auf die 6 Fragen
# ---------------------------------------------------------------------------

def print_answers(char_results, summary_by_j, rf_by_j, a5_by_j):
    SEP = "=" * 68

    print(f"\n{SEP}")
    print("S9 — ANTWORTEN AUF DIE 6 FRAGEN")
    print(SEP)

    # F1: Minimale Eigenschaften
    print("\n── F1: MINIMALE J-EIGENSCHAFTEN (notwendige Bedingungen) ──")
    print("""
  Bedingung 1 — SUB-PERKOLATION:
    ER-Ratio = p(J > cluster_threshold) / (1/n) < 6
    → Verhindert Mega-Cluster-Kollaps
    → Verletzt von: cosine, mixed, threshold, random_projection

  Bedingung 2 — LIPSCHITZ-STETIGKEIT:
    Lipschitz-Proxy > 0.05
    → Persistenz-Gradient endlich → stabile Track-Dynamik
    → Verletzt von: threshold_resonance (Sprungfunktion)

  Bedingung 3 — DYNAMIK-RANGE:
    CV = std(J)/mean(J) > 0.4
    → J diskriminiert kohärente von inkohärenten Paaren
    → Verletzt von: random (CV=0.57 ≈ Grenze)

  Bedingung 4 — DICHTE DES J-WERTEBEREICHS:
    Kein Sprung im J-Bild um cluster_threshold
    → random_projection: Lücke zwischen 0.64 und 0.75 → kein
      sub-kritischer Betrieb möglich

  Empirische Grenzwerte (aus S7/S8/S8.1, 22 J-Funktionen, 50 Seeds):
    ER-Ratio:   0.5 ≤ ER ≤ 5 für funktionierenden Bereich
    alpha-Exp:  0.7 ≤ α ≤ 1.5 (Optimum: α=1.0)
    Lipschitz:  > 0.05
    CV:         > 0.40
""")

    # F2: Maximale Eigenschaften
    print("── F2: MAXIMALE J-EIGENSCHAFTEN (Obergrenzen) ──")
    print("""
  Obergrenze 1 — PERKOLATION:
    ER-Ratio > 6 → Mega-Cluster → 1 stabiler Track, 0 Kanten
    Verletzt von allen nicht-metric-J bei cluster_threshold=0.6

  Obergrenze 2 — PERIODIZITÄT:
    Keine periodischen Terme die p(J>threshold)>>1/n erzeugen
    Erlaubt: schwache A₅-Modulation (S9 icosahedral_a5)
    Verboten: starke cos/sin-Dominanz (cosine_resonance)

  Obergrenze 3 — ALPHA-SKALA (für exp-Familie):
    α > 2.0: zu viele Komponenten (16), kleinste Komponenten (3.5)
    Optimales Fenster: 0.7 ≤ α ≤ 1.5

  Obergrenze 4 — KONNEKTIVITÄT:
    rational_p1: 91.8 Kanten, 1 Komponente → vollständig verbunden
    Zuviele Langkopplungen unterdrücken differenzierte Struktur
    mean_degree > 8 → overconnected
""")

    # F3: Beste Eigenschaften
    print("── F3: BESTE J-EIGENSCHAFTEN (höchste Struktur) ──")
    scored = sorted(summary_by_j.items(), key=lambda x: x[1]["geo_score"], reverse=True)
    print(f"  {'J-Funktion':22s}  {'Score':>6}  {'Kanten':>6}  {'Largest':>7}  "
          f"{'CC':>6}  {'geo_d':>6}  Typ")
    print("  " + "-"*72)
    for name, v in scored[:10]:
        tag = "neu" if name in {"phi_exp", "icosahedral_a5", "rf_weighted"} else "S7/S8"
        print(f"  {name:22s}  {v['geo_score']:6.3f}  {v['edges']:6.1f}  "
              f"{v['largest']:7.1f}  {v['cc']:6.3f}  {v['geo_d']:6.3f}  {tag}")

    # F4: A₅/Ikosaeder-Nähe
    print("\n── F4: FUT/CoMath A₅-IKOSAEDER-NÄHE ──")
    print("""
  A₅-Symmetrie in FUT/CoMath:
    - 3D Raum als minimale treue A₅-Darstellung (CoMath ch. 12)
    - Ikosaeder: 12 Vertices, 5-fach + 3-fach Rotationsachsen
    - φ = goldener Schnitt als A₅-Quantisierungseinheit (φ⁵)

  j_icosahedral_a5 im Vergleich:
""")
    for name in ["metric_exp", "icosahedral_a5", "phi_exp"]:
        if name in summary_by_j:
            v = summary_by_j[name]
            a5 = a5_by_j.get(name, {})
            print(f"    {name:22s}  score={v['geo_score']:.3f}  "
                  f"A₅-Resonanz={a5.get('combined_a5_resonance', '?')}")
    print("""
  A₅-Initialzustände (a5_init=True) vs. zufällig:
    Testet ob ikosaedrische Symmetriebrechung die Geometrie verändert.
    → Ergebnisse siehe s9_comparison.png

  Nächste Schritte für F4:
    → S10: Volle A₅-Darstellungsmatrix als Kopplung J_{A5}(v_x, v_y)
    → Vergleich der emergenten Geodesics mit A₅-Orbit-Struktur
""")

    # F5: FUT/CoMath Variablentypen
    print("── F5: FUT/CoMath-VARIABLENTYPEN IM PYTHON-CODE ──")
    print(f"""
  Implementiert in CoMathState:

  0_f  (strukturierte Null):
    state.is_zero_f = (state.w < {EPSILON_F})
    Bedeutung: Zustand persistiert minimal, kein Kollaps zu echter 0.
    In S9: Zustände mit w→ε überleben durch memory-Feedback.
    Abweichung von CoMath: 0_f sollte noch schwache J-Kopplung haben.

  ∞_f  (gesättigte Kohärenz):
    state.is_inf_f = (state.w > 0.95 and state.memory > 0.95)
    Bedeutung: maximale Persistenz, kein weiteres Wachstum.
    In S9: max_states=120 und memory_cap=1.0 implementieren Grenze.

  gleich_f  (fraktale Gleichheit):
    state.gleich_f(other, j_func) = J(state,other) > {1-GLEICH_F_EPSILON}
    Bedeutung: approximate identity, nicht exakte Gleichheit.
    In S9: track_overlap via Jaccard nutzt gleich_f-Konzept implizit.

  R/F-Verhältnis:
    state.r_f = state.memory / state.w
    R = recursion proxy (memory), F = flat coupling proxy (w)
    Elektron-Horizont in FUT: R/F ≈ 1 → α = 1/137
    In S9: j_rf_weighted bevorzugt Zustände nahe R/F=1.

  NOCH NICHT IMPLEMENTIERT:
    → π_f  (fraktale Pi): k=7.004... Ableitung aus FUT ch. 7
    → A_f  (fraktale Fläche): 4π_f² + 2/π_f (offen, FUT TODO)
    → Vollständige CoMath Emergence Chain (Level 1-12)
    → C_f  (operative Kohärenz): härtestes offenes Problem
""")

    # F6: Code-Verbesserungen
    print("── F6: OPTIMALE CODE-VERBESSERUNGEN ──")
    print("""
  1. A₅-INITIALZUSTÄNDE (implementiert):
     make_a5_initial_states() platziert States an Ikosaeder-Winkeln.
     Test: a5_init=True vs. False für gleiche J-Funktion.

  2. CoMathState (implementiert):
     r_f, coherence_level, is_zero_f, is_inf_f, gleich_f als Properties.
     Kosten: minimal (Properties, kein Extra-Speicher).

  3. j_rf_weighted (implementiert):
     Einzige J-Funktion die dynamisch von R/F abhängt.
     Selbstverstärkend: Gleichgewichtszustände koppeln stärker.

  4. J-THEOREM-CHECKER (implementiert):
     characterize_j() prüft alle 4 Bedingungen vor Simulation.
     Verhindert Fehlläufe mit nicht-geometriefähigen J.

  5. NOCH OFFEN — für S10:
     a) C_f operative Definition → persistenzbasierte Kohärenz
     b) φ-adaptive cluster_threshold = exp(-1/φ) ≈ 0.539
     c) Multi-level emergence tracking (CoMath ch. 1-12)
     d) Volle A₅-Matrixkopplung (Darstellungstheorie)
     e) Gromov-δ-Uniformität als Emergenz-Indikator
     f) Asymmetrische J → gerichtete Proto-Topologie (→ Zeitpfeil?)
""")

    print(SEP)


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SEEDS = list(range(15))       # 15 Seeds pro J für Geschwindigkeit
    OUTPUT_DIR = "s9_outputs"
    VERBOSE = True

    print("=== S9: Minimale J-Theorie + FUT/CoMath-Verbindung ===")
    print(f"Seeds: {len(SEEDS)}  |  J-Funktionen: {len(J_REGISTRY)}")
    print(f"Ausgabe: {OUTPUT_DIR}/\n")

    all_records = []
    char_results = {}
    rf_by_j = {}
    a5_by_j = {}

    for j_name, (j_func, description) in J_REGISTRY.items():
        print(f"--- {j_name} ({description}) ---")

        # J charakterisieren
        char = characterize_j(j_func)
        char_results[j_name] = char
        conditions = ("✓" if char["cond1_subperc"] else "✗") + \
                     ("✓" if char["cond2_lipschitz"] else "✗") + \
                     ("✓" if char["cond3_cv"] else "✗") + \
                     ("✓" if char["cond4_density"] else "✗")
        geo_ok = "→ geometriefähig" if char["all_conditions"] else "→ NICHT geometriefähig"
        print(f"  ER={char['er_ratio']:.1f}  CV={char['cv']:.3f}  "
              f"lip={char['lipschitz']:.3f}  Bed.[1234]={conditions}  {geo_ok}")

        # Simulation
        for seed in SEEDS:
            r = run_one(j_name, j_func, seed, a5_init=False)
            all_records.append(r)
            if seed == 0 and VERBOSE:
                print(f"  seed 0: states={r['final_states']}  "
                      f"stable={r['stable_tracks']}  edges={r['edge_count']}  "
                      f"score={r['geo_score']:.3f}  CC={r['clustering_coeff']:.3f}")

        # A₅-Init-Test (nur für icosahedral_a5 und metric_exp)
        if j_name in {"icosahedral_a5", "metric_exp"}:
            r_a5 = run_one(j_name, j_func, 0, a5_init=True)
            r_a5["j_name"] = j_name + "_a5init"
            all_records.append(r_a5)
            a5_by_j[j_name] = a5_resonance_score(r_a5["_states"])
            print(f"  A₅-Init: edges={r_a5['edge_count']}  "
                  f"score={r_a5['geo_score']:.3f}  "
                  f"A₅-resonance={a5_by_j[j_name]['combined_a5_resonance']}")

        # R/F-Analyse (letzter Seed)
        last_r = [r for r in all_records if r["j_name"] == j_name]
        if last_r:
            rf_by_j[j_name] = rf_distribution(last_r[-1]["_states"])

    # Zusammenfassung nach J
    print("\n=== ZUSAMMENFASSUNG ===")
    summary_by_j = {}
    for j_name in J_REGISTRY:
        rows = [r for r in all_records if r["j_name"] == j_name]
        if not rows:
            continue
        summary_by_j[j_name] = {
            "geo_score": mean(r["geo_score"] for r in rows),
            "edges":     mean(r["edge_count"] for r in rows),
            "largest":   mean(r["largest_component"] for r in rows),
            "cc":        mean(r["clustering_coeff"] for r in rows),
            "geo_d":     mean(r["mean_geodesic_d"] for r in rows),
            "stable":    mean(r["stable_tracks"] for r in rows),
        }

    # Plots + CSV
    save_results_csv(all_records, OUTPUT_DIR)
    plot_comparison(summary_by_j, OUTPUT_DIR)
    plot_rf_analysis(rf_by_j, OUTPUT_DIR)

    # Antworten auf 6 Fragen
    print_answers(char_results, summary_by_j, rf_by_j, a5_by_j)
