#!/usr/bin/env python3
"""
s12_rf_bifurcation_universality.py
FUT/CoMath Simulation S12

Ziel
----
S12 testet, ob die in S11-A gefundene Bifurkationsform

    r*(alpha) = 1 / (1 - alpha)

nur ein Artefakt des einfachen Modells ist oder ob sie unter realistischeren
Rekohärenz-Netzwerken stabil bleibt. Zusätzlich wird eine kontrollierte
Dämpfung beta eingeführt, deren analytischer Fixpunkt lautet:

    r*(alpha, beta) = (1 + beta) / (1 - alpha + beta)

Für beta = 0 fällt dies exakt auf S11-A zurück.
Für beta -> unendlich geht r* -> 1, also FUT-Elektron-Horizont.

Module
------
A) Universality Sweep:
   alpha × beta × J-Funktion × Seeds.
   Vergleich Simulation gegen analytischen Fixpunkt.

B) Geometry Coupling Check:
   Prüft, ob L∞ / power / metric_exp bei verschiedenen alpha,beta-Regimen
   ihre Spektralordnung behalten oder durch R/F-Selbstverstärkung kollabieren.

C) Phase Classification:
   stable_horizon      : |r-1| < 0.05
   analytic_tracking   : |r-r_theory| < 0.05 und r < saturation
   saturated_memory    : m läuft gegen numerische Obergrenze
   unstable_runaway    : r wächst stark / nicht konvergiert
   undercoupled        : Netzwerk verliert Verbindung

Abhängigkeiten: Python 3.10+, numpy >= 1.22, matplotlib >= 3.5
Autor: Jens Deutschmann (Framework) / OpenAI GPT-5.5 Thinking (Implementierung)
Datum: 30.05.2026
"""

from __future__ import annotations

import csv
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Output ──────────────────────────────────────────────────────────────────

OUT_DIR = Path("s12_output")
OUT_DIR.mkdir(exist_ok=True)

# ─── Configuration ───────────────────────────────────────────────────────────

N_NODES = 12
N_SEEDS = 20
N_STEPS = 500
BURN_IN_AVG = 80
THETA_C = 0.35
BASE_LAMBDA_W = 0.08
BASE_LAMBDA_M = 0.12
GAMMA_FEEDBACK = 0.45
M_CAP = 50.0               # numerical safety cap; saturation is reported
SATURATION_LEVEL = 0.90 * M_CAP

ALPHAS = [0.00, 0.005, 1 / 137.036, 0.02, 0.05, 0.10, 0.20, 0.30,
          0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
BETAS = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00, 2.00]
J_NAMES = ["linf_exp", "power", "metric_exp"]

ICO_EIG_RAW = np.array([5.0, math.sqrt(5.0), -1.0, -math.sqrt(5.0), -4.0])
ICO_EIG_NORM = ICO_EIG_RAW / 5.0

# ─── State ───────────────────────────────────────────────────────────────────

@dataclass
class CoMathState:
    sigma: float
    chi: float
    w: float = 0.50
    m: float = 0.30

    @property
    def r_f(self) -> float:
        return self.m / max(self.w, 1e-12)


def make_nodes(seed: int) -> List[CoMathState]:
    random.seed(seed)
    np.random.seed(seed)
    nodes: List[CoMathState] = []
    for i in range(N_NODES):
        sigma = (i / N_NODES) + random.gauss(0.0, 0.04)
        chi = random.random()
        nodes.append(CoMathState(
            sigma=max(0.0, min(1.0, sigma)),
            chi=max(0.0, min(1.0, chi)),
        ))
    return nodes

# ─── Coupling functions ──────────────────────────────────────────────────────


def j_linf_exp(a: CoMathState, b: CoMathState) -> float:
    return math.exp(-max(abs(a.sigma - b.sigma), abs(a.chi - b.chi)))


def j_power(a: CoMathState, b: CoMathState) -> float:
    d2 = (a.sigma - b.sigma) ** 2 + (a.chi - b.chi) ** 2
    return 1.0 / (1.0 + d2)


def j_metric_exp(a: CoMathState, b: CoMathState) -> float:
    d = math.sqrt((a.sigma - b.sigma) ** 2 + (a.chi - b.chi) ** 2)
    return math.exp(-d)


J_FUNCS: Dict[str, Callable[[CoMathState, CoMathState], float]] = {
    "linf_exp": j_linf_exp,
    "power": j_power,
    "metric_exp": j_metric_exp,
}


def j_with_memory(a: CoMathState, b: CoMathState, base_j: Callable[[CoMathState, CoMathState], float]) -> float:
    # Positive memory feedback as in S10, but clipped only for adjacency safety.
    return base_j(a, b) * (1.0 + GAMMA_FEEDBACK * a.m * b.m / (1.0 + a.m * b.m))

# ─── Analytic theory ─────────────────────────────────────────────────────────


def r_star(alpha: float, beta: float) -> float:
    denom = 1.0 - alpha + beta
    if denom <= 1e-12:
        return float("inf")
    return (1.0 + beta) / denom

# ─── Network diagnostics ─────────────────────────────────────────────────────


def build_adj(nodes: List[CoMathState], j_name: str, use_memory: bool = True) -> np.ndarray:
    base_j = J_FUNCS[j_name]
    n = len(nodes)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            val = j_with_memory(nodes[i], nodes[j], base_j) if use_memory else base_j(nodes[i], nodes[j])
            if val > THETA_C:
                A[i, j] = val
                A[j, i] = val
    return A


def sorted_eigenvalues(A: np.ndarray) -> np.ndarray:
    eigs = np.linalg.eigvalsh(A)
    return np.sort(eigs)[::-1]


def ico_score_from_adj(A: np.ndarray) -> float:
    eigs = sorted_eigenvalues(A)
    if len(eigs) < 5 or abs(eigs[0]) < 1e-12:
        return 0.0
    eigs_n = eigs / abs(eigs[0])
    n = len(eigs_n)
    idx = [0, 1, n // 2, n - 2, n - 1]
    selected = np.array([eigs_n[i] for i in idx])
    diff = np.linalg.norm(selected - ICO_EIG_NORM)
    ref = np.linalg.norm(ICO_EIG_NORM)
    return float(1.0 - diff / ref)


def edge_count(A: np.ndarray) -> int:
    return int(np.count_nonzero(np.triu(A > 0.0, 1)))


def largest_component_size(A: np.ndarray) -> int:
    n = A.shape[0]
    seen = [False] * n
    best = 0
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        size = 0
        while stack:
            v = stack.pop()
            size += 1
            for u in np.nonzero(A[v] > 0.0)[0]:
                if not seen[int(u)]:
                    seen[int(u)] = True
                    stack.append(int(u))
        best = max(best, size)
    return best

# ─── Dynamics ────────────────────────────────────────────────────────────────


def dynamics_step(nodes: List[CoMathState], j_name: str, alpha: float, beta: float) -> None:
    """
    Coupled R/F update.

    Memory target:
        target_m = w + alpha*m - beta*max(0, r-1)*w

    If r > 1 at fixed point:
        r = 1 + alpha*r - beta*(r-1)
        r*(alpha,beta) = (1+beta)/(1-alpha+beta)
    """
    base_j = J_FUNCS[j_name]
    n = len(nodes)
    w_new: List[float] = []
    m_new: List[float] = []

    for i in range(n):
        wsum = 0.0
        jsum = 0.0
        for j in range(n):
            if i == j:
                continue
            jval = j_with_memory(nodes[i], nodes[j], base_j)
            if jval > THETA_C:
                jsum += jval
                wsum += jval * nodes[j].w

        w_nb = wsum / jsum if jsum > 1e-12 else nodes[i].w
        w_i = nodes[i].w + BASE_LAMBDA_W * (w_nb - nodes[i].w)
        w_i = max(1e-9, min(1.0, w_i))

        r = nodes[i].m / max(nodes[i].w, 1e-12)
        damping = beta * max(0.0, r - 1.0) * nodes[i].w
        target_m = nodes[i].w + alpha * nodes[i].m - damping
        m_i = nodes[i].m + BASE_LAMBDA_M * (target_m - nodes[i].m)
        m_i = max(0.0, min(M_CAP, m_i))

        w_new.append(w_i)
        m_new.append(m_i)

    for i, nd in enumerate(nodes):
        nd.w = w_new[i]
        nd.m = m_new[i]


def run_single(seed: int, j_name: str, alpha: float, beta: float) -> Dict[str, float]:
    nodes = make_nodes(seed)
    rf_traj: List[float] = []
    mmax_traj: List[float] = []

    for _ in range(N_STEPS):
        rf_traj.append(float(statistics.mean(nd.r_f for nd in nodes)))
        mmax_traj.append(max(nd.m for nd in nodes))
        dynamics_step(nodes, j_name, alpha, beta)

    tail = rf_traj[-BURN_IN_AVG:]
    r_mean = float(statistics.mean(tail))
    r_std = float(statistics.pstdev(tail))
    r_slope = float((tail[-1] - tail[0]) / max(1, len(tail) - 1))
    m_max = float(max(mmax_traj[-BURN_IN_AVG:]))

    A = build_adj(nodes, j_name, use_memory=True)
    E = edge_count(A)
    L = largest_component_size(A)
    ico = ico_score_from_adj(A)

    return {
        "r_mean": r_mean,
        "r_std": r_std,
        "r_slope": r_slope,
        "m_max": m_max,
        "edges": float(E),
        "largest_component": float(L),
        "ico_score": ico,
    }


def classify(r_mean: float, r_std: float, r_slope: float, m_max: float,
             edges: float, largest_component: float, theory: float) -> str:
    if m_max >= SATURATION_LEVEL:
        return "saturated_memory"
    if abs(r_slope) > 0.002 and r_mean > 3.0:
        return "unstable_runaway"
    if largest_component < max(3, N_NODES // 3) or edges <= 0:
        return "undercoupled"
    if abs(r_mean - 1.0) < 0.05:
        return "stable_horizon"
    if math.isfinite(theory) and abs(r_mean - theory) < 0.05 and r_std < 0.05:
        return "analytic_tracking"
    return "deviating"

# ─── Main sweep ──────────────────────────────────────────────────────────────


def run_sweep() -> List[Dict[str, float | str]]:
    print("\n" + "=" * 72)
    print("S12 — R/F-Bifurkationsuniversality")
    print("=" * 72)
    print(f"N={N_NODES}, seeds={N_SEEDS}, steps={N_STEPS}, theta_c={THETA_C}")
    print(f"alphas={len(ALPHAS)}, betas={len(BETAS)}, J={J_NAMES}")

    rows: List[Dict[str, float | str]] = []
    for j_name in J_NAMES:
        print(f"\nJ = {j_name}")
        for beta in BETAS:
            for alpha in ALPHAS:
                seed_results = [run_single(seed + 10_000, j_name, alpha, beta)
                                for seed in range(N_SEEDS)]
                theory = r_star(alpha, beta)

                def mean(key: str) -> float:
                    return float(statistics.mean(float(r[key]) for r in seed_results))

                def pstdev(key: str) -> float:
                    return float(statistics.pstdev(float(r[key]) for r in seed_results))

                r_mean = mean("r_mean")
                r_std = mean("r_std")
                r_slope = mean("r_slope")
                m_max = mean("m_max")
                edges = mean("edges")
                L = mean("largest_component")
                ico = mean("ico_score")
                cls = classify(r_mean, r_std, r_slope, m_max, edges, L, theory)

                rows.append({
                    "j_func": j_name,
                    "alpha": alpha,
                    "beta": beta,
                    "r_theory": theory,
                    "r_mean": r_mean,
                    "r_between_seed_std": pstdev("r_mean"),
                    "r_within_tail_std": r_std,
                    "r_abs_error": abs(r_mean - theory) if math.isfinite(theory) else float("nan"),
                    "r_slope_tail": r_slope,
                    "m_max_mean": m_max,
                    "edges_mean": edges,
                    "largest_component_mean": L,
                    "ico_score_mean": ico,
                    "ico_score_seed_std": pstdev("ico_score"),
                    "class": cls,
                })

            # short progress line per beta
            beta_rows = [r for r in rows if r["j_func"] == j_name and r["beta"] == beta]
            horizon_count = sum(1 for r in beta_rows if r["class"] == "stable_horizon")
            print(f"  beta={beta:>4.2f}: stable_horizon {horizon_count:>2}/{len(ALPHAS)}")

    return rows


def write_csv(rows: List[Dict[str, float | str]]) -> Path:
    csv_path = OUT_DIR / "s12_rf_bifurcation_universality.csv"
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nCSV: {csv_path}")
    return csv_path

# ─── Plots ───────────────────────────────────────────────────────────────────


def plot_bifurcation_curves(rows: List[Dict[str, float | str]]) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(
        "S12-A — R/F-Bifurkation: Simulation vs. Analytik\n"
        r"$r^*(\alpha,\beta)=(1+\beta)/(1-\alpha+\beta)$, J=linf_exp",
        fontsize=12, fontweight="bold",
    )

    j_name = "linf_exp"
    selected_betas = [0.00, 0.10, 0.50, 1.00, 2.00]
    for beta in selected_betas:
        sub = [r for r in rows if r["j_func"] == j_name and abs(float(r["beta"]) - beta) < 1e-12]
        sub = sorted(sub, key=lambda x: float(x["alpha"]))
        xs = np.array([float(r["alpha"]) for r in sub])
        ys = np.array([float(r["r_mean"]) for r in sub])
        th = np.array([float(r["r_theory"]) for r in sub])
        ax.plot(xs, th, "--", lw=1.2, alpha=0.6)
        ax.plot(xs, ys, "-o", lw=2.0, ms=4, label=f"β={beta:g}")

    ax.axvline(1 / 137.036, color="crimson", ls=":", lw=1.8,
               label=r"$\alpha_{em}=1/137.036$")
    ax.axhline(1.0, color="black", ls=":", lw=1.4, label="R/F=1")
    ax.set_xlabel(r"$\alpha$  (Memory-Selbstverstärkung)")
    ax.set_ylabel("Attraktor R/F")
    ax.set_ylim(0.9, 5.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = OUT_DIR / "s12_a_bifurcation_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PNG: {path}")
    return path


def plot_error_heatmap(rows: List[Dict[str, float | str]]) -> Path:
    j_name = "linf_exp"
    grid = np.zeros((len(BETAS), len(ALPHAS)))
    for i, beta in enumerate(BETAS):
        for k, alpha in enumerate(ALPHAS):
            match = [r for r in rows if r["j_func"] == j_name
                     and abs(float(r["beta"]) - beta) < 1e-12
                     and abs(float(r["alpha"]) - alpha) < 1e-12][0]
            grid[i, k] = min(float(match["r_abs_error"]), 1.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("S12-B — Absoluter Fehler |R/F_sim − R/F_theory|, J=linf_exp",
                 fontsize=12, fontweight="bold")
    im = ax.imshow(grid, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(ALPHAS)))
    ax.set_xticklabels([f"{a:.3f}" if a < 0.1 else f"{a:.1f}" for a in ALPHAS], rotation=45)
    ax.set_yticks(np.arange(len(BETAS)))
    ax.set_yticklabels([f"{b:g}" for b in BETAS])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\beta$")
    fig.colorbar(im, ax=ax, label="abs error clipped at 1.0")
    plt.tight_layout()
    path = OUT_DIR / "s12_b_error_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PNG: {path}")
    return path


def plot_ico_vs_alpha(rows: List[Dict[str, float | str]]) -> Path:
    beta = 0.50
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("S12-C — Spektral-Iko-Score unter stabilisierter R/F-Dynamik\nβ=0.5",
                 fontsize=12, fontweight="bold")
    for j_name in J_NAMES:
        sub = [r for r in rows if r["j_func"] == j_name and abs(float(r["beta"]) - beta) < 1e-12]
        sub = sorted(sub, key=lambda x: float(x["alpha"]))
        xs = [float(r["alpha"]) for r in sub]
        ys = [float(r["ico_score_mean"]) for r in sub]
        ax.plot(xs, ys, "-o", lw=2.0, ms=4, label=j_name)
    ax.axvline(1 / 137.036, color="crimson", ls=":", lw=1.8,
               label=r"$\alpha_{em}$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("Ico-Score aus Adjazenzspektrum")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = OUT_DIR / "s12_c_ico_vs_alpha.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PNG: {path}")
    return path

# ─── Report ──────────────────────────────────────────────────────────────────


def write_report(rows: List[Dict[str, float | str]]) -> Path:
    linf = [r for r in rows if r["j_func"] == "linf_exp"]
    tracking = [r for r in linf if r["class"] in ("analytic_tracking", "stable_horizon")]
    saturated = [r for r in linf if r["class"] == "saturated_memory"]

    alpha_em = 1 / 137.036
    em_rows = [r for r in rows if abs(float(r["alpha"]) - alpha_em) < 1e-12]
    em_lines = []
    for r in em_rows:
        em_lines.append(
            f"| {r['j_func']} | {float(r['beta']):.2f} | {float(r['r_theory']):.5f} | "
            f"{float(r['r_mean']):.5f} | {float(r['r_abs_error']):.5f} | {r['class']} |"
        )

    # Best Ico row at beta=0.5
    beta_half = [r for r in rows if abs(float(r["beta"]) - 0.50) < 1e-12]
    best_ico = max(beta_half, key=lambda r: float(r["ico_score_mean"]))

    report = f"""# S12 Report — R/F-Bifurkationsuniversality
Datum: 30.05.2026 | FUT/CoMath Simulation S12

## Kernfrage

S12 prüft, ob die S11-A-Formel

```text
r*(alpha) = 1 / (1 - alpha)
```

unter Netzwerk-Rekohärenz, unterschiedlichen J-Funktionen und aktiver Dämpfung erhalten bleibt.
Die getestete Erweiterung lautet:

```text
r*(alpha,beta) = (1 + beta) / (1 - alpha + beta)
```

## Setup

- Knoten: {N_NODES}
- Seeds: {N_SEEDS}
- Schritte: {N_STEPS}
- J-Funktionen: {', '.join(J_NAMES)}
- alpha-Werte: {len(ALPHAS)}
- beta-Werte: {len(BETAS)}
- Memory-Cap: {M_CAP}

## Schnellbefund

Für J=linf_exp liegen {len(tracking)} von {len(linf)} Parameterpunkten im Bereich `analytic_tracking` oder `stable_horizon`.
Saturierte Memory-Regime: {len(saturated)} von {len(linf)}.

Bestes Spektrum bei beta=0.5:

```text
J={best_ico['j_func']}, alpha={float(best_ico['alpha']):.5f}, beta={float(best_ico['beta']):.2f}, Ico={float(best_ico['ico_score_mean']):.5f}
```

## alpha_em-Schnitt

| J | beta | r_theory | r_sim | abs_error | class |
|---|------|----------|-------|-----------|-------|
{chr(10).join(em_lines)}

## Interpretation

1. Wenn beta=0, testet S12 direkt die S11-A-Bifurkation.
2. Wenn beta>0, wird geprüft, ob R/F=1 als stabilisierbarer Horizont aus einer allgemeinen Gegenkopplung folgt.
3. Wenn alpha nahe alpha_em liegt, muss r* nahe 1 bleiben. Das ist die entscheidende Brücke zur FUT-Deutung des Elektron-Horizonts.
4. Wenn hohe alpha-Werte saturieren, ist das kein Fehler, sondern die numerische Signatur rekursiver Selbstverstärkung oberhalb des stabilen Fortsetzungsregimes.

## Dateien

- `s12_rf_bifurcation_universality.csv`
- `s12_a_bifurcation_curves.png`
- `s12_b_error_heatmap.png`
- `s12_c_ico_vs_alpha.png`

*S12 abgeschlossen.*
"""
    path = OUT_DIR / "s12_report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report: {path}")
    return path

# ─── Entrypoint ──────────────────────────────────────────────────────────────


def main() -> None:
    rows = run_sweep()
    write_csv(rows)
    plot_bifurcation_curves(rows)
    plot_error_heatmap(rows)
    plot_ico_vs_alpha(rows)
    write_report(rows)
    print("\n" + "=" * 72)
    print("S12 ABGESCHLOSSEN")
    print(f"Ausgaben in: {OUT_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
