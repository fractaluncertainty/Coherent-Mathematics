#!/usr/bin/env python3
"""
s11_alpha_feedback_vs_alpha_em.py
FUT/CoMath Simulation S11

Frage: Gibt es eine Verbindung zwischen dem diskreten Feedback-Parameter alpha
(Memory-Selbstverstärkung) und der Feinstrukturkonstante alpha_em?

Theoretisches Ergebnis aus S10:
  Attraktor-Formel:  r* = 1 / (1 - alpha)
  Taylorentwicklung: r* ≈ 1 + alpha + alpha² + ...
  → Abweichung vom Gleichgewicht (erster Ordnung): r* - 1 ≈ alpha

FUT-Hypothese:
  Wenn alpha_eff = alpha_em = 1/137.036... am Elektron-Horizont,
  dann folgt:  r*(alpha_em) = 1 / (1 - 1/137.036) = 137.036/136.036 ≈ 1.00735
  Interpretation: alpha_em misst die residuale R/F-Asymmetrie des Elektrons.
  Das Elektron sitzt bei r* = 1 + alpha_em, nicht exakt bei r* = 1.

Drei Module:
  S11-A: Bifurkationsdiagramm r*(alpha) — analytisch + Simulation (20 Seeds)
  S11-B: Zoom auf alpha ∈ [0, 0.02] — Bereich alpha_em, Abweichungsanalyse
  S11-C: Residuum-Test: |r*(alpha_em) - 1 - alpha_em| vs. Ordnung alpha²

Abhängigkeiten: Python 3.10+, numpy >= 1.22, matplotlib >= 3.5
Autor: Jens Deutschmann (Framework) / Claude Sonnet (Implementierung)
Datum: 30.05.2026
"""

import math
import random
import csv
import statistics
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Output ───────────────────────────────────────────────────────────────────

OUT = Path("s11_output")
OUT.mkdir(exist_ok=True)

# ─── Konstanten ───────────────────────────────────────────────────────────────

# FUT alpha-Formel: 1/alpha_em = (2+√2)(4π²+2/π) + 3/(4π²+2/π)
_B     = 4 * math.pi**2 + 2 / math.pi
ALPHA_EM_INV = (2 + math.sqrt(2)) * _B + 3.0 / _B
ALPHA_EM     = 1.0 / ALPHA_EM_INV       # ≈ 0.007297...

N_NODES       = 12
N_SEEDS       = 20
THETA_C       = 0.35
N_STEPS       = 400     # mehr Schritte für Konvergenz bei kleinem alpha
BASE_LAMBDA_W = 0.08
BASE_LAMBDA_M = 0.10

print(f"FUT: 1/alpha_em = {ALPHA_EM_INV:.6f}  →  alpha_em = {ALPHA_EM:.8f}")
print(f"     r*(alpha_em) = {1/(1-ALPHA_EM):.8f}  (analytisch)")
print(f"     r* - 1       = {1/(1-ALPHA_EM) - 1:.8f}")
print(f"     alpha_em     = {ALPHA_EM:.8f}")
print(f"     Ordnung alpha²: {ALPHA_EM**2:.2e}\n")

# ─── CoMathState ─────────────────────────────────────────────────────────────

class Node:
    __slots__ = ('sigma', 'chi', 'w', 'm')
    def __init__(self, sigma, chi):
        self.sigma = float(sigma)
        self.chi   = float(chi)
        self.w     = 0.50
        self.m     = 0.30

    @property
    def r_f(self):
        return self.m / max(self.w, 1e-9)

def make_nodes(seed):
    random.seed(seed)
    nodes = []
    for i in range(N_NODES):
        s = max(0.0, min(1.0, i / N_NODES + random.gauss(0, 0.04)))
        c = random.random()
        nodes.append(Node(s, c))
    return nodes

# ─── Kopplungsfunktion ────────────────────────────────────────────────────────

def j_linf(a, b):
    return math.exp(-max(abs(a.sigma - b.sigma), abs(a.chi - b.chi)))

# ─── Dynamik ─────────────────────────────────────────────────────────────────

def step(nodes, alpha):
    """
    Ein Dynamik-Schritt mit Memory-Selbstverstärkungsparameter alpha.
    Undamped: target = w + alpha * m  → Attraktor r* = 1/(1-alpha)
    """
    n = len(nodes)
    w_new, m_new = [], []

    for i in range(n):
        # Kohärenz-Update
        wsum, jsum = 0.0, 0.0
        for j in range(n):
            if i == j:
                continue
            jv = j_linf(nodes[i], nodes[j])
            if jv > THETA_C:
                jsum += jv
                wsum += jv * nodes[j].w
        w_nb = wsum / jsum if jsum > 1e-9 else nodes[i].w
        wi = max(1e-6, min(1.0, nodes[i].w + BASE_LAMBDA_W * (w_nb - nodes[i].w)))

        # Memory-Update mit Selbstverstärkung alpha
        target = nodes[i].w + alpha * nodes[i].m
        mi = max(0.0, min(1.0, nodes[i].m + BASE_LAMBDA_M * (target - nodes[i].m)))

        w_new.append(wi)
        m_new.append(mi)

    for i in range(n):
        nodes[i].w = w_new[i]
        nodes[i].m = m_new[i]

def simulate_attractor(alpha, seed_offset=0):
    """
    Simuliert N_SEEDS Läufe für ein gegebenes alpha.
    Gibt (mean_r_star, std_r_star) zurück (Mittel der letzten 50 Schritte).
    """
    attractors = []
    for seed in range(N_SEEDS):
        nodes = make_nodes(seed + seed_offset)
        traj = []
        for _ in range(N_STEPS):
            rf = statistics.mean(nd.r_f for nd in nodes)
            traj.append(rf)
            step(nodes, alpha)
        attractors.append(float(np.mean(traj[-50:])))
    return float(np.mean(attractors)), float(np.std(attractors))

# ─── S11-A: Bifurkationsdiagramm ─────────────────────────────────────────────

def run_s11a():
    print("=" * 60)
    print("S11-A: Bifurkationsdiagramm r*(alpha)")
    print("=" * 60)

    # Alpha-Sweep: 0 bis 0.95 (Instabilität ab alpha -> 1)
    alphas_coarse = np.linspace(0.0, 0.90, 19)
    # Fein um alpha_em
    alphas_fine   = np.array([0.0, ALPHA_EM/2, ALPHA_EM,
                               2*ALPHA_EM, 5*ALPHA_EM, 0.01, 0.02,
                               0.05, 0.10, 0.20, 0.30, 0.40,
                               0.50, 0.60, 0.70, 0.80, 0.90])
    alphas = np.unique(np.concatenate([alphas_coarse, alphas_fine]))

    analytical = 1.0 / (1.0 - alphas)
    sim_means, sim_stds = [], []

    for i, a in enumerate(alphas):
        mu, sd = simulate_attractor(a, seed_offset=300)
        sim_means.append(mu)
        sim_stds.append(sd)
        print(f"  alpha={a:.5f}  r*(analyt)={1/(1-a):.5f}  r*(sim)={mu:.5f}  std={sd:.5f}")

    sim_means = np.array(sim_means)
    sim_stds  = np.array(sim_stds)

    # CSV
    csv_path = OUT / "s11a_bifurcation.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['alpha', 'r_star_analytical', 'r_star_sim_mean', 'r_star_sim_std'])
        for a, an, sm, ss in zip(alphas, analytical, sim_means, sim_stds):
            wr.writerow([a, an, sm, ss])
    print(f"\n  CSV: {csv_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        r"S11-A — Bifurkationsdiagramm: $r^*(\alpha) = 1/(1-\alpha)$" + "\n"
        r"Analytisch vs. Simulation (20 Seeds, N=12)",
        fontsize=12, fontweight='bold'
    )

    # Analytische Kurve
    a_dense = np.linspace(0, 0.92, 500)
    ax.plot(a_dense, 1/(1-a_dense), '-', color='#1f77b4', lw=2.5,
            label=r'Analytisch: $r^* = 1/(1-\alpha)$', zorder=2)

    # Simulation
    ax.fill_between(alphas, sim_means - sim_stds, sim_means + sim_stds,
                    alpha=0.25, color='#ff7f0e')
    ax.plot(alphas, sim_means, 'o', color='#ff7f0e', ms=5, zorder=3,
            label='Simulation (mean ± σ, 20 Seeds)')

    # alpha_em markieren
    r_em = 1.0 / (1.0 - ALPHA_EM)
    ax.axvline(ALPHA_EM, color='crimson', lw=1.5, ls='--', zorder=4)
    ax.scatter([ALPHA_EM], [r_em], color='crimson', s=120, zorder=5,
               marker='*',
               label=rf'$\alpha_{{em}} = 1/137.036 = {ALPHA_EM:.5f}$'
                     rf'  →  $r^* = {r_em:.5f}$')

    # Gleichgewicht
    ax.axhline(1.0, color='black', lw=1.2, ls=':', label=r'$r^* = 1$ (FUT Elektron-Horizont)')

    ax.set_xlabel(r'$\alpha$ (Memory-Selbstverstärkung)', fontsize=11)
    ax.set_ylabel(r'$r^*$ (Attraktor R/F)', fontsize=11)
    ax.set_xlim(-0.01, 0.93)
    ax.set_ylim(0.8, 12)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png = OUT / "s11a_bifurcation.png"
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PNG: {png}")

    return alphas, sim_means, sim_stds

# ─── S11-B: Zoom alpha_em ─────────────────────────────────────────────────────

def run_s11b():
    print("\n" + "=" * 60)
    print(r"S11-B: Zoom um alpha_em — Abweichungsanalyse")
    print("=" * 60)

    # Sehr feiner Sweep nahe alpha_em
    alphas = np.array([
        0.0,
        ALPHA_EM * 0.25,
        ALPHA_EM * 0.50,
        ALPHA_EM * 0.75,
        ALPHA_EM,
        ALPHA_EM * 1.25,
        ALPHA_EM * 1.50,
        ALPHA_EM * 2.0,
        ALPHA_EM * 3.0,
        ALPHA_EM * 5.0,
        ALPHA_EM * 10.0,
        0.10,
    ])

    print(f"\n  {'alpha':>12}  {'r*(analyt)':>12}  {'r*(sim)':>12}  "
          f"{'r*-1':>10}  {'alpha':>10}  {'|r*-1-alpha|':>14}  {'alpha^2':>10}")
    print("  " + "-" * 90)

    rows = []
    for a in alphas:
        mu, sd = simulate_attractor(a, seed_offset=400)
        r_an  = 1.0 / (1.0 - a) if a < 1.0 else float('inf')
        dev   = r_an - 1.0
        resid = abs(dev - a)     # |r* - 1 - alpha| = Order(alpha^2) expected
        a2    = a**2
        print(f"  {a:>12.7f}  {r_an:>12.7f}  {mu:>12.7f}  "
              f"{dev:>10.7f}  {a:>10.7f}  {resid:>14.2e}  {a2:>10.2e}")
        rows.append([a, r_an, mu, sd, dev, resid, a2])

    # CSV
    csv_path = OUT / "s11b_zoom_alpha_em.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['alpha', 'r_star_analytical', 'r_star_sim', 'r_star_sim_std',
                     'r_star_minus_1', 'residuum', 'alpha_squared'])
        wr.writerows(rows)
    print(f"\n  CSV: {csv_path}")

    # Plot: r* - 1 vs alpha (log-log)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        r"S11-B — Zoom: $r^*(\alpha) - 1$ vs. $\alpha$ nahe $\alpha_{em}$",
        fontsize=12, fontweight='bold'
    )

    alphas_arr = np.array([r[0] for r in rows if r[0] > 0])
    devs_arr   = np.array([r[4] for r in rows if r[0] > 0])
    resid_arr  = np.array([r[5] for r in rows if r[0] > 0])
    a2_arr     = np.array([r[6] for r in rows if r[0] > 0])

    # Panel 1: r* - 1 vs alpha (Linearität prüfen)
    ax = axes[0]
    ax.loglog(alphas_arr, devs_arr, '-o', color='#1f77b4', lw=2, ms=6,
              label=r'$r^* - 1$ (analytisch)')
    ax.loglog(alphas_arr, alphas_arr, '--', color='gray', lw=1.5,
              label=r'$\alpha$ (erste Ordnung)')
    ax.loglog(alphas_arr, a2_arr, ':', color='gray', lw=1.2,
              label=r'$\alpha^2$ (zweite Ordnung)')
    ax.scatter([ALPHA_EM], [1/(1-ALPHA_EM) - 1], color='crimson',
               s=120, marker='*', zorder=5,
               label=rf'$\alpha_{{em}} = 1/137.036$')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$r^* - 1$')
    ax.set_title(r'$r^*(\alpha) - 1 \approx \alpha + \alpha^2 + \ldots$')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # Panel 2: Residuum |r* - 1 - alpha| — soll ~alpha² sein
    ax = axes[1]
    ax.loglog(alphas_arr, resid_arr, '-o', color='#ff7f0e', lw=2, ms=6,
              label=r'$|r^* - 1 - \alpha|$  (Residuum)')
    ax.loglog(alphas_arr, a2_arr, '--', color='#1f77b4', lw=2,
              label=r'$\alpha^2$ (erwartete Ordnung)')
    ax.scatter([ALPHA_EM], [abs(1/(1-ALPHA_EM) - 1 - ALPHA_EM)], color='crimson',
               s=120, marker='*', zorder=5,
               label=rf'$\alpha_{{em}}$: Residuum = {abs(1/(1-ALPHA_EM)-1-ALPHA_EM):.2e}')
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$|r^* - 1 - \alpha|$')
    ax.set_title(r'Residuum $\sim \mathcal{O}(\alpha^2)$ — erwartet')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    png = OUT / "s11b_zoom_alpha_em.png"
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  PNG: {png}")

    return rows

# ─── S11-C: Residuum-Präzisionstest ──────────────────────────────────────────

def run_s11c():
    """
    Präzisionsmessung der FUT-Verbindung bei alpha = alpha_em.

    Analytische Werte:
    - r*(alpha_em) = 1/(1 - alpha_em) = 1/(1 - 1/137.036...) = 137.036/136.036
    - r* - 1       = alpha_em / (1 - alpha_em) = alpha_em * r*
    - Erste Ordnung: r* - 1 ≈ alpha_em
    - Zweite Ordnung: r* - 1 ≈ alpha_em + alpha_em² + ...
    - FUT-Zahl: alpha_em = 1/[(2+√2)(4π²+2/π) + 3/(4π²+2/π)]
    """
    print("\n" + "=" * 60)
    print("S11-C: Präzisionstest bei alpha = alpha_em")
    print("=" * 60)

    r_star   = 1.0 / (1.0 - ALPHA_EM)
    dev      = r_star - 1.0                  # = alpha_em/(1-alpha_em)
    dev_1    = ALPHA_EM                       # erste Ordnung
    dev_2    = ALPHA_EM + ALPHA_EM**2         # zweite Ordnung
    dev_exact = ALPHA_EM / (1.0 - ALPHA_EM)  # exakt = alpha_em * r*

    print(f"\n  1/alpha_em   = {ALPHA_EM_INV:.8f}")
    print(f"  alpha_em     = {ALPHA_EM:.10f}")
    print(f"  r*(alpha_em) = {r_star:.10f}")
    print(f"")
    print(f"  r* - 1 (exakt)         = {dev_exact:.10f}")
    print(f"  r* - 1 (1. Ord.: alpha)= {dev_1:.10f}")
    print(f"  Differenz 1. Ord.      = {abs(dev_exact - dev_1):.4e}  ≈ alpha²={ALPHA_EM**2:.4e}")
    print(f"  r* - 1 (2. Ord.)       = {dev_2:.10f}")
    print(f"  Differenz 2. Ord.      = {abs(dev_exact - dev_2):.4e}  ≈ alpha³={ALPHA_EM**3:.4e}")

    # FUT-Interpretation
    print(f"\n  FUT-Interpretation:")
    print(f"  Der Elektron-Horizont liegt bei r* = 1 + alpha_em = {1 + ALPHA_EM:.8f}")
    print(f"  (exakt: r* = {r_star:.8f}, Abweichung = {abs(r_star - (1+ALPHA_EM)):.2e})")
    print(f"  alpha_em ist die residuale R/F-Asymmetrie des Elektrons.")

    # Simulation bei alpha_em (viele Seeds für Präzision)
    print(f"\n  Simuliere {N_SEEDS} Seeds bei alpha = alpha_em...")
    mu, sd = simulate_attractor(ALPHA_EM, seed_offset=500)
    print(f"  Simulation: r*(alpha_em) = {mu:.7f} ± {sd:.7f}")
    print(f"  Analytisch: r*(alpha_em) = {r_star:.7f}")
    print(f"  Abweichung Sim-Analyt:   = {abs(mu - r_star):.4e}")

    # Zusammenfassungstabelle
    print(f"\n  Zusammenfassung der Ordnungen:")
    print(f"  {'Term':<30}  {'Wert':>14}  {'Relative Fehler':>16}")
    print("  " + "-" * 64)
    terms = [
        ("r* - 1  (exakt)",           dev_exact,  0.0),
        ("alpha_em (1. Ordnung)",      dev_1,      abs(dev_exact-dev_1)/dev_exact),
        ("alpha_em + alpha² (2. Ord)", dev_2,      abs(dev_exact-dev_2)/dev_exact),
        ("Simulation (20 Seeds)",      mu - 1.0,   abs((mu-1.0)-dev_exact)/dev_exact),
    ]
    for name, val, err in terms:
        print(f"  {name:<30}  {val:>14.10f}  {err:>16.4e}" if err > 0 else
              f"  {name:<30}  {val:>14.10f}  {'(Referenz)':>16}")

    # Plot: Illustration der Expansion
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        r"S11-C — Präzisionstest: $r^*(\alpha_{em}) - 1 = \alpha_{em} + \mathcal{O}(\alpha_{em}^2)$" + "\n"
        r"FUT: $\alpha_{em} = 1/[(2+\sqrt{2})(4\pi^2+2/\pi) + 3/(4\pi^2+2/\pi)]$",
        fontsize=11, fontweight='bold'
    )

    # Geometrische Reihe: r* - 1 = sum_{n=1}^{inf} alpha^n
    n_terms = range(1, 12)
    partial_sums = [sum(ALPHA_EM**k for k in range(1, n+1)) for n in n_terms]

    ax.semilogy(list(n_terms), partial_sums, '-o', color='#1f77b4', lw=2, ms=6,
                label=r'$\sum_{k=1}^{n} \alpha_{em}^k$ (geometrische Reihe)')
    ax.axhline(dev_exact, color='crimson', lw=2, ls='--',
               label=rf'Exakter Wert $r^*-1 = {dev_exact:.6f}$')
    ax.axhline(dev_1, color='#ff7f0e', lw=1.5, ls=':',
               label=rf'1. Ordnung $\alpha_{{em}} = {dev_1:.6f}$')

    ax.set_xlabel('Anzahl Terme der Reihe', fontsize=11)
    ax.set_ylabel(r'Partialsumme $\sum \alpha_{em}^k$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    png = OUT / "s11c_precision_test.png"
    plt.savefig(png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  PNG: {png}")

# ─── Report ───────────────────────────────────────────────────────────────────

def write_report():
    r_star = 1.0 / (1.0 - ALPHA_EM)
    dev    = ALPHA_EM / (1.0 - ALPHA_EM)

    report = f"""# S11 Report — Feedback-alpha vs. Feinstrukturkonstante alpha_em
Datum: 30.05.2026 | FUT/CoMath Simulation S11
Autor: Jens Deutschmann · Independent Researcher, Karlsruhe

---

## Kernbefund

Der diskrete Feedback-Parameter alpha der CoMath-Memory-Dynamik ist über die
Attraktor-Formel r* = 1/(1-alpha) direkt mit der FUT-Feinstrukturkonstante verknüpft:

| Größe | Wert |
|-------|------|
| alpha_em (FUT) | {ALPHA_EM:.10f} |
| 1/alpha_em | {ALPHA_EM_INV:.8f} |
| r*(alpha_em) | {r_star:.10f} |
| r* - 1 (exakt) | {dev:.10f} |
| alpha_em (1. Ord.) | {ALPHA_EM:.10f} |
| Residuum |r*-1 - alpha_em| | {abs(dev - ALPHA_EM):.4e} ≈ alpha_em² = {ALPHA_EM**2:.4e} |

**Hauptaussage:**
r*(alpha_em) - 1 = alpha_em + O(alpha_em²)

Die Abweichung des Elektron-Horizonts vom exakten Gleichgewicht r*=1 ist
in erster Ordnung gleich alpha_em. Die Feinstrukturkonstante misst die
residuale R/F-Asymmetrie des Elektrons.

---

## FUT-Verbindung

1/alpha_em = (2+√2)(4π²+2/π) + 3/(4π²+2/π) = {ALPHA_EM_INV:.8f}

Geometrische Reihe:  r* - 1 = sum_{{n=1}}^{{inf}} alpha^n = alpha/(1-alpha)
Bei alpha = alpha_em:  r* - 1 = {dev:.8f}

**Interpretation:**
- r* = 1 (exakt) wäre alpha_eff = 0 — perfekte Kohärenz, kein Feedback
- r* = 1 + alpha_em = {1+ALPHA_EM:.8f} — physikalischer Elektron-Horizont
- alpha_em ist die "minimale residuale Asymmetrie" die mit der beobachteten EM-Kopplung konsistent ist

---

## Offene Frage (S12)

Kann alpha_em aus den CoMath-Axiomen abgeleitet werden, ohne die FUT-Formel
als Input zu verwenden? Dies würde bedeuten: die Simulation "kennt" alpha_em
ohne externe Kalibrierung.

Kandidat: die geometrische Bedingung B4 (dichte J-Abbildung, keine Lücke um theta_c)
könnte alpha_em als kritischen Threshold auszeichnen.
"""

    rp = OUT / "s11_report.md"
    with open(rp, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n  Report: {rp}")

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("S11: alpha_feedback vs. alpha_em — FUT-Verbindungsanalyse")
    print(f"  N={N_NODES} | Seeds={N_SEEDS} | Steps={N_STEPS}")
    print()

    run_s11a()
    run_s11b()
    run_s11c()
    write_report()

    print("\n" + "=" * 60)
    print("S11 ABGESCHLOSSEN")
    print(f"  Ausgaben in: {OUT}/")
    print("=" * 60)
