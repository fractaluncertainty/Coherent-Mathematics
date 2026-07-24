# S9 — Recoherence Geometry Classifier Report
## Purpose
S9 consolidates the three S8 scripts into one classifier for J-geometriefaehigkeit.
It evaluates collapse, fragmentation, overconnection, good proto-geometry, and approximate icosahedrality.
## Direct answers to the six guiding questions
### 1. Minimal properties J must possess
The empirical minimal profile is: monotone, gradual, local, difference-sensitive recoherence order. J must preserve a usable neighborhood ranking: nearer/compatible structures couple stronger than far/incompatible structures.
### 2. Maximal properties J may possess
J must not overcouple. Strong long-range tails can create a globally connected graph that loses differentiated geometry. The acceptable regime is subcritical or near-critical percolation, not total percolation.
### 3. Best J-properties for highest structure
The best candidates are those with high geometry_quality and nontrivial icosahedrality while avoiding FAILED_OVERCONNECTED.
### 4. FUT/CoMath icosahedral closeness
S9 uses an Icosahedrality proxy: closeness to 12 nodes, 30 edges, degree ~5, triangular closure, shell-like density, connectedness, and degree regularity. This is not a proof of an icosahedron; it is a diagnostic for triangulated recoherence shells.
### 5. Are CoMath/FUT formula types used in Python?
S9 explicitly introduces toy wrappers for 0_f, infinity_f, =_f and CoherenceValue. However, Python still computes with classical floats internally. The wrappers prevent conceptual conflation, but they are not yet a full formal implementation of CoMath/FUT variable ontology.
### 6. How to improve qualitative strength by code changes?
Use sweeps over coupling_threshold, finite-size scaling, perturbation robustness, directed geodesics for asymmetric J, real icosahedral graph matching, and explicit CoMath/FUT symbolic type propagation.
## Top geometry-quality candidates
- **anisotropic_exp**: quality=0.996, ico=0.542, edges=40.50, largest=14.80, class=ICO_CANDIDATE
- **power**: quality=0.987, ico=0.602, edges=22.40, largest=11.55, class=ICO_CANDIDATE
- **metric_exp**: quality=0.984, ico=0.504, edges=41.65, largest=14.70, class=ICO_CANDIDATE
- **exp_alpha_1_0**: quality=0.984, ico=0.504, edges=41.65, largest=14.70, class=ICO_CANDIDATE
- **asymmetric**: quality=0.978, ico=0.588, edges=27.20, largest=9.75, class=ICO_CANDIDATE

## Top icosahedrality candidates
- **linf_exp_1_0**: ico=0.633, quality=0.978, edges=25.65, degree=2.76, triangles=12.10
- **power**: ico=0.602, quality=0.987, edges=22.40, degree=2.36, triangles=9.45
- **asymmetric**: ico=0.588, quality=0.978, edges=27.20, degree=2.70, triangles=17.70
- **rational_p2**: ico=0.570, quality=0.923, edges=11.30, degree=1.65, triangles=3.90
- **anisotropic_exp**: ico=0.542, quality=0.996, edges=40.50, degree=2.79, triangles=20.40

## Overconnected candidates
- **rational_p1**: density=0.449, edges=84.65, largest=19.50
- **random_hash**: density=0.667, edges=2.00, largest=3.00

## Failed / weak candidates
- **rational_p1**: class=FAILED_OVERCONNECTED, edges=84.65, largest=19.50
- **exp_alpha_0_5**: class=FAILED_FRAGMENTED, edges=2.95, largest=3.00
- **random_hash**: class=FAILED_OVERCONNECTED, edges=2.00, largest=3.00
- **linear_cutoff_0_35**: class=FAILED_COLLAPSE, edges=0.25, largest=1.25
- **step**: class=FAILED_COLLAPSE, edges=0.10, largest=1.10
- **gaussian_l2_0_5**: class=FAILED_COLLAPSE, edges=0.05, largest=1.05
- **linear_cutoff_0_20**: class=FAILED_COLLAPSE, edges=0.00, largest=1.00
- **cosine_resonance**: class=FAILED_COLLAPSE, edges=0.00, largest=1.00
- **mixed_resonance**: class=FAILED_COLLAPSE, edges=0.00, largest=1.00
- **threshold_resonance**: class=FAILED_COLLAPSE, edges=0.00, largest=1.00
- **random_projection**: class=FAILED_COLLAPSE, edges=0.00, largest=1.00

## Core S9 thesis
Proto-geometry appears when J produces graduated local recoherence order without falling below the connectivity threshold or exceeding the overpercolation threshold. The precise analytic form is secondary; the functional regime is primary.
