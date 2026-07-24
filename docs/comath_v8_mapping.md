# Mapping of experiments to CoMath V8

This map connects the numerical sequence to the current conceptual organization of CoMath V8. Because chapter numbering may still move during editorial consolidation, the conceptual anchor and symbol chain are authoritative; chapter numbers are editorial cross-references to the July 2026 manuscript state.

| Code | Experiment | CoMath V8 anchor | Current chapter relation | Status |
|---|---|---|---|---|
| `simulation.py` | Minimal continuation/recoherence toy model | recursive continuation, compatibility, persistence | Early motivation and continuation dynamics | Illustrative baseline |
| `s0_recoherence_simulation_full.py` | Full baseline dynamics | \(\Delta_f \to \Omega_f\), continuation and persistence | Ch. 9 transition from incomplete self-coincidence to recursive openness | Exploratory |
| `s0_recoherence_cluster_proto_metric.py` | Cluster proto-metric | persistence, recoherence cost, local proximity | Emergence-of-structures insertion after Ch. 9; later geometry bridge | Exploratory |
| `s1_recoherence_visualization.py` | Visual diagnostics | partial visibility and horizon-relative observation | Ch. 3 knowledge windows; local manifestation of partial visibility | Illustrative diagnostic |
| `s2_recoherence_hierarchy.py` | Threshold hierarchy | nested possibility corridors and organization levels | Emergence of structures; \(\kappa_f\), \(C_f\), \(P_f\) transition | Exploratory |
| `s3_recoherence_multiseed_hierarchy.py` | Multi-seed hierarchy | persistence under varied initial conditions | Same emergence transition; falsifiability/robustness material | Robustness test |
| `s4_temporal_cluster_identity.py` | Temporal cluster identity | persistence \(P_f\) before identity \(I_f\) | Identity-emergence chapters following persistence | Constructive test |
| `s5_self_stabilizing_recoherence_identity.py` | History-stabilized identity | past corridors as guide rails for future corridors | Persistence/identity transition; no primitive memory required | Constructive test |
| `s6_coupled_identity_networks.py` | Coupled identity networks | combinable continuations and higher organization | Emergence of structures; relation \(\bowtie_f\) and identity \(I_f\) | Exploratory |
| `s7_emergent_track_proto_geometry.py` | Track-induced proto-geometry | proximity as local recoherence cost | Geometry bridge; downstream of stable structure and relation | Research bridge |
| `s8_j_agnosticity_proto_geometry.py` | Coupling pluralism | non-uniqueness of admissible mathematics/geometry | Horizon-relative mathematics and multiple compatible structures | Ablation study |
| `s8_j_ablation_coupling_pluralism.py` | \(J\)-functional ablation | dependence on observer/model choice | Same pluralism question; tests non-privileged coupling choice | Ablation study |
| `s8_1_minimal_j_geometry_candidates.py` | Minimal \(J\) candidates | minimal sufficient structure | Geometry bridge; candidate selection without ontological uniqueness | Candidate screen |
| `s9_recoherence_geometry_classifier.py` | Geometry property classifier | explicit criteria for emergent geometry | Geometry bridge and falsifiability conditions | Classifier |
| `s9_minimal_j_theory_comath.py` | Minimal-\(J\) theory comparison | recoherence-derived proximity and geometry | Research material beyond the frozen V8 mathematical core | Research bridge |
| `s10_l_inf_rf_stabilization.py` | \(L_\infty\)/\(R_f\) stabilization | limiting behavior and stability | FUT-facing continuation of the geometry experiments | FUT research, not V8 core |
| `s11_alpha_feedback_vs_alpha_em.py` | Feedback comparison with \(\alpha_{\rm em}\) | speculative physics bridge | FUT only; excluded from the V8 book's mathematical endpoint | Conjectural physics test |
| `s12_rf_bifurcation_universality.py` | \(R_f\) bifurcation universality | sensitivity of the S11 bridge | FUT only; independent robustness check | Conjectural robustness test |

## Relation to the V8 emergence spine

The simulations do not implement the entire ontological spine. Their main operational range is:

\[
\Omega_f \longrightarrow \rho_f \longrightarrow \delta_f
\longrightarrow \kappa_f \longrightarrow C_f \longrightarrow P_f
\longrightarrow I_f,
\]

followed by exploratory relations and proto-geometric structures. The earlier ontological status of \(\Delta_f\), and the later emergence of equality, number, measure, and logic, cannot be established by these finite programs.

## Manuscript citation convention

Recommended wording in CoMath V8:

> An executable toy-model test of this mechanism is provided in experiment Sx of the accompanying Coherent Mathematics code repository. The experiment demonstrates computational possibility under explicit assumptions; it is not used as a proof of the ontological claim.

