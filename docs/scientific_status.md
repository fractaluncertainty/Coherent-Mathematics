# Scientific status and interpretation

## What the repository establishes

Each experiment is a constructive computational test. It can show that a declared rule set, parameter region, and random initialization produce a particular pattern. Multi-seed experiments additionally test whether that observation survives a specified sample of initial conditions.

The experiments are useful for:

- exposing hidden assumptions in verbal formulations;
- finding counterexamples and unstable parameter regions;
- comparing alternative coupling functionals;
- formulating falsifiable follow-up questions;
- checking whether a proposed emergence chain is computationally coherent.

## What the repository does not establish

The simulations do not by themselves prove:

- that the implemented rules are uniquely implied by CoMath;
- that reality follows the simulated dynamics;
- that a robust numerical regularity is a theorem;
- that a fitted or recovered number is derived from first principles;
- that a finite-seed test establishes universal behavior;
- that similarity to geometry establishes a unique metric geometry.

This distinction is particularly important for S10–S12. Those programs are exploratory bridges and sensitivity tests. They must not be cited as derivations of a physical constant.

## Epistemic classification

| Label | Meaning in this repository |
|---|---|
| Definition | A rule explicitly introduced by the program |
| Assumption | A parameter, functional form, threshold, or initialization chosen before the run |
| Observation | A value or pattern computed from a run |
| Robustness result | An observation reproduced over the declared seeds or parameter sweep |
| Interpretation | A proposed connection between an observation and CoMath terminology |
| Conjecture | A claim requiring proof, independent derivation, or broader testing |

## Relation to partial visibility

Every run exposes only a finite, parameterized observation window. The output is therefore an instance of partial visibility: it reports what becomes visible under the chosen state representation, thresholds, duration, statistics, and plots. It does not provide a total view of the model's full state space, still less of the ontology the model is intended to explore.

