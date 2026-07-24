# Coherent Mathematics — Numerical Experiments

Reference implementations and reproducible numerical experiments accompanying the development of **CoMath (Coherent Mathematics)** by Jens Deutschmann.

CoMath studies how stable mathematical structure may emerge from recursive openness, continuation, compatibility, persistence, and recoherence. This repository contains the exploratory simulation sequence **S0–S12**, developed to test whether specific qualitative mechanisms can occur in explicit computational toy models.

> [!IMPORTANT]
> These programs are numerical experiments, not proofs of the CoMath ontology or derivations of physical constants. A positive simulation result establishes only that the implemented mechanism can produce the reported behavior under the stated assumptions and parameters.

## Repository contents

| Series | Main question | Principal output |
|---|---|---|
| S0 | Can local compatibility and recoherence yield stable clusters? | Baseline dynamics and proto-metric |
| S1 | How can the baseline dynamics be inspected visually? | State, cluster, history, and distance plots |
| S2 | Does structure persist across thresholds? | Threshold hierarchy |
| S3 | Is the hierarchy robust across random seeds? | Multi-seed statistics |
| S4 | Can clusters retain temporal identity? | Identity tracks and lifetimes |
| S5 | Can retained history stabilize identity? | Memory-dependent persistence |
| S6 | Can persistent identities form coupled networks? | Components, degrees, and edges |
| S7 | Can those networks induce a proto-geometry? | Direct and geodesic distance structures |
| S8 | How dependent is the geometry on the chosen coupling functional \(J\)? | Ablation and candidate comparison |
| S9 | Which candidate geometries satisfy the declared criteria? | Classifier and minimal-\(J\) analysis |
| S10 | Does the selected dynamics stabilize under an \(L_\infty\)-style test? | Spectral and trajectory diagnostics |
| S11 | What happens in an exploratory feedback comparison involving \(\alpha_{\rm em}\)? | Bifurcation and precision plots |
| S12 | Are the reported bifurcation features robust across \(R_f\) variants? | Universality sweep and error heatmap |

The complete experiment-to-manuscript map is in [`docs/comath_v8_mapping.md`](docs/comath_v8_mapping.md). Methodological limits are documented in [`docs/scientific_status.md`](docs/scientific_status.md).

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/fractaluncertainty/Coherent-Mathematics.git
cd Coherent-Mathematics
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the experiments

Run scripts from the repository root. Most early scripts use fixed seeds and create their own output directories relative to the current working directory.

```bash
python experiments/s0_recoherence_simulation_full.py
python experiments/s1_recoherence_visualization.py
python experiments/s9_recoherence_geometry_classifier.py --help
```

Some later experiments are computationally heavier because they perform multi-seed or parameter sweeps. Begin with the S0 baseline or use the S9 classifier's command-line options to reduce the number of runs.

Reference outputs produced during development are retained under [`results/reference/`](results/reference/). Newly generated result directories are ignored by Git unless deliberately added.

## Reproducibility

- Random seeds are set explicitly in the experiment series.
- Source files are syntax-checked in the test suite.
- Published reference CSV and image outputs are kept separate from source code.
- The repository records the minimal runtime dependencies.
- No external data or network access is required to execute the experiments.

See [`docs/reproducibility.md`](docs/reproducibility.md) for the recommended verification sequence and known limitations.

## Citation

If you use this code, cite the repository using [`CITATION.cff`](CITATION.cff). A release DOI can be added after the first archived GitHub/Zenodo release.

## Authorship and implementation history

The CoMath framework, experiment design, interpretation, and publication responsibility belong to **Jens Deutschmann**. Individual scripts were developed with implementation assistance from generative AI systems, including Claude and OpenAI models. Such assistance is disclosed in source headers and [`NOTICE.md`](NOTICE.md); all scientific claims and responsibility remain with the human author.

## License

Code and repository documentation are released under the [MIT License](LICENSE). The CoMath books, manuscripts, terminology, figures, and theory are not relicensed by this repository unless explicitly stated.

