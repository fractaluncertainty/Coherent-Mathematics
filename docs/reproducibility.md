# Reproducibility guide

## Environment

- Python 3.10 or newer
- NumPy and Matplotlib versions constrained by `requirements.txt`
- No network access or external dataset required

For stricter archival reproduction, record the output of:

```bash
python --version
python -m pip freeze
```

## Verification sequence

1. Create an isolated virtual environment and install the dependencies.
2. Run `python -m pytest` after installing `pytest` as a development tool.
3. Execute the lightweight baseline `experiments/simulation.py`.
4. Execute S0 and compare the textual summary qualitatively.
5. Run S1–S9 in order because the experiment design becomes progressively richer.
6. Run S10–S12 separately and interpret them under the stronger limitations stated in `scientific_status.md`.
7. Compare generated CSV tables before comparing rendered PNG files; plotting-library versions can cause harmless pixel-level differences.

## Seeds and determinism

The scripts explicitly initialize Python's pseudorandom generator; scripts using NumPy also initialize NumPy where required. Repeated runs in the same supported environment should reproduce the tabular results. Floating-point values and image rendering may vary slightly across platforms or library versions.

## Reference outputs

`results/reference/` contains curated outputs from the supplied development archive. They are evidence of prior runs, not golden files guaranteed to match byte-for-byte on all systems.

## Known limitations

- Several programs are intentionally standalone and repeat shared simulation code.
- Default parameter sweeps can be computationally expensive.
- Most early scripts use constants in the source rather than a shared configuration format.
- Formal unit tests of individual mathematical functions remain future work.
- The current integrity test validates source parsing and executable entry points; it does not rerun the full experimental series in continuous integration.

