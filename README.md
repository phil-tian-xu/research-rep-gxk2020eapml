# Gu, Kelly, and Xiu (2020) Replication
Research project to replicate Empirical Asset Pricing via Machine Learning by Gu, Kelly, and Xiu (2020).
The project prioritizes correctness, reproducibility, and readable research code.

---
## Current Status
PCR and PLS workflows support local 5% checks and full-universe SCRP runs.
Replication results and execution performance are being validated.
---

## Structure
- `config/`: Configuration files and settings
- `core/`: Main logic and implementation
- `data/`: Input data
- `results/`: Outputs and generated results
- `utils/`: General helpers independent of the research
- `materials/`: Local exploratory files and scratch work
- `docs/`: Original paper and research documentation
- `main.py`: Entry point 

---

## Environment
The project uses Python 3.12 and uv.
Install the project environment:
```shell
uv sync
```
Run Python within the environment:
```shell
uv run python
```

## Usage

For new PCR/PLS replication experiments use the unified command:

```bash
python -B run_experiment.py pcr --local --sample p05 --years 1987:1988
python -B run_experiment.py list
```

On SCRP, from the repository with its environment active:

```bash
python -B run_experiment.py pcr --scrp --sample full --years 1987:2016 --parallel 4 --cpus 8 --mem 128G
```

See [Running Experiments](docs/Running%20Experiments.md) for PLS, results locations,
resource settings, submission history, resume and collection. These commands use
the original complete-array models. Existing running jobs retain their original
commands and output directories.

Run the configured models on a saved dataset:
```bash
python main.py --config model_demo
python main.py --config model_demo_nonlinear
```

Both synthetic checks and real replication use this underlying entry point.
`DATA.DATASET` selects the prepared panel; `RUN` sets models, sample periods,
evaluation and output saving. The example configs above use synthetic data;
the unified launcher uses the prepared GKX Parquet datasets.

See [run configuration](docs/working_notes/Experiment%20Configuration.md) for
dataset preparation, config settings, and saved outputs.

---

## Notes

(Optional)
