# Gu, Kelly, and Xiu (2020) Replication
Research project to replicate Empirical Asset Pricing via Machine Learning by Gu, Kelly, and Xiu (2020).
The project prioritizes correctness, reproducibility, and readable research code.

---
## Current Status
Initial project setup and structure design. The replication workflow is under development.
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

Run the configured models on a saved dataset:
```bash
python main.py --config model_demo
python main.py --config model_demo_nonlinear
```

Both synthetic checks and real replication use this entry point. `DATA.DATASET`
selects the prepared panel; `RUN` sets models, sample periods, evaluation, and
optional output saving. The example configs use synthetic data. The real `gkx`
panel and its empirical config still need to be prepared.

See [run configuration](docs/working_notes/Experiment%20Configuration.md) for
dataset preparation, config settings, and saved outputs.

---

## Notes

(Optional)
