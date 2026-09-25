# Research configuration entry points

For new experiments use `run_experiment.py` and the single
[run guide](../docs/Running%20Experiments.md).

| CLI model | Current baseline config | Research settings |
| --- | --- | --- |
| `pcr` | `gkx_pcr_full_v5.yaml` | Original complete-array PCR, K=0–100 |
| `pls` | `gkx_pls_p05_v1.yaml` | Original complete-array PLS, K=0–10 |

The CLI explicitly selects `--sample p05/full` and `--years`; a filename's
historical sample label does not override those choices. It saves the resolved
settings with each experiment. CPU, memory and concurrency options belong to
submission records, not research model configs. New yearly configs are generated
only inside experiment result directories.

`gkx_pcr_scrp.yaml` supplies the existing SCRP results location. Other configs
are retained for earlier pilots, memory checks, synthetic examples or existing
run/resume commands. Do not delete or rename them while those workflows are in
use. Create a new versioned research config only for a substantive specification
change; select it using `--config NAME`.
