# iGSAFT Cox simulations

Simulation code for **“Identification and Inference for Structural Accelerated
Failure Time Models via Instrument Interactions.”**

This repository implements the current manuscript estimator:

1. An independent 20% subsample selects pairwise interactions by adaptive
   lasso.
2. The remaining observations are split into two 40% folds. Each fold supplies
   nuisance estimates for the other fold.
3. Ordinary Cox models with Breslow baseline cumulative hazards estimate the
   censoring survival function and conditional failure-time survival function.
4. Held-out moments from both folds are concatenated. One GEL fit produces the
   CUE, EL, or ET estimate, standard error, and overidentification statistic.

This public directory is a compact, configurable implementation intended for
one-setting runs and code inspection. The controlled-access real data and the
full 500-replication manuscript archive are not included.

## Files

- `run_simulation.py`: configurable single-setting or simulation-grid entry.
- `igsaft_simulation/dgp.py`: manuscript DGP and Cases 1--4.
- `igsaft_simulation/estimator.py`: adaptive lasso, Cox nuisance estimation,
  cross-fitted moments, and GEL inference.
- `igsaft_simulation/diagnostics.py`: interaction-strength F statistic and the
  conventional Weibull AFT benchmark.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Quick run

```bash
python run_simulation.py
```

The default runs one replication with `n=10000`, `p=10`, Case 1, and 20%
censoring. Results are written immediately to `results/replicates.csv`; the
summary is saved to `results/summary.csv`. Each completed replication is
flushed to disk immediately. Re-running the same command resumes incomplete
settings from the existing replicate file.

## Examples

Run 500 replications of the four primary cases and censoring rates:

```bash
python run_simulation.py \
  --n 10000 \
  --p 10 \
  --case 1 2 3 4 \
  --censor-rate 0.2 0.4 0.6 0.8 \
  --replications 500 \
  --workers 16
```

Run the supplementary settings:

```bash
python run_simulation.py --n 10000 --p 20 --case 1 2 3 4 \
  --censor-rate 0.2 0.4 0.6 0.8 --replications 500 --workers 16

python run_simulation.py --n 20000 --p 10 --case 1 2 3 4 \
  --censor-rate 0.2 --replications 500 --workers 16 \
  --output-dir results_n20000
```

Use `python run_simulation.py --help` for all options. Censoring rates may be
written as proportions or percentages.

## Output

The replicate file reports the estimate, standard error, 95% confidence
interval, coverage indicator, selected-interaction count,
`F_iGSAFT`, and the GEL overidentification statistic and p-value. Numerical
failures are recorded in the `status` and `error` columns without stopping the
remaining replications.
