# iGSAFT

Simulation code for **“Identification and Inference for Structural Accelerated
Failure Time Models via Instrument Interactions”**.

The code generates one right-censored data set and runs one simulation replicate
using CUE, EL, and ET. The conventional Weibull AFT estimator is included as a
benchmark. 

## Files

- `run_simulation.py`: run one simulation experiment.
- `igsaft_simulation/dgp.py`: data-generating process and four simulation cases.
- `igsaft_simulation/estimator.py`: interaction selection, cross-fitting, GEL
  estimation, standard errors, and overidentification tests.
- `igsaft_simulation/censoring.py`: Kaplan–Meier and AIPCW nuisance estimation.
- `igsaft_simulation/basis.py`: interaction terms and regression utilities.
- `igsaft_simulation/diagnostics.py`: interaction F test and AFT benchmark.
- `igsaft_simulation/config.py`: default DGP and estimator settings.

## Run

Install the required libraries and run the default setting:

```bash
python -m pip install -r requirements.txt
python run_simulation.py
```

The default is one run with `n=10000`, `p=10`, Case 1, and censoring rate 0.2.
Results are printed and saved to `results/single_run.csv`.

Example with different settings:

```bash
python run_simulation.py \
  --n 20000 \
  --p 20 \
  --case 4 \
  --censor-rate 0.6 \
  --active-fraction 0.4 \
  --seed 2025
```

Use `python run_simulation.py --help` to see all options.

## Main parameters

- `--n`: sample size.
- `--p`: number of instrumental variables.
- `--case`: DGP case, from 1 to 4.
- `--censor-rate`: target censoring rate.
- `--beta-true`: true causal effect.
- `--interaction-scale`: scale of the interaction coefficients.
- `--active-fraction`: proportion of nonzero pairwise interactions.
- `--selection`: `lasso`, `truth`, or `all`.
- `--methods`: any combination of `CUE`, `EL`, and `ET`.
- `--seed`: data-generation seed.
- `--split-seed`: sample-splitting seed.
- `--skip-aft`: omit the AFT benchmark.
- `--output`: output CSV file.

## Main functions and variables

- `simulate_data`: generates `Z`, exposure `D`, failure time `T`, censoring time
  `C`, observed time `Y`, and event indicator `delta`.
- `select_interactions`: selects pairwise instrument interactions.
- `fit_aipcw_nuisance`: estimates the censoring and augmentation nuisance
  functions on an auxiliary fold.
- `apply_aipcw_nuisance`: applies those nuisance estimates to the evaluation fold.
- `prepare_igsaft_moments`: constructs the two-fold cross-fitted AIPCW moments.
- `fit_igsaft`: estimates the causal effect `beta_hat` and its standard error.
- `interaction_relevance_test`: performs the robust interaction F test.
- `fit_weibull_aft`: fits the conventional AFT benchmark.

The output contains the estimate, standard error, confidence interval, bias,
selected interaction count, and overidentification-test result for each method.

