"""Run exactly one configurable simulation replicate."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from igsaft_simulation.config import DGPConfig, EstimatorConfig
from igsaft_simulation.dgp import simulate_data
from igsaft_simulation.diagnostics import fit_weibull_aft, interaction_relevance_test
from igsaft_simulation.estimator import fit_prepared_igsaft, prepare_igsaft_moments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one iGSAFT simulation setting once.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    dgp = parser.add_argument_group("data-generating process")
    dgp.add_argument("--n", type=int, default=10_000, help="sample size")
    dgp.add_argument("--p", type=int, default=10, help="number of candidate instruments")
    dgp.add_argument("--case", type=int, choices=range(1, 5), default=1, help="paper DGP case")
    dgp.add_argument("--censor-rate", type=float, default=0.20, help="target censoring rate")
    dgp.add_argument("--beta-true", type=float, default=1.0, help="true causal effect")
    dgp.add_argument("--interaction-scale", type=float, default=4.0, help="c in c*n^(-1/4)")
    dgp.add_argument(
        "--active-fraction",
        type=float,
        default=None,
        help="fraction of nonzero pairwise interactions; default is 1 for p=10 and 0.4 otherwise",
    )
    dgp.add_argument("--seed", type=int, default=240, help="DGP random seed")

    estimation = parser.add_argument_group("estimation")
    estimation.add_argument(
        "--methods", nargs="+", choices=["CUE", "EL", "ET"], default=["CUE", "EL", "ET"]
    )
    estimation.add_argument(
        "--selection", choices=["lasso", "truth", "all"], default="lasso",
        help="interaction-selection rule",
    )
    estimation.add_argument("--split-seed", type=int, default=1, help="two-fold split seed")
    estimation.add_argument("--lasso-jobs", type=int, default=1, help="parallel LASSO CV jobs")
    estimation.add_argument("--skip-aft", action="store_true", help="omit the naive AFT benchmark")
    parser.add_argument(
        "--output", type=Path, default=Path("results/single_run.csv"), help="result CSV path"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_fraction = args.active_fraction
    if active_fraction is None:
        active_fraction = 1.0 if args.p == 10 else 0.4

    dgp_config = DGPConfig(
        n=args.n,
        p=args.p,
        beta_true=args.beta_true,
        scenario=args.case,
        censor_rate=args.censor_rate,
        interaction_scale=args.interaction_scale,
        active_interaction_fraction=active_fraction,
        seed=args.seed,
    )
    estimator_config = EstimatorConfig(
        selection=args.selection,
        split_seed=args.split_seed,
        lasso_jobs=args.lasso_jobs,
    )
    data = simulate_data(dgp_config)
    observed_censoring = float(1.0 - data.event.mean())
    f_test = interaction_relevance_test(data.exposure, data.instruments)
    prepared = prepare_igsaft_moments(
        data.observed_time,
        data.exposure,
        data.instruments,
        data.event,
        estimator_config,
        data.active_interactions,
    )
    run_metadata = {
        "n": args.n,
        "p": args.p,
        "case": args.case,
        "target_censor_rate": args.censor_rate,
        "observed_censor_rate": observed_censoring,
        "beta_true": args.beta_true,
        "dgp_seed": args.seed,
        "split_seed": args.split_seed,
        "selection": args.selection,
        "active_interaction_fraction": active_fraction,
        "interaction_relevance_F": f_test.statistic,
        "interaction_relevance_pvalue": f_test.pvalue,
    }

    rows: list[dict[str, object]] = []
    for method in args.methods:
        result = fit_prepared_igsaft(
            prepared, data.observed_time.size, method, estimator_config
        )
        rows.append(
            {
                **run_metadata,
                "method": method,
                "beta_hat": result.beta_hat,
                "standard_error": result.standard_error,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "covers_beta_true": result.ci_lower <= args.beta_true <= result.ci_upper,
                "bias": result.beta_hat - args.beta_true,
                "selected_interactions": result.selected_interaction_count,
                "overid_statistic": result.overid_statistic,
                "overid_df": result.overid_df,
                "overid_pvalue": result.overid_pvalue,
            }
        )

    if not args.skip_aft:
        try:
            aft = fit_weibull_aft(data.observed_time, data.event, data.exposure)
        except Exception as error:
            print(f"Warning: AFT benchmark failed to converge: {error.__class__.__name__}")
            aft = None
        rows.append(
            {
                **run_metadata,
                "method": "AFT",
                "beta_hat": aft.beta_hat if aft else None,
                "standard_error": aft.standard_error if aft else None,
                "ci_lower": aft.ci_lower if aft else None,
                "ci_upper": aft.ci_upper if aft else None,
                "covers_beta_true": (
                    aft.ci_lower <= args.beta_true <= aft.ci_upper if aft else None
                ),
                "bias": aft.beta_hat - args.beta_true if aft else None,
                "selected_interactions": None,
                "overid_statistic": None,
                "overid_df": None,
                "overid_pvalue": None,
            }
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)

    print("DGP configuration:")
    print(json.dumps(asdict(dgp_config), indent=2))
    print(f"Observed censoring rate: {observed_censoring:.4f}")
    print(
        f"Interaction relevance test: F={f_test.statistic:.4f}, "
        f"df=({f_test.numerator_df}, {f_test.denominator_df}), p={f_test.pvalue:.4g}"
    )
    print("\nEstimation results:")
    print(frame.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
