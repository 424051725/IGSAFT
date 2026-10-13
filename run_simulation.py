import argparse
import csv
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("IGSAFT_LASSO_JOBS", "1")

import numpy as np
import pandas as pd

from igsaft_simulation.dgp import DGPConfig, simulate_data
from igsaft_simulation.diagnostics import fit_weibull_aft, interaction_f_statistic
from igsaft_simulation.estimator import fit_igsaft


def deterministic_seed(master_seed, n, p, case, censor_rate, replication):
    payload = (
        f"igsaft-cox|{master_seed}|{n}|{p}|{case}|"
        f"{censor_rate:.12g}|{replication}"
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=16, person=b"iGSAFT-Cox").digest()
    return int.from_bytes(digest[:4], "little")


def run_replication(job):
    n, p, case, censor_rate, replication, args = job
    seed = deterministic_seed(args.master_seed, n, p, case, censor_rate, replication)
    config = DGPConfig(
        n=n, p=p, case=case, censor_rate=censor_rate,
        beta_true=args.beta_true, interaction_scale=args.interaction_scale,
        censor_width=args.censor_width, seed=seed,
    )
    base = {
        "n": n, "p": p, "case": case, "censor_rate": censor_rate,
        "replication": replication, "seed": seed,
    }
    try:
        data = simulate_data(config)
        results, prepared = fit_igsaft(
            data["Y"], data["D"], data["Z"], data["delta"],
            methods=args.methods, split_seed=seed ^ 0xA5A5A5A5,
            selection=args.selection, active=data["active_interactions"],
            alpha_scale=args.lasso_alpha_scale,
        )
        inference = prepared["inference_indices"]
        f_value = interaction_f_statistic(
            data["D"][inference], data["Z"][inference], prepared["I"]
        )
        common = {
            **base, "observed_censoring": data["observed_censoring"],
            "F_iGSAFT": f_value, "status": "ok", "error": "",
        }
        rows = []
        for result in results:
            rows.append({
                **common, "method": result.method,
                "beta_hat": result.beta_hat,
                "standard_error": result.standard_error,
                "ci_lower": result.ci_lower, "ci_upper": result.ci_upper,
                "coverage": int(result.ci_lower <= args.beta_true <= result.ci_upper),
                "selected_interactions": result.selected_interactions,
                "overid_statistic": result.overid_statistic,
                "overid_df": result.overid_df,
                "overid_pvalue": result.overid_pvalue,
            })
        if not args.skip_aft:
            try:
                aft = fit_weibull_aft(data["Y"], data["delta"], data["D"])
                rows.append({
                    **common, "method": "AFT", "beta_hat": aft.beta_hat,
                    "standard_error": aft.standard_error,
                    "ci_lower": aft.beta_hat - 1.96 * aft.standard_error,
                    "ci_upper": aft.beta_hat + 1.96 * aft.standard_error,
                    "coverage": int(abs(aft.beta_hat - args.beta_true)
                                    <= 1.96 * aft.standard_error),
                    "selected_interactions": np.nan,
                    "overid_statistic": np.nan, "overid_df": np.nan,
                    "overid_pvalue": np.nan,
                })
            except Exception as error:
                rows.append({
                    **common, "method": "AFT", "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                })
        return rows
    except Exception as error:
        methods = list(args.methods) + ([] if args.skip_aft else ["AFT"])
        return [{
            **base, "method": method, "status": "error",
            "error": f"{type(error).__name__}: {error}",
        } for method in methods]


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "n", "p", "case", "censor_rate", "replication", "seed", "method",
        "beta_hat", "standard_error", "ci_lower", "ci_upper", "coverage",
        "selected_interactions", "F_iGSAFT", "observed_censoring",
        "overid_statistic", "overid_df", "overid_pvalue", "status", "error",
    ]
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def summarize(raw_path, output_path, beta_true):
    frame = pd.read_csv(raw_path)
    frame = frame.drop_duplicates(
        ["n", "p", "case", "censor_rate", "replication", "method"],
        keep="last",
    )
    valid = frame[(frame.status == "ok") & frame.beta_hat.notna()].copy()
    valid["bias"] = 100 * (valid.beta_hat - beta_true)
    summary = (
        valid.groupby(["n", "p", "case", "censor_rate", "method"])
        .agg(
            Bias_percent=("bias", "mean"), SD=("beta_hat", "std"),
            SE=("standard_error", "mean"), CP=("coverage", "mean"),
            F_iGSAFT=("F_iGSAFT", "mean"),
            selected_interactions=("selected_interactions", "mean"),
            overid_rejection=("overid_pvalue", lambda x: np.mean(x.dropna() < 0.05)),
            successful_replications=("replication", "nunique"),
        ).reset_index()
    )
    summary.to_csv(output_path, index=False)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Cox-based iGSAFT simulation experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, nargs="+", default=[10_000])
    parser.add_argument("--p", type=int, nargs="+", default=[10])
    parser.add_argument("--case", type=int, nargs="+", choices=range(1, 5), default=[1])
    parser.add_argument("--censor-rate", type=float, nargs="+", default=[0.20])
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=["CUE", "EL", "ET"],
                        default=["CUE", "EL", "ET"])
    parser.add_argument("--selection", choices=["lasso", "truth", "all"],
                        default="lasso")
    parser.add_argument("--lasso-alpha-scale", type=float, default=0.5)
    parser.add_argument("--interaction-scale", type=float, default=0.4)
    parser.add_argument("--censor-width", type=float, default=110.0)
    parser.add_argument("--beta-true", type=float, default=1.0)
    parser.add_argument("--master-seed", type=int, default=2025)
    parser.add_argument("--skip-aft", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    rates = [rate / 100 if rate > 1 else rate for rate in args.censor_rate]
    jobs = [
        (n, p, case, rate, replication, args)
        for n in args.n for p in args.p for case in args.case for rate in rates
        for replication in range(args.replications)
    ]
    raw = args.output_dir / "replicates.csv"
    if raw.exists():
        previous = pd.read_csv(raw, on_bad_lines="skip")
        expected = len(args.methods) + (0 if args.skip_aft else 1)
        counts = previous.groupby(
            ["n", "p", "case", "censor_rate", "replication"]
        ).method.nunique()
        complete = set(counts[counts >= expected].index)
        jobs = [job for job in jobs if job[:5] not in complete]
    if not jobs:
        summary = summarize(raw, args.output_dir / "summary.csv", args.beta_true)
        print(summary.to_string(index=False))
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_replication, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            append_rows(raw, future.result())
            print(f"completed {completed}/{len(jobs)}", flush=True)
    summary = summarize(raw, args.output_dir / "summary.csv", args.beta_true)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
