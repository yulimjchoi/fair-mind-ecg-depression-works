#!/usr/bin/env python3
"""
Weber spectral proxy coefficient calculation.

This script reproduces the validation flow for deriving LF, HF, LFnu, and TP
proxy equations from published Weber summary statistics.

Source values:
- R_xx and r_xy: manually extracted from Weber Fig. 1.
- Cluster N, mean, SD: manually transcribed from Weber Table 2.

Variable order is fixed everywhere:
BPM, pNN20, pNN50, MAD, RMSSD, BR
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PREDICTORS = ["BPM", "pNN20", "pNN50", "MAD", "RMSSD", "BR"]
TARGETS = ["LF", "HF", "LFnu", "TP"]


# [Step 1 note] R_xx is the predictor-predictor correlation matrix from Weber
# Fig. 1. The row/column order must stay identical to PREDICTORS. This matrix is
# inverted so the target correlations can be converted into standardized
# multiple-regression coefficients.
R_XX = [
    [1.00, -0.40, -0.28, -0.23, -0.22, 0.02],
    [-0.40, 1.00, 0.69, 0.66, 0.64, 0.07],
    [-0.28, 0.69, 1.00, 0.59, 0.70, 0.02],
    [-0.23, 0.66, 0.59, 1.00, 0.54, -0.11],
    [-0.22, 0.64, 0.70, 0.54, 1.00, 0.03],
    [0.02, 0.07, 0.02, -0.11, 0.03, 1.00],
]


# [Step 1 note] r_xy contains the predictor-target correlations from Weber
# Fig. 1. Each vector is aligned to PREDICTORS and is used with inv(R_xx).
R_XY = {
    "LF": [-0.11, 0.41, 0.41, 0.55, 0.40, -0.19],
    "HF": [-0.10, 0.50, 0.55, 0.43, 0.66, 0.06],
    "LFnu": [0.03, -0.24, -0.16, 0.10, -0.19, -0.41],
    "TP": [-0.12, 0.52, 0.57, 0.58, 0.60, -0.07],
}


# [Step 3 note] Table 2 provides cluster-level N, mean, and SD. These values
# are pooled to reconstruct Weber-wide means and SDs for predictors and targets.
# pNN20/pNN50 are proportions, and BR is frequency-scale breathing rate.
CLUSTERS = {
    "MP": {
        "n": 3731,
        "BPM": (56.5, 8.3),
        "RMSSD": (45.0, 20.5),
        "pNN20": (0.6, 0.1),
        "pNN50": (0.2, 0.1),
        "MAD": (27.6, 10.1),
        "BR": (0.20, 0.06),
        "LF": (459.8, 445.1),
        "HF": (449.9, 340.7),
        "TP": (896.6, 565.1),
        "LFnu": (45.0, 25.7),
    },
    "HRS": {
        "n": 5158,
        "BPM": (64.0, 9.9),
        "RMSSD": (18.1, 9.8),
        "pNN20": (0.2, 0.2),
        "pNN50": (0.02, 0.03),
        "MAD": (16.2, 7.4),
        "BR": (0.15, 0.04),
        "LF": (274.4, 309.4),
        "HF": (77.7, 97.2),
        "TP": (358.4, 374.1),
        "LFnu": (74.0, 15.7),
    },
    "LRS": {
        "n": 5339,
        "BPM": (63.8, 9.5),
        "RMSSD": (19.4, 10.9),
        "pNN20": (0.3, 0.2),
        "pNN50": (0.02, 0.04),
        "MAD": (12.6, 5.2),
        "BR": (0.23, 0.05),
        "LF": (72.2, 95.0),
        "HF": (127.2, 143.9),
        "TP": (200.2, 208.2),
        "LFnu": (35.5, 20.2),
    },
    "HP": {
        "n": 1540,
        "BPM": (59.2, 12.0),
        "RMSSD": (76.5, 29.9),
        "pNN20": (0.7, 0.2),
        "pNN50": (0.3, 0.2),
        "MAD": (39.4, 15.4),
        "BR": (0.19, 0.06),
        "LF": (1425.0, 964.6),
        "HF": (1592.8, 1042.7),
        "TP": (3204.4, 1239.9),
        "LFnu": (49.0, 28.3),
    },
}


def dot(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(left, right))


def mat_vec(matrix: List[List[float]], vector: List[float]) -> List[float]:
    return [dot(row, vector) for row in matrix]


def invert_matrix(matrix: List[List[float]]) -> List[List[float]]:
    """Invert a square matrix with Gauss-Jordan elimination."""
    n = len(matrix)
    augmented = [
        list(row) + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]

    for col in range(n):
        pivot = max(range(col, n), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("R_xx is singular or numerically unstable.")

        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [value / pivot_value for value in augmented[col]]

        for row in range(n):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [
                current - factor * pivot_current
                for current, pivot_current in zip(augmented[row], augmented[col])
            ]

    return [row[n:] for row in augmented]


def pooled_mean_sd(variable: str) -> Tuple[float, float]:
    """Pool cluster means and SDs using within-cluster and between-cluster SS."""
    total_n = sum(cluster["n"] for cluster in CLUSTERS.values())
    pooled_mean = (
        sum(cluster["n"] * cluster[variable][0] for cluster in CLUSTERS.values())
        / total_n
    )

    within_ss = sum(
        (cluster["n"] - 1) * cluster[variable][1] ** 2
        for cluster in CLUSTERS.values()
    )
    between_ss = sum(
        cluster["n"] * (cluster[variable][0] - pooled_mean) ** 2
        for cluster in CLUSTERS.values()
    )
    pooled_sd = ((within_ss + between_ss) / (total_n - 1)) ** 0.5
    return pooled_mean, pooled_sd


def clip_proxy(target: str, value: float) -> float:
    """Apply physical constraints after a proxy value has been predicted."""
    if target == "LFnu":
        return min(100.0, max(0.0, value))
    return max(0.0, value)


def calculate() -> Dict[str, object]:
    # [Step 1 calculation] Invert R_xx after preserving the fixed predictor
    # order. The inverse is the linear algebra bridge from correlations to
    # standardized regression coefficients.
    r_xx_inv = invert_matrix(R_XX)

    # [Step 2 calculation] Compute standardized beta:
    # beta_std = inv(R_xx) * r_xy.
    beta_std = {target: mat_vec(r_xx_inv, R_XY[target]) for target in TARGETS}

    # [Step 3 calculation] Reconstruct Weber-wide pooled means and SDs from
    # cluster N/mean/SD. Raw beta uses sigma_y / sigma_x_j.
    pooled = {
        variable: pooled_mean_sd(variable)
        for variable in [*PREDICTORS, *TARGETS]
    }
    beta_raw = {
        target: [
            beta_std[target][idx]
            * pooled[target][1]
            / pooled[predictor][1]
            for idx, predictor in enumerate(PREDICTORS)
        ]
        for target in TARGETS
    }

    # [Step 4 calculation] Convert the raw beta vector into a usable raw-scale
    # equation by anchoring it at Weber's pooled predictor and target means.
    # intercept = mu_y - sum(beta_raw_j * mu_x_j).
    intercept = {
        target: pooled[target][0]
        - dot(beta_raw[target], [pooled[predictor][0] for predictor in PREDICTORS])
        for target in TARGETS
    }

    # [Step 5 calculation] Internal reproduction check:
    # R^2 = r_xy' * beta_std in standardized multiple regression.
    r_squared = {
        target: dot(R_XY[target], beta_std[target])
        for target in TARGETS
    }

    # [Step 6 calculation] Apply the raw equations to Table 2 cluster centroids,
    # then apply physical constraints to the predicted proxy values. Coefficients
    # are not clipped; only predicted LF/HF/LFnu/TP proxy values are clipped.
    # The LFnu validation order is checked after clipping.
    cluster_predictions_raw = {}
    cluster_predictions_clipped = {}
    cluster_clipping_flags = {}
    for cluster_name, cluster in CLUSTERS.items():
        x = [cluster[predictor][0] for predictor in PREDICTORS]
        raw_values = {
            target: intercept[target] + dot(beta_raw[target], x)
            for target in TARGETS
        }
        clipped_values = {
            target: clip_proxy(target, value)
            for target, value in raw_values.items()
        }
        cluster_predictions_raw[cluster_name] = raw_values
        cluster_predictions_clipped[cluster_name] = clipped_values
        cluster_clipping_flags[cluster_name] = {
            target: raw_values[target] != clipped_values[target]
            for target in TARGETS
        }

    lfnu_order = sorted(
        cluster_predictions_clipped,
        key=lambda name: cluster_predictions_clipped[name]["LFnu"],
        reverse=True,
    )

    return {
        "predictors": PREDICTORS,
        "targets": TARGETS,
        "r_xx": R_XX,
        "r_xx_inverse": r_xx_inv,
        "r_xy": R_XY,
        "beta_std": beta_std,
        "pooled": {
            variable: {"mean": mean, "sd": sd}
            for variable, (mean, sd) in pooled.items()
        },
        "beta_raw": beta_raw,
        "intercept": intercept,
        "r_squared": r_squared,
        "cluster_predictions_raw": cluster_predictions_raw,
        "cluster_predictions_clipped": cluster_predictions_clipped,
        "cluster_clipping_flags": cluster_clipping_flags,
        "lfnu_proxy_order": lfnu_order,
        "lfnu_order_matches_weber": lfnu_order == ["HRS", "HP", "MP", "LRS"],
    }


def write_outputs(results: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "weber_proxy_results.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    coef_path = output_dir / "weber_proxy_coefficients.csv"
    with coef_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "intercept", *PREDICTORS, "R2"])
        for target in TARGETS:
            writer.writerow(
                [
                    target,
                    results["intercept"][target],
                    *results["beta_raw"][target],
                    results["r_squared"][target],
                ]
            )

    pooled_path = output_dir / "weber_pooled_stats.csv"
    with pooled_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variable", "pooled_mean", "pooled_sd"])
        for variable in [*PREDICTORS, *TARGETS]:
            writer.writerow(
                [
                    variable,
                    results["pooled"][variable]["mean"],
                    results["pooled"][variable]["sd"],
                ]
            )

    cluster_path = output_dir / "weber_cluster_proxy_check.csv"
    with cluster_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "cluster",
                *[f"{target}_raw" for target in TARGETS],
                *[f"{target}_clipped" for target in TARGETS],
                *[f"{target}_clipped_flag" for target in TARGETS],
                "LFnu_rank_after_clipping",
            ]
        )
        for rank, cluster_name in enumerate(results["lfnu_proxy_order"], start=1):
            raw_row = results["cluster_predictions_raw"][cluster_name]
            clipped_row = results["cluster_predictions_clipped"][cluster_name]
            flag_row = results["cluster_clipping_flags"][cluster_name]
            writer.writerow(
                [
                    cluster_name,
                    *[raw_row[target] for target in TARGETS],
                    *[clipped_row[target] for target in TARGETS],
                    *[flag_row[target] for target in TARGETS],
                    rank,
                ]
            )


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs"
    results = calculate()
    write_outputs(results, output_dir)

    print("Weber spectral proxy calculation complete.")
    print(f"Output directory: {output_dir}")
    print("LFnu proxy order:", " > ".join(results["lfnu_proxy_order"]))
    print("LFnu order matches Weber:", results["lfnu_order_matches_weber"])
    print("R^2:")
    for target in TARGETS:
        print(f"  {target}: {results['r_squared'][target]:.6f}")


if __name__ == "__main__":
    main()
