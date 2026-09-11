#!/usr/bin/env python3
"""Assign MIMIC subjects to fixed Weber Table 2 centroids.

Self-z analysis:
  - Weber centroids are Table 2 measured cluster means.
  - No K-means is fit in MIMIC.
  - MIMIC LF/HF/LFnu/TP proxy columns are mapped to Weber LF/HF/LFnu/TP
    for standardization and distance calculation.
  - MIMIC subjects are standardized using the MIMIC sample mean/SD.
  - Weber centroids are standardized using the Weber pooled mean/SD.

Core functions:
  - load_pooled: Load Weber pooled mean/SD.
  - build_centroids: Standardize the Weber Table 2 centroids as z-score.
  - read_subjects: Standardize MIMIC subject values using the MIMIC sample mean/SD.
  - assign_subjects: Calculate the Euclidean distance between each subject and the four centroids.
  - write_assignments: Save the final assignment CSV.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import subprocess
from pathlib import Path
from typing import Any


WORKS_DIR = Path(__file__).resolve().parent / "outputs"
# DEFAULT_INPUT = the result of step 2. subject-level aggregation (required, no default)
DEFAULT_WEBER_JSON = WORKS_DIR / "weber_proxy_results.json" # the result of step 1. spectral proxy estimation
DEFAULT_OUTPUT = WORKS_DIR / "hf_s3_selfz.csv"
DEFAULT_SUMMARY = WORKS_DIR / "hf_s3_selfz_summary.json"
DEFAULT_CENTROIDS = WORKS_DIR / "hf_s3_selfz_weber_centroids_table2.csv"
DEFAULT_SCALER = WORKS_DIR / "hf_s3_selfz_scaler.csv"
DEFAULT_PCA = WORKS_DIR / "hf_s3_selfz_pca_assignment.png"

CLUSTER_ORDER = ["MP", "HRS", "LRS", "HP"]
FEATURE_ORDER = [
    "BPM",
    "pNN20",
    "pNN50",
    "MAD",
    "RMSSD",
    "BR",
    "LF",
    "HF",
    "LFnu",
    "TP",
]
MIMIC_TO_WEBER = {
    "BPM": "BPM",
    "pNN20": "pNN20",
    "pNN50": "pNN50",
    "MAD": "MAD",
    "RMSSD": "RMSSD",
    "BR": "BR",
    "LF_proxy": "LF",
    "HF_proxy": "HF",
    "LFnu_proxy": "LFnu",
    "TP_proxy": "TP",
}
MIMIC_FEATURE_ORDER = [
    "BPM",
    "pNN20",
    "pNN50",
    "MAD",
    "RMSSD",
    "BR",
    "LF_proxy",
    "HF_proxy",
    "LFnu_proxy",
    "TP_proxy",
]
OUTPUT_COLUMNS = [
    "subject_id",
    "BPM",
    "pNN20",
    "pNN50",
    "MAD",
    "RMSSD",
    "BR",
    "LF_proxy",
    "HF_proxy",
    "LFnu_proxy",
    "TP_proxy",
    "assigned_cluster",
    "distance_MP",
    "distance_HRS",
    "distance_LRS",
    "distance_HP",
    "nearest_distance",
    "second_nearest_distance",
    "ambiguity_gap",
]
COLORS = {
    "MP": "#4C78A8",
    "HRS": "#F58518",
    "LRS": "#54A24B",
    "HP": "#E45756",
}


# Weber Table 2 measured means and SDs transcribed in spectral_proxy_estimation.py.
WEBER_TABLE2 = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign MIMIC subjects to fixed Weber centroids.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--weber-json", type=Path, default=DEFAULT_WEBER_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--centroids", type=Path, default=DEFAULT_CENTROIDS)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--pca", type=Path, default=DEFAULT_PCA)
    return parser.parse_args()


def to_float(value: str | None, column: str, row_number: int) -> float:
    if value is None or value.strip() == "":
        raise ValueError(f"Missing value for {column} at row {row_number}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite value for {column} at row {row_number}: {value!r}")
    return number


def fmt(value: float) -> str:
    if value == 0:
        return "0"
    return format(value, ".15g")


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norm(vector: list[float]) -> float:
    return math.sqrt(dot(vector, vector))


def describe(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    n = len(ordered)
    mean = sum(ordered) / n if n else math.nan
    sd = math.sqrt(sum((x - mean) ** 2 for x in ordered) / (n - 1)) if n > 1 else 0.0

    def pct(q: float) -> float:
        if not ordered:
            return math.nan
        if n == 1:
            return ordered[0]
        position = (n - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[int(position)]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "min": ordered[0] if n else math.nan,
        "q1": pct(0.25),
        "median": pct(0.50),
        "q3": pct(0.75),
        "max": ordered[-1] if n else math.nan,
    }


def load_pooled(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as f:
        data = json.load(f)
    pooled = data.get("pooled")
    if not isinstance(pooled, dict):
        raise ValueError(f"{path} missing pooled statistics")
    missing = [feature for feature in FEATURE_ORDER if feature not in pooled]
    if missing:
        raise ValueError(f"{path} pooled statistics missing: {missing}")
    return {
        feature: {
            "mean": float(pooled[feature]["mean"]),
            "sd": float(pooled[feature]["sd"]),
        }
        for feature in FEATURE_ORDER
    }


def validate_pooled_against_table2(pooled: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    reconstructed: dict[str, dict[str, float]] = {}
    total_n = sum(WEBER_TABLE2[cluster]["n"] for cluster in CLUSTER_ORDER)
    for feature in FEATURE_ORDER:
        pooled_mean = sum(
            WEBER_TABLE2[cluster]["n"] * WEBER_TABLE2[cluster][feature][0]
            for cluster in CLUSTER_ORDER
        ) / total_n
        within_ss = sum(
            (WEBER_TABLE2[cluster]["n"] - 1) * WEBER_TABLE2[cluster][feature][1] ** 2
            for cluster in CLUSTER_ORDER
        )
        between_ss = sum(
            WEBER_TABLE2[cluster]["n"] * (WEBER_TABLE2[cluster][feature][0] - pooled_mean) ** 2
            for cluster in CLUSTER_ORDER
        )
        pooled_sd = math.sqrt((within_ss + between_ss) / (total_n - 1))
        reconstructed[feature] = {
            "mean": pooled_mean,
            "sd": pooled_sd,
            "json_mean": pooled[feature]["mean"],
            "json_sd": pooled[feature]["sd"],
            "mean_abs_diff": abs(pooled_mean - pooled[feature]["mean"]),
            "sd_abs_diff": abs(pooled_sd - pooled[feature]["sd"]),
        }
    return reconstructed


def build_centroids(pooled: dict[str, dict[str, float]]) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    raw: dict[str, list[float]] = {}
    z: dict[str, list[float]] = {}
    for cluster in CLUSTER_ORDER:
        raw[cluster] = [WEBER_TABLE2[cluster][feature][0] for feature in FEATURE_ORDER]
        z[cluster] = [
            (WEBER_TABLE2[cluster][feature][0] - pooled[feature]["mean"]) / pooled[feature]["sd"]
            for feature in FEATURE_ORDER
        ]
    return raw, z


def read_subjects(path: Path) -> tuple[list[dict[str, Any]], list[list[float]], dict[str, dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = [column for column in ["subject_id", *MIMIC_FEATURE_ORDER] if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing required columns: {missing}")
        for row_number, row in enumerate(reader, start=2):
            values = {
                column: to_float(row.get(column), column, row_number)
                for column in MIMIC_FEATURE_ORDER
            }
            rows.append({"subject_id": row["subject_id"], "values": values}) # get raw data before z

    if not rows:
        raise ValueError(f"{path} has no rows")

    mimic_scaler: dict[str, dict[str, float]] = {} # dictionary to store mean, sd for each MIMIC feature
    for column in MIMIC_FEATURE_ORDER: # calculate mean and sd for each MIMIC feature
        values = [float(row["values"][column]) for row in rows]
        mean = sum(values) / len(values) 
        sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
        if sd <= 0 or not math.isfinite(sd):
            raise ValueError(f"Cannot self-z standardize {column}: sample SD={sd}")
        mimic_scaler[column] = {"mean": mean, "sd": sd} # store mean and sd, can be found in 'hf_s3_selfz_scaler.csv'

    z_matrix = [
        [
            (float(row["values"][mimic_column]) - mimic_scaler[mimic_column]["mean"])
            / mimic_scaler[mimic_column]["sd"]
            for mimic_column in MIMIC_FEATURE_ORDER
        ]
        for row in rows
    ] # apply self-z standardization
    return rows, z_matrix, mimic_scaler

# rows: raw data
# z_matrix: self-z standardized data for distance calculation
# mimic_scaler: mean and sd for each MIMIC feature, used for self-z standardization

def assign_subjects(
    rows: list[dict[str, Any]],
    z_matrix: list[list[float]],
    z_centroids: dict[str, list[float]],
) -> list[dict[str, Any]]:
    assigned: list[dict[str, Any]] = []
    for row, z_values in zip(rows, z_matrix):
        distances = {
            cluster: math.sqrt(sum((value - center) ** 2 for value, center in zip(z_values, z_centroids[cluster])))
            for cluster in CLUSTER_ORDER
        }
        sorted_distances = sorted(distances.items(), key=lambda item: item[1])
        nearest_cluster, nearest_distance = sorted_distances[0]
        second_nearest_distance = sorted_distances[1][1]
        ambiguity_gap = (
            (second_nearest_distance - nearest_distance) / nearest_distance
            if nearest_distance > 0
            else math.nan
        )
        assigned.append(
            {
                "subject_id": row["subject_id"],
                **row["values"],
                "assigned_cluster": nearest_cluster,
                **{f"distance_{cluster}": distances[cluster] for cluster in CLUSTER_ORDER},
                "nearest_distance": nearest_distance,
                "second_nearest_distance": second_nearest_distance,
                "ambiguity_gap": ambiguity_gap,
            }
        )
    return assigned


def write_assignments(path: Path, assignments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in assignments:
            writer.writerow(
                {
                    "subject_id": row["subject_id"],
                    **{column: fmt(float(row[column])) for column in MIMIC_FEATURE_ORDER},
                    "assigned_cluster": row["assigned_cluster"],
                    **{f"distance_{cluster}": fmt(float(row[f"distance_{cluster}"])) for cluster in CLUSTER_ORDER},
                    "nearest_distance": fmt(float(row["nearest_distance"])),
                    "second_nearest_distance": fmt(float(row["second_nearest_distance"])),
                    "ambiguity_gap": "" if math.isnan(row["ambiguity_gap"]) else fmt(float(row["ambiguity_gap"])),
                }
            )


def write_centroids(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster", "n", *FEATURE_ORDER])
        for cluster in CLUSTER_ORDER:
            writer.writerow(
                [
                    cluster,
                    WEBER_TABLE2[cluster]["n"],
                    *[WEBER_TABLE2[cluster][feature][0] for feature in FEATURE_ORDER],
                ]
            )


def write_scaler(
    path: Path,
    pooled: dict[str, dict[str, float]],
    validation: dict[str, dict[str, float]],
    mimic_scaler: dict[str, dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mimic_feature",
                "weber_feature",
                "mimic_mean",
                "mimic_sd",
                "weber_pooled_mean",
                "weber_pooled_sd",
                "weber_reconstructed_mean",
                "weber_reconstructed_sd",
                "mean_abs_diff",
                "sd_abs_diff",
                "subject_standardization",
                "centroid_standardization",
            ],
        )
        writer.writeheader()
        for mimic_feature in MIMIC_FEATURE_ORDER:
            weber_feature = MIMIC_TO_WEBER[mimic_feature]
            writer.writerow(
                {
                    "mimic_feature": mimic_feature,
                    "weber_feature": weber_feature,
                    "mimic_mean": mimic_scaler[mimic_feature]["mean"],
                    "mimic_sd": mimic_scaler[mimic_feature]["sd"],
                    "weber_pooled_mean": pooled[weber_feature]["mean"],
                    "weber_pooled_sd": pooled[weber_feature]["sd"],
                    "weber_reconstructed_mean": validation[weber_feature]["mean"],
                    "weber_reconstructed_sd": validation[weber_feature]["sd"],
                    "mean_abs_diff": validation[weber_feature]["mean_abs_diff"],
                    "sd_abs_diff": validation[weber_feature]["sd_abs_diff"],
                    "subject_standardization": "MIMIC sample mean/SD (self-z)",
                    "centroid_standardization": "Weber pooled mean/SD",
                }
            )


def cluster_feature_summary(assignments: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    out: dict[str, dict[str, dict[str, float | int]]] = {}
    for cluster in CLUSTER_ORDER:
        cluster_rows = [row for row in assignments if row["assigned_cluster"] == cluster]
        out[cluster] = {
            feature: describe([float(row[feature]) for row in cluster_rows])
            for feature in MIMIC_FEATURE_ORDER
        }
    return out


def covariance_matrix(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    p = len(matrix[0])
    means = [sum(row[j] for row in matrix) / n for j in range(p)]
    centered = [[row[j] - means[j] for j in range(p)] for row in matrix]
    cov = [[0.0 for _ in range(p)] for _ in range(p)]
    for row in centered:
        for i in range(p):
            for j in range(i, p):
                cov[i][j] += row[i] * row[j]
    for i in range(p):
        for j in range(i, p):
            cov[i][j] /= n - 1
            cov[j][i] = cov[i][j]
    return cov


def mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [dot(row, vector) for row in matrix]


def outer(vector: list[float]) -> list[list[float]]:
    return [[x * y for y in vector] for x in vector]


def first_eigenpair(matrix: list[list[float]], seed_index: int) -> tuple[float, list[float]]:
    p = len(matrix)
    vector = [0.0 for _ in range(p)]
    vector[seed_index % p] = 1.0
    for _ in range(200):
        next_vector = mat_vec(matrix, vector)
        length = norm(next_vector)
        if length == 0:
            break
        next_vector = [value / length for value in next_vector]
        if norm([a - b for a, b in zip(vector, next_vector)]) < 1e-12:
            vector = next_vector
            break
        vector = next_vector
    eigenvalue = dot(vector, mat_vec(matrix, vector))
    return eigenvalue, vector


def pca_2d(matrix: list[list[float]], centroids: dict[str, list[float]]) -> dict[str, Any]:
    cov = covariance_matrix(matrix)
    eig1, pc1 = first_eigenpair(cov, 0)
    deflated = [
        [cov[i][j] - eig1 * outer(pc1)[i][j] for j in range(len(cov))]
        for i in range(len(cov))
    ]
    eig2, pc2 = first_eigenpair(deflated, 1)
    total_var = sum(cov[i][i] for i in range(len(cov)))
    subject_scores = [[dot(row, pc1), dot(row, pc2)] for row in matrix]
    centroid_scores = {
        cluster: [dot(values, pc1), dot(values, pc2)]
        for cluster, values in centroids.items()
    }
    return {
        "subject_scores": subject_scores,
        "centroid_scores": centroid_scores,
        "explained_variance_ratio": [
            eig1 / total_var if total_var else math.nan,
            eig2 / total_var if total_var else math.nan,
        ],
        "components": [pc1, pc2],
    }


def save_pca_plot(
    path: Path,
    assignments: list[dict[str, Any]],
    pca: dict[str, Any],
) -> None:
    svg_path = path.with_suffix(".svg")
    scores = pca["subject_scores"]
    centroid_scores = pca["centroid_scores"]
    width, height = 1200, 900
    left, right, top, bottom = 90, 250, 55, 85
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [score[0] for score in scores] + [score[0] for score in centroid_scores.values()]
    ys = [score[1] for score in scores] + [score[1] for score in centroid_scores.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad = (x_max - x_min) * 0.06 or 1.0
    y_pad = (y_max - y_min) * 0.06 or 1.0
    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfbfb" stroke="#333" stroke-width="1"/>',
        '<style>text{font-family:Arial, Helvetica, sans-serif; fill:#222}</style>',
        f'<text x="{left + plot_w / 2}" y="30" text-anchor="middle" font-size="20">MIMIC assignment to fixed Weber Table 2 centroids</text>',
    ]
    for i in range(6):
        x = left + i * plot_w / 5
        y = top + i * plot_h / 5
        parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_h}" stroke="#e7e7e7"/>')
        parts.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e7e7e7"/>')

    for row, score in zip(assignments, scores):
        cluster = row["assigned_cluster"]
        parts.append(
            f'<circle cx="{sx(score[0]):.2f}" cy="{sy(score[1]):.2f}" r="2.1" '
            f'fill="{COLORS[cluster]}" fill-opacity="0.25" stroke="none"/>'
        )

    for cluster in CLUSTER_ORDER:
        x, y = centroid_scores[cluster]
        px, py = sx(x), sy(y)
        color = COLORS[cluster]
        parts.append(
            f'<path d="M {px - 10:.1f} {py:.1f} L {px + 10:.1f} {py:.1f} '
            f'M {px:.1f} {py - 10:.1f} L {px:.1f} {py + 10:.1f}" '
            f'stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
        )
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="12" fill="none" stroke="#111" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{px + 15:.1f}" y="{py - 12:.1f}" font-size="14" font-weight="bold">{html.escape(cluster)}</text>'
        )

    evr = pca["explained_variance_ratio"]
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 24}" text-anchor="middle" font-size="16">'
        f'PC1 ({evr[0] * 100:.1f}% variance)</text>'
    )
    label_x = 27
    label_y = top + plot_h / 2
    parts.append(
        f'<text x="{label_x}" y="{label_y:.1f}" text-anchor="middle" font-size="16" '
        f'transform="rotate(-90 {label_x} {label_y:.1f})">PC2 ({evr[1] * 100:.1f}% variance)</text>'
    )
    legend_x = left + plot_w + 45
    parts.append(f'<text x="{legend_x}" y="{top + 20}" font-size="17" font-weight="bold">assigned cluster</text>')
    for idx, cluster in enumerate(CLUSTER_ORDER):
        y = top + 55 + idx * 32
        parts.append(f'<circle cx="{legend_x + 9}" cy="{y}" r="6" fill="{COLORS[cluster]}" fill-opacity="0.75"/>')
        parts.append(f'<text x="{legend_x + 28}" y="{y + 5}" font-size="15">{cluster}</text>')
    y = top + 55 + len(CLUSTER_ORDER) * 32 + 22
    parts.append(f'<path d="M {legend_x + 1} {y} L {legend_x + 17} {y} M {legend_x + 9} {y - 8} L {legend_x + 9} {y + 8}" stroke="#111" stroke-width="4" stroke-linecap="round"/>')
    parts.append(f'<text x="{legend_x + 28}" y="{y + 5}" font-size="15">Weber centroid</text>')
    parts.append("</svg>")

    svg_path.write_text("\n".join(parts), encoding="utf-8")
    if path.suffix.lower() != ".svg":
        subprocess.run(
            ["sips", "-s", "format", "png", str(svg_path), "--out", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )


def main() -> int:
    args = parse_args()
    pooled = load_pooled(args.weber_json)
    pooled_validation = validate_pooled_against_table2(pooled)
    _, z_centroids = build_centroids(pooled)
    subject_rows, z_matrix, mimic_scaler = read_subjects(args.input)
    assignments = assign_subjects(subject_rows, z_matrix, z_centroids)

    write_assignments(args.output, assignments)
    write_centroids(args.centroids)
    write_scaler(args.scaler, pooled, pooled_validation, mimic_scaler)

    pca = pca_2d(z_matrix, z_centroids)
    args.pca.parent.mkdir(parents=True, exist_ok=True)
    save_pca_plot(args.pca, assignments, pca)

    cluster_counts = {cluster: 0 for cluster in CLUSTER_ORDER}
    for row in assignments:
        cluster_counts[row["assigned_cluster"]] += 1
    total = len(assignments)
    ambiguity_values = [float(row["ambiguity_gap"]) for row in assignments if math.isfinite(row["ambiguity_gap"])]
    ambiguity_by_cluster = {
        cluster: describe(
            [float(row["ambiguity_gap"]) for row in assignments if row["assigned_cluster"] == cluster and math.isfinite(row["ambiguity_gap"])]
        )
        for cluster in CLUSTER_ORDER
    }

    summary = {
        "metadata": {
            "step": "Step 3",
            "analysis_name": "hf_s3_selfz",
            "input": str(args.input),
            "weber_json": str(args.weber_json),
            "centroid_source": "Weber Table 2 measured cluster means",
            "assignment_method": "nearest fixed Weber centroid by Euclidean distance in 10D HRV space; MIMIC subjects use self-z, Weber centroids use Weber pooled-z",
            "subject_standardization": "MIMIC sample mean/SD (self-z)",
            "centroid_standardization": "Weber pooled mean/SD",
            "no_kmeans_refit": True,
            "feature_order": MIMIC_FEATURE_ORDER,
            "mimic_to_weber_mapping": MIMIC_TO_WEBER,
            "outputs": {
                "assignments": str(args.output),
                "summary": str(args.summary),
                "centroids": str(args.centroids),
                "scaler": str(args.scaler),
                "pca_plot_png": str(args.pca),
                "pca_plot_svg": str(args.pca.with_suffix(".svg")),
            },
        },
        "cluster_counts": {
            cluster: {
                "n": cluster_counts[cluster],
                "percent": cluster_counts[cluster] / total * 100 if total else math.nan,
            }
            for cluster in CLUSTER_ORDER
        },
        "cluster_feature_summary_subject_level": cluster_feature_summary(assignments),
        "weber_table2_centroids": {
            cluster: {
                feature: WEBER_TABLE2[cluster][feature][0]
                for feature in FEATURE_ORDER
            }
            for cluster in CLUSTER_ORDER
        },
        "mimic_selfz_scaler": mimic_scaler,
        "weber_pooled_scaler": pooled,
        "pooled_scaler_validation_against_table2": pooled_validation,
        "ambiguity_gap": {
            "overall": describe(ambiguity_values),
            "by_cluster": ambiguity_by_cluster,
            "lt_0_05_count": sum(1 for value in ambiguity_values if value < 0.05),
            "lt_0_05_percent": sum(1 for value in ambiguity_values if value < 0.05) / total * 100 if total else math.nan,
            "lt_0_10_count": sum(1 for value in ambiguity_values if value < 0.10),
            "lt_0_10_percent": sum(1 for value in ambiguity_values if value < 0.10) / total * 100 if total else math.nan,
        },
        "nearest_distance": describe([float(row["nearest_distance"]) for row in assignments]),
        "pca": {
            "explained_variance_ratio": pca["explained_variance_ratio"],
            "components_feature_order": MIMIC_FEATURE_ORDER,
            "components": pca["components"],
        },
    }

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"summary={args.summary}")
    print(f"centroids={args.centroids}")
    print(f"scaler={args.scaler}")
    print(f"pca={args.pca}")
    print(f"subjects={total}")
    for cluster in CLUSTER_ORDER:
        pct = cluster_counts[cluster] / total * 100 if total else math.nan
        print(f"cluster_{cluster}_n={cluster_counts[cluster]} ({pct:.2f}%)")
    print(f"ambiguity_gap_lt_0.05={summary['ambiguity_gap']['lt_0_05_count']} ({summary['ambiguity_gap']['lt_0_05_percent']:.2f}%)")
    print(f"ambiguity_gap_lt_0.10={summary['ambiguity_gap']['lt_0_10_count']} ({summary['ambiguity_gap']['lt_0_10_percent']:.2f}%)")
    print(f"pca_explained_variance={pca['explained_variance_ratio'][0]:.4f},{pca['explained_variance_ratio'][1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
