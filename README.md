# fair-mind-ecg-depression-works

This directory contains the core code for the publication "ECG-based Autonomic Profiles for Depression: Partial Reproduction, Nonlinear Extension, and Commentary."

The two scripts cover the central analysis: deriving spectral HRV proxies (LF, HF,
LFnu, TP) from the published summary statistics of Weber et al., and assigning subjects
to the four Weber autonomic profiles without refitting a clustering model. Upstream ECG
preprocessing and subject-level aggregation are not included here.

## spectral_proxy_estimation.py

This code has been used to derive the LF, HF, LFnu and TP proxy equations from the
published Weber summary statistics as described in the methods part of the manuscript
under "Spectral proxy reconstruction and cluster assignment." The predictor-predictor and predictor-target correlations are
taken from Weber Fig. 1, and the cluster-level N, mean and SD from Weber Table 2; both
are transcribed directly into the script, so no input file is required.

Outputs are written to `outputs/`:

- `weber_proxy_results.json` - all intermediate and final quantities (input to `cluster_assignment.py`)
- `weber_proxy_coefficients.csv` - intercept, raw coefficients and R2 per target
- `weber_pooled_stats.csv` - pooled means and SDs reconstructed from Weber Table 2
- `weber_cluster_proxy_check.csv` - proxy values predicted at the Weber cluster
  centroids, with the physical-constraint check (raw vs clipped) and the resulting
  LFnu rank order

## cluster_assignment.py

This code has been used for the cluster assignment as described in the methods part of
the manuscript under "Spectral proxy reconstruction and cluster assignment." Subjects are assigned to the fixed Weber Table 2
centroids by Euclidean distance in the 10-dimensional HRV space; no K-means is refitted.

The subject-level input CSV is produced by the upstream preprocessing and aggregation
step, which is not part of this directory. It therefore has no default path in the
script - the corresponding `DEFAULT_INPUT` line is left commented out.

Outputs are written to `outputs/`:

- `hf_s3_selfz.csv` - per-subject cluster assignment, distances and ambiguity gap
- `hf_s3_selfz_summary.json` - cluster counts, per-cluster feature summaries, PCA components
- `hf_s3_selfz_weber_centroids_table2.csv` - the fixed Weber Table 2 centroids
- `hf_s3_selfz_scaler.csv` - standardization parameters for subjects and centroids
- `hf_s3_selfz_pca_assignment.svg` - PCA projection of subjects and centroids
