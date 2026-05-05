```markdown
# NethoBench

NethoBench evaluates neural, behavioural and multimodal neural-behavioural generative models, such as autoregressive causal transformers, by measuring the realism or plausibility of generated traces. Each of the three modes is covered by its own command, and which is appropriate depends on the nature of the generated traces: synthetic neural traces should be evaluated using "neuro-scores", "fidelity-scores" and "neuro-analysis", synthetic behavioural traces should use "etho-scores" and "etho-analysis" and multimodal traces "cross-scores" and "cross-analysis".

The project is licensed under the MIT license. 



![NethoBench Logo](assets/nethobench.png)

**Outputs:**
1. **Neuro:** Structural neural realism (Distribution, Temporal, Relational, Geometry, State Dynamics).
2. **Fidelity:** Direct trace-alignment (Error, Mutual Information).
3. **Behavior:** Pose and kinematics realism.
4. **Cross-modal:** Neural-behavioral coupling (CCA, Predictive $R^2$, Lead-Lag).
5. **Multimodal Composite:** Average across available domains.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Data Format
- - Must include alignment columns for sequence ID and time step (defaults to sequenceId and itemPosition, but can be overridden in config.json).
- **Neural:** One column per region.
- **Behavior:** Keypoint columns with `_X`/`_Y` suffixes (e.g., `CENTER_X`).
- **Multimodal:** Provide a single merged CSV or a config JSON identifying column types.

### Config Schema (`config.json`)

The configuration options are broadly divided into two categories: **Data Schema Mappings** (which tell NethoBench how to read your multimodal datasets) and **Data-Dependent Hyperparameters** (which adjust evaluation bounds based on your dataset's frame rate, length, or available compute).

### 1. Data Schema Mappings
These options map your dataset's columns to the semantic features NethoBench expects. If omitted, NethoBench will attempt to auto-infer them from the dataset headers.

*   **`sequence_key`**: *(string)* The column name identifying discrete sequences or trials. Defaults to `"sequenceId"`.
*   **`time_key`**: *(string)* The column name tracking the chronological frame or time step within a sequence. Defaults to `"itemPosition"`.
*   **`neuro_cols`**: *(array of strings)* A list of column names corresponding to continuous neural regions, sensors, or ROIs.
*   **`behavior_parts`**: *(array of strings)* A list of tracked body parts. NethoBench expects these to have `_X` and `_Y` coordinate suffixes in the data (e.g., `["CENTER", "NOSE", "TAIL_BASE"]`).
*   **`center_part`**: *(string)* The primary point used for tracking global speed and trajectory kinematics. Defaults to `"CENTER"`.
*   **`body_axis`**: *(array of strings)* A two-element list defining the front and back of the subject for directional heading calculations. Defaults to `["NOSE", "TAIL_BASE"]`.

### 2. Data-Dependent Hyperparameters
These values fine-tune the statistical bounds and metrics algorithms. They are typically adjusted based on your dataset's sampling frequency, memory constraints, or expected physiological bounds. 

**Time & Frame-Rate Settings**
*   **`WELCH_SAMPLING_FREQUENCY`**: *(float)* Base sampling rate used for PSD and bandpower estimations. Default: `30.0`.
*   **`WELCH_NPERSEG`**: *(int)* Window size for Welch's spectral density estimation. Default: `256`.
*   **`LEAD_LAG_DEFAULT_MAX_LAG`**: *(int)* The maximum frame offset searched when calculating cross-modal neural-behavioral lead/lag relationships. Default: `30`.
*   **`CROSS_CORR_MAX_LAG_SMALL`**: *(int)* Maximum lag for targeted top-edge cross-correlation profiles. Default: `16`.
*   **`AUTOCORR_MAX_LAG`**: *(int)* Maximum lag calculated for autocorrelation agreement. Default: `48`.
*   **`DFC_STATE_CONFIGS`**: *(list of tuples)* Defines `(window_size, step_size, n_states)` for Dynamic Functional Connectivity calculations. Default: `[[45, 15, 8], [60, 30, 8], [90, 30, 8]]`.
*   **`CHUNK_SIZE_EMBEDDINGS`**: *(int)* Frame window size used to generate movement manifold embeddings. Default: `5`.

**Compute & Subsampling Limits**
*   **`MAX_TIME_STEPS_SUBSAMPLE`**: *(int)* Maximum time steps sampled for heavy distribution bounds (like Quantile and MI scores) to prevent memory overload. Default: `1200`.
*   **`LATENT_SUBSAMPLE_SIZE`**: *(int)* Number of points sampled for global topology/manifold metrics. Default: `96`.
*   **`N_POINTS_MANIFOLD_PH_KNN`**: *(int)* Points sampled for K-Nearest Neighbors manifold comparisons. Default: `128`.
*   **`MAX_SEQUENCES_STRATIFIED`**: *(int)* Number of sequences evaluated during stratified topology metrics. Default: `48`.
*   **`MAX_POINTS_MANIFOLD_ALIGNMENT`**: *(int)* Upper point limit for MMD similarity tracking. Default: `2000`.
*   **`MI_MAX_POINTS`**: *(int)* Point limit evaluated when generating pairwise Mutual Information matrices. Default: `1500`.

**Algorithm Heuristics & Thresholds**
*   **`PCA_VARIANCE_THRESHOLD`**: *(float)* The target explained variance used to dynamically select the number of principal components. Default: `0.85`.
*   **`STATIONARY_THRESHOLD_PERCENTILE`**: *(int)* The global speed percentile below which a subject is considered "stationary". Default: `10`.
*   **`SPEED_OUTLIER_PERCENTILE`**: *(int)* Used to cap extreme velocity outliers in histograms. Default: `99`.
*   **`HISTOGRAM_BINS_DEFAULT`**: *(int)* Bin counts for standard marginal distributions. Default: `12`.
*   **`KMEANS_K_TRAJECTORY`**: *(int)* Target cluster count for whole-sequence trajectory shape scoring. Default: `6`.
*   **`KMEANS_K_SYLLABLE`**: *(int)* Target cluster count for local kinematic "syllable" scoring. Default: `8`.
*   **`MI_N_NEIGHBORS`**: *(int)* The number of neighbors used in continuous Mutual Information regression. Default: `5`.
*   **`LEAD_LAG_MIN_ARRAY_SIZE`**: *(int)* Minimum valid overlap window needed to compute lead/lag cross-correlation. Default: `5`.

---

### Example `config.json`
```json
{
  "sequence_key": "trial_id",
  "time_key": "frame_idx",
  "neuro_cols": ["vis_ctx_1", "vis_ctx_2", "motor_ctx"],
  "behavior_parts": ["HEAD", "BODY", "TAIL"],
  "center_part": "BODY",
  "body_axis": ["HEAD", "TAIL"],
  "WELCH_SAMPLING_FREQUENCY": 60.0,
  "LEAD_LAG_DEFAULT_MAX_LAG": 60,
  "PCA_VARIANCE_THRESHOLD": 0.90
}
```

## CLI Usage
All commands output JSON scores; `-analysis` commands also generate figures.

**Neural**
```bash
nethobench neuro-scores --gt gt.csv --preds pred.csv
nethobench fidelity-scores --gt gt.csv --preds pred.csv
nethobench neuro-analysis --gt gt.csv --preds pred.csv
```

**Behavioral**
```bash
nethobench etho-scores --gt-dir gt/ --inf-dir inf/
nethobench etho-analysis --gt-dir gt/ --inf-dir inf/
```

**Multimodal**
```bash
nethobench cross-scores --gt gt.csv --preds pred.csv --config config.json
nethobench cross-analysis --gt gt.csv --preds pred.csv --config config.json
```

```

## Metric Composites

The active neural benchmark bounds all scores to `[0, 1]` (higher is better) and computes a weighted final score:

- **Distribution ($D$):** KL/JSD, Quantiles, Moments, Mean.
- **Temporal ($T$):** Trajectory distribution.
- **Relational ($R$):** Graph, Cross-region MI, Lagged Covariance, Impulse Response.
- **Geometry ($G$):** Manifold persistence, Subspace Angle.
- **State Dynamics ($S$):** Occupancy and Transition.

$$ \mathrm{Neuro} = 0.22D + 0.18T + 0.24R + 0.18G + 0.18S $$

Fidelity is calculated separately:

$$ \mathrm{Fidelity} = 0.65 \, s_{Error} + 0.35 \, s_{MI} $$


## Metric Dictionary

### 1. Neuro Metrics (`neuro-scores`)
Evaluates the structural, dynamic, and relational realism of generated neural traces.

**Distribution Family ($D$)**
*   **`KL_or_JSD_score`**: Measures the tail-binned histogram overlap between ground truth and predicted traces using Symmetric KL Divergence.
*   **`Mean_score`**: Region-wise mean shift, normalized by the robust Interquartile Range (IQR).
*   **`QNT_score`**: Quantile distance focusing on the tail distributions.
*   **`MOM_score`**: Agreement across statistical moments (variance, skewness, and kurtosis).

**Temporal / Spectral Family ($T$)**
*   **`TRJDIST_score`**: Pooled latent occupancy and velocity-distribution agreement stabilized by latent path features (speed, turning, path length).

**Relational Family ($R$)**
*   **`GRAPH_score`**: Graph network similarity evaluating top-edge topology, weighted degree, and local clustering agreement based on functional connectivity.
*   **`CrossRegionMI_score`**: Pairwise mutual information matrix similarity between regions.
*   **`LaggedCovariance_score`**: Cross-region lagged covariance similarity at multiple time steps.
*   **`ImpulseResponse_score`**: Similarity of Vector Autoregressive (VAR(1)) system coefficients.

**Geometry Family ($G$)**
*   **`MANI_score`**: Persistent-homology lifetime agreement stabilized by local-neighborhood geometry (K-Nearest Neighbors).
*   **`SubspaceAngle_score`**: Principal angle similarity between the PCA-derived subspaces.

**State Dynamics Family ($S$)**
*   **`LatentStateOccupancy...`**: Similarity of state residence histograms using K-Means clustering (evaluated at $K=11, 12$).
*   **`LatentStateTransition...`**: Transition matrix similarity mapping how states evolve over multiple temporal lags.

---

### 2. Fidelity Metrics (`fidelity-scores`)
Measures the direct, frame-by-frame alignment of predicted traces to ground truth.

*   **`Error_score`**: Trace-level Normalized Root Mean Square Error (nRMSE), scaled by the IQR of the ground truth.
*   **`MI_score`**: Trace-level Mutual Information to assess direct predictive dependency.

---

### 3. Behavioral Metrics (`etho-scores`)
Evaluates the kinematic, spatial, and sequence-level realism of generated body poses.

*   **`position_kl_score`**: 2D spatial position density overlap (measured via KL divergence).
*   **`quadrant_score`**: Distribution agreement across four geometric spatial quadrants.
*   **`stationary_score`**: Agreement on the fraction of time the subject spends strictly stationary.
*   **`velocity_score` / `acceleration_score`**: Distributional similarities for 1st-order (speed) and 2nd-order (acceleration) kinematic magnitudes.
*   **`direction_score`**: Similarity of movement heading orientation (cosine similarity between velocity vector and body axis).
*   **`syllable_score`**: K-Means clustering overlap of local kinematic features (speed and acceleration syllables).
*   **`trajectory_shape_score`**: Similarity of geometric path characteristics (path length, net displacement, straightness, turn angles).
*   **`dtw_similarity_score`**: Dynamic Time Warping (DTW) distance measuring overall trajectory alignment.
*   **`procrustes_similarity` / `mmd_similarity`**: Local movement manifold alignment leveraging Procrustes analysis and Maximum Mean Discrepancy (MMD).

---

### 4. Cross-Modal Metrics (`cross-scores`)
Evaluates how well the generated data captures the latent coupling between neural activity and behavior.

*   **`cca_alignment_score`**: Canonical Correlation Analysis (CCA) similarity, measuring the shared linear spatial subspace between neural and behavioral streams.
*   **`neural_to_behavior_similarity`**: The difference in predictive $R^2$ when mapping neural sequences to behavioral states via linear regression.
*   **`behavior_to_neural_similarity`**: The difference in predictive $R^2$ when mapping behavioral sequences back to neural activity.
*   **`lead_lag_score`**: Agreement on the optimal temporal offset (lead/lag) measured via cross-correlation between the first neural principal component (PC1) and behavioral speed.

## License
MIT License. See `LICENSE`.
```