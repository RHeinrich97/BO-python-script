# bo_chemistry.py

**Bayesian Optimisation for Chemical Synthesis** — a self-contained Python script for optimising reaction conditions with a human-in-the-loop workflow. No GUI, no cloud dependency: one script, one YAML, one CSV.

> Inspired by Lavayssiere, Bantreil & Lamaty, *RSC Mechanochemistry* 2026 and Shields, Stevens, Li et al., *Nature* 2021 (EDBO).

---

## Overview

`bo_chemistry.py` wraps a GPyTorch Gaussian Process surrogate model into a practical command-line workflow for synthetic chemists. You define your reaction parameters in a YAML file, fill experimental results into a CSV, and the script proposes the next conditions to test — iterating until convergence.

```
┌──────────────────────────────────────────────────────┐
│  1. Edit YAML  →  2. Run  →  3. Fill CSV  →  repeat  │
│            ↑_________________________________↓        │
└──────────────────────────────────────────────────────┘
```

**Key design choices:**
- **No web interface, no database** — everything is a plain CSV you can open in Excel
- **Single file** — copy `bo_chemistry.py` next to your YAML and you're done
- **GPyTorch GP** — GammaPriors, learned noise, ADAM optimiser (same backbone as EDBO)
- **Physicochemical encoding** — solvents, bases and ligands encoded with real descriptors (ε, pKa, TEP…) instead of one-hot vectors, so the GP understands that DMF and DMAc are similar

---

## Requirements

```bash
pip install numpy pandas scipy scikit-learn pyyaml matplotlib gpytorch torch
```

Python ≥ 3.9. GPU not required — CPU runs comfortably for typical chemistry campaigns of 5–50 experiments.

---

## Quick start

```bash
# 1. Copy the template config and rename it
cp config_test.yaml my_reaction.yaml

# 2. Edit my_reaction.yaml — set your parameters and outputs

# 3. First run — creates the CSV and suggests initial experiments
python bo_chemistry.py my_reaction.yaml --init-doe

# 4. Fill your experimental results in the CSV (set statut → 'fait')

# 5. Run again — suggests next conditions
python bo_chemistry.py my_reaction.yaml

# 6. Repeat until convergence
```

---

## Repository contents

| File | Description |
|---|---|
| `bo_chemistry.py` | Main optimisation script (v3.3) |
| `config_test.yaml` | Annotated configuration template |
| `descripteurs_chimiques.xlsx` | Physicochemical descriptor database (22 solvents, 21 bases, 21 ligands) |
| `bayes_learning_guide_v3.html` | 18-chapter interactive learning guide |

---

## Configuration (YAML)

The YAML is the only file you edit between campaigns. A minimal setup:

```yaml
campaign_name: "acylation_ball_mill"

parameters:
  - name: "Temperature (C)"
    type: continuous
    min: 25
    max: 80

  - name: "Temps de reaction (min)"
    type: continuous
    min: 15
    max: 120

  - name: "Equivalents acyl chloride"
    type: continuous
    min: 1.0
    max: 3.0

outputs:
  - name: "Rendement (%)"
    direction: maximize
    weight: 1.0

bo_settings:
  n_suggestions: 2
  acquisition: "EI"
  n_restarts: 50
  boxcox: true
```

### Physicochemical encoding of categorical variables

Instead of one-hot encoding (which treats DMF and DMAc as orthogonal), you can encode solvents, bases and ligands with their real physicochemical properties. The GP then learns that compounds with similar descriptors behave similarly — and can extrapolate to untested compounds.

```yaml
parameters:
  - name: "Solvant"
    type: categorical
    categories: ["DMF", "THF", "MeCN", "Toluene"]
    descriptors_file:    "descripteurs_chimiques.xlsx"
    descriptors_sheet:   "Solvants"
    descriptors_columns: ["ε (25°C)", "μ (D)", "δP (MPa½)", "δH (MPa½)"]

  - name: "Base"
    type: categorical
    categories: ["K2CO3", "Cs2CO3", "DBU", "DIPEA"]
    descriptors_file:    "descripteurs_chimiques.xlsx"
    descriptors_sheet:   "Bases"
    descriptors_columns: ["pKa (H₂O)", "Encombrement"]

  - name: "Ligand"
    type: categorical
    categories: ["PPh3", "PCy3", "SPhos", "XPhos"]
    descriptors_file:    "descripteurs_chimiques.xlsx"
    descriptors_sheet:   "Ligands"
    descriptors_columns: ["TEP (cm⁻¹)", "%Vbur (3.5Å)"]
```

The `descripteurs_chimiques.xlsx` file ships with the repository. Add your own compounds by appending rows to the relevant sheet — the script falls back to one-hot encoding with a warning if a compound is not found.

---

## Acquisition functions

| Key | Method | When to use |
|---|---|---|
| `"EI"` | Expected Improvement | Default — always a good choice |
| `"EI-TS"` | EI-TS hybrid (EDBO) | Batch n ≥ 3 — faster, natural diversity |
| `"TEI"` | Tempered EI (Lavayssiere 2026) | Multimodal landscapes, after ~10 iterations |
| `"UCB-AUTO"` | UCB with auto-β (Srinivas 2010) | Adaptive exploration without tuning κ |
| `"Thompson"` | Thompson Sampling | Auto-triggered on stagnation |
| `"VARMAX"` | Variance maximisation | Forced exploration of unknown regions |
| `"PI"` | Probability of Improvement | Very limited budget, conservative gains |

**EI-TS** is the recommended choice for batch experiments (n ≥ 3): EI selects the best exploitation point, then Thompson Sampling independently draws the remaining n−1 suggestions from the GP posterior — 2–3× faster than Kriging Believer with equivalent diversity.

**TEI** reduces redundant sampling near already-exploited regions by multiplying EI by a sigmoid factor g(x, β) that depends on the GP prediction at each point. On multimodal benchmarks it continues improving when EI plateaus. The advantage over EI becomes visible after ~10–12 iterations.

---

## Multi-objective optimisation

```yaml
outputs:
  - name: "Rendement (%)"
    direction: maximize
    weight: 0.6

  - name: "ee (%)"
    direction: maximize
    weight: 0.4
    threshold: 90        # must exceed 90% before yield is optimised

bo_settings:
  multi_objective_mode: "weighted_sum"   # or "pareto", "threshold", "ehvi"
```

Four modes are available:

- **`weighted_sum`** — linear combination of objectives with user-defined weights
- **`pareto`** — ParEGO with random weight vectors, traces the Pareto front
- **`threshold`** — constrained optimisation: some objectives must meet a threshold before others are optimised
- **`ehvi`** — MC-Expected Hypervolume Improvement (pure NumPy, no extra dependencies), theoretically superior for 2–3 objectives

---

## Initialisation strategies

```bash
# LHS (Latin Hypercube Sampling) — default
python bo_chemistry.py my_reaction.yaml --init-doe

# Specify number of initial experiments
python bo_chemistry.py my_reaction.yaml --init-doe --doe-n 9
```

```yaml
bo_settings:
  init_method: "lhs"      # uniform coverage (default)
  # init_method: "kmeans" # k-means on encoded space — better when
                          # categoricals are physicochemically encoded
```

K-means initialisation clusters the full encoded space (including physicochemical descriptors) and selects the point closest to each centroid. This ensures the initial experiments cover chemically distinct regions rather than a uniform grid — particularly relevant when ligands or solvents span very different electronic/steric spaces.

---

## Automatic failure detection

The script monitors the optimisation trajectory and adjusts automatically:

| Detector | What it catches | Response |
|---|---|---|
| `stagnation_detection` | A variable is stuck across rounds | Switch to Thompson + n=3 |
| `drift_detection` | A variable drifts without improvement | Switch to UCB κ=0.2 |
| `outlier_detection` | Measurements inconsistent with GP (LOO) | Flag in report |
| `boundary_saturation` | Suggestions pile up at parameter bounds | Warning |
| `multimodal_detection` | Multiple acquisition peaks | Multi-start optimisation |

All detectors are optional and independent. Enable them individually in the YAML:

```yaml
bo_settings:
  stagnation_detection: true
  stagnation_patience:  3
  drift_detection:      true
  drift_patience:       3
  outlier_detection:    false   # computationally expensive (N GP refits)
```

---

## Outputs

Every run produces or updates:

```
campaign_dashboard.png      ← always overwritten, always current
campaign_FINAL_report.png   ← generated at convergence or with --final
campaign_iter_XX_report.txt
campaign.csv
```

The **dashboard** is a 16:9 PNG with four panels:

- **Convergence** — best-so-far curve annotated at each new record
- **ARD importance** — which parameters the GP considers most influential
- **Scatter** — all experiments coloured by yield, with current suggestions marked ★
- **PDP** — Partial Dependence Plot of the most influential variable (continuous: curve ±σ; encoded categorical: bar chart with observed points overlaid)

```bash
# Generate high-resolution final report at any time
python bo_chemistry.py my_reaction.yaml --final

# Read-only status summary (no new suggestions)
python bo_chemistry.py my_reaction.yaml --status

# Benchmark acquisition functions on your current data
python bo_chemistry.py my_reaction.yaml --benchmark
```

---

## GP architecture (v3.3)

```
GPModel
├── Kernel  : Matérn 5/2 ARD
│              GammaPrior(3, 0.5) on lengthscales   ← Bayesian regularisation
│              GammaPrior(3, 0.5) on outputscale
├── Noise   : GaussianLikelihood                     ← learned from data
└── Optim.  : ADAM  ×  150 iterations  ×  5 restarts
               ExactMarginalLogLikelihood
```

**Why GPyTorch instead of sklearn?** sklearn's GP uses fixed bounds on lengthscales to prevent divergence — effective but blunt. GPyTorch uses GammaPriors that *gently pull* hyperparameters toward reasonable values while letting the data speak. With 5–15 training points (the typical chemistry regime), this makes a material difference in the quality of predictions. The noise level is also learned from data rather than fixed, so the GP automatically adapts its confidence to experimental reproducibility.

All GP hyperparameters are set internally — nothing to configure in the YAML.

---

## Descriptor database

`descripteurs_chimiques.xlsx` covers the most common reagents in organic synthesis and catalysis:

| Sheet | Compounds | Descriptors |
|---|---|---|
| **Solvants** | 22 solvents | ε (25°C), μ (D), δD/δP/δH (Hansen), log P, Eb |
| **Bases** | 21 bases | pKa (H₂O + MeCN Bordwell scale), steric hindrance score, nucleophilicity |
| **Ligands** | 21 phosphines + NHCs | TEP (cm⁻¹), Tolman cone angle (°), %V_bur at 3.5 Å |

Sources: CRC Handbook, Bordwell pKa tables, Tolman (1977), SambVca (Cavallo), Clavier & Nolan *Chem. Soc. Rev.* 2010.

To add a compound: append a row to the relevant sheet following the existing format. Columns marked `n.d.` are accepted (treated as zero).

---

## References

- Lavayssiere M., Bantreil X., Lamaty F. *RSC Mechanochemistry* 2026 — TEI acquisition function, diversity filter
- Shields B.J., Stevens J., Li J. et al. *Nature* **590**, 89–96 (2021) — EDBO framework, EI-TS, Kriging Believer
- Gardner J.R. et al. *NeurIPS* 2018 — GPyTorch
- Srinivas N. et al. *ICML* 2010 — UCB theoretical bound (auto-β)
- Hansen C.M. *Hansen Solubility Parameters: A User's Handbook*, 2007
- Tolman C.A. *Chem. Rev.* **77**, 313–348 (1977) — TEP, cone angle
- Clavier H., Nolan S.P. *Chem. Soc. Rev.* **39**, 3305 (2010) — %V_bur

---

## Changelog

| Version | Highlights |
|---|---|
| v2.9 | MinMax normalisation, alpha=0.5, ls_bounds, differential_evolution optimiser |
| v3.0 | Dashboard 16:9 (single overwritten file), PDP, physicochemical encoding |
| v3.1 | TEI acquisition (Lavayssiere 2026), diversity filter, domain reduction (opt-in) |
| v3.2 | EI-TS hybrid, UCB auto-β, VARMAX pure exploration, k-means initialisation |
| v3.3 | 100% GPyTorch — GammaPriors, learned noise, ADAM, efficient Thompson Sampling |

---

## License

Have fun with this
