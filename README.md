**English** | [Français](README.fr.md)

# BO Chemistry - Bayesian Optimization for Experimental Chemistry

A Bayesian optimization engine for synthetic chemistry campaigns (cross-couplings, catalysis, photochemistry), with a desktop interface to run campaigns without the command line.

The goal: suggest the most promising experimental conditions to test next, based on the results already obtained, to reach a target yield (or any other objective) in as few experiments as possible.

> Developed alongside research in synthetic organic chemistry. The optimization engine is the core of the project; the desktop interface is a layer on top of it for day-to-day lab use.

<!-- ![Interface](docs/screenshot.png) -->

## Features

- **Single- and multi-objective Bayesian optimization** over mixed parameter spaces (continuous variables such as temperature or time, categorical ones such as ligand, solvent or base).
- **Adaptive autopilot**: automatic exploration/exploitation strategy depending on the progress of the campaign (EARLY / MID / LATE phases).
- **Physicochemical descriptors**: categorical variables encoded through molecular properties (RDKit / xTB / Mordred) instead of one-hot vectors, so the model can generalise between similar compounds.
- **Experimental constraints**: constraints known in advance on the parameters (bounded sum of equivalents, ratio between reagents, forbidden combination) and constraints on measured results (keep a by-product low while optimising the yield). The acquisition only proposes feasible points, and the initial design respects them too.
- **Logarithmic scale** per variable, for quantities whose effect is log-linear (concentration, equivalents, catalyst loading, ratios).
- **Adjustable measurement reliability**: the GP noise prior follows the declared analytical precision, so the model does not smooth away a real peak as if it were noise.
- **Dimension-aware initial design**: the size of the LHS design accounts for continuous variables and categorical levels, with a maximin criterion for coverage.
- **Desktop interface**: campaign creation, result entry, suggestions, convergence and acquisition plots, summary of the experimental cost before launch, all offline.

## Tech stack

| Area | Tools |
|---|---|
| Model | Gaussian Process (GPyTorch / PyTorch), Matérn ARD kernel |
| Acquisitions | EI, UCB, PI, Thompson, TEI, EI-TS, EHVI / qNEHVI (multi-objective) |
| Advanced methods | TuRBO (trust region), Box-Cox warping, prior mean injection, Kumaraswamy input warping |
| Constraints | Known constraints (constrained acquisition) and constraints on measured outputs (one GP per constraint) |
| Descriptors | RDKit, xTB, Mordred, morfeus |
| Interface | pywebview (native desktop), HTML/CSS/JS, Plotly.js |
| Data | NumPy, pandas, scikit-learn, SciPy |

## How the engine works

The engine (`bo_chemistry.py`) implements a complete Bayesian optimization loop:

1. **Encoding** of a mixed parameter space (continuous + categorical) into a normalised numerical space, with optional physicochemical descriptors for categorical variables.
2. **Surrogate model**: GPyTorch Gaussian Process with learned noise, Gamma priors and a Matérn 5/2 ARD kernel (or a composite continuous x categorical kernel). Noise priors are calibrated on standardised `y`: a prior tuned for the raw scale makes the GP overestimate the noise and flatten the best points (see changelog v3.9).
3. **Acquisition function** selected from context (autopilot) or chosen manually, optimised by multi-start search.
4. **Robustness strategies**: stagnation, drift and outlier detection; adaptive trust region; anti-duplication filtering of batches.

At each iteration the autopilot computes a `progress` (best / target) and a `coverage` (fraction of the space explored), and switches the acquisition strategy accordingly.

### Constraints

Two families are handled, following the distinction made by Hickman, Aldeghi, Häse and Aspuru-Guzik (*Bayesian Optimization with Known Experimental and Design Constraints for Chemistry Applications*, Digital Discovery 2022, 1, 732):

- **Known constraints** (`constraints` in the YAML): boolean expressions on the parameters, evaluated before any experiment. The optimisation of the acquisition function is restricted to the feasible domain, and the initial design is filtered the same way. They encode equipment or process limits, and more generally the chemist's expert knowledge.
- **Constraints on measured outputs** (`soft_constraints`): one GP per constraint models the probability of satisfying it, which weights the acquisition (Gardner / Gelbart approach). Useful when the constrained quantity is only known after the experiment, for example a by-product to keep below a threshold.

## Validation

The engine was validated in emulation on published HTE datasets (Reizman-Jensen Suzuki, Doyle Buchwald-Hartwig) and on synthetic benchmark functions. On the Doyle coupling (4 categorical dimensions), the optimizer with RDKit descriptors reaches ~96 % yield in 30 experiments, versus ~88 % for random sampling.

> The datasets are not redistributed in this repository. They are available from the original publications.

## Installation

Python 3.10+; Anaconda or Miniconda recommended. PyTorch runs on CPU, no GPU needed.

```bash
git clone https://github.com/RHeinrich97/BO-python-script.git
cd BO-python-script
conda create -n bo python=3.11 -y
conda activate bo
pip install -r requirements.txt

# Plotly.js for the interface charts (downloaded once into static/)
python scripts/setup_assets.py
```

Step-by-step instructions and troubleshooting: see the *Installation & launch* chapter of the guide, `bayes_learning_guide_v3.html` (open it in a browser).

## Launch

**Desktop interface:**
```bash
python app_bo.py
```

**Command-line engine:**
```bash
python bo_chemistry.py my_campaign.yaml --init-doe   # first run: initial design
python bo_chemistry.py my_campaign.yaml              # next runs: suggestions
```

Start from `config_test.yaml` (annotated template) or from the examples in `examples/`.

## Repository layout

```
.
├── bo_chemistry.py               # Bayesian optimization engine
├── app_bo.py                     # desktop interface backend
├── static/
│   └── index.html                # interface (HTML/CSS/JS)
├── config_test.yaml              # annotated YAML template (command line)
├── bayes_learning_guide_v3.html  # learning guide (19 chapters + installation)
├── examples/                     # test campaigns, demo data, descriptor library
├── docs/                         # architecture notes and diagrams
├── scripts/setup_assets.py       # downloads Plotly.js
├── tests/test_app.py             # interface and API test suite
└── requirements.txt
```

## Tests

```bash
python tests/test_app.py
```

The suite runs in a temporary folder and never touches your campaigns.

## Known limitations

- The benefit of Bayesian optimization depends on the problem: for small spaces or reactions with an obvious strong gradient, a classic design of experiments may be enough. The approach is most useful on mixed spaces with non-trivial interactions and a high experimental cost.
- RDKit descriptors alone capture part of the signal; electronic descriptors (xTB) bring a measurable gain but require a heavier installation.
- The engine is a large single file; a modular refactoring is planned (see `docs/architecture.md`).

## License

MIT, see [LICENSE](LICENSE).
