# Architecture

Ce document décrit l'organisation du moteur `bo_chemistry.py` et de
l'interface `app_bo.py`. Le moteur est un fichier monolithique conséquent
(~7500 lignes) ; cette documentation sert de carte pour s'y retrouver.

> Une refonte modulaire (séparation en package) est envisagée. Le choix du
> fichier unique vient de l'historique du projet (développement itératif en
> contexte de recherche, déploiement simple sur postes de laboratoire sans
> gestion d'installation de package).

## Vue d'ensemble du flux

```
YAML config ─> SpaceEncoder ─> GPModel (fit) ─> Acquisition ─> suggestions
     │              │               │                │
 paramètres    encodage         Gaussian        EI / UCB / TEI
 + objectifs   mixte +          Process         / EHVI / qNEHVI
               descripteurs     (GPyTorch)
```

## `bo_chemistry.py` - sections principales

### Configuration et données
- `load_config()` - parsing et validation du YAML de campagne
- `generate_lhs_design()` - plan d'expériences initial (Latin Hypercube)
- `detect_multimodality()` - détection de plusieurs zones prometteuses

### Encodage
- `SpaceEncoder` - conversion espace mixte (continu + catégoriel) ↔ espace
  numérique normalisé. Gère le one-hot et l'encodage par descripteurs
  physico-chimiques (`descriptors_file`).
- `_KumaraswamyWarping` - warping non-linéaire optionnel des entrées.

### Modèle (Gaussian Process)
- `_ExactGPModel` / `_ExactGPModelComposite` - modèles GPyTorch (noyau Matérn
  ARD simple ou composite cont×cat).
- `GPModel` - wrapper principal (fit, predict, gestion du bruit appris et des
  priors Gamma).
- `GPModelWithPrior` - variante avec prior mean injecté (connaissance a priori).
- `SAASGP` - variante haute dimension (Sparse Axis-Aligned Subspace).
- `build_gp()` - fabrique le bon modèle selon la config.

### Fonctions d'acquisition
- Mono-objectif : `acq_ei`, `acq_ucb`, `acq_pi`, `acq_tei`, `acq_thompson`,
  `acq_kg`, `acq_qei`, `acq_qnei`
- Multi-objectif : `acq_qnehvi`, `_maximise_ehvi_grid`, `_maximise_qnehvi_grid`
- Portfolio : `gp_hedge_select` (sélection automatique d'acquisition)
- `maximise_acquisition()` - optimisation multi-start de l'acquisition

### Contraintes
- `check_feasibility(point, constraints)` - évalue les contraintes connues a
  priori (`cfg["constraints"]`, expressions Python sur les noms de
  paramètres) dans un scope restreint, sans builtins.
- `_feasible_acq` - wrapper autour de la fonction d'acquisition : met `-inf`
  sur les candidats infaisables, donc l'optimisation ne peut proposer que des
  points du domaine faisable.
- Garde-fou de sortie dans `suggest_experiments()` : chaque suggestion est
  revérifiée et, si infaisable (cas d'un raffinement local franchissant une
  frontière serrée), remplacée par échantillonnage de rejet.
- `generate_lhs_design()` applique le même filtrage au plan initial, sinon les
  premiers points violeraient les contraintes.
- `fit_constraint_gps` - contraintes sur sorties mesurées
  (`cfg["soft_constraints"]`) : un GP par contrainte, pondération de
  l'acquisition par la probabilité de satisfaction.

### Trust region (TuRBO)
- `TrustRegionState`, `turbo_get_trust_region`, `turbo_should_use` -
  restriction adaptative de la région de recherche en haute dimension.

### Orchestration
- `suggest_experiments()` - point d'entrée principal : choisit l'acquisition,
  applique l'autopilot, gère le batch et l'anti-duplication.
- `suggest_experiments_ehvi()` - variante multi-objectif.
- `autopilot_decide()` - décision de phase (EARLY/MID/LATE) et choix de
  stratégie selon progress + coverage.
- `suggest_ei_ts()` - hybride EI + Thompson Sampling.
- `apply_domain_reduction()` - réduction de domaine optionnelle.

### Visualisation
- `plot_dashboard`, `plot_convergence_animation`, `plot_trajectory_pca`,
  `compute_pdp`, `get_influential_params` - graphiques et analyse.

### Robustesse (déclencheurs réactifs)
Détection de stagnation, dérive, outliers, et ajustements automatiques
(`auto_xi`, `noise_prior` auto, switch EI→UCB) intégrés dans la boucle de
suggestion.

## `app_bo.py` - interface desktop

Backend pywebview exposant une API Python au JavaScript :
- Gestion des campagnes (CRUD, persistance disque YAML + CSV)
- Validation des entrées (noms sûrs, configs cohérentes)
- `suggest()` - appelle le moteur en capturant ses logs (badges + console)
- `get_convergence()`, `get_acq_1d()` - données pour les graphes

L'interface (`static/index.html`) est en HTML/CSS/JS vanilla avec Plotly.js
pour les graphes. Toute la logique de présentation y est ; le calcul reste
côté Python.

## Persistance

Chaque campagne = un dossier `campaigns/<nom>/` contenant `config.yaml`
(configuration) et `results.csv` (expériences + résultats, source de vérité).
Toute action écrit immédiatement sur disque ; l'interface ne conserve pas
d'état volatil critique.
