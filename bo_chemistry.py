#!/usr/bin/env python3
"""
bo_chemistry.py  -  Bayesian Optimisation for Chemistry
========================================================
Version  : 3.0
Reference: inspire de Gallego, Lavayssiere et al. (RSC Mechanochem. 2026)

Usage
-----
    python bo_chemistry.py                          <- detecte le .yaml automatiquement
    python bo_chemistry.py config.yaml              <- specifie le fichier
    python bo_chemistry.py config.yaml --benchmark  <- benchmark des acquisitions
    python bo_chemistry.py config.yaml --status     <- resume lecture seule
    python bo_chemistry.py config.yaml --init-doe   <- plan Latin Hypercube initial
    python bo_chemistry.py config.yaml --init-doe --doe-n 9
    python bo_chemistry.py config.yaml --final        <- rapport final + animation GIF (v3.6)
    python bo_chemistry.py config.yaml --what-if T=65 t=90 eq=2.0 <- prediction GP hypothetique
    python bo_chemistry.py config.yaml --explore      <- force VARMAX ce round (exploration pure)
    python bo_chemistry.py config.yaml --what-if T=65 t=90 --neighbors <- + sensibilites & voisins

Historique des versions
-----------------------
v2.3  MC-EHVI multi-objectif (numpy pur), rapport texte *_iter_XX_report.txt
v2.4  Analyse GP ARD (_explain_suggestion), palette daltonien, fig_dpi/fig_width,
      timestamp_outputs, mode --status, _write_run_report
v2.5  Detection de stagnation de variable (Thompson+3 automatique)
v2.6  Detection de derive directionnelle (UCB kappa=0.2 automatique),
      Latin Hypercube Sampling (--init-doe)
v2.7  Detection d'outliers GP (Leave-One-Out), avertissement saturation de borne,
      transformation Box-Cox optionnelle, detection multi-modalite (multi-start)
v2.8  Box-Cox activable (boxcox: true), detection multi-modal landscape
v2.9  Refactorisation (_get_recent_bo_window), predictions en espace reel, fusion multimodal
v3.0  Dashboard 16:9 unique ecrase (dashboard.png), PDP continu+categoriel,
      encodage physicochimique (descriptors_file), --final rapport haute resolution
v3.1  TEI (Tempered Expected Improvement), diversity filter (select_top_diverse),
      domain reduction optionnel (desactive par defaut, risque maxima locaux)
v3.2  EI-TS hybrid, UCB auto-beta, VARMAX, k-means init
v3.3  Architecture 100% GPyTorch (bruit appris, GammaPriors, ADAM) — sklearn GP supprime
v3.4  Benchmark sklearn (vitesse), log_ratio encoding, seed global, hyperparams GP dans rapport
v3.5  ADAM 100×3 (-60% temps), dashboard 3x2 avec heatmaps trio, prediction_std_BO
      dans CSV/txt, batch intelligent (diversity aware), rapport HTML parallele,
      mode --what-if, suppression plot_final (code mort)
v3.6  Incertitude epistemique vs aleatoire (.txt + HTML), dashboard secondaire
      PCA trajectoire, SHAP manuel (pourquoi ce point ?), Warped GP Kumaraswamy
      (toujours actif), animation GIF convergence (--final uniquement),
      Knowledge Gradient exact (acquisition: "KG")
v3.7  Mode --explore (VARMAX forcee), --what-if --neighbors (sensibilites +
      voisins prometteurs), contraintes de faisabilite (YAML constraints),
      prior knowledge injection (prior_mean linear/threshold),
      kernel composite (additif + multiplicatif, bo_settings.kernel)
v3.8  Auto-Pilot Orchestrator : decision de phase a 3 criteres (progres,
      couverture, stagnation) avec seuils qui scalent en dimension ;
      acquisition choisie par phase (EARLY/MID/LATE) ; lengthscales ARD
      exploitees pour la couverture
v3.9  Correctif majeur GP : les priors de bruit etaient calibres pour des
      rendements bruts (0-100) alors que y est standardise (variance 1).
      Le prior "low" GammaPrior(1.1, 0.05) attendait un bruit de variance
      moyenne 22 -> le GP surestimait le bruit d'un facteur ~2 et lissait
      les vrais pics (residu +7.5 sur le meilleur point mesure).
      Recalibrage sur y standardise : low Gamma(2.0, 100) mode 0.01,
      medium Gamma(2.0, 20) mode 0.05, high Gamma(2.0, 6.67) mode 0.15.
      Verifie : bruit estime 5.2 -> 1.7 points, residu sur le hit 7.5 -> 1.0,
      le GP apprend desormais la penalite haute temperature.
      Aussi : design initial LHS dimension-aware (compte les niveaux
      categoriels) + maximin (30 candidats, maximise la distance minimale) ;
      LHS et suggestions filtres par les contraintes hard (check_feasibility,
      remplacement par echantillonnage de rejet) ; contraintes soft exposees
      (GP par contrainte, a la Gardner/Gelbart) ; encodage log_ratio et
      noise_prior pilotables depuis l'interface

Workflow
--------
    1. Creez votre .yaml (copiez config_template.yaml, adaptez-le)
    2. Placez bo_chemistry.py et votre .yaml dans le MEME dossier
    3. Premier lancement -> le CSV est cree et s'ouvre automatiquement
    4. Remplissez vos donnees initiales, sauvegardez le CSV
    5. Relancez -> l'algo propose N nouvelles conditions + graphiques
    6. Faites les experiences, remplissez le CSV, changez statut -> 'fait'
    7. Relancez -> iteration suivante
    8. Quand converge : graphiques finaux automatiques

Nouveautes v2.2
---------------
    - Graphiques non-ecrasants : chaque round sauvegarde iteration_XX_convergence.png
    - Figure 2b style : courbe meilleur cumule + potentiel GP + experiences
    - Figure 2c style : carte contour 2D + surface 3D du paysage GP
      (2 parametres les plus influents selon les length-scales ARD)
    - Detection automatique de convergence (patience configurable)
    - --benchmark : comparaison EI / UCB / PI / Thompson / Random sur donnees actuelles

Dependances
-----------
    pip install numpy pandas scipy scikit-learn pyyaml matplotlib gpytorch torch
"""

import sys
import os
import subprocess
import platform
import warnings
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401  (active la projection 3D)
from scipy.stats import norm
from scipy.optimize import minimize, differential_evolution

# GPyTorch — requis pour le GP principal
# Force CPU-only avant import (évite init CUDA)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    import gpytorch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

warnings.filterwarnings("ignore")

# Palette coherente pour les graphiques
_COLORS = {
    "best_line" : "#2176AE",   # bleu steelblue -> courbe meilleur cumule
    "initial"   : "#888888",   # gris -> experiences initiales
    "bo_dot"    : "#8B1C30",   # bordeaux -> experiences BO mesurees
    "potential" : "#E8A838",   # or -> potentiel GP (prediction)
    "best_star" : "#5C0033",   # violet -> meilleur point
    "EI"        : "#2176AE",
    "UCB"       : "#8B1C30",
    "PI"        : "#E8A838",
    "Thompson"  : "#D62839",
    "Random"    : "#3A7D44",
}

# ── Palette dalton-friendly (Wong 2011, Nature Methods) ──────────────────────
_COLORS_CB = {
    "best_line" : "#0072B2",
    "initial"   : "#999999",
    "bo_dot"    : "#D55E00",
    "potential" : "#E69F00",
    "best_star" : "#CC79A7",
    "EI"        : "#0072B2",
    "UCB"       : "#D55E00",
    "PI"        : "#E69F00",
    "Thompson"  : "#009E73",
    "Random"    : "#56B4E9",
}


def _fig_opts(cfg: dict) -> dict:
    """Lit les options graphiques du YAML (dpi, fig_width, colorblind, timestamp)."""
    bo = cfg.get("bo_settings", {})
    return {
        "dpi"       : int(bo.get("fig_dpi", 150)),
        "fig_width" : float(bo.get("fig_width", 14.0)),
        "colorblind": bool(bo.get("colorblind_mode", False)),
        "timestamp" : bool(bo.get("timestamp_outputs", False)),
    }


def _ts_path(campaign: str, suffix: str, opts: dict) -> str:
    """Construit le nom de fichier de sortie, avec timestamp optionnel."""
    import datetime
    prefix = (datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
               if opts.get("timestamp") else "")
    return _unique_filename(f"{campaign}_{prefix}{suffix}")




# =============================================================================
# 0b.  LATIN HYPERCUBE SAMPLING — GENERATION DE POINTS INITIAUX
# =============================================================================
#
# Le Latin Hypercube Sampling (LHS) divise chaque dimension en N intervalles
# egaux et place exactement un point par intervalle dans chaque dimension.
# Cela garantit une couverture uniforme de l'espace, contrairement aux coins
# ou aux points aleatoires qui laissent souvent des zones vides.
#
# Exemple pour 3 variables, N=6 :
#   Temperature : [25-34] [34-43] [43-52] [52-61] [61-70] [70-80]
#   Temps       : [15-35] [35-55] [55-75] [75-95] [95-115] [115-140]
#   Equivalents : [0.5-1] [1-1.5] [1.5-2] [2-2.5] [2.5-3] [3-3.5]
#   → Un seul point par intervalle, en permutat aléatoirement les intervalles.
#   → Aucune dimension n'est sous-représentée.
#
# Variables categorielles : chaque categorie est representee
# autant que possible (round-robin aleatoire).
#
# Usage :
#   python bo_chemistry.py config.yaml --init-doe          (N auto = 2*n_params)
#   python bo_chemistry.py config.yaml --init-doe --doe-n 9
# =============================================================================

def generate_lhs_design(cfg, n_points=None, rng=None):
    """
    Genere un plan Latin Hypercube pour les points initiaux.

    v3.9 :
      - Le nombre de points par defaut scale avec la dimension reelle :
        continus ET niveaux categoriels (au lieu de 2*n_cont+1).
        Regle : max(2*(d_cont+1), n_cat_max, ~4*d_cont) plafonne raisonnablement.
      - LHS *maximin* : on tire plusieurs LHS candidats et on garde celui qui
        maximise la distance minimale entre points (meilleure couverture
        multidimensionnelle, evite les trous d'un LHS vanilla).

    Retourne un DataFrame CSV-ready avec les colonnes du CSV de campagne
    et statut = 'initial'.

    Parametres :
        cfg      : configuration chargee depuis le YAML
        n_points : nombre de points (defaut : calcule selon la dimension)
        rng      : generateur aleatoire (reproductibilite)
    """
    if rng is None:
        rng = np.random.default_rng(cfg["bo_settings"].get("seed", 42))

    params       = cfg["parameters"]
    output_names = [o["name"] for o in cfg["outputs"]]
    cont_params  = [p for p in params if p["type"] == "continuous"]
    cat_params   = [p for p in params if p["type"] == "categorical"]

    n_cont = len(cont_params)
    # Nombre maximal de niveaux parmi les variables categorielles
    n_cat_max = max([len(p["categories"]) for p in cat_params], default=0)

    if n_points is None:
        # Scaling dimensionnel :
        #   - au moins 2*(n_cont+1) pour identifier le GP continu,
        #   - au moins n_cat_max pour voir chaque niveau categoriel au moins 1x,
        #   - une marge ~ 3*n_cont pour la couverture.
        # On prend le max, borne par un plafond raisonnable (labo : <= 24 init).
        base_cont = max(2 * (n_cont + 1), 3 * n_cont)
        n_points = max(base_cont, n_cat_max, 5)
        # Si beaucoup de niveaux categoriels, garantir ~1.5 point/niveau
        if n_cat_max > 0:
            n_points = max(n_points, int(np.ceil(1.5 * n_cat_max)))
        n_points = int(min(n_points, 24))

    # ── LHS maximin pour variables continues ─────────────────────────────────
    def _one_lhs(n_pts, n_dim, _rng):
        m = np.zeros((n_pts, n_dim))
        for j in range(n_dim):
            intervals = (np.arange(n_pts) + _rng.uniform(0, 1, n_pts)) / n_pts
            perm      = _rng.permutation(n_pts)
            m[:, j] = intervals[perm]
        return m

    if n_cont > 0:
        # Tirer plusieurs candidats, garder le meilleur (maximin)
        n_candidates = 30
        best_m = None
        best_score = -np.inf
        for _ in range(n_candidates):
            m = _one_lhs(n_points, n_cont, rng)
            # distance min entre paires (a maximiser)
            from scipy.spatial.distance import pdist
            if n_points > 1:
                dmin = float(pdist(m).min())
            else:
                dmin = 0.0
            if dmin > best_score:
                best_score = dmin
                best_m = m
        lhs_matrix = best_m
    else:
        lhs_matrix = np.zeros((n_points, 0))

    # Denormaliser vers les vraies valeurs
    rows = []
    for i in range(n_points):
        row = {}
        for j, p in enumerate(cont_params):
            lo, hi     = float(p["min"]), float(p["max"])
            val        = lo + lhs_matrix[i, j] * (hi - lo)
            row[p["name"]] = round(val, 4)

        for o in output_names:
            row[o] = ""
        row["statut"]        = "initial"
        row["iteration_BO"]  = ""
        row["prediction_BO"] = ""
        rows.append(row)

    # ── Stratification categorielle (v3.7b) ───────────────────────────────
    _strat = cfg.get("bo_settings", {}).get("stratified_init", True)
    for p in cat_params:
        cats   = p["categories"]
        n_cats = len(cats)
        if not _strat or n_cats < 2:
            for i in range(n_points):
                rows[i][p["name"]] = cats[int(rng.integers(0, n_cats))]
            continue
        n_repeat = max(1, n_points // n_cats)
        stratified = []
        for _ in range(n_repeat):
            stratified.extend(cats)
        while len(stratified) < n_points:
            stratified.append(cats[int(rng.integers(0, n_cats))])
        perm_cats = list(rng.permutation(len(stratified))[:n_points])
        stratified_perm = [stratified[k] for k in perm_cats]
        for i in range(n_points):
            rows[i][p["name"]] = stratified_perm[i]

    # ── Filtrage par contraintes hard (v3.9) ──────────────────────────────
    # Le LHS brut ignore les contraintes ; on remplace chaque point infaisable
    # par un echantillon aleatoire faisable (rejet, jusqu'a 300 essais/point).
    _hard = cfg.get("constraints", []) or []
    if _hard:
        for i in range(len(rows)):
            point = {p["name"]: rows[i][p["name"]] for p in params}
            feas, _ = check_feasibility(point, _hard)
            if feas:
                continue
            for _ in range(300):
                cand = {}
                for p in cont_params:
                    lo, hi = float(p["min"]), float(p["max"])
                    cand[p["name"]] = round(float(rng.uniform(lo, hi)), 4)
                for p in cat_params:
                    cats = p["categories"]
                    cand[p["name"]] = cats[int(rng.integers(0, len(cats)))]
                f2, _ = check_feasibility(cand, _hard)
                if f2:
                    for k, v in cand.items():
                        rows[i][k] = v
                    break

    return pd.DataFrame(rows)


def generate_kmeans_design(cfg, n_points=None, rng=None, cfg_dir="."):
    """
    Sélection initiale par k-means sur l espace encodé.

    Avantages sur LHS :
      - LHS : couvre uniformément l espace des paramètres continus,
        mais ignore les relations entre variables et les clusters naturels.
      - K-means : identifie des groupes cohérents dans l espace encodé,
        puis retourne le point le plus proche du centroïde de chaque cluster.
        Assure que les expériences initiales couvrent des zones chimiquement
        distinctes (surtout utile avec des descripteurs physicochimiques).

    Algorithme (d après EDBO k_means) :
      1. Construire l espace encodé complet (grille dense ou LHS dense)
      2. Appliquer k-means pour batch_size clusters
      3. Pour chaque cluster, prendre le point le plus proche du centroïde
      4. Décoder vers les paramètres réels

    Retourne un DataFrame CSV-ready (mêmes colonnes que generate_lhs_design).
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    if rng is None:
        rng = np.random.default_rng(cfg.get("bo_settings", {}).get("seed", 42))

    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    bo           = cfg.get("bo_settings", {})
    enc          = SpaceEncoder(cfg, cfg_dir=cfg_dir)

    if n_points is None:
        n_cont = sum(1 for p in cfg["parameters"] if p["type"] == "continuous")
        n_cat  = sum(1 for p in cfg["parameters"] if p["type"] == "categorical")
        n_points = max(2 * n_cont + n_cat + 1, 6)

    # ── Grille de candidats encodés ──────────────────────────────────────────
    # Pour les continues : LHS dense
    # Pour les catégorielles : toutes les combinaisons
    cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]
    cat_params  = [p for p in cfg["parameters"] if p["type"] == "categorical"]

    # LHS dense (10 000 points) pour les continues
    n_cand = int(bo.get("kmeans_n_candidates", 10000))
    lhs_rng = rng.random((n_cand, len(cont_params)))  # [0,1]

    # Dénormaliser les continues
    cont_vals = np.zeros_like(lhs_rng)
    for k, p in enumerate(cont_params):
        lo, hi = float(p["min"]), float(p["max"])
        cont_vals[:, k] = lhs_rng[:, k] * (hi - lo) + lo

    if cat_params:
        # Tirer les catégorielles aléatoirement pour chaque candidat
        cat_choices = []
        for p in cat_params:
            cats = p["categories"]
            idx  = rng.integers(0, len(cats), n_cand)
            cat_choices.append([cats[i] for i in idx])
        cat_arr = np.array(cat_choices).T  # (n_cand, n_cat)
    else:
        cat_arr = np.zeros((n_cand, 0), dtype=object)

    # Assembler X_raw dans l ordre des paramètres
    X_raw_list = []
    ci = 0
    ki = 0
    for p in cfg["parameters"]:
        if p["type"] == "continuous":
            X_raw_list.append(cont_vals[:, ci].reshape(-1,1))
            ci += 1
        else:
            X_raw_list.append(cat_arr[:, ki].reshape(-1,1))
            ki += 1
    X_raw = np.hstack(X_raw_list).astype(object)
    for ci2, p in enumerate(cont_params):
        col_idx = [j for j,p2 in enumerate(cfg["parameters"]) if p2["name"]==p["name"]][0]
        X_raw[:, col_idx] = X_raw[:, col_idx].astype(float)

    # Encoder
    X_enc = enc.encode(X_raw)

    # ── K-means sur l espace encodé ───────────────────────────────────────────
    seed_km = int(rng.integers(0, 2**31))
    kmeans  = KMeans(n_clusters=n_points, random_state=seed_km,
                     n_init=10, max_iter=300)
    kmeans.fit(X_enc)

    # Point le plus proche de chaque centroïde
    chosen_rows = []
    for ci3 in range(n_points):
        cluster_mask = kmeans.labels_ == ci3
        if cluster_mask.sum() == 0:
            continue
        cluster_X    = X_enc[cluster_mask]
        cluster_raw  = X_raw[cluster_mask]
        center       = kmeans.cluster_centers_[[ci3]]
        closest_idx, _ = pairwise_distances_argmin_min(center, cluster_X)
        chosen_rows.append(cluster_raw[closest_idx[0]])

    # ── Décoder et formater ───────────────────────────────────────────────────
    rows = []
    for raw_row in chosen_rows:
        decoded = enc.decode(enc.encode(raw_row.reshape(1,-1)))[0]
        row = {}
        for p in cfg["parameters"]:
            row[p["name"]] = decoded[p["name"]]
        for o in output_names:
            row[o] = ""
        row["statut"]       = "initial"
        row["iteration_BO"] = ""
        row["prediction_BO"]= ""
        rows.append(row)

    df_init = pd.DataFrame(rows)
    print(f"  [OK]   K-means init : {len(df_init)} pts dans {n_points} clusters "
          f"(espace {X_enc.shape[1]}D encodé)")
    return df_init


def run_init_doe(cfg, csv_path, n_points, rng):
    """Genere le CSV avec le plan LHS et l'ouvre."""
    print(f"\n  Generation du plan LHS : {n_points} points")
    cont  = [p for p in cfg["parameters"] if p["type"] == "continuous"]
    cat   = [p for p in cfg["parameters"] if p["type"] == "categorical"]
    print(f"  Variables continues    : {len(cont)}")
    print(f"  Variables categorielles: {len(cat)}")
    print(f"  Couverture garantie    : chaque dimension divisee en {n_points} intervalles")

    init_method = cfg.get("bo_settings", {}).get("init_method", "lhs").lower()
    if init_method == "kmeans":
        cfg_dir = cfg.get("_cfg_dir", ".")
        df = generate_kmeans_design(cfg, n_points=n_points, rng=rng, cfg_dir=cfg_dir)
    else:
        df = generate_lhs_design(cfg, n_points, rng)
    save_csv(df, csv_path, cfg=cfg)

    print(f"\n  Plan LHS genere : {csv_path}")
    print(f"  Etapes suivantes :")
    print(f"    1. Ouvrez {csv_path}")
    print(f"    2. Realisez les {n_points} experiences")
    print(f"    3. Remplissez les colonnes de sortie")
    print(f"    4. Changez statut -> 'fait'")
    print(f"    5. Relancez : python bo_chemistry.py {os.path.basename(csv_path.replace('.csv','.yaml'))}")
    print(f"\n  Conseil : respectez l'ordre du plan (points bien espaces).")
    print(f"  Si une condition est irrealisable, remplacez-la par la valeur")
    print(f"  la plus proche testable et notez-le dans une colonne 'notes'.")

    # Ouvrir le CSV
    try:
        if platform.system() == "Windows":
            os.startfile(csv_path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", csv_path])
        else:
            subprocess.Popen(["xdg-open", csv_path])
    except Exception:
        pass

    print()


# =============================================================================
# 0c.  BOX-COX TRANSFORMATION (bruit heteroscedastique)
# =============================================================================
#
# Probleme : le GP standard suppose que le bruit experimental est uniforme
# sur tout l'espace (homoscedastique). En chimie, ce n'est presque jamais vrai :
#   - Les rendements bas (0-20%) ont souvent plus de bruit relatif que 80-100%.
#   - Les mesures d'ee% sont tres stables pres de 0 et bruitees pres de 100%.
#
# Solution Box-Cox :
#   Avant de fitter le GP, on transforme y → y_bc = (y^λ - 1) / λ
#   Cette transformation stabilise la variance et normalise la distribution.
#   Les predictions sont ensuite retransformees dans l'espace original.
#
# Le parametre λ est trouve automatiquement par maximum de vraisemblance.
# λ=1 : pas de transformation (bruit deja homoscedastique)
# λ=0 : transformation log
# λ≈0.3-0.5 : typique pour des rendements en chimie
#
# Active via : boxcox_transform: true dans bo_settings
# =============================================================================

def detect_multimodality(df_done, cfg, encoder, bo_settings):
    """
    Detection heuristique de multi-modalite dans le paysage.
    Retourne (detected: bool, message: str).
    """
    if not bo_settings.get("multimodal_detection", False):
        return False, ""

    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return False, ""

    output_names = [o["name"] for o in cfg["outputs"]]
    param_names  = [p["name"] for p in cfg["parameters"]]
    out_name     = output_names[0]
    out_cfg      = cfg["outputs"][0]
    maximize     = out_cfg.get("direction", "maximize") == "maximize"

    if len(df_done) < 10:
        return False, ""  # pas assez de points

    Y = df_done[out_name].values.astype(float)
    X = encoder.encode(df_done[param_names].values.astype(object))

    # Top 25% des points (ou au moins 6)
    n_top   = max(8, int(len(df_done) * 0.40))  # 40% -> couvre mieux les 2 bassins
    idx_top = np.argsort(Y)[-n_top:] if maximize else np.argsort(Y)[:n_top]
    X_top   = X[idx_top]
    Y_top   = Y[idx_top]

    if len(X_top) < 4:
        return False, ""

    # K-means K=1 vs K=2
    km1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(X_top)
    km2 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_top)
    inertia_ratio = km2.inertia_ / (km1.inertia_ + 1e-9)

    # Score moyen par bassin
    labels = km2.labels_
    basin_scores = []
    for k in range(2):
        mask = labels == k
        if mask.sum() < 2:
            return False, ""
        basin_scores.append(Y_top[mask].mean())

    score_gap = abs(basin_scores[0] - basin_scores[1])
    y_range   = float(Y.max() - Y.min())

    # Criteres : bassins geometriquement separes + scores proches
    geom_separated = inertia_ratio < 0.35
    scores_similar = score_gap < 0.15 * y_range  # moins de 15% d ecart

    if not (geom_separated and scores_similar):
        return False, ""

    sep62 = "=" * 62
    sep2  = "-" * 62
    lines = [f"\n  {sep62}"]
    lines.append(f"  [MULTI-MODALITE POTENTIELLE DETECTEE]")
    lines.append(f"  {len(idx_top)} meilleurs points analyses  |  2 bassins identifies")
    lines.append(f"  {sep2}")
    lines.append(f"  Bassin A : score moyen = {basin_scores[0]:.2f}%")
    lines.append(f"  Bassin B : score moyen = {basin_scores[1]:.2f}%")
    lines.append(f"  Ecart entre bassins  : {score_gap:.2f}%  (< 15% → bassins similaires)")
    lines.append(f"  Separation geometrique : {inertia_ratio:.3f}  (< 0.35 → bassins distincts)")
    lines.append(f"  {sep2}")
    lines.append(f"  Interpretation : l espace chimique contient peut-etre 2 zones")
    lines.append(f"  distinctes avec des resultats similaires.")
    lines.append(f"  Le GP se concentre probablement sur l une d elles.")
    lines.append(f"  {sep2}")
    lines.append(f"  Suggestion : lancez quelques experiences manuelles dans")
    lines.append(f"  l autre bassin pour confirmer. Ou utilisez Thompson n=3")
    lines.append(f"  pour echantillonner les deux zones en parallele.")
    lines.append(f"  {sep62}")

    return True, "\n".join(lines)


# =============================================================================
# 1.  CHARGEMENT DE LA CONFIGURATION
# =============================================================================

def trouver_config(argument):
    """
    Resout le chemin du fichier YAML de configuration.
    Ordre : argument fourni -> a cote du script -> detection automatique.
    """
    dossier_script = os.path.dirname(os.path.abspath(__file__))

    if argument is not None:
        if os.path.isfile(argument):
            return os.path.abspath(argument)
        cote_script = os.path.join(dossier_script, os.path.basename(argument))
        if os.path.isfile(cote_script):
            return cote_script
        print(f"\n  [ERREUR] Fichier de configuration introuvable : '{argument}'")
        print(f"  Recherche dans : {os.path.abspath(argument)}")
        print(f"             et : {cote_script}\n")
        sys.exit(1)

    yamls = sorted(f for f in os.listdir(dossier_script) if f.endswith((".yaml", ".yml")))
    if len(yamls) == 0:
        print(f"\n  [ERREUR] Aucun fichier .yaml trouve dans : {dossier_script}\n")
        sys.exit(1)
    if len(yamls) == 1:
        chemin = os.path.join(dossier_script, yamls[0])
        print(f"\n  [CONFIG] Fichier detecte automatiquement : {yamls[0]}")
        return chemin
    print(f"\n  Plusieurs fichiers de configuration trouves :")
    for i, y in enumerate(yamls, 1):
        print(f"    {i}. {y}")
    while True:
        choix = input("\n  Entrez le numero de votre campagne : ").strip()
        if choix.isdigit() and 1 <= int(choix) <= len(yamls):
            return os.path.join(dossier_script, yamls[int(choix) - 1])
        print("  Numero invalide, reessayez.")


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key in ["campaign_name", "parameters", "outputs", "bo_settings"]:
        if key not in cfg:
            raise ValueError(f"Cle manquante dans le fichier de config : '{key}'")
    for i, p in enumerate(cfg["parameters"]):
        if "name" not in p or "type" not in p:
            raise ValueError(f"Parametre #{i} : champs 'name' et 'type' obligatoires.")
        if p["type"] == "continuous" and ("min" not in p or "max" not in p):
            raise ValueError(f"Parametre '{p['name']}' : 'min' et 'max' obligatoires.")
        if p["type"] == "categorical" and "categories" not in p:
            raise ValueError(f"Parametre '{p['name']}' : 'categories' obligatoire.")
    for i, o in enumerate(cfg["outputs"]):
        if "name" not in o:
            raise ValueError(f"Output #{i} : champ 'name' obligatoire.")
    return cfg


# =============================================================================
# 2.  GESTION DU CSV
# =============================================================================

def get_csv_path(cfg):
    return cfg["campaign_name"].strip() + ".csv"


def get_all_columns(cfg):
    """Colonnes obligatoires du CSV (sans prediction_BO pour compat. arriere)."""
    return (
        [p["name"] for p in cfg["parameters"]] +
        [o["name"] for o in cfg["outputs"]] +
        ["statut", "iteration_BO"]
    )




def ouvrir_fichier(chemin):
    try:
        if platform.system() == "Windows":
            os.startfile(chemin)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", chemin])
        else:
            subprocess.Popen(["xdg-open", chemin],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def create_csv(cfg, csv_path):
    """Cree le CSV vide, l'ouvre, affiche les instructions."""
    # prediction_BO : prediction du GP au moment de la suggestion (stocke pour les graphiques)
    cols = get_all_columns(cfg) + ["prediction_BO"]
    pd.DataFrame(columns=cols).to_csv(csv_path, index=False, encoding="utf-8-sig")

    n_params   = len(cfg["parameters"])
    n_min      = int(cfg["bo_settings"].get("min_initial_points", 2 * n_params))
    param_noms = [p["name"] for p in cfg["parameters"]]
    out_noms   = [o["name"] for o in cfg["outputs"]]

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  ETAPE 1  -  Fichier de donnees cree : {csv_path}")
    print(f"{sep}")
    print(f"\n  Le fichier s'ouvre dans votre tableur (Excel / LibreOffice).")
    print(f"\n  COLONNES du fichier :")
    print(f"  Parametres (vos conditions) : {', '.join(param_noms)}")
    print(f"  Sorties    (vos mesures)    : {', '.join(out_noms)}")
    print(f"  Gestion    : statut, iteration_BO, prediction_BO (auto)")
    print(f"""
  CONSIGNES :
    1. Ajoutez une ligne par experience deja realisee au labo
    2. Remplissez toutes les colonnes de parametres ET de sorties
    3. Dans la colonne  statut  : ecrivez  fait
    4. Laissez  iteration_BO  et  prediction_BO  vides
    5. Minimum recommande : {n_min} experiences ({n_params} params x 2)
    6. Sauvegardez le CSV puis relancez le script
""")
    ouvrir_fichier(csv_path)


def load_csv(csv_path):
    return pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str)


def save_csv(df, csv_path, cfg=None):
    """Sauvegarde le CSV en garantissant l ordre canonique des colonnes."""
    if cfg is not None:
        canonical = (
            [p["name"] for p in cfg["parameters"]] +
            [o["name"] for o in cfg["outputs"]] +
            ["statut", "iteration_BO", "prediction_BO", "prediction_std_BO"]
        )
        ordered = [c for c in canonical if c in df.columns]
        extra   = [c for c in df.columns if c not in ordered]
        df = df[ordered + extra]
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


# =============================================================================
# 3.  ENCODAGE DE L'ESPACE (continu + categoriel -> [0,1]^d)
# =============================================================================

def _load_physchem_descriptors(param, cfg_dir="."):
    """
    Charge les descripteurs physicochimiques pour une variable categorielle.

    Le YAML peut specifier :
        descriptors_file    : chemin vers l Excel (relatif au YAML)
        descriptors_sheet   : nom de la feuille
        descriptors_columns : liste des colonnes a utiliser

    Retourne un dict {nom_categorie: np.array(descripteurs)} ou None.
    """
    desc_file = param.get("descriptors_file", None)
    if desc_file is None:
        return None
    import pandas as pd_inner

    path = os.path.join(cfg_dir, desc_file)
    if not os.path.isfile(path):
        print(f"  [WARN] Fichier descripteurs introuvable : {path} — encodage one-hot utilise")
        return None

    sheet   = param.get("descriptors_sheet", None)
    columns = param.get("descriptors_columns", None)

    try:
        df_desc = pd_inner.read_excel(path, sheet_name=sheet) if sheet else pd_inner.read_excel(path)
    except Exception as e:
        print(f"  [WARN] Impossible de lire {path} : {e} — one-hot utilise")
        return None

    # Premiere colonne = noms des composés
    name_col = df_desc.columns[0]
    if columns is None:
        # Toutes les colonnes numeriques sauf la premiere et SMILES
        columns = [c for c in df_desc.columns[1:]
                   if pd_inner.api.types.is_numeric_dtype(df_desc[c])]

    result = {}
    for _, row in df_desc.iterrows():
        name = str(row[name_col]).strip()
        vals = []
        for col in columns:
            v = row.get(col, None)
            try:
                v = float(v)
                if pd_inner.isna(v): v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            vals.append(v)
        result[name] = np.array(vals, dtype=float)

    print(f"  [OK]   Descripteurs physicochim. : {len(result)} composés,"
          f" {len(columns)} descripteurs ({', '.join(str(c) for c in columns[:3])}...)")
    return result


class SpaceEncoder:
    """
    Encodeur de l espace experimental.

    Variables continues     : normalisation [0,1] via min/max du YAML.
    Variables categorielles : two modes —
        - one-hot ordinal (defaut) : cats[i] / (n-1)
        - physicochimique (si descriptors_file) : valeurs numeriques
          normalisees en [0,1] par colonne, concat apres la variable

    Avec encodage physicochimique, n_params augmente :
      chaque catégorielle avec N descripteurs remplace 1 dim par N dims.
    """
    def __init__(self, cfg, cfg_dir="."):
        self.params          = cfg["parameters"]
        self.continuous_idx  = []
        self.categorical_idx = []
        self.bounds_phys     = []
        self.cat_lists       = []
        self._cat_pos        = {}
        # Physicochemical encoding per categorical param
        self._physchem       = {}   # cat_pos -> {name: np.array}
        self._physchem_scaler= {}   # cat_pos -> (min_arr, scale_arr)
        self._physchem_ncols = {}   # cat_pos -> n_descriptors
        self._encoded_dims   = []   # dim count per param (in order)
        self._log_ratio_idx  = set()  # indices de params avec encoding: log_ratio

        cat_count = 0
        for i, p in enumerate(self.params):
            if p["type"] == "continuous":
                self.continuous_idx.append(i)
                self.bounds_phys.append((float(p["min"]), float(p["max"])))
                self._encoded_dims.append(1)
                # Log ratio : stocker le flag pour encode/decode
                if p.get("encoding") == "log_ratio":
                    self._log_ratio_idx.add(i)
            else:
                self.categorical_idx.append(i)
                self.cat_lists.append([str(c) for c in p["categories"]])
                self._cat_pos[i] = cat_count

                desc = _load_physchem_descriptors(p, cfg_dir)
                if desc:
                    # Ajouter les categories manquantes avec vecteur zero
                    cats = self.cat_lists[-1]
                    for cat in cats:
                        if cat not in desc:
                            # Fallback: one-hot like — utilise le centre
                            n_d = len(next(iter(desc.values())))
                            desc[cat] = np.zeros(n_d)
                            print(f"  [WARN] '{cat}' absent de la BDD — encodage nul utilise")

                    # Normaliser chaque colonne de descripteurs en [0,1]
                    all_vals = np.array(list(desc.values()))
                    v_min    = all_vals.min(axis=0)
                    v_range  = all_vals.max(axis=0) - v_min
                    v_range[v_range < 1e-9] = 1.0   # eviter /0
                    self._physchem[cat_count]        = desc
                    self._physchem_scaler[cat_count] = (v_min, v_range)
                    self._physchem_ncols[cat_count]  = all_vals.shape[1]
                    self._encoded_dims.append(all_vals.shape[1])
                else:
                    self._encoded_dims.append(1)
                cat_count += 1

    @property
    def n_params(self):
        """Nombre de dimensions apres encodage (peut differ du nb de parametres)."""
        return sum(self._encoded_dims)

    @property
    def n_raw_params(self):
        return len(self.params)

    def _encode_cat(self, cat_pos, val):
        """Encode une valeur categorielle -> vecteur 1D normalise."""
        cats = self.cat_lists[cat_pos]
        if cat_pos in self._physchem:
            desc   = self._physchem[cat_pos]
            v_min, v_range = self._physchem_scaler[cat_pos]
            raw    = desc.get(str(val).strip(), np.zeros(self._physchem_ncols[cat_pos]))
            return (raw - v_min) / v_range
        else:
            n_c = len(cats)
            val = str(val).strip()
            if val not in cats:
                lower_map = {c.lower(): c for c in cats}
                val = lower_map.get(val.lower(), cats[0])
            return np.array([cats.index(val) / max(n_c - 1, 1)])

    def encode(self, X_raw):
        """Encode X_raw (n, n_raw_params) -> X_norm (n, n_encoded_dims)."""
        n      = X_raw.shape[0]
        X_norm = np.zeros((n, self.n_params), dtype=float)
        dim_cursor = 0

        for param_idx, p in enumerate(self.params):
            if p["type"] == "continuous":
                k  = self.continuous_idx.index(param_idx)
                lo, hi = self.bounds_phys[k]
                raw_vals = X_raw[:, param_idx].astype(float)
                if param_idx in self._log_ratio_idx:
                    # Log ratio encoding : GP travaille en espace log
                    # Symétrique autour de 1:1, meilleur pour les stoichiométries
                    # x_enc = (ln(x) - ln(lo)) / (ln(hi) - ln(lo))
                    raw_clipped = np.clip(raw_vals, max(lo, 1e-9), hi)
                    ln_lo = np.log(max(lo, 1e-9))
                    ln_hi = np.log(max(hi, 1e-9))
                    X_norm[:, dim_cursor] = (np.log(raw_clipped) - ln_lo) / max(ln_hi - ln_lo, 1e-9)
                else:
                    X_norm[:, dim_cursor] = (raw_vals - lo) / max(hi - lo, 1e-9)
                dim_cursor += 1
            else:
                cat_pos = self._cat_pos[param_idx]
                n_dims  = self._encoded_dims[param_idx]
                for i in range(n):
                    vec = self._encode_cat(cat_pos, X_raw[i, param_idx])
                    X_norm[i, dim_cursor:dim_cursor+n_dims] = vec
                dim_cursor += n_dims

        return X_norm

    def decode(self, X_norm):
        """Decode X_norm (n, n_encoded_dims) -> list[dict]."""
        results    = []
        dim_cursor = 0
        for i in range(X_norm.shape[0]):
            row = {}
            d   = dim_cursor
            for param_idx, p in enumerate(self.params):
                n_dims = self._encoded_dims[param_idx]
                if p["type"] == "continuous":
                    k      = self.continuous_idx.index(param_idx)
                    lo, hi = self.bounds_phys[k]
                    enc_v  = float(np.clip(X_norm[i, d], 0, 1))
                    if param_idx in self._log_ratio_idx:
                        ln_lo = np.log(max(lo, 1e-9))
                        ln_hi = np.log(max(hi, 1e-9))
                        val   = float(np.exp(ln_lo + enc_v * (ln_hi - ln_lo)))
                    else:
                        val   = enc_v * (hi - lo) + lo
                    row[p["name"]] = round(val, 4)
                    d += 1
                else:
                    cat_pos = self._cat_pos[param_idx]
                    cats    = self.cat_lists[cat_pos]
                    if cat_pos in self._physchem:
                        # Trouver la catégorie la plus proche dans l espace des descripteurs
                        desc     = self._physchem[cat_pos]
                        v_min, v_range = self._physchem_scaler[cat_pos]
                        query    = X_norm[i, d:d+n_dims]
                        best_cat = min(
                            cats,
                            key=lambda c: np.sum((
                                (desc.get(c, np.zeros(n_dims)) - v_min) / v_range
                                - query) ** 2)
                        )
                        row[p["name"]] = best_cat
                    else:
                        n_c = len(cats)
                        ri  = int(np.clip(round(float(X_norm[i, d]) * max(n_c-1,1)), 0, n_c-1))
                        row[p["name"]] = cats[ri]
                    d += n_dims
            results.append(row)
        return results

    def unit_bounds(self):
        return [(0.0, 1.0)] * self.n_params

    def physchem_info(self):
        """
        Retourne les infos d encodage physico pour chaque param categoriel.
        Utile pour les PDP : dict param_name -> {categories, descriptors, values_norm}
        """
        info = {}
        for param_idx, p in enumerate(self.params):
            if p["type"] != "categorical":
                continue
            cat_pos = self._cat_pos[param_idx]
            if cat_pos not in self._physchem:
                continue
            desc         = self._physchem[cat_pos]
            v_min, v_rng = self._physchem_scaler[cat_pos]
            cats         = self.cat_lists[cat_pos]
            info[p["name"]] = {
                "categories":  cats,
                "n_desc":      self._physchem_ncols[cat_pos],
                "desc_raw":    {c: desc.get(c, np.zeros(self._physchem_ncols[cat_pos]))
                                for c in cats},
                "desc_norm":   {c: (desc.get(c, np.zeros(self._physchem_ncols[cat_pos]))
                                    - v_min) / v_rng
                                for c in cats},
                "scaler":      (v_min, v_rng),
                "param_cfg":   p,
            }
        return info


# =============================================================================
# 4.  SCALARISATION MULTI-OBJECTIF
# =============================================================================

def _sign(output_cfg):
    return 1.0 if output_cfg.get("direction", "maximize") == "maximize" else -1.0


def _apply_utility(y_col, out_cfg):
    """
    v3.7c : applique une transformation d'utilité non-lineaire par output.

    Permet d'exprimer "un gain de 50->60% vaut plus qu'un gain de 90->95%"
    pour orienter le GP vers l'amelioration des conditions encore mediocres
    plutot que les derniers % d'optimisation deja avancee.

    Champs YAML par output :
        utility: "linear" (defaut) | "log" | "sqrt"
        utility_threshold: float (par defaut 0.0) — applique seulement
                           au-dessus de ce seuil. En-dessous, lineaire.

    Pour direction='minimize', l'utilite est ignoree (toujours lineaire) :
    on veut le minimum coute que coute, pas de rendement decroissant.

    Retourne y_col transforme (meme shape).
    """
    utility = str(out_cfg.get("utility", "linear")).lower()
    if utility == "linear" or utility == "none":
        return y_col

    # Seulement sur maximize — les minimize gardent l'echelle lineaire
    direction = str(out_cfg.get("direction", "maximize")).lower()
    if direction != "maximize":
        return y_col

    threshold = float(out_cfg.get("utility_threshold", 0.0))
    y = np.asarray(y_col, dtype=float).copy()

    # Sous le seuil : echelle lineaire d'origine
    # Au-dessus : transformation concave centree sur le seuil
    above = y > threshold
    if not above.any():
        return y

    delta = y[above] - threshold
    if utility == "log":
        # log(1 + delta) — gain decroissant ; equivaut a delta pour delta<<1
        y[above] = threshold + np.log1p(np.maximum(delta, 0.0))
    elif utility == "sqrt":
        y[above] = threshold + np.sqrt(np.maximum(delta, 0.0))
    else:
        # type inconnu -> ignore silencieusement, pas de transformation
        return np.asarray(y_col, dtype=float)
    return y


def scalarise(Y, cfg, pareto_weights=None):
    outputs = cfg["outputs"]
    mode    = cfg["bo_settings"].get("multi_objective_mode", "weighted_sum")
    n_out   = len(outputs)
    Y_s = np.zeros_like(Y, dtype=float)
    for j, out in enumerate(outputs):
        # v3.7c : transformation d'utilite (log/sqrt/linear) avant signe
        y_transformed = _apply_utility(Y[:, j], out)
        Y_s[:, j] = y_transformed * _sign(out)
    if n_out == 1:
        return Y_s[:, 0]
    if mode == "weighted_sum":
        weights = np.array([out.get("weight", 1.0) for out in outputs], dtype=float)
        weights /= weights.sum()
        return Y_s @ weights
    elif mode == "pareto":
        if pareto_weights is None:
            pareto_weights = np.ones(n_out) / n_out
        w      = np.array(pareto_weights, dtype=float) / np.sum(pareto_weights)
        rho    = 0.05
        z_star = Y_s.max(axis=0)
        diff   = Y_s - z_star
        return -(np.max(w * diff, axis=1) + rho * (w * diff).sum(axis=1))
    elif mode == "threshold":
        feasible    = np.ones(len(Y), dtype=bool)
        primary     = np.zeros(len(Y), dtype=float)
        has_primary = False
        for j, out in enumerate(outputs):
            thresh = out.get("threshold", None)
            if thresh is not None:
                if _sign(out) > 0:
                    feasible &= (Y[:, j] >= float(thresh))
                else:
                    feasible &= (Y[:, j] <= float(thresh))
            else:
                primary += out.get("weight", 1.0) * Y_s[:, j]
                has_primary = True
        if not has_primary:
            weights = np.array([out.get("weight", 1.0) for out in outputs], dtype=float)
            primary = Y_s @ (weights / weights.sum())
        penalty_base = Y_s.min() - np.abs(Y_s).max() * 2.0
        return np.where(feasible, primary, penalty_base)
    else:
        raise ValueError(f"multi_objective_mode inconnu : '{mode}'.")




# =============================================================================
# 5.  MODELE SURROGATE – GP Matern 5/2
# =============================================================================


# =============================================================================
# 3b.  TRANSFORMATION BOX-COX (optionnelle)
# =============================================================================
#
# Pourquoi : le GP standard suppose que le bruit experimental est gaussien
# et UNIFORME dans tout l'espace (homoscedastique). En chimie, le bruit
# est souvent heteroscedastique : faible quand le rendement est proche de
# l'optimum (reactions bien controlees), fort dans les zones marginales.
# De plus, les rendements suivent souvent une distribution asymetrique
# (beaucoup de 0-20%, peu de 80-100%).
#
# Solution Box-Cox : transformer y avant de fitter le GP, de facon a
# rendre la distribution plus gaussienne et plus homoscedastique.
#   y_bc = (y^lambda - 1) / lambda   si lambda != 0
#   y_bc = log(y)                    si lambda = 0 (cas log)
#
# Le parametre lambda est estime automatiquement par maximum de vraisemblance.
# Apres prediction, on inverse la transformation pour revenir a l'echelle reelle.
#
# Condition : toutes les valeurs de sortie doivent etre > 0.
# Le script applique automatiquement un offset si des valeurs <= 0 existent.
#
# Parametre YAML : boxcox: true  (defaut false)
#
# Quand l'utiliser :
#   - Distribution des rendements tres asymetrique
#   - GP tres mal calibre (predictions trop optimistes ou trop pessimistes)
#   - Bruit experimental variable selon la zone de l'espace
# =============================================================================

def apply_boxcox(y, lam):
    """Applique la transformation Box-Cox."""
    if lam == 0:
        return np.log(y)
    return (y ** lam - 1.0) / lam


def invert_boxcox(y_bc, lam):
    """
    Inverse la normalisation MinMax de fit_boxcox.
    lam est ici l'echelle (y_scale), pas un lambda Box-Cox.
    """
    if lam is None:
        return np.asarray(y_bc, dtype=float)
    y_bc = np.asarray(y_bc, dtype=float)
    return np.maximum(y_bc * float(lam), 0.0)


def fit_boxcox(y_raw, bo_settings, verbose=False):
    """
    Normalisation de la sortie quand boxcox: true est active.

    Utilise une normalisation MinMax vers [0, y_scale] :
      y_t = y / y_scale
    ou y_scale = max(1.5 * max_observe, 100)

    Pourquoi MinMax plutot que log1p :
      - log1p ecrase le gradient entre les bons points (68% vs 100%
        deviennent 4.22 vs 4.61 — difference de 0.36 seulement)
      - MinMax preserve les differences relatives : 68% → 0.68, 100% → 1.00
        Le gradient vers l'optimum reste 3x plus visible pour le GP
      - Stable numeriquement : inversion = multiplication simple
      - normalize_y=True dans le GP est desactive pour eviter
        la double compression (minmax + standardisation interne)

    Retourne (y_transformed, y_scale, 0.0) ou (y_raw, None, 0.0).
    """
    if not bo_settings.get("boxcox", False):
        return y_raw, None, 0.0

    y_pos   = np.maximum(y_raw, 0.0)
    # Echelle fixee a max(1.5 * max_observe, 100) pour rester stable
    # quand de nouveaux points arriveront (evite que l'echelle change a chaque iter)
    y_scale = max(float(y_pos.max()) * 1.5, 100.0)
    y_t     = y_pos / y_scale
    if verbose:
        print(f"  [OK]   MinMax : [{y_pos.min():.2f}, {y_pos.max():.2f}]"
              f" → [0.00, {y_pos.max()/y_scale:.3f}]  (echelle={y_scale:.1f})")
    return y_t, y_scale, 0.0



# =============================================================================
# 5c.  CLASSE GP GPYTORCH (architecture principale, remplace sklearn)
# =============================================================================

def _require_torch():
    if not _TORCH_OK:
        raise ImportError(
            "GPyTorch est requis mais n est pas installe.\n"
            "Installez-le avec : pip install gpytorch torch"
        )


class _KumaraswamyWarping:
    """
    Input warping via Kumaraswamy CDF (v3.6).

    Chaque dimension d'entree est transformee independamment :
      x' = 1 - (1 - x^a)^b, avec x dans [0,1]

    a=b=1 -> identite (initialisation neutre).
    Apprise conjointement avec les hyperparametres GP via ADAM.

    Permet au GP de s'adapter a des effets non-lineaires sur les inputs
    (ex: plateau de temperature, seuil d'equivalents).
    """
    def __init__(self, n_dims):
        self.n_dims = n_dims
        # Parametres torch (raw), sigmoid pour contraindre > 0
        # init: raw=0 -> a=b=1 (identite)
        self._a_raw = None  # rempli au fit
        self._b_raw = None

    def init_params(self):
        """Cree les parametres torch (appele au fit)."""
        import torch
        # raw_a, raw_b : transformes via softplus pour rester positifs
        # softplus(0) = ln(2) ~ 0.693, proche de 1 apres exp -> warping doux
        self._a_raw = torch.nn.Parameter(torch.zeros(self.n_dims))
        self._b_raw = torch.nn.Parameter(torch.zeros(self.n_dims))

    def parameters(self):
        return [self._a_raw, self._b_raw]

    def transform(self, X_t):
        """Applique Kumaraswamy CDF par dimension. X_t : torch.Tensor (n, d)"""
        import torch
        if self._a_raw is None:
            return X_t
        # Contraintes : a, b > 0.3 (eviter effondrement) et < 5 (eviter warping extreme)
        a = 0.3 + torch.nn.functional.softplus(self._a_raw) * 1.5
        b = 0.3 + torch.nn.functional.softplus(self._b_raw) * 1.5
        # Clip x strict dans (eps, 1-eps) pour stabilite
        X_clamped = torch.clamp(X_t, min=1e-6, max=1.0 - 1e-6)
        # Kumaraswamy CDF
        return 1.0 - torch.pow(1.0 - torch.pow(X_clamped, a), b)

    def get_params_numpy(self):
        """Retourne (a, b) apres activation, en numpy (pour diagnostic)."""
        import torch
        if self._a_raw is None:
            return None, None
        with torch.no_grad():
            a = 0.3 + torch.nn.functional.softplus(self._a_raw) * 1.5
            b = 0.3 + torch.nn.functional.softplus(self._b_raw) * 1.5
        return a.numpy(), b.numpy()

class _ExactGPModel(gpytorch.models.ExactGP if _TORCH_OK else object):
    """Modele GP exact GPyTorch — Matern 5/2 ARD avec GammaPriors + warping."""

    def __init__(self, X, y, likelihood, n_dims, warper=None):
        if not _TORCH_OK:
            raise ImportError("GPyTorch requis")
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ls_prior = gpytorch.priors.GammaPrior(3.0, 0.5)
        os_prior = gpytorch.priors.GammaPrior(3.0, 0.5)
        base_kernel = gpytorch.kernels.MaternKernel(
            nu=2.5, ard_num_dims=n_dims,
            lengthscale_prior=ls_prior
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, outputscale_prior=os_prior
        )
        # Warping Kumaraswamy (v3.6) : si fourni, applique transformation avant kernel
        self._warper = warper
        if warper is not None:
            # Enregistrer les parametres du warper comme modules de ce GP
            # pour qu'ils soient optimises par ADAM
            for i, p in enumerate(warper.parameters()):
                self.register_parameter(f"warp_param_{i}", p)

    def forward(self, x):
        # Applique le warping avant le kernel si actif
        if self._warper is not None:
            x = self._warper.transform(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )



class _ExactGPModelComposite(gpytorch.models.ExactGP if _TORCH_OK else object):
    """
    GP avec kernel composite (v3.7) : additif + multiplicatif.

    Structure :
      k_composite = k_all_matern + k_cont_matern × k_cat_matern

    Ou :
      k_all_matern  : Matern 5/2 ARD sur toutes les dimensions (effets additifs)
      k_cont_matern : Matern 5/2 ARD sur dimensions continues uniquement
      k_cat_matern  : Matern 5/2 ARD sur dimensions catégorielles uniquement
      le produit croisé capture les interactions cont × cat (ex : T × solvant)

    Fallback si pas de dim categorielle : kernel single Matern (comportement v3.5).

    Plus coûteux à fitter (~2x) mais capture mieux les interactions en chimie.
    """
    def __init__(self, X, y, likelihood, n_dims, continuous_dims, categorical_dims,
                 warper=None):
        if not _TORCH_OK:
            raise ImportError("GPyTorch requis")
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        ls_prior = gpytorch.priors.GammaPrior(3.0, 0.5)
        os_prior = gpytorch.priors.GammaPrior(3.0, 0.5)

        # Kernel global (sur toutes les dims)
        self._k_all = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_dims,
                                          lengthscale_prior=ls_prior),
            outputscale_prior=os_prior
        )

        # Kernels separes sur sous-ensembles de dimensions
        self._has_cat = len(categorical_dims) > 0
        self._has_cont = len(continuous_dims) > 0
        self._cont_dims = continuous_dims
        self._cat_dims  = categorical_dims

        if self._has_cont and self._has_cat:
            self._k_cont = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=len(continuous_dims),
                active_dims=continuous_dims,
                lengthscale_prior=ls_prior
            )
            self._k_cat = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=len(categorical_dims),
                active_dims=categorical_dims,
                lengthscale_prior=ls_prior
            )
            self._k_interaction = gpytorch.kernels.ScaleKernel(
                self._k_cont * self._k_cat,
                outputscale_prior=os_prior
            )
            # covar_module exposé pour compat avec GPModel.fit (lengthscale extraction)
            self.covar_module = self._k_all
        else:
            # Pas de dim categorielle ou pas de continue : fallback single kernel
            self.covar_module = self._k_all
            self._k_interaction = None

        # Warping Kumaraswamy (comme _ExactGPModel)
        self._warper = warper
        if warper is not None:
            for i, p in enumerate(warper.parameters()):
                self.register_parameter(f"warp_param_{i}", p)

    def forward(self, x):
        if self._warper is not None:
            x = self._warper.transform(x)
        # Combine additif + multiplicatif
        if self._k_interaction is not None:
            covar = self._k_all(x) + self._k_interaction(x)
        else:
            covar = self._k_all(x)
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), covar)

class GPModel:
    """
    Processus Gaussien GPyTorch — interface compatible avec le reste du script.

    Architecture (v3.3) :
      • Kernel Matérn 5/2 ARD (longueurs de corrélation par paramètre)
      • GammaPrior(3, 0.5) sur lengthscales ET outputscale
        → Régularisation bayésienne : empêche les hyperparamètres de diverger
          avec peu de données (remplace nos bornes arbitraires ls_bounds)
      • GaussianLikelihood : le bruit est APPRIS depuis les données
        (remplace notre alpha=0.5 fixe)
      • Optimisation ADAM sur la log-vraisemblance marginale exacte
        avec n_restarts restarts aléatoires

    Interface sklearn-compatible :
      fit(X_np, y_np)
      predict(X_np, return_std=False)  → np.array ou (np.array, np.array)
      predict(X_np, return_cov=True)   → (np.array, np.array cov matrice)
      .kernel_.k2.length_scale         → np.array (pour ARD)
      .alpha                           → float (noise estimé)
    """

    # Hyperparamètres fixes — calibrés pour la chimie de synthèse
    _TRAIN_ITERS   = 100    # iterations ADAM par restart (v3.5 : reduit de 150 → 100)
    _N_RESTARTS    = 3      # restarts aléatoires (v3.5 : reduit de 5 → 3, -60% de temps)
                            # v3.7d : remplace par self._n_restarts adaptatif au fit
    _LR            = 0.05   # learning rate ADAM
    _NOISE_FLOOR   = 1e-4   # bruit minimal (évite la dégénérescence)

    @staticmethod
    def _adaptive_n_restarts(n_train):
        """v3.7d : adapte le nombre de restarts ADAM selon la taille des donnees.
        Petits datasets : peu de restarts (variance de fit dominee par les donnees).
        Grands datasets : plus de restarts (paysage MLL plus complexe).
        """
        if n_train < 10:    return 3
        if n_train < 30:    return 5
        return 8

    def __init__(self, n_dims, normalize_y=True,
                 kernel_type="matern", continuous_dims=None, categorical_dims=None,
                 noise_prior="low"):
        _require_torch()
        self.n_dims       = n_dims
        self.normalize_y  = normalize_y
        self._y_mean = 0.0
        self._y_std  = 1.0
        # kernel_ stub initial (avant fit)
        self.kernel_ = _KernelStub(np.ones(n_dims))
        self.alpha   = self._NOISE_FLOOR
        self._warper = None  # rempli au fit (Kumaraswamy v3.6)
        # v3.7 : type de kernel et separation cont/cat pour composite
        self._kernel_type = kernel_type
        self._cont_dims   = continuous_dims if continuous_dims is not None else []
        self._cat_dims    = categorical_dims if categorical_dims is not None else []
        # v3.7c : niveau de bruit attendu pour le prior de la likelihood
        # "low"    : GammaPrior(1.1, 0.05)  - bruit faible (défaut historique)
        # "medium" : GammaPrior(2.0, 0.15)  - bruit modéré (CV 5-15%)
        # "high"   : GammaPrior(2.5, 0.5)   - bruit fort   (CV > 15%, données bruitées)
        if noise_prior not in ("low", "medium", "high"):
            noise_prior = "low"
        self._noise_prior = noise_prior

    # ── Conversion numpy → torch ──────────────────────────────────────────────
    @staticmethod
    def _t(arr):
        return torch.tensor(np.array(arr, dtype=np.float32))

    # ── Construction du modèle GPyTorch ──────────────────────────────────────
    def _build(self, X_t, y_t):
        # v3.9 : priors de bruit RECALIBRES pour y standardise (variance=1).
        # ANCIEN BUG : les priors etaient calibres pour des rendements bruts
        # (0-100), donnant un bruit attendu enorme une fois y standardise
        # (mode 2.0, moyenne 22 sur une variance de signal de 1.0). Resultat :
        # le GP surestimait le bruit (~2x) et lissait les vrais pics.
        # Nouveaux priors : mode du bruit-variance a 1% / 5% / 15% de la
        # variance du signal, soit ~±1 / ±2.5 / ±4 points sur un rendement
        # d'ecart-type ~11. (mode = (alpha-1)/beta)
        #   "low"    : Gamma(2.0, 100) mode=0.01  - mesures fiables (±1-2%)
        #   "medium" : Gamma(2.0, 20)  mode=0.05  - bruit modere (±3-7%)
        #   "high"   : Gamma(2.0, 6.67) mode=0.15 - donnees bruitees (±10%+)
        _np = getattr(self, "_noise_prior", "low")
        if _np == "high":
            noise_prior_obj = gpytorch.priors.GammaPrior(2.0, 6.67)
        elif _np == "medium":
            noise_prior_obj = gpytorch.priors.GammaPrior(2.0, 20.0)
        else:  # "low"
            noise_prior_obj = gpytorch.priors.GammaPrior(2.0, 100.0)

        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self._NOISE_FLOOR),
            noise_prior=noise_prior_obj,
        )
        # Warper Kumaraswamy (v3.6+) : toujours actif
        # v3.7b : init leger vers identite si peu de donnees (<=10 pts)
        # pour eviter le warping aggressif sur données rares
        warper = _KumaraswamyWarping(self.n_dims)
        warper.init_params()
        if hasattr(self, "_n_train") and self._n_train <= 10:
            import torch
            with torch.no_grad():
                warper._a_raw.fill_(-0.3)  # softplus(-0.3) -> a≈0.52, warping doux
                warper._b_raw.fill_(-0.3)
        # v3.7 : kernel composite si demande et dims separables
        if self._kernel_type == "composite" and (self._cont_dims or self._cat_dims):
            model = _ExactGPModelComposite(
                X_t, y_t, likelihood, self.n_dims,
                continuous_dims=self._cont_dims,
                categorical_dims=self._cat_dims,
                warper=warper
            )
        else:
            model = _ExactGPModel(X_t, y_t, likelihood, self.n_dims, warper=warper)
        # Garder reference au warper pour diagnostic
        self._warper = warper
        return model, likelihood

    # ── Entrainement ──────────────────────────────────────────────────────────
    def fit(self, X, y):
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)
        self._n_train = len(X_np)  # v3.7b : pour warping adaptatif

        # Standardisation y si demandée (évite double compression avec MinMax)
        if self.normalize_y and len(y_np) >= 2:
            self._y_mean = float(y_np.mean())
            self._y_std  = max(float(y_np.std()), 1e-6)
            y_np = (y_np - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std  = 1.0

        X_t = self._t(X_np)
        y_t = self._t(y_np).view(-1)

        best_loss  = np.inf
        best_state = None
        n_failed_restarts = 0  # v3.7d : compte les fits divergents

        # v3.7d : n_restarts adaptatif selon taille des donnees
        n_restarts_eff = self._adaptive_n_restarts(len(X_np))

        for restart in range(n_restarts_eff + 1):
            model, likelihood = self._build(X_t, y_t)
            model.train(); likelihood.train()

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()),
                lr=self._LR
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            torch.manual_seed(restart * 17 + 42)
            restart_failed = False
            for _ in range(self._TRAIN_ITERS):
                optimizer.zero_grad()
                try:
                    loss = -mll(model(X_t), y_t)
                    loss.backward()
                    optimizer.step()
                except Exception:
                    restart_failed = True
                    break

            try:
                final_loss = float(loss.item())
                if not np.isfinite(final_loss):
                    final_loss = np.inf
                    restart_failed = True
            except Exception:
                final_loss = np.inf
                restart_failed = True

            if restart_failed:
                n_failed_restarts += 1

            if final_loss < best_loss:
                best_loss  = final_loss
                best_state = {
                    "model":      {k: v.clone().detach() for k,v in model.state_dict().items()},
                    "likelihood": {k: v.clone().detach() for k,v in likelihood.state_dict().items()},
                }

        # v3.7d : si tous les restarts ont divergé, warning explicite
        if n_failed_restarts >= n_restarts_eff and best_state is None:
            print(f"  [WARN GP] tous les restarts MLL ont diverge "
                  f"({n_failed_restarts}/{n_restarts_eff+1}). "
                  f"Le GP est probablement degenere. Inspecter les donnees.")

        # Recharger le meilleur état
        self._model, self._likelihood = self._build(X_t, y_t)
        if best_state:
            self._model.load_state_dict(best_state["model"])
            self._likelihood.load_state_dict(best_state["likelihood"])
        self._model.eval(); self._likelihood.eval()
        self._X_t = X_t; self._y_t = y_t

        # Mettre à jour kernel_ et alpha pour compatibilité
        try:
            ls = self._model.covar_module.base_kernel.lengthscale.detach().numpy().ravel()
        except Exception:
            ls = np.ones(self.n_dims)
        self.kernel_ = _KernelStub(ls)

        try:
            self.alpha = float(self._likelihood.noise.item())
        except Exception:
            self.alpha = self._NOISE_FLOOR

        return self

    # ── Prédiction ────────────────────────────────────────────────────────────
    def predict(self, X, return_std=False, return_cov=False, return_std_split=False):
        """
        return_std_split=True : renvoie (mu, sigma_epistemic, sigma_aleatoric)
            sigma_epistemic : incertitude reductible (manque de donnees)
            sigma_aleatoric : bruit intrinseque (appris via likelihood.noise)
        """
        X_t = self._t(np.array(X, dtype=np.float32))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Prediction sans likelihood = epistemique seul
            pred_f = self._model(X_t)
            mu = pred_f.mean.numpy().astype(np.float64) * self._y_std + self._y_mean

            if return_std_split:
                sigma_ep = pred_f.stddev.numpy().astype(np.float64) * self._y_std
                noise_var = float(self._likelihood.noise.item())
                sigma_al = float(np.sqrt(noise_var)) * self._y_std
                sigma_al_arr = np.full_like(sigma_ep, sigma_al)
                return mu, sigma_ep, sigma_al_arr

            # Chemin standard : total = epistemique + aleatoire via likelihood
            pred = self._likelihood(pred_f)

            if return_cov:
                cov = pred.covariance_matrix.numpy().astype(np.float64)
                cov = cov * (self._y_std ** 2)
                return mu, cov

            if return_std:
                std = pred.stddev.numpy().astype(np.float64) * self._y_std
                return mu, std

            return mu


class _KernelStub:
    """Stub kernel_ pour compatibilité ARD avec le reste du script."""
    def __init__(self, length_scale):
        self.k2 = type("k2", (), {"length_scale": np.asarray(length_scale)})()
        self.k1 = type("k1", (), {"constant_value": 1.0})()




class GPModelWithPrior(GPModel):
    """
    Extension de GPModel qui ajoute un prior mean physico-chimique (v3.7).

    Au fit : centre les y par y_prior(X_train) avant de passer au GP
    Au predict : ajoute y_prior(X_test) au mu retourne par le GP

    Le GP apprend ainsi les residus par rapport au prior (plus facile a modeliser,
    convergence plus rapide si le prior est correct).

    L activation est transparente : cette classe est instanciee uniquement
    si cfg["prior_mean"]["type"] != "none".

    Compatible avec kernel_type="composite" (v3.7) via super().__init__.
    """
    def __init__(self, cfg, encoder, **kwargs):
        super().__init__(**kwargs)
        self._prior_cfg = cfg
        self._prior_encoder = encoder

    def fit(self, X, y):
        prior = _compute_prior_mean(X, self._prior_cfg, self._prior_encoder)
        y_centered = np.asarray(y, dtype=float) - prior
        return super().fit(X, y_centered)

    def predict(self, X, return_std=False, return_cov=False, return_std_split=False):
        prior = _compute_prior_mean(X, self._prior_cfg, self._prior_encoder)
        if return_std_split:
            mu, se, sa = super().predict(X, return_std_split=True)
            return mu + prior, se, sa
        if return_cov:
            mu, cov = super().predict(X, return_cov=True)
            return mu + prior, cov
        if return_std:
            mu, std = super().predict(X, return_std=True)
            return mu + prior, std
        mu = super().predict(X)
        return mu + prior


# =============================================================================
# SAASBO — Sparse Axis-Aligned Subspace BO (Eriksson & Jankowiak, UAI 2021)
# =============================================================================
# Identifie automatiquement les dimensions importantes via un prior log-normal
# half-Cauchy sur les inverse-lengthscales. Echantillonne le posterior via NUTS.
# Utile pour HAUTE DIMENSION (>= 6 params) — surpasse ARD classique sur des
# problemes ou peu de variables sont actives (cas typique en chimie quand on
# code des descripteurs ou des solvants par one-hot).
#
# Active via cfg["bo_settings"]["use_saasbo"]: true (opt-in).
# Necessite Pyro : conda install -c conda-forge pyro-ppl

class SAASGP:
    """
    GP avec prior sparse sur les inverse-lengthscales (kappa_d), echantillonne
    par NUTS. La prediction marginalize sur n_samples du posterior.

    Interface compatible GPModel.predict(X, return_std=True).

    Hyperparametres :
      tau    : controle global de la sparsite (default 0.1)
      n_warmup : NUTS burn-in (default 256)
      n_samples : nombre d'echantillons posterieurs gardes (default 256)
      thinning : 1 par defaut (pas de thinning)
    """

    def __init__(self, n_dims, normalize_y=True,
                 tau=0.1, n_warmup=256, n_samples=256, thinning=1, **kwargs):
        self.n_dims        = n_dims
        self.normalize_y   = normalize_y
        self._tau          = float(tau)
        self._n_warmup     = int(n_warmup)
        self._n_samples    = int(n_samples)
        self._thinning     = int(thinning)
        self._y_mean = 0.0
        self._y_std  = 1.0
        self._samples = None  # rempli par fit
        self._X_train = None
        self._y_train = None
        self.kernel_  = _KernelStub(np.ones(n_dims))
        self.alpha    = 1e-4

    @staticmethod
    def _check_pyro():
        try:
            import pyro       # noqa: F401
            import pyro.infer # noqa: F401
            import torch      # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, X, y):
        """Fit via NUTS sampling. Lent (~30s pour n=20, n_dims=10)."""
        if not self._check_pyro():
            raise RuntimeError(
                "SAASBO necessite pyro-ppl. Installer via :\n"
                "  conda install -c conda-forge pyro-ppl\n"
                "ou : pip install pyro-ppl"
            )

        import torch
        import pyro
        import pyro.distributions as dist
        from pyro.infer.mcmc import MCMC, NUTS

        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

        # Standardisation y
        if self.normalize_y and len(y_np) >= 2:
            self._y_mean = float(y_np.mean())
            self._y_std  = max(float(y_np.std()), 1e-6)
            y_np = (y_np - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std  = 1.0

        X_t = torch.tensor(X_np, dtype=torch.float64)
        y_t = torch.tensor(y_np, dtype=torch.float64)
        D   = X_t.shape[1]
        N   = X_t.shape[0]

        def saas_model(X, y):
            # Outputscale
            outputscale = pyro.sample("outputscale", dist.HalfNormal(scale=1.0))
            # Mean
            mean = pyro.sample("mean", dist.Normal(0.0, 1.0))
            # Noise
            noise = pyro.sample("noise", dist.HalfNormal(scale=1.0))
            # Sparse prior : tau * Cauchy^+, then individual log-normal
            tau = pyro.sample("tau", dist.HalfCauchy(scale=self._tau))
            # Inverse-lengthscales kappa_d
            with pyro.plate("kappas", D):
                kappa = pyro.sample("kappa",
                                    dist.HalfCauchy(scale=tau))

            # Compute K matrix : Matern 5/2 with kappa_d (inverse lengthscale)
            ls = 1.0 / (kappa + 1e-9)  # lengthscales
            # squared distances normalisees par ls
            diff = (X.unsqueeze(1) - X.unsqueeze(0)) / ls.unsqueeze(0).unsqueeze(0)
            r2   = (diff ** 2).sum(-1)
            r    = torch.sqrt(r2 + 1e-12)
            sqrt5 = float(np.sqrt(5.0))
            K = outputscale * (1 + sqrt5 * r + (5.0 / 3.0) * r2) * torch.exp(-sqrt5 * r)
            K = K + (noise + 1e-6) * torch.eye(N, dtype=K.dtype)

            pyro.sample("y",
                        dist.MultivariateNormal(loc=mean * torch.ones(N, dtype=K.dtype),
                                                covariance_matrix=K),
                        obs=y)

        pyro.clear_param_store()
        nuts_kernel = NUTS(saas_model, jit_compile=False, max_tree_depth=8)
        mcmc = MCMC(nuts_kernel,
                    num_samples=self._n_samples * self._thinning,
                    warmup_steps=self._n_warmup,
                    disable_progbar=True)
        mcmc.run(X_t, y_t)
        samples = mcmc.get_samples(group_by_chain=False)

        # Thin
        for k in samples:
            samples[k] = samples[k][::self._thinning][:self._n_samples]

        self._samples = samples
        self._X_train = X_t
        self._y_train = y_t

        # Mettre a jour kernel_ avec lengthscales moyennes pour compatibility
        ls_mean = (1.0 / (samples["kappa"].mean(0) + 1e-9)).numpy()
        self.kernel_ = _KernelStub(ls_mean)
        self.alpha   = float(samples["noise"].mean().item())
        return self

    def predict(self, X, return_std=False, return_cov=False, return_std_split=False):
        """Marginalize sur les samples NUTS. Plus lent que GPModel mais plus precis."""
        if self._samples is None:
            raise RuntimeError("SAASGP doit etre fit() avant predict()")
        import torch

        X_t = torch.tensor(np.array(X, dtype=np.float32), dtype=torch.float64)
        N_test = X_t.shape[0]
        S = self._samples["kappa"].shape[0]
        D = X_t.shape[1]

        means_all = np.zeros((S, N_test))
        vars_all  = np.zeros((S, N_test))

        sqrt5 = float(np.sqrt(5.0))
        with torch.no_grad():
            for s in range(S):
                kappa = self._samples["kappa"][s]  # (D,)
                outputscale = float(self._samples["outputscale"][s])
                noise = float(self._samples["noise"][s])
                mean_const = float(self._samples["mean"][s])

                ls = 1.0 / (kappa + 1e-9)

                # K(X_train, X_train)
                diff_tr = (self._X_train.unsqueeze(1) - self._X_train.unsqueeze(0)) / ls
                r2_tr = (diff_tr ** 2).sum(-1)
                r_tr  = torch.sqrt(r2_tr + 1e-12)
                K_tr = outputscale * (1 + sqrt5 * r_tr + (5.0/3.0)*r2_tr) * torch.exp(-sqrt5 * r_tr)
                K_tr = K_tr + (noise + 1e-6) * torch.eye(K_tr.shape[0], dtype=K_tr.dtype)

                # K(X_test, X_train)
                diff_x = (X_t.unsqueeze(1) - self._X_train.unsqueeze(0)) / ls
                r2_x = (diff_x ** 2).sum(-1)
                r_x  = torch.sqrt(r2_x + 1e-12)
                K_x = outputscale * (1 + sqrt5 * r_x + (5.0/3.0)*r2_x) * torch.exp(-sqrt5 * r_x)

                # K(X_test, X_test) diag
                K_xx_diag = torch.full((N_test,), float(outputscale), dtype=K_tr.dtype)

                # Solve
                try:
                    L_tr = torch.linalg.cholesky(K_tr)
                    alpha = torch.cholesky_solve((self._y_train - mean_const).unsqueeze(-1), L_tr).squeeze(-1)
                    mu = mean_const + (K_x @ alpha)
                    v  = torch.cholesky_solve(K_x.T, L_tr)
                    var_diag = K_xx_diag - (K_x * v.T).sum(dim=1)
                    var_diag = torch.clamp(var_diag, min=1e-9)
                except Exception:
                    mu = torch.zeros(N_test, dtype=K_tr.dtype)
                    var_diag = torch.full((N_test,), float(outputscale), dtype=K_tr.dtype)

                means_all[s] = mu.numpy()
                vars_all[s]  = var_diag.numpy()

        # Marginalize : E[mu], V[mu] = E[var] + V[mu_s] (loi de variance totale)
        mu_post = means_all.mean(axis=0)
        var_post = vars_all.mean(axis=0) + means_all.var(axis=0)

        # De-normaliser
        mu_real = mu_post * self._y_std + self._y_mean
        std_real = np.sqrt(var_post) * self._y_std

        if return_std_split:
            return mu_real, std_real, np.full_like(std_real, self.alpha * self._y_std)
        if return_cov:
            # Approximation : matrice diagonale (vraie cov inter-points trop coûteuse)
            return mu_real, np.diag(std_real**2)
        if return_std:
            return mu_real, std_real
        return mu_real


def build_gp(n_dims, normalize_y=True, cfg=None, encoder=None,
             y_train=None, **kwargs):
    """
    Crée un GPModel GPyTorch.
    Interface identique à l ancienne fonction sklearn pour compatibilité.
    Les kwargs (alpha, etc.) sont ignorés — le GP gère son bruit en interne.

    v3.7 routage :
    - cfg["bo_settings"]["kernel"] = "composite" -> kernel additif+multiplicatif
    - cfg["prior_mean"]["type"] != "none"        -> GPModelWithPrior

    v3.7c noise_prior :
    - cfg["bo_settings"]["noise_prior"] = "low" / "medium" / "high" / "auto"
    - "auto" (defaut) : detecte selon CV des y_train. CV > 15% -> medium ;
       CV > 30% -> high. Fournir y_train pour activer la detection.

    v3.7d use_saasbo :
    - cfg["bo_settings"]["use_saasbo"]: true active SAASGP a la place du GP standard.
       Recommande pour n_dims >= 6 quand certaines variables sont peu actives.
       Necessite pyro-ppl. Plus lent (~30s vs ~3s par fit).
    """
    # Determiner kernel_type et identifier dims cont/cat
    kernel_type = "matern"
    continuous_dims = []
    categorical_dims = []
    noise_prior_setting = "auto"
    if cfg is not None and encoder is not None:
        bo_cfg = cfg.get("bo_settings", {})
        kernel_type = bo_cfg.get("kernel", "matern")
        noise_prior_setting = bo_cfg.get("noise_prior", "auto")
        # Identifier les dimensions encodees par type
        dim_cursor = 0
        for pi, p in enumerate(cfg["parameters"]):
            nd = encoder._encoded_dims[pi]
            if p["type"] == "continuous":
                continuous_dims.extend(range(dim_cursor, dim_cursor + nd))
            else:
                categorical_dims.extend(range(dim_cursor, dim_cursor + nd))
            dim_cursor += nd

    # v3.7c : auto-detection noise_prior selon CV
    if noise_prior_setting == "auto":
        if y_train is not None and len(y_train) >= 5:
            y_arr = np.array(y_train, dtype=float).ravel()
            mu    = abs(y_arr.mean())
            cv    = (y_arr.std() / max(mu, 1e-6)) if mu > 1e-6 else 0.0
            if cv > 0.30:
                noise_prior_resolved = "high"
                print(f"  [NOISE-AUTO] CV={cv*100:.1f}% > 30% -> noise_prior=high")
            elif cv > 0.15:
                noise_prior_resolved = "medium"
                print(f"  [NOISE-AUTO] CV={cv*100:.1f}% > 15% -> noise_prior=medium")
            else:
                noise_prior_resolved = "low"
        else:
            noise_prior_resolved = "low"
    else:
        noise_prior_resolved = noise_prior_setting

    gp_kwargs = dict(n_dims=n_dims, normalize_y=normalize_y,
                     kernel_type=kernel_type,
                     continuous_dims=continuous_dims,
                     categorical_dims=categorical_dims,
                     noise_prior=noise_prior_resolved)

    # v3.7d : Route vers SAASGP si demande (opt-in via use_saasbo: true)
    if cfg is not None:
        bo_cfg2 = cfg.get("bo_settings", {})
        if bo_cfg2.get("use_saasbo", False):
            if SAASGP._check_pyro():
                saas_kwargs = dict(
                    n_dims=n_dims, normalize_y=normalize_y,
                    tau       = float(bo_cfg2.get("saasbo_tau", 0.1)),
                    n_warmup  = int(bo_cfg2.get("saasbo_warmup", 256)),
                    n_samples = int(bo_cfg2.get("saasbo_samples", 256)),
                    thinning  = int(bo_cfg2.get("saasbo_thinning", 1)),
                )
                return SAASGP(**saas_kwargs)
            else:
                print("  [WARN] use_saasbo=true mais pyro-ppl non installe. "
                      "Fallback GPModel standard. Installer via :")
                print("         conda install -c conda-forge pyro-ppl")

    # Route vers GPModelWithPrior si prior actif
    if cfg is not None and encoder is not None:
        prior_cfg = cfg.get("prior_mean", {})
        if prior_cfg and prior_cfg.get("type", "none") != "none":
            # Note : GPModelWithPrior herite GPModel, donc accepte les memes kwargs
            return GPModelWithPrior(cfg=cfg, encoder=encoder, **gp_kwargs)
    return GPModel(**gp_kwargs)


# =============================================================================
# 5b.  BACKEND GPYTORCH (optionnel — gp_backend: "torch" dans le YAML)
# =============================================================================
def acq_ei(X_norm, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_norm, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    Z  = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-9] = 0.0
    return ei


def acq_ucb(X_norm, gp, kappa=2.0):
    mu, sigma = gp.predict(X_norm, return_std=True)
    return mu + kappa * sigma


def acq_pi(X_norm, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_norm, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    return norm.cdf((mu - y_best - xi) / sigma)


def acq_tei(X_norm, gp, y_best, xi=0.001, beta=8):
    """
    Tempered Expected Improvement (TEI) — Lavayssiere, Bantreil, Lamaty 2026.

    TEI(x) = (μ-f*-ξ)·Φ(Z) + g(x,β)·σ·φ(Z)

    ou Z = (μ - f* - ξ) / σ  (identique a EI standard)

    et g(x,β) = 1 / (1 + exp[β·(1 - μ(x)/f*)])

    g est un facteur sigmoid qui varie POINT PAR POINT selon μ(x) :
      - μ(x) >> f*  →  g ≈ 1  →  TEI ≈ EI  (explore librement)
      - μ(x) ≈ f*   →  g ≈ 0.5 →  exploration moderee
      - μ(x) < f*   →  g → 0   →  exploration supprimee

    Effet pratique : EI tend a revenir autour du meilleur point connu
    meme quand il a deja ete bien explore (l incertitude locale reste faible
    mais le gain espere est positif). TEI penalise ces retours redondants.

    Sur paysage multi-modal (benchmark modele B du papier de reference) :
      TEI > EI > UCB > PI  — TEI continue d ameliorer quand EI stagne.

    Sans modele physique (contrairement au papier original), f_max(x) = μ(x)
    du GP — le plafond atteignable en x est estime par la prediction surrogate.

    beta=8 : valeur fixee dans le papier original. Ne pas modifier sauf pour
    des systemes a tres haute reproducibilite (flux automatise).

    xi=0.001 : valeur recommandee (papier original). 0 possible mais 0.001
    donne de meilleures performances en pratique.

    NOTE sur le comportement en debut de campagne :
    Avec peu de donnees (<12 pts) ou un GP encore peu differencie, g varie
    peu entre les points et TEI ≈ k*EI (meme argmax). L avantage de TEI sur EI
    emerge a partir d environ 10-15 iterations quand le GP a cartographie
    suffisamment l espace pour differencier les zones prometteuses des zones
    exploitees. Sur paysage multi-modal (benchmark papier modele B), TEI evite
    de rester piege dans des maxima locaux : il monte aussi vite qu EI
    initialement, mais continue d ameliorer quand EI stagne.
    Recommandation : utiliser EI en debut de campagne (<10 pts), TEI ensuite.
    """
    from scipy.stats import norm as sp_norm
    mu, sigma = gp.predict(X_norm, return_std=True)
    mu    = mu.ravel()
    sigma = np.maximum(sigma.ravel(), 1e-9)

    Z   = (mu - y_best - xi) / sigma
    ei  = (mu - y_best - xi) * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)

    # Facteur de temperage : f_max(x) = mu(x) + sigma(x)  (estimation optimiste)
    # Pourquoi mu + sigma plutot que mu seul :
    #   - mu seul chute vers 0 dans les zones non explorees (prior GP)
    #     → g → 0 partout → exploration entierement supprimee (mauvais)
    #   - mu + sigma reste eleve ou sigma est grand (zones inconnues)
    #     → g → 1 dans les zones inexplorees (exploration pleine)
    #     → g → 0 pres du meilleur point deja connu (redondance evitee)
    # Comportement resultant :
    #   Zone inconnue (sigma grand)  : f_max >> y_best → g ≈ 1 → explore
    #   Pres du meilleur (sigma ≈ 0) : f_max ≈ mu ≈ y_best → g ≈ 0.5 → modere
    #   Zone clairement mauvaise     : f_max << y_best → g → 0 → supprime
    f_max = mu + sigma
    g   = 1.0 / (1.0 + np.exp(beta * (1.0 - f_max / max(float(y_best), 1e-9))))

    tei = ei * g
    tei = np.where(sigma < 1e-9, 0.0, tei)
    tei = np.maximum(tei, 0.0)
    return tei


def acq_qei(X_batch, gp, y_best, xi=0.01, n_samples=512):
    """
    v3.7d : quasi-MC Expected Improvement pour batch (n_suggestions >= 2).
    
    **REQUIS** : torch et gpytorch (utilisés par GPModel).
    
    Calcule EI du batch via échantillonnage GP multivarié. Plus rapide que
    Kriging Believer (pas de refit GP), plus précis que Local Penalization.
    
    Args:
        X_batch   : array (n_batch, n_dims) — candidats batch normalisés
        gp        : GPModel entraîné
        y_best    : float — meilleur y observé
        xi        : float — paramètre exploration
        n_samples : int — nb échantillons MC (défaut 512, min 128)
    
    Returns:
        float — valeur qEI du batch (pas par point, batch global)
    
    Usage dans suggest_experiments:
        Pour générer un batch de taille N, on optimise x1, puis on ajoute x1
        au batch candidat et on optimise qEI([x1, x2]), etc. séquentiellement.
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only
    import torch
    
    n_batch = X_batch.shape[0]
    if n_batch == 1:
        # Fallback EI standard si batch=1
        return acq_ei(X_batch, gp, y_best, xi)
    
    # Prédiction GP : mu et covariance complète du batch
    with torch.no_grad():
        X_t = torch.tensor(X_batch, dtype=torch.float32)
        gp._model.eval()
        pred = gp._model(X_t)
        mu = pred.mean.numpy()  # (n_batch,)
        cov = pred.covariance_matrix.numpy()  # (n_batch, n_batch)
    
    # Échantillonnage multivarié N(mu, cov)
    try:
        samples = np.random.multivariate_normal(mu, cov, size=n_samples)  # (n_samples, n_batch)
    except np.linalg.LinAlgError:
        # Fallback si cov singulière : diagonale uniquement
        std = np.sqrt(np.maximum(np.diag(cov), 1e-9))
        samples = mu[None, :] + std[None, :] * np.random.randn(n_samples, n_batch)
    
    # EI du batch : max(0, max_batch(sample) - y_best - xi) moyenné sur samples
    max_per_sample = samples.max(axis=1)  # (n_samples,) — meilleur du batch par sample
    ei_batch = np.maximum(0, max_per_sample - y_best - xi).mean()
    
    return float(ei_batch)


def acq_qnei(X_batch, gp, y_best, xi=0.01, n_samples=512):
    """
    v3.7d : quasi-MC Noisy Expected Improvement pour batch.
    
    Comme qEI mais tient compte du bruit observationnel (utile si noise_prior=high).
    Marginalise sur le bruit en plus de l'incertitude GP.
    
    Args: identiques à acq_qei
    Returns: float — valeur qNEI du batch
    """
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    
    n_batch = X_batch.shape[0]
    if n_batch == 1:
        return acq_ei(X_batch, gp, y_best, xi)
    
    with torch.no_grad():
        X_t = torch.tensor(X_batch, dtype=torch.float32)
        gp._model.eval()
        pred = gp._model(X_t)
        mu = pred.mean.numpy()
        cov = pred.covariance_matrix.numpy()
        
        # Ajouter bruit observationnel à la diagonale
        noise_var = float(gp._likelihood.noise.item())
        cov_noisy = cov + np.eye(n_batch) * noise_var
    
    try:
        samples = np.random.multivariate_normal(mu, cov_noisy, size=n_samples)
    except np.linalg.LinAlgError:
        std = np.sqrt(np.maximum(np.diag(cov_noisy), 1e-9))
        samples = mu[None, :] + std[None, :] * np.random.randn(n_samples, n_batch)
    
    max_per_sample = samples.max(axis=1)
    ei_batch = np.maximum(0, max_per_sample - y_best - xi).mean()
    return float(ei_batch)



def acq_kg(X_norm, gp, y_best, n_fantasy=32, n_grid=200, rng=None):
    """
    Knowledge Gradient (KG) exact via Monte Carlo avec fantaisies (v3.6).

    Pour chaque point candidat x :
      1. Echantillonner n_fantasy valeurs y_fantasy ~ N(mu(x), sigma(x))
      2. Pour chaque y_fantasy, refit hypothetique du GP (formule de Kriging update)
      3. Trouver le nouveau max sur une grille de reference
      4. KG(x) = E[max_posterior] - max_prior

    Utilise les formules de update Bayesien sans re-fit complet (rapide).

    Retourne : KG(x) >= 0, plus grand = plus d'information attendue
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # Guards : X_norm doit etre 2D et non vide
    X_norm = np.atleast_2d(X_norm)
    if len(X_norm) == 0:
        return np.array([])

    # Predictions aux candidats
    mu_X, sigma_X = gp.predict(X_norm, return_std=True)

    # Grille de reference (pour chercher le nouveau max)
    # Utilise les candidats eux-memes comme reference : O(n^2) mais n petit
    X_ref = X_norm
    mu_ref_prior, cov_ref_prior = gp.predict(X_ref, return_cov=True)
    max_prior = float(mu_ref_prior.max())

    # Fantasy sampling : normal reduit
    # KG(x) = E_z [ max(mu_ref + sigma_tilde(x, X_ref) * z) ] - max_prior
    # où sigma_tilde = cov(x, X_ref) / sigma(x)  (effet d'une observation en x)
    n_x = len(X_norm)
    n_ref = len(X_ref)

    # Covariances croisees cov(x_i, X_ref)
    # On refait une prediction jointe pour obtenir la cov complete
    # Approximation pragmatique : pour chaque x, cov(x, X_ref) via predict joint
    kg_vals = np.zeros(n_x)
    # z samples
    Z = rng.standard_normal(n_fantasy)

    for i in range(n_x):
        # Stack [X[i], X_ref] et obtenir cov
        X_joint = np.vstack([X_norm[i:i+1], X_ref])
        _, cov_joint = gp.predict(X_joint, return_cov=True)
        # cov(x_i, X_ref) est la 1ere ligne (sans [0,0])
        cov_x_ref = cov_joint[0, 1:]  # shape (n_ref,)
        sigma_x = np.sqrt(max(cov_joint[0, 0], 1e-12))
        if sigma_x < 1e-6:
            kg_vals[i] = 0.0
            continue
        sigma_tilde = cov_x_ref / sigma_x  # shape (n_ref,)

        # Pour chaque z, calculer max(mu_ref + sigma_tilde * z)
        # broadcasting : (n_ref, 1) + (n_ref, 1) * (n_fantasy,)
        post_means = mu_ref_prior[:, None] + sigma_tilde[:, None] * Z[None, :]
        max_per_z = post_means.max(axis=0)  # shape (n_fantasy,)
        expected_max = float(max_per_z.mean())
        kg_vals[i] = max(0.0, expected_max - max_prior)

    return kg_vals

def acq_thompson(X_norm, gp, rng):
    """
    Thompson Sampling : tire une fonction du posterior GP.

    Pour GPyTorch : utilise fast_pred_samples (Lanczos) au lieu de la
    covariance complète — évite la matrice 4000×4000 en mémoire.
    Fallback : multivariate_normal numpy si la méthode torch échoue.
    """
    if _TORCH_OK and isinstance(gp, GPModel):
        try:
            X_t = gp._t(np.array(X_norm, dtype=np.float32))
            with torch.no_grad(), gpytorch.settings.fast_pred_samples():
                sample = gp._model(X_t).rsample().numpy().astype(np.float64)
            # Remettre à l échelle
            sample = sample * gp._y_std + gp._y_mean
            return sample
        except Exception:
            pass
    # Fallback numpy (petit nombre de points)
    mu, cov = gp.predict(X_norm, return_cov=True)
    cov += 1e-8 * np.eye(len(X_norm))
    return rng.multivariate_normal(mu, cov)



# =============================================================================
# 6b.  HYPERVOLUME UTILITIES + MC-EHVI
#      (utilise uniquement quand multi_objective_mode = "ehvi")
# =============================================================================
#
# EHVI (Expected Hypervolume Improvement) maximise directement le volume
# de l'espace de performance domine par le front de Pareto.
# C'est theoriquement superieur a ParEGO, surtout avec 3+ objectifs.
#
# IMPLEMENTATION CHOISIE : Monte Carlo pure numpy
#   - Pour 2 objectifs : hypervolume exact par balayage O(n log n)
#   - Pour 3+ objectifs : hypervolume Monte Carlo
#   - L'EHVI d'un candidat = moyenne de l'amelioration HV sur N echantillons
#     du posterior GP
#
# COUT COMPUTATIONNEL :
#   - 2 objectifs : ~5-15s par suggestion (grille 2000 pts x 64 echantillons)
#   - 3 objectifs : ~30-90s par suggestion (HV MC dans la boucle)
#   - Preferer n_suggestions = 1 pour ne pas multiplier le temps de calcul
# =============================================================================

def _hv_2d(Y_signed, ref):
    """
    Hypervolume exact pour 2 objectifs (all in maximize convention).
    Algorithme de balayage O(n log n).
    """
    mask = np.all(Y_signed > ref, axis=1)
    Yf   = Y_signed[mask]
    if len(Yf) == 0:
        return 0.0

    # Sort by obj0 descending → obj1 necessarily ascending on Pareto front
    idx     = np.argsort(-Yf[:, 0])
    Yf      = Yf[idx]

    # Pareto filter (keep only points with strictly increasing obj1)
    pareto  = []
    max_y1  = -np.inf
    for pt in Yf:
        if pt[1] > max_y1:
            pareto.append(pt)
            max_y1 = pt[1]
    pareto = np.array(pareto)   # sorted obj0 desc / obj1 asc

    # Hypervolume = sum of horizontal strips
    # Strip i: height = (pareto[i].y1 - pareto[i-1].y1), width = pareto[i].y0 - ref[0]
    # Caution: pareto is sorted by obj0 DESC, so pareto[0] has HIGHEST obj0
    # Strip between consecutive obj1 values:
    hv      = 0.0
    prev_y1 = ref[1]
    for pt in pareto:            # iterating obj0 desc = obj1 asc
        hv      += (pt[1] - prev_y1) * (pt[0] - ref[0])
        prev_y1  = pt[1]
    return float(hv)


def _hv_mc(Y_signed, ref, rng, n_mc=2000):
    """
    Hypervolume Monte Carlo pour n_obj >= 3.
    Estimateur sans biais par echantillonnage uniforme dans la boite englobante.
    """
    mask = np.all(Y_signed > ref, axis=1)
    Yf   = Y_signed[mask]
    if len(Yf) == 0:
        return 0.0
    upper = Yf.max(axis=0)
    vol   = float(np.prod(upper - ref))
    if vol <= 0:
        return 0.0
    samples   = rng.uniform(ref, upper, (n_mc, Y_signed.shape[1]))
    dominated = np.any(
        np.all(Yf[:, np.newaxis, :] >= samples[np.newaxis, :, :], axis=2),
        axis=0
    )
    return vol * float(dominated.mean())


def _hypervolume(Y_signed, ref, rng=None, n_mc=2000):
    """Dispatche vers le calcul HV adapte au nombre d'objectifs."""
    n_obj = Y_signed.shape[1]
    if n_obj == 1:
        mask = Y_signed[:, 0] > ref[0]
        return float(np.maximum(0, Y_signed[mask, 0] - ref[0]).max()) if mask.any() else 0.0
    if n_obj == 2:
        return _hv_2d(Y_signed, ref)
    return _hv_mc(Y_signed, ref, rng if rng is not None else np.random.default_rng(0), n_mc)


def _maximise_ehvi_grid(gps, ref_point, Y_signed, cfg, encoder, rng):
    """
    Maximise MC-EHVI par recherche sur une grille aleatoire dense.
    Pas d'evolution differentielle (EHVI non-differentiable).

    Retourne x_best en espace normalise [0,1]^d.
    """
    bo     = cfg["bo_settings"]
    n_mc   = int(bo.get("ehvi_n_mc", 64))
    N_GRID = 2000
    n_obj  = len(gps)
    n_params = encoder.n_params

    X_grid = rng.uniform(0, 1, (N_GRID, n_params))

    # Batch GP predictions
    mus  = np.zeros((N_GRID, n_obj))
    stds = np.zeros((N_GRID, n_obj))
    for j, gp in enumerate(gps):
        mu, std = gp.predict(X_grid, return_std=True)
        mus[:, j]  = mu
        stds[:, j] = np.maximum(std, 1e-9)

    # Current hypervolume (precompute once)
    current_hv = _hypervolume(Y_signed, ref_point, rng)

    # MC-EHVI on grid
    ehvi_grid = np.zeros(N_GRID)
    for i in range(N_GRID):
        samples  = rng.normal(mus[i], stds[i], (n_mc, n_obj))
        hvi_sum  = 0.0
        for k in range(n_mc):
            y_new = samples[k]
            if np.all(y_new > ref_point):
                Y_aug  = np.vstack([Y_signed, y_new.reshape(1, -1)])
                hv_new = _hypervolume(Y_aug, ref_point, rng)
                hvi_sum += max(0.0, hv_new - current_hv)
        ehvi_grid[i] = hvi_sum / n_mc

    best_idx = int(np.argmax(ehvi_grid))
    return X_grid[best_idx]


def acq_qnehvi(X_batch, gps, Y_pareto, ref_point, n_samples=64, rng=None):
    """
    v3.8 : q-Noisy Expected Hypervolume Improvement (Daulton et al. 2021).

    Difference avec EHVI standard :
      - q-batch : evalue HVI du batch entier (pas point par point)
      - Noisy   : ajoute bruit observationnel, marginalise sur observation
                  bruitee de Y_pareto (utile sans replicats)

    Args:
        X_batch    : (n_batch, n_dims) candidats normalises
        gps        : liste de GPModel (un par objectif)
        Y_pareto   : (n_pareto, n_obj) front Pareto actuel (signe : direction)
        ref_point  : (n_obj,) point reference
        n_samples  : MC samples (defaut 64)

    Returns:
        float : qNEHVI du batch
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_batch = X_batch.shape[0]
    n_obj   = len(gps)

    # Predictions : mu et covariance complete pour chaque objectif
    mus  = np.zeros((n_batch, n_obj))
    covs = np.zeros((n_obj, n_batch, n_batch))
    noise_vars = np.zeros(n_obj)

    try:
        import torch
        for j, gp in enumerate(gps):
            with torch.no_grad():
                X_t = torch.tensor(X_batch, dtype=torch.float32)
                gp._model.eval()
                pred = gp._model(X_t)
                mus[:, j]  = pred.mean.numpy()
                cov_j      = pred.covariance_matrix.numpy()
                noise_j    = float(gp._likelihood.noise.item())
            # Noisy : ajoute le bruit observationnel
            covs[j] = cov_j + np.eye(n_batch) * noise_j
            noise_vars[j] = noise_j
    except Exception:
        # Fallback : utilise predict() standard (variance diagonale)
        for j, gp in enumerate(gps):
            mu, std = gp.predict(X_batch, return_std=True)
            mus[:, j] = mu.ravel()
            covs[j]   = np.diag(np.maximum(std.ravel(), 1e-9) ** 2)

    # Current hypervolume
    hv_current = _hypervolume(Y_pareto, ref_point, rng)

    # MC sampling : pour chaque sample, compute HVI du batch augmente
    hvi_acc = 0.0
    for s in range(n_samples):
        # Echantillonner un Y_batch pour ce sample
        y_batch = np.zeros((n_batch, n_obj))
        for j in range(n_obj):
            try:
                y_batch[:, j] = rng.multivariate_normal(mus[:, j], covs[j])
            except np.linalg.LinAlgError:
                std_j = np.sqrt(np.maximum(np.diag(covs[j]), 1e-9))
                y_batch[:, j] = mus[:, j] + std_j * rng.standard_normal(n_batch)

        # Filtrer les points qui dominent le ref_point
        mask_above_ref = np.all(y_batch > ref_point, axis=1)
        if not mask_above_ref.any():
            continue

        # Augmenter Pareto avec le batch et calculer nouveau HV
        Y_aug = np.vstack([Y_pareto, y_batch[mask_above_ref]])
        hv_new = _hypervolume(Y_aug, ref_point, rng)
        hvi_acc += max(0.0, hv_new - hv_current)

    return hvi_acc / n_samples


def _maximise_qnehvi_grid(gps, ref_point, Y_pareto, cfg, encoder, rng,
                           n_batch=1):
    """
    Maximise qNEHVI par recherche grille (analogue _maximise_ehvi_grid).

    Pour batch=1 : equivalent a noisy EHVI.
    Pour batch>1 : optimisation greedy sequentielle (1 point a la fois).
    """
    bo     = cfg["bo_settings"]
    n_mc   = int(bo.get("qnehvi_n_mc", 64))
    N_GRID = int(bo.get("qnehvi_n_grid", 2000))
    n_params = encoder.n_params

    if n_batch == 1:
        # Single point : grille standard
        X_grid = rng.uniform(0, 1, (N_GRID, n_params))
        scores = np.zeros(N_GRID)
        for i in range(N_GRID):
            scores[i] = acq_qnehvi(X_grid[i:i+1], gps, Y_pareto, ref_point,
                                    n_samples=n_mc, rng=rng)
        return X_grid[int(np.argmax(scores))]
    else:
        # Batch : greedy sequentiel
        selected = []
        for _ in range(n_batch):
            X_grid = rng.uniform(0, 1, (N_GRID, n_params))
            scores = np.zeros(N_GRID)
            for i in range(N_GRID):
                if selected:
                    X_test = np.vstack([np.array(selected), X_grid[i:i+1]])
                else:
                    X_test = X_grid[i:i+1]
                scores[i] = acq_qnehvi(X_test, gps, Y_pareto, ref_point,
                                        n_samples=n_mc, rng=rng)
            selected.append(X_grid[int(np.argmax(scores))])
        return np.array(selected)


# =============================================================================
# 7.  MAXIMISATION DE L'ACQUISITION
# =============================================================================

def select_top_diverse_candidates(X_grid, scores, n, min_dist=None):
    """
    Selectionne les n meilleurs candidats en imposant une diversite spatiale.

    Algorithme (Lavayssiere et al. 2026) :
      1. Prendre le candidat avec le score le plus eleve (ancre)
      2. Pour chaque candidat suivant : parmi les non-selectionnes,
         prendre celui qui maximise la distance minimale aux deja selectionnes
         (greedy max-min distance)

    Assure que les suggestions de batch explorent des zones distinctes
    plutot que de se regrouper autour du meme maximum local.

    min_dist : distance minimale entre candidats (en espace normalise [0,1]).
               Si None : pas de contrainte stricte (juste le tri greedy).
               Recommande : 0.1 pour n=2, 0.05 pour n=3+.
    """
    from scipy.spatial.distance import cdist
    scores    = np.asarray(scores).ravel()
    sorted_i  = np.argsort(-scores)
    selected  = [sorted_i[0]]

    for _ in range(1, n):
        remaining = [i for i in sorted_i if i not in selected]
        if not remaining:
            break
        dists     = cdist(X_grid[remaining], X_grid[selected])
        min_dists = dists.min(axis=1)
        if min_dist is not None:
            # Filtrer les candidats trop proches
            far_enough = [remaining[j] for j, d in enumerate(min_dists)
                          if d >= min_dist]
            # Si aucun n'est assez loin, prendre le plus eloigne disponible
            pool = far_enough if far_enough else remaining
            dists2   = cdist(X_grid[pool], X_grid[selected])
            selected.append(pool[np.argmax(dists2.min(axis=1))])
        else:
            selected.append(remaining[np.argmax(min_dists)])

    return selected


def gp_hedge_select(acq_fns_dict, encoder, gp, df_done, cfg, rng,
                    eta=None, n_train_for_max=None):
    """
    v3.7d (FIX) : GP-Hedge portfolio (Hoffman, Brochu, de Freitas, UAI 2011).

    Lance plusieurs acquisitions en parallele, choisit l'une des recommandations
    selon une probabilite softmax basee sur leur 'gain' historique cumule.

    CORRECTIONS v3.7d.1 :
    1. Le gain est maintenant standardise par mu_train (zero-mean) et sigma_train.
       Avant : gain = mu(x_proposed) brut -> divergence si echelle non-normalisee.
    2. eta calcule selon Hoffman 2011 : eta = sqrt(8 * log(K) / T_max).
       Avant : eta=1.0 fixe -> winner-takes-all premature.
    3. Si eta fourni explicitement, il est respecte.

    Args:
        acq_fns_dict   : dict {nom_acq: fn(X)->vals}
        gp             : GP entraine (pour calculer mu standardise)
        eta            : temperature softmax. None = auto (Hoffman 2011).
        n_train_for_max: passe a maximise_acquisition pour adapter n_restarts

    Retourne : (x_chosen_norm, name_chosen, all_proposals_dict)
    """
    proposals = {}
    for name, fn in acq_fns_dict.items():
        try:
            x_prop = maximise_acquisition(fn, encoder.n_params, n_restarts=20, rng=rng,
                                          n_train=n_train_for_max)
            proposals[name] = x_prop
        except Exception as e:
            print(f"  [HEDGE] acq '{name}' a echoue: {e}")

    if not proposals:
        # Fallback : random
        return rng.uniform(0, 1, encoder.n_params), "random", {}

    # Charger gains historiques
    bo_cfg = cfg.setdefault("bo_settings", {})
    _debug_hedge = bo_cfg.get("_debug_hedge", False)
    gains_state = bo_cfg.setdefault("_hedge_gains", {})
    if _debug_hedge and len(gains_state) > 0:
        print(f"    [HEDGE-DEBUG] gains_state LOADED from prev iter: {gains_state}")
    for name in proposals:
        gains_state.setdefault(name, 0.0)

    # FIX #2 : eta auto selon Hoffman 2011, theoreme regret bound :
    # eta = sqrt(8 * log(K) / T) avec K=nb acquireurs, T=nb iterations max
    K = len(proposals)
    if eta is None:
        # On estime T comme une borne raisonnable (50 iterations BO typique)
        T_est = max(50, n_train_for_max or 50)
        eta = float(np.sqrt(8.0 * np.log(max(K, 2)) / T_est))

    # Softmax pour selectionner l'acquireur
    names_list = list(proposals.keys())
    g_arr = np.array([gains_state[n] for n in names_list], dtype=float)
    g_arr = g_arr - g_arr.max()  # stabilite numerique
    probs = np.exp(eta * g_arr)
    probs = probs / probs.sum()

    chosen_idx = int(rng.choice(len(names_list), p=probs))
    chosen_name = names_list[chosen_idx]
    x_chosen = proposals[chosen_name]
    
    # DEBUG logging
    if _debug_hedge:
        print(f"    [HEDGE-DEBUG] eta={eta:.3f}  gains_cumul={dict(zip(names_list, [gains_state[n] for n in names_list]))}")
        print(f"    [HEDGE-DEBUG] gains_rel={dict(zip(names_list, g_arr))}")
        print(f"    [HEDGE-DEBUG] probs={dict(zip(names_list, probs))}")
        print(f"    [HEDGE-DEBUG] chosen={chosen_name}")

    # FIX #1 : gains clippes pour stabilite (GP predict retourne deja en espace normalise)
    # Pas besoin de re-standardiser, juste borner a +/- 3.
    for name, x_prop in proposals.items():
        try:
            mu_at_x, _ = gp.predict(x_prop.reshape(1, -1), return_std=True)
            # Le GP a ete fit sur y normalise, donc mu est deja N(0,1)
            gain_std = float(np.clip(mu_at_x[0], -3.0, 3.0))
            old_gain = gains_state[name]
            gains_state[name] = float(old_gain) + gain_std
            if _debug_hedge:
                print(f"    [HEDGE-DEBUG] {name}: mu={mu_at_x[0]:.3f} -> gain_delta={gain_std:.3f}, cumul={gains_state[name]:.3f}")
        except Exception:
            pass

    bo_cfg["_hedge_gains"] = gains_state
    return x_chosen, chosen_name, proposals


# =============================================================================
# TuRBO — Trust Region Bayesian Optimization (Eriksson et al., NeurIPS 2019)
# =============================================================================

class TrustRegionState:
    """
    v3.7d : Etat d'une trust region TuRBO pour BO en haute dimension.

    Maintient un cube/rectangle local centre sur le meilleur point observe.
    La taille L (en espace normalise [0,1]^D) se contracte/dilate selon
    les succes/echecs accumules :
      - succes consecutifs >= n_success_thresh -> L *= 2
      - echecs consecutifs >= n_failure_thresh -> L /= 2
      - L < L_min -> redemarrage (re-tirage LHS dans tout l'espace)

    Etat persiste entre runs via cfg["bo_settings"]["_turbo_state"].
    """

    L_init           = 0.8     # taille initiale (fraction du cube unite)
    L_min            = 0.5**7  # ~0.0078 — seuil de redemarrage
    L_max            = 1.6     # plafond
    success_factor   = 2.0
    failure_factor   = 0.5

    def __init__(self, n_dims, batch_size=1):
        self.n_dims        = n_dims
        self.batch_size    = batch_size
        # Seuils standards : tau_success = 3, tau_failure = max(4, ceil(d / batch))
        self.n_success_thresh = 3
        self.n_failure_thresh = max(4, int(np.ceil(n_dims / max(batch_size, 1))))
        self.L                = self.L_init
        self.success_count    = 0
        self.failure_count    = 0
        self.best_y           = -np.inf
        self.restart_count    = 0

    def to_dict(self):
        return dict(n_dims=self.n_dims, batch_size=self.batch_size,
                    L=float(self.L),
                    success_count=self.success_count,
                    failure_count=self.failure_count,
                    best_y=float(self.best_y),
                    restart_count=self.restart_count)

    @classmethod
    def from_dict(cls, d, n_dims=None, batch_size=None):
        s = cls(n_dims or d["n_dims"], batch_size or d.get("batch_size", 1))
        s.L              = float(d.get("L", s.L_init))
        s.success_count  = int(d.get("success_count", 0))
        s.failure_count  = int(d.get("failure_count", 0))
        s.best_y         = float(d.get("best_y", -np.inf))
        s.restart_count  = int(d.get("restart_count", 0))
        return s

    def update(self, y_observed_max):
        """Met a jour l'etat selon le nouveau meilleur observe."""
        # Si y observed n'est pas fini, on ignore (rien a apprendre)
        if not np.isfinite(y_observed_max):
            return
        # Tolerance numerique relative — gerer le cas best_y = -inf au demarrage
        if not np.isfinite(self.best_y):
            tol = 1e-9
        else:
            tol = 1e-9 * max(abs(self.best_y), 1e-3)
        if y_observed_max > self.best_y + tol:
            self.success_count += 1
            self.failure_count  = 0
            self.best_y         = y_observed_max
        else:
            self.failure_count += 1
            self.success_count  = 0

        if self.success_count >= self.n_success_thresh:
            self.L = min(self.L * self.success_factor, self.L_max)
            self.success_count = 0
        elif self.failure_count >= self.n_failure_thresh:
            self.L = self.L * self.failure_factor
            self.failure_count = 0

    def needs_restart(self):
        return self.L < self.L_min


def turbo_get_trust_region(state, X_norm_train, Y_scalar_train, gp, encoder):
    """
    Retourne (lo, hi) en [0,1]^D, bornes de la trust region centree sur
    le meilleur point observe et de taille L ponderee par les lengthscales
    inverses du GP (ARD).

    Le poids w_i = ls_i / mean(ls) — plus une dim a un grand lengthscale
    (peu sensible), plus la TR s'etend dans cette direction.
    """
    best_idx = int(np.argmax(Y_scalar_train))
    x_center = X_norm_train[best_idx].copy()

    # Lengthscales depuis le GP (peuvent etre uniques ou ARD)
    try:
        ls = gp.kernel_.length_scale
        ls = np.asarray(ls, dtype=float).ravel()
        if ls.size != encoder.n_params:
            ls = np.full(encoder.n_params, float(ls.mean()))
    except Exception:
        ls = np.ones(encoder.n_params)

    # Normaliser : weights centres autour de 1.0
    w = ls / max(ls.mean(), 1e-6)
    # Eviter degenerescence
    w = np.clip(w, 0.2, 5.0)
    half = (state.L * w / 2.0)
    lo   = np.clip(x_center - half, 0.0, 1.0)
    hi   = np.clip(x_center + half, 0.0, 1.0)
    return lo, hi, x_center


def turbo_should_use(cfg, n_dims, n_obs):
    """v3.7d : auto-activation TuRBO si :
       - haute dim (>=5 params) ET >=10 observations,
       - OU activation explicite via bo_settings.use_turbo: true.
    Desactive si bo_settings.use_turbo: false.
    """
    bo_cfg = cfg.get("bo_settings", {})
    explicit = bo_cfg.get("use_turbo", None)
    if explicit is True:
        return True
    if explicit is False:
        return False
    return (n_dims >= 5) and (n_obs >= 10)


def make_local_penalizer(X_selected_norm, gp, y_train):
    """
    v3.7d : Local Penalization (Gonzalez et al., AISTATS 2016).

    Pour selectionner un batch sans Kriging Believer (refit GP coute cher),
    on multiplie l'acquisition par un facteur de penalisation autour de
    chaque point deja selectionne dans le batch.

    Le facteur de Lipschitz L est estime depuis le GP par max |grad mu|.
    Le rayon r_j depend de la distance |M - mu(x_j)| / L, avec M = max y.
    psi(x; x_j) = phi((|x - x_j| - r_j) / s_j) avec phi = CDF(N(0,1)).

    Retourne une fonction acq_penalized(X) qui multiplie l'acq par
    le produit des psi pour chaque point deja choisi.
    """
    if len(X_selected_norm) == 0:
        return lambda acq_vals, X: acq_vals

    M_obs = float(np.max(y_train))
    # Estimer L : max norme du gradient de mu sur un echantillonnage
    n_grid = 200
    rng_local = np.random.default_rng(0)
    Xg = rng_local.uniform(0, 1, (n_grid, X_selected_norm.shape[1]))
    eps = 1e-3
    mu_g, _ = gp.predict(Xg, return_std=True)
    # Approximation du gradient par differences finies dim par dim
    grads = np.zeros_like(Xg)
    for d in range(Xg.shape[1]):
        Xp = Xg.copy(); Xp[:, d] = np.clip(Xp[:, d] + eps, 0, 1)
        Xm = Xg.copy(); Xm[:, d] = np.clip(Xm[:, d] - eps, 0, 1)
        mp, _ = gp.predict(Xp, return_std=True)
        mm, _ = gp.predict(Xm, return_std=True)
        grads[:, d] = (mp - mm) / (2 * eps)
    L = max(np.linalg.norm(grads, axis=1).max(), 1e-3)

    # Pre-calculer les centres et leurs (mu, sigma)
    mu_sel, sigma_sel = gp.predict(X_selected_norm, return_std=True)
    sigma_sel = np.maximum(sigma_sel, 1e-6)

    def penalize(acq_vals, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Distance aux points selectionnes
        # dist[i, j] = ||X[i] - X_selected[j]||
        dist = np.linalg.norm(X[:, None, :] - X_selected_norm[None, :, :], axis=2)
        r_j = (M_obs - mu_sel) / L  # (n_sel,)
        s_j = sigma_sel / L
        # phi((dist - r_j) / s_j)
        z = (dist - r_j[None, :]) / np.maximum(s_j[None, :], 1e-9)
        from scipy.stats import norm as _norm
        psi = _norm.cdf(z)  # (n_X, n_sel)
        # Multiplie l'acquisition par produit des psi
        return acq_vals * psi.prod(axis=1)

    return penalize


def maximise_acquisition(acq_fn, n_params, n_restarts=20, rng=None, n_train=None,
                         bounds=None):
    """
    v3.7d : si n_train fourni, n_restarts est ajuste automatiquement.
            si bounds fourni (np.array shape (n_params, 2)), restreint la
            recherche a ces bornes (utile pour TuRBO trust region).
            Sinon, comportement historique.
    """
    if rng is None:
        rng = np.random.default_rng()
    # v3.7d : adaptation automatique selon la taille des donnees
    if n_train is not None:
        if n_train < 10:    n_restarts_eff = 20
        elif n_train < 30:  n_restarts_eff = 50
        else:               n_restarts_eff = 100
        # On respecte le minimum demande
        n_restarts = max(n_restarts, n_restarts_eff)
    # v3.7d : bornes restreintes (TuRBO) ou unitaires
    if bounds is not None:
        unit_bounds = [(float(bounds[i, 0]), float(bounds[i, 1]))
                       for i in range(n_params)]
        lo_arr = np.array([b[0] for b in unit_bounds])
        hi_arr = np.array([b[1] for b in unit_bounds])
    else:
        unit_bounds = [(0.0, 1.0)] * n_params
        lo_arr = np.zeros(n_params)
        hi_arr = np.ones(n_params)
    seed_val    = int(rng.integers(0, 2**31))
    de_result   = differential_evolution(
        func=lambda x: -float(acq_fn(x.reshape(1, -1)).item()),
        bounds=unit_bounds, seed=seed_val, maxiter=300, tol=1e-6, popsize=15, workers=1,
    )
    best_x, best_val = de_result.x, -de_result.fun
    for _ in range(n_restarts):
        x0  = lo_arr + rng.uniform(0.0, 1.0, size=n_params) * (hi_arr - lo_arr)
        res = minimize(
            fun=lambda x: -float(acq_fn(x.reshape(1, -1)).item()),
            x0=x0, bounds=unit_bounds, method="L-BFGS-B",
        )
        if res.success and (-res.fun) > best_val:
            best_val, best_x = -res.fun, res.x
    return best_x


def acq_ucb_auto(X_norm, gp, iteration, n_dims, delta=0.5):
    """
    Upper Confidence Bound avec β auto-calculé (Srinivas 2010).

    β = √(2 · log(d · t² · π² / 6δ))

    Contrairement à notre UCB à kappa fixe, β croit avec le nombre
    d iterations t et la dimensionnalité d.
    Au debut (t petit) : β petit → exploitation favorisée.
    Apres (t grand)    : β augmente → exploration croissante.

    Avantage : calibration automatique sans YAML à régler.
    Paramètre : delta=0.5 (valeur standard de la borne théorique).
    """
    import math
    mu, sigma = gp.predict(X_norm, return_std=True)
    t    = max(int(iteration), 1)
    beta = np.sqrt(2 * np.log(n_dims * t**2 * math.pi**2 / (6 * delta)))
    return mu.ravel() + beta * sigma.ravel()


def suggest_ei_ts(gp, encoder, df_done, cfg, y_best, n_suggest,
                  xi_val, rng, n_restarts, X_norm_base):
    """
    Stratégie EI-TS (Hybrid Thompson Sampling d après EDBO).

    Batch de n suggestions :
      - Suggestion 1 : argmax EI classique (meilleure exploitation)
      - Suggestions 2..n : Thompson Sampling indépendants
        (tire n-1 fonctions du posterior GP, prend l argmax de chacune)

    Avantages sur Kriging Believer :
      - Pas de refit du GP pour chaque suggestion supplémentaire
        (2-3x plus rapide pour n>=3)
      - Diversité naturelle via TS sans contrainte de distance artificielle
      - Robuste : si EI est dans une zone peu informative, TS peut l'être ailleurs

    Retourne (suggestions_norm: list[np.array], predictions: list[float])
    """
    from scipy.spatial.distance import cdist as _cdist

    suggestions_norm = []
    predictions      = []

    # ── Suggestion 1 : EI classique ──────────────────────────────────────────
    acq_fn_ei = lambda x, _g=gp, _yb=y_best: acq_ei(x, _g, _yb, xi_val)
    x1 = maximise_acquisition(acq_fn_ei, encoder.n_params, n_restarts, rng)
    suggestions_norm.append(x1)
    mu1, _ = gp.predict(x1.reshape(1, -1), return_std=True)
    predictions.append(float(mu1[0]))

    # ── Suggestions 2..n : Thompson Sampling ─────────────────────────────────
    if n_suggest > 1:
        # Grille dense pour évaluer TS
        n_grid = max(5000, n_suggest * 2000)
        X_grid = rng.uniform(0, 1, (n_grid, encoder.n_params))

        for _ in range(n_suggest - 1):
            # Tirer une fonction du posterior
            ts_sample = acq_thompson(X_grid, gp, rng)

            # Exclure les zones déjà sélectionnées (distance minimale)
            already = np.array(suggestions_norm)
            dists   = _cdist(X_grid, already).min(axis=1)
            # Zéro-er les zones trop proches (diversity implicite)
            ts_penalized = ts_sample * (dists > 0.05).astype(float)

            best_idx = np.argmax(ts_penalized) if ts_penalized.max() > 0 else np.argmax(ts_sample)
            x_ts = X_grid[best_idx]
            suggestions_norm.append(x_ts)
            mu_ts, _ = gp.predict(x_ts.reshape(1, -1), return_std=True)
            predictions.append(float(mu_ts[0]))

    return suggestions_norm, predictions


# =============================================================================
# 8.  BOUCLE BO PRINCIPALE
# =============================================================================

def apply_domain_reduction(encoder, df_done, cfg, bo, iteration):
    """
    Reduction progressive du domaine de recherche autour de la zone prometteuse.

    !! ATTENTION : risque de piege dans des maxima locaux !!
    Desactive par defaut. N utiliser qu en phase d affinage (iterations >= 15)
    quand on est certain d etre dans le bon bassin.

    domain_reduction: true         # activer dans le YAML
    reduction_factor: 0.8          # chaque borne se resserre de 20%
    reduction_start_iter: 15       # ne commence pas avant N iterations BO
    reduction_min_range: 0.3       # plage minimale (30% de la plage initiale)
    """
    if not bo.get("domain_reduction", False):
        return encoder  # Pas de reduction

    start_iter = int(bo.get("reduction_start_iter", 15))
    if iteration < start_iter:
        return encoder

    factor   = float(bo.get("reduction_factor", 0.8))
    min_rng  = float(bo.get("reduction_min_range", 0.3))

    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    df_bo        = df_done.copy()
    for col in output_names:
        df_bo[col] = pd.to_numeric(df_bo[col], errors="coerce")
    df_bo = df_bo.dropna(subset=output_names)

    if len(df_bo) < 3:
        return encoder

    # Centre : meilleur point observe
    best_idx  = df_bo[output_names[0]].idxmax()
    best_row  = df_bo.loc[best_idx]

    # Nouveaux bounds autour du centre
    new_params = []
    for p in cfg["parameters"]:
        if p["type"] != "continuous":
            new_params.append(p)
            continue
        lo_orig = float(p["min"])
        hi_orig = float(p["max"])
        center  = float(best_row[p["name"]])
        rng_orig = hi_orig - lo_orig
        rng_new  = max(rng_orig * factor, rng_orig * min_rng)
        lo_new   = max(lo_orig, center - rng_new / 2)
        hi_new   = min(hi_orig, lo_new + rng_new)
        lo_new   = max(lo_orig, hi_new - rng_new)  # readjust si on touche hi
        pnew = dict(p); pnew["min"] = round(lo_new, 4); pnew["max"] = round(hi_new, 4)
        new_params.append(pnew)
        if abs(lo_new - lo_orig) > 0.01 or abs(hi_new - hi_orig) > 0.01:
            print(f"  [DR]   {p['name']}: [{lo_orig:.2f}, {hi_orig:.2f}]"
                  f" → [{lo_new:.2f}, {hi_new:.2f}]  (centre={center:.2f})")

    cfg_reduced = dict(cfg); cfg_reduced["parameters"] = new_params
    return type(encoder)(cfg_reduced)


def suggest_experiments(cfg, df_done, encoder, n_suggest, rng):
    """
    Retourne (suggestions: list[dict], gp_predictions: list[float]).
    gp_predictions[i] = score predit par le GP au point suggere i.
    Batch : strategie Kriging Believer.
    """
    # ── Dispatch EHVI ────────────────────────────────────────────────────────
    mode = cfg["bo_settings"].get("multi_objective_mode", "weighted_sum")
    if mode == "ehvi" and len(cfg["outputs"]) > 1:
        return suggest_experiments_ehvi(cfg, df_done, encoder, n_suggest, rng)

    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    bo           = cfg["bo_settings"]

    acq_name   = bo.get("acquisition", "EI").upper()
    # v3.7d : auto-activation HEDGE si campagne mature et acq par defaut
    if "acquisition" not in bo and len(df_done) >= 15:
        acq_name = "HEDGE"
        print(f"  [AUTO] {len(df_done)} points -> HEDGE active "
              f"(force YAML acquisition: \"EI\" pour desactiver)")
    kappa      = float(bo.get("ucb_kappa", 2.0))
    xi_val     = float(bo.get("xi", 0.01))
    n_restarts = int(bo.get("n_restarts", 50))
    mode       = bo.get("multi_objective_mode", "weighted_sum")

    # ── v3.8 : AUTO-PILOT (si active) ─────────────────────────────────────
    # Override acq/xi/kappa selon decision orchestrateur (phase, target, coverage).
    # S'applique avant les triggers reactifs de main() pour fonctionner aussi
    # quand suggest_experiments est appele directement (bench, scripts custom).
    if bo.get("autopilot", False):
        try:
            # Calcul stagnation : nb de rounds BO recents sans amelioration
            _no_impr_sx = 0
            try:
                if "iteration_BO" in df_done.columns and len(output_names) > 0:
                    _oc = output_names[0]
                    _d = df_done.copy()
                    _d["_it"] = pd.to_numeric(_d["iteration_BO"], errors="coerce")
                    _d = _d[_d["_it"].notna()]
                    if len(_d) > 0:
                        _by_round = _d.groupby("_it")[_oc].max().sort_index()
                        _running = _by_round.cummax()
                        # rounds consecutifs en fin sans nouveau record
                        _no_impr_sx = 0
                        _best = -np.inf
                        _streak = 0
                        for _v in _running.values:
                            if _v > _best + 1e-9:
                                _best = _v
                                _streak = 0
                            else:
                                _streak += 1
                        _no_impr_sx = _streak
            except Exception:
                _no_impr_sx = 0
            _ap_dec = autopilot_decide(df_done, cfg, bo, encoder,
                                       _no_impr_existing=_no_impr_sx)
            if _ap_dec.get('override'):
                acq_name = _ap_dec['acquisition'].upper()
                if _ap_dec.get('xi') is not None:
                    xi_val = float(_ap_dec['xi'])
                if _ap_dec.get('kappa') is not None:
                    kappa = float(_ap_dec['kappa'])
                if bo.get("autopilot_verbose", True):
                    print(autopilot_format_status(_ap_dec))
        except Exception as _e:
            if bo.get("autopilot_verbose", True):
                print(f"  [AUTOPILOT] erreur, fallback config standard: {_e}")

    # v3.7d : auto-adaptation de xi selon stagnation
    # Si les 3 dernieres iterations BO n'ont rien ameliore -> xi *= 5 (jusqu'a 0.1)
    # Reset des qu'une amelioration est observee.
    auto_xi = bo.get("auto_xi", True)
    if auto_xi and "iteration_BO" in df_done.columns:
        df_bo_x = df_done.copy()
        df_bo_x["iteration_BO"] = pd.to_numeric(df_bo_x["iteration_BO"],
                                                 errors="coerce")
        df_bo_x = df_bo_x.dropna(subset=["iteration_BO"]).sort_values("iteration_BO")
        if len(df_bo_x) >= 4:
            # Calculer le best-so-far par iteration (en espace reel via scalarise)
            Y_seq = scalarise(df_bo_x[output_names].values.astype(float), cfg)
            best_so_far = np.maximum.accumulate(Y_seq)
            # Stagnation : derniere amelioration vs il y a 3 iterations
            n_stag = 0
            tol = 1e-6 * max(abs(best_so_far[-1]), 1e-3)
            for k in range(len(best_so_far) - 1, 0, -1):
                if best_so_far[k] - best_so_far[k - 1] > tol:
                    break
                n_stag += 1
            stagnation_threshold = int(bo.get("xi_stagnation_threshold", 3))
            if n_stag >= stagnation_threshold:
                # Multiplier xi, mais plafonner a 0.10
                multiplier = min(5.0 ** ((n_stag - stagnation_threshold) // 2 + 1), 10.0)
                xi_new = min(xi_val * multiplier, 0.10)
                if xi_new > xi_val * 1.5:
                    print(f"  [AUTO-XI] {n_stag} iter sans amelioration "
                          f"-> xi {xi_val:.3f} -> {xi_new:.3f} (force exploration)")
                    xi_val = xi_new

    X_raw        = df_done[param_names].values.astype(object)
    Y            = df_done[output_names].values.astype(float)
    X_norm_base  = encoder.encode(X_raw)

    suggestions     = []
    gp_predictions  = []   # NEW: prediction GP a chaque suggestion
    X_norm_train    = X_norm_base.copy()
    Y_scalar_train  = None
    y_phantom_list  = []
    _sc_models      = None   # v3.7d : cache des GPs contraintes (cBO)

    # v3.7d : TuRBO — initialisation de l'etat de la trust region
    # Note : TuRBO desactive sous HEDGE (les sous-acq de Hedge ne respecteraient
    # pas la trust region). Les deux features sont complementaires mais s'utilisent
    # separement.
    _use_turbo = (turbo_should_use(cfg, encoder.n_params, len(df_done))
                  and acq_name != "HEDGE")
    _turbo_state = None
    if _use_turbo:
        _turbo_dict = bo.get("_turbo_state", None)
        if _turbo_dict and _turbo_dict.get("n_dims") == encoder.n_params:
            _turbo_state = TrustRegionState.from_dict(
                _turbo_dict, n_dims=encoder.n_params, batch_size=n_suggest)
            # Mettre a jour selon le meilleur observe
            _y_best_so_far = float(scalarise(Y, cfg).max())
            _turbo_state.update(_y_best_so_far)
            print(f"  [TuRBO] L={_turbo_state.L:.4f}  "
                  f"succ={_turbo_state.success_count}  "
                  f"fail={_turbo_state.failure_count}  "
                  f"best_y={_turbo_state.best_y:.3f}")
        else:
            _turbo_state = TrustRegionState(encoder.n_params, batch_size=n_suggest)
            _y_best_so_far = float(scalarise(Y, cfg).max())
            _turbo_state.best_y = _y_best_so_far
            print(f"  [TuRBO] init L={_turbo_state.L:.3f}  "
                  f"(n_dims={encoder.n_params}, n_obs={len(df_done)})")

        # Restart si trust region trop petite
        if _turbo_state.needs_restart():
            _turbo_state = TrustRegionState(encoder.n_params, batch_size=n_suggest)
            _turbo_state.restart_count += 1
            print(f"  [TuRBO] RESTART trust region (L_min atteint, "
                  f"restart #{_turbo_state.restart_count})")

    for s in range(n_suggest):
        pareto_w = None
        if mode == "pareto" and len(output_names) > 1:
            pareto_w = rng.dirichlet(np.ones(len(output_names)))

        if s == 0:
            Y_scalar_raw   = scalarise(Y, cfg, pareto_w)
            Y_scalar_train, _bc_lam, _bc_off = fit_boxcox(Y_scalar_raw, bo, verbose=(s==0))
            _bc_train_max  = float(Y_scalar_train.max()) if _bc_lam is not None else 0.0
        else:
            # v3.7d : fix bug Box-Cox phantom — toujours appliquer la transformation
            # Avant : Y_real (espace reel) + y_phantom (espace BC) -> mix incoherent
            # Maintenant : on transforme Y_real avec le meme lambda que s=0
            Y_real = scalarise(Y, cfg, pareto_w)
            if _bc_lam is not None:
                # Re-applique Box-Cox avec le lambda fixe a s=0
                Y_real_bc = apply_boxcox(Y_real + _bc_off, _bc_lam)
                Y_scalar_train = np.append(Y_real_bc, y_phantom_list)
            else:
                Y_scalar_train = np.append(Y_real, y_phantom_list)

        y_best = float(Y_scalar_train.max())
        gp = build_gp(encoder.n_params, normalize_y=(_bc_lam is None),
                      cfg=cfg, encoder=encoder, y_train=Y_scalar_train)
        gp.fit(X_norm_train, Y_scalar_train)

        if acq_name == "EI":
            acq_fn = lambda x, _g=gp, _yb=y_best: acq_ei(x, _g, _yb, xi_val)

        elif acq_name == "EI-TS":
            # ── EI-TS hybrid (EDBO) : EI pour le 1er point, TS pour les suivants ──
            # On court-circuite le reste de la boucle et on retourne directement.
            # Plus rapide que Kriging Believer (pas de refit GP intermédiaire).
            if s == 0:
                _ei_ts_sugg_norm, _ei_ts_preds = suggest_ei_ts(
                    gp, encoder, df_done, cfg, y_best, n_suggest,
                    xi_val, rng, n_restarts, X_norm_base)
                for _xn, _pred_raw in zip(_ei_ts_sugg_norm, _ei_ts_preds):
                    _d = encoder.decode(_xn.reshape(1, -1))[0]
                    suggestions.append(_d)
                    _p_real = invert_boxcox(np.array([_pred_raw]), _bc_lam)
                    _p_scalar = float(np.asarray(_p_real).ravel()[0])
                    gp_predictions.append(round(float(np.clip(_p_scalar, 0, 9999)), 2))
                break   # sort de la boucle for s
            else:
                # Ne devrait pas arriver (break ci-dessus), mais fallback EI
                acq_fn = lambda x, _g=gp, _yb=y_best: acq_ei(x, _g, _yb, xi_val)

        elif acq_name == "UCB":
            acq_fn = lambda x, _g=gp: acq_ucb(x, _g, kappa)

        elif acq_name == "UCB-AUTO":
            # UCB avec β auto-calculé (Srinivas 2010)
            _iter_cur = int(pd.to_numeric(
                df_done.get("iteration_BO", pd.Series(dtype=float)),
                errors="coerce").fillna(0).max() or 0)
            acq_fn = lambda x, _g=gp, _it=_iter_cur, _nd=encoder.n_params:                 acq_ucb_auto(x, _g, _it, _nd)

        elif acq_name == "PI":
            acq_fn = lambda x, _g=gp, _yb=y_best: acq_pi(x, _g, _yb, xi_val)

        elif acq_name == "TEI":
            beta_tei = float(bo.get("tei_beta", 8))
            acq_fn   = lambda x, _g=gp, _yb=y_best: acq_tei(x, _g, _yb, xi_val, beta_tei)

        elif acq_name == "THOMPSON":
            grid   = rng.uniform(0.0, 1.0, size=(4000, encoder.n_params))
            sample = acq_thompson(grid, gp, rng)
            acq_fn = lambda x, _gr=grid, _sm=sample: np.array([
                _sm[np.argmin(np.sum((_gr - pt) ** 2, axis=1))]
                for pt in (x if x.ndim == 2 else x.reshape(1, -1))
            ])

        elif acq_name == "VARMAX":
            # Pure exploration : point de variance maximale
            # Utile comme mode forcé quand stagnation détectée manuellement
            acq_fn = lambda x, _g=gp: _g.predict(x, return_std=True)[1].ravel()

        elif acq_name == "KG":
            # Knowledge Gradient exact (v3.6). Cout CPU eleve (~30s/suggestion)
            # car refit hypothetique du GP pour chaque point candidat.
            # Utile pour phase d exploitation fine (> 15 iterations).
            n_fantasy = int(bo.get("kg_n_fantasy", 32))
            acq_fn = lambda x, _g=gp, _yb=y_best, _nf=n_fantasy, _r=rng: acq_kg(
                x if x.ndim == 2 else x.reshape(1, -1),
                _g, _yb, n_fantasy=_nf, rng=_r
            )

        elif acq_name == "HEDGE":
            # v3.7d : portfolio GP-Hedge — choisit dynamiquement parmi
            # EI, UCB-AUTO, TEI selon leur historique de performance.
            # Construire les acq individuelles
            _iter_cur_h = int(pd.to_numeric(
                df_done.get("iteration_BO", pd.Series(dtype=float)),
                errors="coerce").fillna(0).max() or 0)
            beta_tei_h = float(bo.get("tei_beta", 8))
            _hedge_pool = {
                "EI":       lambda x, _g=gp, _yb=y_best: acq_ei(x, _g, _yb, xi_val),
                "UCB-AUTO": lambda x, _g=gp, _it=_iter_cur_h, _nd=encoder.n_params:
                            acq_ucb_auto(x, _g, _it, _nd),
                "TEI":      lambda x, _g=gp, _yb=y_best, _b=beta_tei_h:
                            acq_tei(x, _g, _yb, xi_val, _b),
            }
            # WARNING: GP-Hedge actuellement experimental/non-optimal
            # Les 3 acquireurs proposent souvent le meme point (mu identiques)
            # -> pas de differentiation -> random walk
            # Cause: maximise_acquisition converge vers meme optimum global
            # TODO: ajouter perturbation ou diversite dans proposals
            # Hedge selectionne directement le point ; on court-circuite la suite
            x_chosen, name_chosen, _props = gp_hedge_select(
                _hedge_pool, encoder, gp, df_done, cfg, rng,
                eta=bo.get("hedge_eta", None),     # None = auto Hoffman 2011
                n_train_for_max=len(X_norm_train))
            print(f"  [HEDGE] iter={_iter_cur_h}  acq choisie={name_chosen}  "
                  f"(parmi {len(_hedge_pool)})")
            x_next_dict = encoder.decode(x_chosen.reshape(1, -1))[0]
            suggestions.append(x_next_dict)
            # Prediction GP au point choisi (espace reel)
            y_phantom_bc = float(gp.predict(x_chosen.reshape(1, -1))[0])
            if _bc_lam is not None:
                _y_real_max = float(invert_boxcox(np.array([_bc_train_max]), _bc_lam)[0]) - _bc_off
                _y_real_clip = max(_y_real_max * 1.30, _y_real_max + 10.0)
                y_raw_inv = float(invert_boxcox(np.array([y_phantom_bc]), _bc_lam)[0]) - _bc_off
                y_phantom_real = float(np.clip(y_raw_inv, 0.0, _y_real_clip))
            else:
                y_phantom_real = y_phantom_bc
            gp_predictions.append(y_phantom_real)
            y_phantom_list.append(y_phantom_bc)
            X_norm_train = np.vstack([X_norm_train, x_chosen.reshape(1, -1)])
            continue  # passe a s suivant sans le reste de la boucle

        else:
            raise ValueError(f"Acquisition inconnue : '{acq_name}'.")

        # ── Feasibility constraints (v3.7) : wrapper acq_fn pour exclure ─────
        # les points qui violent les contraintes YAML (renvoie -inf)
        _constraints = cfg.get("constraints", []) or []
        if _constraints:
            _base_acq = acq_fn
            def _feasible_acq(X, _ba=_base_acq, _cstr=_constraints, _enc=encoder):
                Xa = X if X.ndim == 2 else X.reshape(1, -1)
                scores = _ba(Xa).ravel().astype(float)
                # Decoder et filtrer
                try:
                    decoded_all = _enc.decode(Xa)
                    for k, dec in enumerate(decoded_all):
                        feas, _ = check_feasibility(dec, _cstr)
                        if not feas:
                            scores[k] = -np.inf
                except Exception:
                    pass
                return scores
            acq_fn = _feasible_acq

        # ── Boundary avoidance (v3.7b) : penaliser zones deja mauvaises ─────
        _ba_active = cfg.get("bo_settings", {}).get("boundary_avoidance", True)
        if _ba_active and len(X_norm_train) >= 5:
            _base_acq2 = acq_fn
            _yt_ba = Y_scalar_train
            def _ba_acq(X, _ba=_base_acq2, _Xt=X_norm_train, _yt=_yt_ba):
                Xa   = X if X.ndim == 2 else X.reshape(1, -1)
                vals = _ba(Xa).ravel().astype(float)
                # threshold 0.75 : mauvais si y < 75% du meilleur
                return _apply_boundary_avoidance(vals, Xa, _Xt, _yt,
                                                  threshold_pct=0.75, radius=0.20, penalty=0.1)
            acq_fn = _ba_acq

        # v3.7d : cBO — multiplier par P(contraintes respectees)
        # Active automatiquement si soft_constraints est defini dans le YAML.
        if _sc_models is None:  # fit GPs contraintes 1 fois par batch
            _sc_models = fit_constraint_gps(cfg, df_done, encoder)
            if _sc_models:
                print(f"  [cBO] {len(_sc_models)} contrainte(s) active(s) : "
                      + ", ".join(cm["name"] for cm in _sc_models))
        if _sc_models:
            acq_fn = constrained_acquisition(acq_fn, _sc_models)

        # v3.7d : TuRBO bounds (si actif, restreint la recherche)
        _turbo_bounds = None
        if _turbo_state is not None:
            lo_tr, hi_tr, _ = turbo_get_trust_region(
                _turbo_state, X_norm_train, Y_scalar_train, gp, encoder)
            _turbo_bounds = np.column_stack([lo_tr, hi_tr])
        
        # v3.7d : Batch strategies alternatives au Kriging Believer
        # v3.7d : défaut "qei" (validé +58% vs kriging_believer sur Branin)
        # Legacy: "kriging_believer", "local_penalization"
        _batch_strat = bo.get("batch_strategy", "qei")
        
        # qEI/qNEI : quasi-MC Expected Improvement pour batch
        if _batch_strat in ("qei", "qnei") and s > 0 and len(suggestions) > 0:
            # Construire batch actuel (points déjà sélectionnés)
            X_batch_current = np.array([encoder.encode(np.array([[v for v in d.values()]],
                                        dtype=object))[0] for d in suggestions])
            
            # Fonction batch : optimise qEI([batch_current + x_new])
            acq_batch_fn = acq_qei if _batch_strat == "qei" else acq_qnei
            n_samples_qei = int(bo.get("qei_n_samples", 512))
            
            def batch_acq_wrapper(x_new):
                """qEI du batch complet : retourne np.array (compat .item())."""
                x_new_2d = x_new.reshape(1, -1) if x_new.ndim == 1 else x_new
                X_full_batch = np.vstack([X_batch_current, x_new_2d])
                val = acq_batch_fn(X_full_batch, gp, y_best, xi_val, n_samples=n_samples_qei)
                return np.asarray([float(val)])
            
            # Optimiser x_new tel que qEI([batch + x_new]) max
            # On passe la fonction qui retourne qEI directement (positif, a maximiser)
            x_next_norm = maximise_acquisition(
                batch_acq_wrapper, encoder.n_params, n_restarts, rng,
                n_train=len(X_norm_train), bounds=_turbo_bounds
            )
            # Skip le reste de la boucle (pas de Local Penalization ni acq_fn standard)
            
        # Local Penalization : pénalise autour des points déjà sélectionnés
        elif _batch_strat == "local_penalization" and s > 0 and len(suggestions) > 0:
            X_sel_arr = np.array([encoder.encode(np.array([[v for v in d.values()]],
                                  dtype=object))[0] for d in suggestions])
            _lp_fn = make_local_penalizer(X_sel_arr, gp, Y_scalar_train[:len(Y)])
            _base_acq_lp = acq_fn
            def _lp_acq(X, _ba=_base_acq_lp, _lp=_lp_fn):
                Xa = X if X.ndim == 2 else X.reshape(1, -1)
                vals = _ba(Xa).ravel().astype(float)
                return _lp(vals, Xa)
            acq_fn = _lp_acq

        # Pour kriging_believer ou si s=0 : utiliser acq_fn standard
        # Pour qEI/qNEI : on a déjà x_next_norm, skip cette section
        if _batch_strat not in ("qei", "qnei") or s == 0 or len(suggestions) == 0:
            x_next_norm = maximise_acquisition(acq_fn, encoder.n_params, n_restarts, rng,
                                               n_train=len(X_norm_train),
                                               bounds=_turbo_bounds)
        # Multi-modal detection : K-means rapide → multi-start si confirme
        if s == 0 and acq_name not in ("THOMPSON",) and len(df_done) >= 8:
            mm_detected, mm_msg = detect_multimodality(df_done, cfg, encoder, bo)
            if mm_detected:
                print(mm_msg)
                modal_cands = find_modal_candidates(acq_fn, encoder,
                                                    n_restarts=max(n_restarts, 15),
                                                    rng=rng)
                if len(modal_cands) > 1:
                    _report_modal_candidates(modal_cands, encoder, cfg,
                                             best_acq_val=modal_cands[0][1])

        # ── Diversity filter pour les batchs (n_suggest >= 2) ────────────────
        # Au lieu du Kriging Believer seul, on evalue l acquisition sur
        # une grille dense et on selectionne les candidats les plus diversifies.
        # Active quand : n_suggest >= 2 et pas deja au dernier point du batch.
        diversity_on = bo.get("diversity_filter", True)
        if diversity_on and n_suggest >= 2 and len(suggestions) == s:
            n_grid_div   = int(bo.get("diversity_n_grid", 3000))
            X_div_grid   = rng.uniform(0, 1, (n_grid_div, encoder.n_params))
            scores_div   = acq_fn(X_div_grid).ravel()
            min_d_div    = float(bo.get("diversity_min_dist",
                                        0.10 if n_suggest == 2 else 0.06))
            # Inclure le point deja choisi par maximise_acquisition
            X_div_full   = np.vstack([x_next_norm.reshape(1,-1), X_div_grid])
            sc_full      = np.concatenate([[scores_div.max() + 1e-6], scores_div])
            top_idxs     = select_top_diverse_candidates(
                X_div_full, sc_full, n=1, min_dist=min_d_div)
            x_next_norm  = X_div_full[top_idxs[0]]

        x_next_dict = encoder.decode(x_next_norm.reshape(1, -1))[0]
        suggestions.append(x_next_dict)

        y_phantom_bc = float(gp.predict(x_next_norm.reshape(1, -1))[0])
        # Inverser vers l espace reel (MinMax)
        if _bc_lam is not None:
            # Clip le resultat en espace reel : max 30% au-dela du max observe
            _y_real_max = float(invert_boxcox(np.array([_bc_train_max]), _bc_lam)[0]) - _bc_off
            _y_real_clip = max(_y_real_max * 1.30, _y_real_max + 10.0)
            y_raw_inv = float(invert_boxcox(np.array([y_phantom_bc]), _bc_lam)[0]) - _bc_off
            y_phantom_real = float(np.clip(y_raw_inv, 0.0, _y_real_clip))
        else:
            y_phantom_real = y_phantom_bc
        gp_predictions.append(y_phantom_real)
        y_phantom_list.append(y_phantom_bc)  # kriging believer reste en espace transforme
        X_norm_train = np.vstack([X_norm_train, x_next_norm.reshape(1, -1)])

    # v3.7d : persister l'etat TuRBO pour la prochaine iteration
    if _turbo_state is not None:
        cfg.setdefault("bo_settings", {})["_turbo_state"] = _turbo_state.to_dict()

    # v3.9 : garde-fou final de faisabilite (hard constraints)
    # Le filtre dans l'acquisition peut laisser passer un point infaisable
    # lors d'un raffinement local serre (contraintes continues etroites).
    # On re-verifie chaque suggestion ; si infaisable, on la remplace par un
    # echantillon aleatoire faisable (jusqu'a 200 essais).
    _hard = cfg.get("constraints", []) or []
    if _hard and suggestions:
        cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]
        cat_params = [p for p in cfg["parameters"] if p["type"] == "categorical"]
        for si, sug in enumerate(suggestions):
            feas, _ = check_feasibility(sug, _hard)
            if feas:
                continue
            # tenter de reparer par echantillonnage rejet
            repaired = None
            for _ in range(200):
                cand = {}
                for p in cont_params:
                    cand[p["name"]] = round(float(rng.uniform(p["min"], p["max"])), 4)
                for p in cat_params:
                    cand[p["name"]] = p["categories"][int(rng.integers(0, len(p["categories"])))]
                f2, _ = check_feasibility(cand, _hard)
                if f2:
                    repaired = cand
                    break
            if repaired is not None:
                # conserver les cles supplementaires eventuelles
                for k in sug:
                    if k not in repaired:
                        repaired[k] = sug[k]
                suggestions[si] = repaired
                print(f"  [CONSTRAINT] suggestion {si+1} infaisable -> remplacee "
                      f"par un point faisable")

    return suggestions, gp_predictions   # NEW: retourne le tuple



# =============================================================================
# 8b.  BOUCLE BO EHVI  (multi-objectif, un GP par objectif)
# =============================================================================

def suggest_experiments_ehvi(cfg, df_done, encoder, n_suggest, rng):
    """
    Variante EHVI de suggest_experiments.
    - Entraine un GP SEPARE par objectif (pas de scalarisation).
    - Maximise MC-EHVI sur une grille aleatoire dense.
    - Retourne (suggestions, gp_predictions) comme suggest_experiments.
      gp_predictions[i] = prediction de l'objectif principal (index 0)
                          au point suggere (echelle originale).
    """
    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    bo           = cfg["bo_settings"]
    n_obj        = len(output_names)

    X_raw    = df_done[param_names].values.astype(object)
    Y_raw    = df_done[output_names].values.astype(float)
    X_norm   = encoder.encode(X_raw)

    # Convert all objectives to "maximise" convention
    signs    = np.array([_sign(o) for o in cfg["outputs"]])
    Y_signed = Y_raw * signs   # shape (n_obs, n_obj)

    # Reference point : auto-calcule (min observe - 10% de la plage)
    if "ref_point" in bo and bo["ref_point"] is not None:
        # User-specified ref_point (in ORIGINAL direction, one per objective)
        ref_point = np.array([float(v) * s for v, s in zip(bo["ref_point"], signs)])
    else:
        ranges    = np.abs(Y_signed.max(axis=0) - Y_signed.min(axis=0)) + 1e-6
        ref_point = Y_signed.min(axis=0) - 0.1 * ranges

    suggestions    = []
    gp_predictions = []
    X_aug          = X_norm.copy()
    Y_aug_signed   = Y_signed.copy()

    for s in range(n_suggest):

        # Fit one GP per objective on current (augmented) data
        gps = []
        for j in range(n_obj):
            gp_j = build_gp(encoder.n_params, normalize_y=True)
            gp_j.fit(X_aug, Y_aug_signed[:, j])
            gps.append(gp_j)

        # Maximise EHVI ou qNEHVI selon config
        _mo_acq = cfg["bo_settings"].get("mo_acquisition", "ehvi").lower()
        if _mo_acq == "qnehvi":
            print(f"    qNEHVI : calcul suggestion {s+1}/{n_suggest} "
                  f"(noisy + q-batch)...", flush=True)
            x_next_norm = _maximise_qnehvi_grid(gps, ref_point, Y_aug_signed,
                                                 cfg, encoder, rng, n_batch=1)
        else:
            print(f"    EHVI : calcul suggestion {s+1}/{n_suggest} "
                  f"({'exact 2D' if n_obj == 2 else 'MC 3D+'})...", flush=True)
            x_next_norm = _maximise_ehvi_grid(gps, ref_point, Y_aug_signed,
                                               cfg, encoder, rng)
        x_next_dict = encoder.decode(x_next_norm.reshape(1, -1))[0]
        suggestions.append(x_next_dict)

        # Prediction du 1er objectif (echelle originale) pour stockage/affichage
        y_prim_signed = float(gps[0].predict(x_next_norm.reshape(1, -1))[0])
        gp_predictions.append(y_prim_signed * signs[0])

        # Kriging Believer : halluciner tous les objectifs
        y_phantom = np.array([
            float(gps[j].predict(x_next_norm.reshape(1, -1))[0])
            for j in range(n_obj)
        ])
        X_aug        = np.vstack([X_aug, x_next_norm.reshape(1, -1)])
        Y_aug_signed = np.vstack([Y_aug_signed, y_phantom.reshape(1, -1)])

    return suggestions, gp_predictions


# =============================================================================
# 9.  PARAMETRES LES PLUS INFLUENTS (ARD length-scales)
# =============================================================================

def get_influential_params(gp, cfg, n=2):
    """
    Retourne les indices (dans cfg['parameters']) des n parametres continus
    les plus influents selon les ARD length-scales du GP.
    Un length-scale court signifie une variation rapide -> forte influence.
    """
    cont_indices = [i for i, p in enumerate(cfg["parameters"])
                    if p["type"] == "continuous"]
    if len(cont_indices) < 2:
        return cont_indices[:n]

    try:
        ls = gp.kernel_.k2.length_scale   # (n_params,)
        # Trier les continus par length-scale croissant (plus influent en premier)
        cont_sorted = sorted(cont_indices, key=lambda i: ls[i])
        return cont_sorted[:n]
    except Exception:
        return cont_indices[:n]


# =============================================================================
# 10.  GRAPHIQUE PAR ROUND – style Figure 2b  (non-ecrasant)
# =============================================================================

# =============================================================================
# 9b.  PARTIAL DEPENDENCE PLOTS + DASHBOARD 16:9
# =============================================================================

def compute_pdp(gp, encoder, cfg, df_done, param_name,
                n_grid=80, n_mc=30, rng=None):
    """
    Calcule la courbe de dépendance partielle pour un paramètre.

    Pour chaque valeur de param_name sur sa plage, on fixe les autres
    variables à leur distribution (Monte Carlo sur les données observées)
    et on prédit mu(x) avec le GP.

    Retourne (grid_values, mu_mean, mu_std) dans l espace réel.
    Pour les variables catégorielles, retourne (categories, mu, None).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    param_idx    = param_names.index(param_name)
    p_cfg        = cfg["parameters"][param_idx]

    # Encodage des données observées
    X_obs = encoder.encode(df_done[param_names].values.astype(object))

    # Déterminer la position encodée du paramètre dans X_norm
    dim_cursor = 0
    param_start_dim = 0
    param_n_dims    = 1
    for i, p in enumerate(cfg["parameters"]):
        nd = encoder._encoded_dims[i]
        if p["name"] == param_name:
            param_start_dim = dim_cursor
            param_n_dims    = nd
            break
        dim_cursor += nd

    if p_cfg["type"] == "continuous":
        lo, hi   = float(p_cfg["min"]), float(p_cfg["max"])
        grid_raw = np.linspace(lo, hi, n_grid)
        grid_enc = (grid_raw - lo) / max(hi - lo, 1e-9)

        # Pour chaque valeur de la grille, marginalise sur les autres vars
        mu_all = []
        for g_val in grid_enc:
            X_mc = X_obs[rng.integers(0, len(X_obs), n_mc)].copy()
            X_mc[:, param_start_dim] = g_val
            mu, _ = gp.predict(X_mc, return_std=True)
            mu_all.append(mu)
        mu_arr = np.array(mu_all)  # (n_grid, n_mc)

        # Inverser la normalisation de Y si BoxCox actif
        bo = cfg["bo_settings"]
        if bo.get("boxcox", False):
            Y_ref    = df_done[output_names].values.astype(float)
            y_raw    = scalarise(Y_ref, cfg)
            _, scale, _ = fit_boxcox(y_raw, bo)
            if scale is not None:
                mu_arr = invert_boxcox(mu_arr, scale)

        return grid_raw, mu_arr.mean(axis=1), mu_arr.std(axis=1)

    else:  # catégorielle
        cats     = p_cfg["categories"]
        cat_pos  = encoder._cat_pos[param_idx]
        mu_cats  = []
        for cat in cats:
            X_mc = X_obs[rng.integers(0, len(X_obs), n_mc)].copy()
            vec  = encoder._encode_cat(cat_pos, cat)
            X_mc[:, param_start_dim:param_start_dim+param_n_dims] = vec
            mu, _ = gp.predict(X_mc, return_std=True)
            mu_cats.append(float(mu.mean()))

        bo = cfg["bo_settings"]
        if bo.get("boxcox", False):
            Y_ref    = df_done[output_names].values.astype(float)
            y_raw    = scalarise(Y_ref, cfg)
            _, scale, _ = fit_boxcox(y_raw, bo)
            if scale is not None:
                mu_cats = list(invert_boxcox(np.array(mu_cats), scale))
        return cats, np.array(mu_cats), None


def _colors_for_values(vals, cmap_name="RdYlGn"):
    """Mappe des valeurs scalaires vers des couleurs RGBA."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cmap = cm.get_cmap(cmap_name)
    vmin, vmax = np.min(vals), np.max(vals)
    norm = mcolors.Normalize(vmin=vmin, vmax=max(vmax, vmin+1e-9))
    return [cmap(norm(v)) for v in vals]



def _plot_gp_heatmap_trio(fig, gs, gp_display, encoder, cfg, df_done,
                          df_sugg_cur, out_name, maximize):
    """
    Trace 3 heatmaps côte-à-côte dans gs[2, :] montrant le paysage GP
    pour les 3 paires des variables continues les plus influentes.

    Pour chaque paire (p_i, p_j) :
      - axe X = valeur physique de p_i
      - axe Y = valeur physique de p_j
      - couleur = prédiction GP (mu) avec les autres paramètres fixés au point optimal
      - points réels superposés (cercles)
      - suggestions du round (étoiles)

    Si < 3 paramètres continus, adapte au nombre de paires disponibles.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    param_names = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]

    # Identifier les paramètres continus
    cont_idx = [i for i,p in enumerate(cfg["parameters"]) if p["type"] == "continuous"]
    if len(cont_idx) < 2:
        ax_empty = fig.add_subplot(gs[2, :])
        ax_empty.text(0.5, 0.5,
                     "Heatmaps : nécessite >= 2 paramètres continus",
                     ha="center", va="center", transform=ax_empty.transAxes,
                     fontsize=10, color="#888888")
        ax_empty.axis("off")
        return

    # Prioriser par importance ARD (lengthscale faible = important)
    try:
        ls = gp_display.kernel_.k2.length_scale
        cont_importance = [(i, 1.0 / (ls[encoder.continuous_idx.index(i):
                                         encoder.continuous_idx.index(i)+1].mean() + 1e-9))
                           for i in cont_idx]
        cont_importance.sort(key=lambda x: -x[1])
        cont_sorted = [c[0] for c in cont_importance]
    except Exception:
        cont_sorted = cont_idx

    # Générer les paires (max 3)
    pairs = []
    n_cont = len(cont_sorted)
    if n_cont >= 3:
        pairs = [(cont_sorted[0], cont_sorted[1]),
                 (cont_sorted[0], cont_sorted[2]),
                 (cont_sorted[1], cont_sorted[2])]
    elif n_cont == 2:
        pairs = [(cont_sorted[0], cont_sorted[1])]
    n_panels = len(pairs)

    # Meilleur point (pour fixer les autres paramètres)
    Y = df_done[output_names].values.astype(float)
    scores = scalarise(Y, cfg)
    best_i = int(np.argmax(scores)) if maximize else int(np.argmin(scores))
    X_best_raw  = df_done[param_names].values.astype(object)[best_i:best_i+1]
    X_best_norm = encoder.encode(X_best_raw)[0]

    # Sous-grille pour les 3 heatmaps (ou moins)
    sub_gs = GridSpecFromSubplotSpec(1, n_panels, subplot_spec=gs[2, :],
                                      wspace=0.35)

    n_grid = 40
    grid = np.linspace(0, 1, n_grid)
    Xn, Yn = np.meshgrid(grid, grid)

    for panel_idx, (pi, pj) in enumerate(pairs):
        ax = fig.add_subplot(sub_gs[0, panel_idx])
        p_i = cfg["parameters"][pi]
        p_j = cfg["parameters"][pj]

        # Positions normalisées dans l'espace encoder
        # Pour un paramètre continu, l'index encoder = sa position dans continuous_idx
        ki = encoder.continuous_idx.index(pi)
        kj = encoder.continuous_idx.index(pj)

        # Construire la grille dans l'espace encoder complet
        n_pts = n_grid * n_grid
        X_grid = np.tile(X_best_norm, (n_pts, 1))
        # Remplacer les 2 dimensions ki et kj par la grille
        # Note : dans l'encoder, continuous occupent les positions ki et kj
        # On doit mapper depuis param_idx pi → dimension dans X_norm
        # Pour continuous : dimension = cumul des encoded_dims des params avant pi
        dim_i = sum(encoder._encoded_dims[k] for k in range(pi))
        dim_j = sum(encoder._encoded_dims[k] for k in range(pj))
        X_grid[:, dim_i] = Xn.ravel()
        X_grid[:, dim_j] = Yn.ravel()

        mu_grid = gp_display.predict(X_grid).reshape(n_grid, n_grid)

        # Axes physiques
        lo_i, hi_i = float(p_i["min"]), float(p_i["max"])
        lo_j, hi_j = float(p_j["min"]), float(p_j["max"])
        # Gérer log_ratio pour l affichage
        if pi in getattr(encoder, "_log_ratio_idx", set()):
            X_phys = np.exp(np.log(max(lo_i,1e-9)) + grid * (np.log(max(hi_i,1e-9)) - np.log(max(lo_i,1e-9))))
        else:
            X_phys = grid * (hi_i - lo_i) + lo_i
        if pj in getattr(encoder, "_log_ratio_idx", set()):
            Y_phys = np.exp(np.log(max(lo_j,1e-9)) + grid * (np.log(max(hi_j,1e-9)) - np.log(max(lo_j,1e-9))))
        else:
            Y_phys = grid * (hi_j - lo_j) + lo_j

        XP, YP = np.meshgrid(X_phys, Y_phys)

        # Heatmap
        cmap = "viridis" if maximize else "viridis_r"
        im = ax.pcolormesh(XP, YP, mu_grid, cmap=cmap, shading="auto", alpha=0.92)

        # Points réels superposés
        df_plot = df_done.copy()
        vals_x = pd.to_numeric(df_plot[p_i["name"]], errors="coerce")
        vals_y = pd.to_numeric(df_plot[p_j["name"]], errors="coerce")
        vals_c = pd.to_numeric(df_plot[out_name], errors="coerce")
        mask = vals_x.notna() & vals_y.notna() & vals_c.notna()
        ax.scatter(vals_x[mask], vals_y[mask], c=vals_c[mask], cmap=cmap,
                   edgecolors="white", linewidths=1.2, s=55, zorder=3,
                   vmin=mu_grid.min(), vmax=mu_grid.max())

        # Suggestions (étoiles)
        if df_sugg_cur is not None and len(df_sugg_cur) > 0:
            sx = pd.to_numeric(df_sugg_cur[p_i["name"]], errors="coerce")
            sy = pd.to_numeric(df_sugg_cur[p_j["name"]], errors="coerce")
            m2 = sx.notna() & sy.notna()
            ax.scatter(sx[m2], sy[m2], marker="*", s=240,
                       c="#D55E00", edgecolors="white", linewidths=1.5,
                       zorder=5, label="Suggestions")

        ax.set_xlabel(p_i["name"], fontsize=8)
        ax.set_ylabel(p_j["name"], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Paysage GP : {p_i['name']} × {p_j['name']}",
                     fontsize=9, fontweight="bold", pad=4)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label(f"μ({out_name})", fontsize=7)
        cbar.ax.tick_params(labelsize=6)



def plot_convergence_animation(df, cfg, gp_display, encoder, opts=None):
    """
    Genere une animation GIF de l'evolution du paysage GP round apres round.
    Fichier : {campaign}_convergence.gif
    Declenche uniquement par --final ou a convergence automatique.

    Contenu : paysage GP 2D (2 params les plus influents) + points accumules
    round apres round. Chaque frame = 1 itération BO.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import warnings
    warnings.filterwarnings("ignore")

    if opts is None: opts = {}
    campaign  = cfg.get("campaign_name", "campaign")
    out_name  = cfg["outputs"][0]["name"]
    maximize  = cfg["outputs"][0].get("direction", "maximize") == "maximize"
    param_names = [p["name"] for p in cfg["parameters"]]

    df_done = df[df["statut"].astype(str).str.strip() == "fait"].copy()
    df_done[out_name] = pd.to_numeric(df_done[out_name], errors="coerce")
    df_done = df_done.dropna(subset=[out_name])
    if len(df_done) < 4:
        return  # pas assez pour animer

    # Identifier les 2 params continus les plus influents (ARD)
    cont_idx = [i for i,p in enumerate(cfg["parameters"]) if p["type"] == "continuous"]
    if len(cont_idx) < 2:
        return  # impossible d'animer avec < 2 variables continues

    try:
        ls = gp_display.kernel_.k2.length_scale if hasattr(gp_display.kernel_, "k2") \
             else gp_display.kernel_.length_scale
        cont_imp = [(i, 1.0 / (ls[encoder.continuous_idx.index(i):
                                encoder.continuous_idx.index(i)+1].mean() + 1e-9))
                    for i in cont_idx]
        cont_imp.sort(key=lambda x: -x[1])
        pi_idx, pj_idx = cont_imp[0][0], cont_imp[1][0]
    except Exception:
        pi_idx, pj_idx = cont_idx[0], cont_idx[1]

    p_i = cfg["parameters"][pi_idx]
    p_j = cfg["parameters"][pj_idx]

    # Bornes physiques
    lo_i, hi_i = float(p_i["min"]), float(p_i["max"])
    lo_j, hi_j = float(p_j["min"]), float(p_j["max"])

    # Point optimal pour fixer autres vars
    y_vals = df_done[out_name].values.astype(float)
    best_i = int(np.argmax(y_vals)) if maximize else int(np.argmin(y_vals))
    X_best_raw = df_done[param_names].values.astype(object)[best_i:best_i+1]
    X_best_norm = encoder.encode(X_best_raw)[0]

    # Iterations disponibles
    iter_col = pd.to_numeric(df_done["iteration_BO"], errors="coerce").fillna(0).astype(int).values
    max_iter = int(iter_col.max())

    # Pre-calculer positions physiques pour le scatter
    X_all_raw = df_done[param_names].values
    xi_all = df_done[p_i["name"]].astype(float).values
    yj_all = df_done[p_j["name"]].astype(float).values

    # Grille pour la heatmap (fixe pour toutes les frames)
    n_grid = 40
    grid = np.linspace(0, 1, n_grid)
    Xn, Yn = np.meshgrid(grid, grid)

    # Echelle physique
    if pi_idx in getattr(encoder, "_log_ratio_idx", set()):
        X_phys = np.exp(np.log(max(lo_i,1e-9)) + grid*(np.log(max(hi_i,1e-9))-np.log(max(lo_i,1e-9))))
    else:
        X_phys = grid * (hi_i - lo_i) + lo_i
    if pj_idx in getattr(encoder, "_log_ratio_idx", set()):
        Y_phys = np.exp(np.log(max(lo_j,1e-9)) + grid*(np.log(max(hi_j,1e-9))-np.log(max(lo_j,1e-9))))
    else:
        Y_phys = grid * (hi_j - lo_j) + lo_j
    XP, YP = np.meshgrid(X_phys, Y_phys)

    # Grille d'evaluation GP (encoder complet)
    n_pts = n_grid * n_grid
    X_grid_base = np.tile(X_best_norm, (n_pts, 1))
    dim_i = sum(encoder._encoded_dims[k] for k in range(pi_idx))
    dim_j = sum(encoder._encoded_dims[k] for k in range(pj_idx))
    X_grid_base[:, dim_i] = Xn.ravel()
    X_grid_base[:, dim_j] = Yn.ravel()
    mu_grid = gp_display.predict(X_grid_base).reshape(n_grid, n_grid)

    # Preparation figure
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    fig.patch.set_facecolor("#F8F9FA")
    cmap = "viridis" if maximize else "viridis_r"

    # Frame 0 : background heatmap fixe
    im = ax.pcolormesh(XP, YP, mu_grid, cmap=cmap, shading="auto", alpha=0.88)
    scat = ax.scatter([], [], c=[], cmap=cmap, edgecolors="white", linewidths=1.5,
                      s=110, zorder=3, vmin=y_vals.min(), vmax=y_vals.max())
    star = ax.scatter([], [], marker="*", s=320, c="#FFD700",
                      edgecolors="#1A3A5C", linewidths=1.8, zorder=5)
    ax.set_xlabel(p_i["name"], fontsize=11)
    ax.set_ylabel(p_j["name"], fontsize=11)
    title = ax.set_title("", fontsize=12, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f"mu({out_name}) GP", fontsize=10)

    def update(frame):
        # frame = iteration BO courante
        mask = iter_col <= frame
        if mask.sum() == 0:
            return scat, star, title
        xi_cur = xi_all[mask]
        yj_cur = yj_all[mask]
        y_cur  = y_vals[mask]
        scat.set_offsets(np.c_[xi_cur, yj_cur])
        scat.set_array(y_cur)
        # Best point sous ce masque
        bi = int(np.argmax(y_cur)) if maximize else int(np.argmin(y_cur))
        star.set_offsets(np.c_[[xi_cur[bi]], [yj_cur[bi]]])
        title.set_text(f"{campaign} — Iteration {frame}  |  "
                       f"meilleur = {y_cur[bi]:.2f}  |  n = {mask.sum()}")
        return scat, star, title

    anim = animation.FuncAnimation(
        fig, update, frames=range(0, max_iter + 1),
        interval=800, blit=False, repeat=True
    )

    # Sauvegarde GIF
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(get_csv_path(cfg))),
        f"{campaign}_convergence.gif"
    )
    try:
        anim.save(out_path, writer="pillow", fps=1.3)
        print(f"  [OK]   Animation       : {os.path.basename(out_path)}")
    except Exception as e:
        print(f"  [WARN] Animation non generee : {e}")
    plt.close(fig)

def plot_trajectory_pca(df, cfg, gp_display, encoder, iteration, opts=None):
    """
    Dashboard secondaire v3.6 : projection PCA 2D de toutes les expériences.
    Fichier : {campaign}_dashboard2.png

    Contenu :
      - Panel gauche : PCA(parametres) coloré par Y, annoté par iteration
      - Panel droit  : histogramme Y par round (distribution de la qualité)
    Permet de voir si l'algo explore tout l'espace chimique ou reste coincé.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings("ignore")

    if opts is None: opts = {}
    dpi       = opts.get("dpi", 150)
    campaign  = cfg.get("campaign_name", "campaign")
    out_name  = cfg["outputs"][0]["name"]
    maximize  = cfg["outputs"][0].get("direction", "maximize") == "maximize"
    param_names = [p["name"] for p in cfg["parameters"]]

    df_done = df[df["statut"].astype(str).str.strip() == "fait"].copy()
    df_done[out_name] = pd.to_numeric(df_done[out_name], errors="coerce")
    df_done = df_done.dropna(subset=[out_name])
    if len(df_done) < 3:
        return  # pas assez de points pour PCA

    # Encoder en espace normalisé
    X_raw = df_done[param_names].values.astype(object)
    X_enc = encoder.encode(X_raw)
    y     = df_done[out_name].values.astype(float)

    # Numéro d'itération
    iter_col = pd.to_numeric(df_done["iteration_BO"], errors="coerce").fillna(0).astype(int).values

    # PCA 2D (ou moins si espace de dimension 1)
    if X_enc.shape[1] < 2:
        return  # pas utile en 1D
    n_comp = min(2, X_enc.shape[1])
    pca = PCA(n_components=n_comp)
    X_2d = pca.fit_transform(X_enc)
    var_ratio = pca.explained_variance_ratio_

    # ── Figure 16:9 2 panels ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(1, 2, figure=fig, left=0.06, right=0.96,
                           top=0.90, bottom=0.10, wspace=0.26,
                           width_ratios=[1.4, 1.0])

    # ── Panel 1 : PCA 2D ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FEFEFE")
    cmap = "viridis" if maximize else "viridis_r"
    sc = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=cmap, s=120,
                     edgecolors="white", linewidths=1.5, zorder=3)
    # Annotations iteration pour les derniers points
    for i in range(len(X_2d)):
        ax1.annotate(str(iter_col[i]), (X_2d[i, 0], X_2d[i, 1]),
                     fontsize=7, ha="center", va="center", color="white",
                     fontweight="bold", zorder=4)
    # Meilleur point : etoile dorée
    best_i = int(np.argmax(y)) if maximize else int(np.argmin(y))
    ax1.scatter(X_2d[best_i, 0], X_2d[best_i, 1], marker="*", s=400,
                c="#FFD700", edgecolors="#1A3A5C", linewidths=2, zorder=5,
                label=f"Meilleur ({out_name}={y[best_i]:.1f})")
    ax1.set_xlabel(f"PC1 ({var_ratio[0]*100:.0f}% variance)", fontsize=10)
    ax1.set_ylabel(f"PC2 ({var_ratio[1]*100:.0f}% variance)" if len(var_ratio) > 1 else "PC2",
                   fontsize=10)
    ax1.set_title(f"Trajectoire PCA des expériences — espace paramétrique",
                  fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.legend(loc="upper right", fontsize=8)
    cbar = fig.colorbar(sc, ax=ax1, shrink=0.85)
    cbar.set_label(out_name, fontsize=9)

    # ── Panel 2 : Distribution Y par round ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FEFEFE")
    max_iter = int(iter_col.max())
    for it in range(max_iter + 1):
        mask = iter_col == it
        if mask.sum() > 0:
            ax2.scatter(np.full(mask.sum(), it), y[mask], alpha=0.6, s=60,
                        color="#2176AE" if it > 0 else "#888888",
                        edgecolors="white", linewidths=1)
    # Ligne best cumulé
    sorted_idx = np.argsort(iter_col)
    y_sorted = y[sorted_idx]
    iter_sorted = iter_col[sorted_idx]
    best_cum = np.maximum.accumulate(y_sorted) if maximize else np.minimum.accumulate(y_sorted)
    ax2.plot(iter_sorted, best_cum, color="#008080", lw=2.2, label="Meilleur cumulé")
    ax2.set_xlabel("Round BO", fontsize=10)
    ax2.set_ylabel(out_name, fontsize=10)
    ax2.set_title("Distribution par round & convergence",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, alpha=0.3, ls="--")
    ax2.legend(loc="lower right", fontsize=8)

    # ── Titre global ──────────────────────────────────────────────────────────
    fig.suptitle(f"{campaign}  |  Iteration {iteration}  |  Dashboard secondaire",
                 fontsize=12, fontweight="bold", y=0.97, color="#1A3A5C")

    # Sauvegarde
    out_path = os.path.join(os.path.dirname(os.path.abspath(
               get_csv_path(cfg))), f"{campaign}_dashboard2.png")
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK]   Dashboard 2     : {os.path.basename(out_path)}")

def plot_dashboard(df, cfg, gp_display, encoder, iteration, opts=None):
    """
    Génère un dashboard 16:9 écrasé à chaque itération.
    Fichier : {campaign_name}_dashboard.png

    Layout 2x2 :
      ┌─────────────┬─────────────┐
      │ Convergence │ ARD         │
      ├─────────────┼─────────────┤
      │ Scatter exp.│ PDP         │
      └─────────────┴─────────────┘
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    if opts is None:
        opts = {}

    dpi       = opts.get("dpi", 150)
    campaign  = cfg.get("campaign_name", "campaign")
    colors    = list(_COLORS_CB.values())
    out_name  = cfg["outputs"][0]["name"]
    maximize  = cfg["outputs"][0].get("direction", "maximize") == "maximize"
    param_names = [p["name"] for p in cfg["parameters"]]

    # Données
    df_done = df[df["statut"].str.strip() == "fait"].copy()
    for col in [out_name]:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=[out_name])
    if len(df_done) < 2:
        return

    df_sugg = df[df["statut"].str.strip() == "suggere"].copy()
    df_sugg_cur = df_sugg[pd.to_numeric(df_sugg.get("iteration_BO", pd.Series()),
                          errors="coerce") == iteration]

    fig = plt.figure(figsize=(16, 12), dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")
    # Layout 3x2 : 2x2 existant + ligne heatmaps full-width
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            left=0.07, right=0.97, top=0.95, bottom=0.06,
                            hspace=0.42, wspace=0.28,
                            height_ratios=[1, 1, 1.15])

    # ── Panel 1 : Convergence ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FEFEFE")

    df_done_sorted = df_done.copy()
    df_done_sorted["_iter"] = pd.to_numeric(
        df_done_sorted.get("iteration_BO", pd.Series(dtype=float)),
        errors="coerce").fillna(0)
    df_done_sorted = df_done_sorted.sort_values("_iter")

    y_vals    = df_done_sorted[out_name].values.astype(float)
    best_so_far = np.maximum.accumulate(y_vals) if maximize else np.minimum.accumulate(y_vals)

    ax1.scatter(range(len(y_vals)), y_vals, s=25, alpha=0.6,
                c=colors[0], zorder=3, label="Expériences")
    ax1.plot(range(len(best_so_far)), best_so_far,
             color=colors[2], lw=2.0, zorder=4, label="Meilleur cumulatif")
    ax1.fill_between(range(len(best_so_far)), y_vals.min(), best_so_far,
                     alpha=0.08, color=colors[2])

    # Annoter le record
    best_idx = np.argmax(best_so_far) if maximize else np.argmin(best_so_far)
    ax1.annotate(f"{best_so_far[best_idx]:.1f}%",
                 xy=(best_idx, best_so_far[best_idx]),
                 xytext=(5, 8), textcoords="offset points",
                 fontsize=8, color=colors[2], fontweight="bold")

    ax1.set_title(f"Convergence — {out_name}", fontsize=10, fontweight="bold", pad=6)
    ax1.set_xlabel("Expérience #", fontsize=8)
    ax1.set_ylabel(out_name, fontsize=8)
    ax1.legend(fontsize=7, framealpha=0.7)
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.tick_params(labelsize=7)

    # ── Panel 2 : ARD importance ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FEFEFE")

    try:
        ls = gp_display.kernel_.k2.length_scale
        if np.isscalar(ls): ls = np.full(len(param_names), ls)
        importance = 1.0 / (np.array(ls) + 1e-9)
        # Normaliser par param (pas par dim encodée)
        raw_params = cfg["parameters"]
        imp_per_param = []
        dim_cur = 0
        for p_i, p in enumerate(raw_params):
            nd  = encoder._encoded_dims[p_i]
            val = importance[dim_cur:dim_cur+nd].mean()
            imp_per_param.append((p["name"], val))
            dim_cur += nd
        imp_arr   = np.array([v for _, v in imp_per_param])
        imp_arr  /= imp_arr.sum()
        labels    = [n for n, _ in imp_per_param]
        sorted_i  = np.argsort(imp_arr)

        bar_colors = [colors[0], colors[1], colors[2], colors[3],
                      colors[4], colors[5]][:len(labels)]
        bars = ax2.barh([labels[i] for i in sorted_i],
                        [imp_arr[i] for i in sorted_i],
                        color=[bar_colors[j % len(bar_colors)] for j in range(len(labels))],
                        alpha=0.85, edgecolor="white", height=0.55)
        for bar, val in zip(bars, [imp_arr[i] for i in sorted_i]):
            ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.1%}", va="center", fontsize=8, fontweight="bold")

        ax2.set_title("Importance ARD des paramètres", fontsize=10,
                      fontweight="bold", pad=6)
        ax2.set_xlabel("Importance relative", fontsize=8)
        ax2.set_xlim(0, imp_arr.max() * 1.25)
        ax2.grid(True, alpha=0.3, axis="x", ls="--")
        ax2.tick_params(labelsize=8)
    except Exception:
        ax2.text(0.5, 0.5, "ARD non disponible (< 3 exp.)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=10)

    # ── Panel 3 : Scatter expériences ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#FEFEFE")

    # Choisir les 2 params continus les plus importants
    cont_params = [p["name"] for p in cfg["parameters"] if p["type"] == "continuous"]
    if len(cont_params) >= 2:
        try:
            imp_cont = {n: v for n, v in imp_per_param if n in cont_params}
            sorted_cont = sorted(imp_cont, key=lambda x: imp_cont[x], reverse=True)
            px, py = sorted_cont[0], sorted_cont[1]
        except Exception:
            px, py = cont_params[0], cont_params[1]

        x_vals = pd.to_numeric(df_done[px], errors="coerce").values
        y_real = pd.to_numeric(df_done[out_name], errors="coerce").values
        y_2d   = pd.to_numeric(df_done[py], errors="coerce").values

        sc = ax3.scatter(x_vals, y_2d, c=y_real, cmap="RdYlGn",
                         s=45, alpha=0.85, zorder=3,
                         vmin=np.percentile(y_real, 5),
                         vmax=np.percentile(y_real, 95))
        plt.colorbar(sc, ax=ax3, label=out_name, shrink=0.85, pad=0.02)

        # Suggestions courantes
        if len(df_sugg_cur) > 0:
            sx = pd.to_numeric(df_sugg_cur[px], errors="coerce").values
            sy = pd.to_numeric(df_sugg_cur[py], errors="coerce").values
            ax3.scatter(sx, sy, marker="*", s=200, color=colors[3],
                       zorder=5, label=f"Suggestion iter {iteration}", edgecolors="black", lw=0.5)
            ax3.legend(fontsize=7, framealpha=0.7)

        # Meilleur point
        best_row = df_done.loc[df_done[out_name].idxmax() if maximize
                               else df_done[out_name].idxmin()]
        ax3.scatter(float(best_row[px]), float(best_row[py]),
                    marker="D", s=90, color=colors[2], zorder=6,
                    edgecolors="black", lw=0.8, label="Meilleur")

        ax3.set_xlabel(px, fontsize=8)
        ax3.set_ylabel(py, fontsize=8)
        ax3.set_title(f"Espace expérimental — iter {iteration}", fontsize=10,
                      fontweight="bold", pad=6)
        ax3.grid(True, alpha=0.3, ls="--")
        ax3.tick_params(labelsize=7)
    else:
        ax3.text(0.5, 0.5, "Pas assez de variables continues",
                 ha="center", va="center", transform=ax3.transAxes, fontsize=10)

    # ── Panel 4 : PDP ─────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#FEFEFE")

    # Choisir quoi afficher
    physchem_info = encoder.physchem_info()
    pdp_done = False

    if physchem_info and len(df_done) >= 5:
        # PDP des descripteurs physicochimiques — catégorielles encodées
        # Prendre la première catégorielle avec physchem (la plus importante si ARD dispo)
        try:
            cat_importances = {n: v for n, v in imp_per_param if n in physchem_info}
            pdp_param = max(cat_importances, key=lambda x: cat_importances[x])
        except Exception:
            pdp_param = next(iter(physchem_info))

        info = physchem_info[pdp_param]
        cats = info["categories"]
        mu_cats = []
        rng_pdp = np.random.default_rng(42)
        for cat in cats:
            try:
                _, mu, _ = compute_pdp(gp_display, encoder, cfg, df_done,
                                       pdp_param, rng=rng_pdp)
                # On passe la valeur de la catégorie via son encodage
                mu_cats.append(float(mu[0]) if hasattr(mu, '__len__') else float(mu))
            except Exception:
                mu_cats.append(0.0)

        # Recalculer proprement via compute_pdp pour catégorielles
        try:
            grid, mu_arr, _ = compute_pdp(gp_display, encoder, cfg, df_done,
                                           pdp_param, rng=np.random.default_rng(42))
            if isinstance(grid, list) and all(isinstance(g, str) for g in grid):
                # Catégorielle : bar chart
                bar_c = _colors_for_values(mu_arr)
                bars4 = ax4.bar(range(len(grid)), mu_arr, color=bar_c,
                                alpha=0.85, edgecolor="white")
                ax4.set_xticks(range(len(grid)))
                ax4.set_xticklabels(grid, rotation=35, ha="right", fontsize=7)

                # Valeurs observées par catégorie
                for j, cat in enumerate(grid):
                    cat_data = df_done[df_done[pdp_param] == cat][out_name]
                    if len(cat_data) > 0:
                        ax4.scatter([j]*len(cat_data), cat_data.values,
                                    c="black", s=20, zorder=5, alpha=0.6)

                ax4.set_title(f"PDP — {pdp_param}", fontsize=10,
                              fontweight="bold", pad=6)
                ax4.set_ylabel(f"{out_name} prédit (GP)", fontsize=8)
                ax4.grid(True, alpha=0.3, axis="y", ls="--")
                ax4.tick_params(labelsize=7)
                pdp_done = True
        except Exception:
            pass

    if not pdp_done and len(df_done) >= 5:
        # PDP de la variable continue la plus importante
        try:
            best_cont = sorted_cont[0] if cont_params else None
            if best_cont:
                grid, mu, sigma = compute_pdp(gp_display, encoder, cfg, df_done,
                                              best_cont, rng=np.random.default_rng(42))
                ax4.plot(grid, mu, color=colors[0], lw=2)
                ax4.fill_between(grid, mu-sigma, mu+sigma,
                                 alpha=0.2, color=colors[0])
                # Points observés
                obs_x = pd.to_numeric(df_done[best_cont], errors="coerce").values
                obs_y = pd.to_numeric(df_done[out_name], errors="coerce").values
                ax4.scatter(obs_x, obs_y, s=25, color=colors[1],
                           zorder=4, alpha=0.7)
                ax4.set_title(f"PDP — {best_cont}", fontsize=10,
                              fontweight="bold", pad=6)
                ax4.set_xlabel(best_cont, fontsize=8)
                ax4.set_ylabel(f"{out_name} prédit", fontsize=8)
                ax4.grid(True, alpha=0.3, ls="--")
                ax4.tick_params(labelsize=7)
                pdp_done = True
        except Exception:
            pass

    if not pdp_done:
        ax4.text(0.5, 0.5, "PDP disponible apres 5 expériences GP",
                 ha="center", va="center", transform=ax4.transAxes,
                 fontsize=10, color="#888888")

    # ── Panel 5 : Heatmaps GP (3 paires de variables) ─────────────────────────
    try:
        _plot_gp_heatmap_trio(fig, gs, gp_display, encoder, cfg, df_done,
                              df_sugg_cur, out_name, maximize)
    except Exception as e:
        # Fallback : afficher un message si les heatmaps ne peuvent etre generees
        ax5 = fig.add_subplot(gs[2, :])
        ax5.text(0.5, 0.5,
                 f"Heatmaps GP indisponibles : {str(e)[:80]}",
                 ha="center", va="center", transform=ax5.transAxes,
                 fontsize=10, color="#888888")
        ax5.axis("off")

    # ── Titre global ──────────────────────────────────────────────────────────
    best_val = df_done[out_name].max() if maximize else df_done[out_name].min()
    fig.suptitle(
        f"{campaign}  |  Iteration BO : {iteration}  |  "
        f"Meilleur : {best_val:.2f}%  |  {len(df_done)} expériences",
        fontsize=12, fontweight="bold", y=0.97, color="#1A3A5C"
    )

    # ── Sauvegarde (écrasement) ───────────────────────────────────────────────
    ts_path = _ts_path(campaign, "dashboard", opts)
    # On écrase toujours le même fichier dashboard sauf si timestamp_outputs
    if not opts.get("timestamp", False):
        ts_path = os.path.join(os.path.dirname(ts_path),
                               f"{campaign}_dashboard.png")
    fig.savefig(ts_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK]   Dashboard         : {os.path.basename(ts_path)}")


def plot_final_report(df, cfg, gp_display, encoder, iteration, opts=None):
    """
    Rapport final haute résolution — déclenché par convergence ou --final.
    Fichier : {campaign}_FINAL_report.png  (jamais écrasé si déjà existant)
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import warnings
    warnings.filterwarnings("ignore")

    if opts is None:
        opts = {}

    dpi      = max(opts.get("dpi", 150), 200)
    campaign = cfg.get("campaign_name", "campaign")
    colors   = list(_COLORS_CB.values())
    out_name = cfg["outputs"][0]["name"]
    maximize = cfg["outputs"][0].get("direction","maximize") == "maximize"
    param_names = [p["name"] for p in cfg["parameters"]]

    df_done = df[df["statut"].str.strip() == "fait"].copy()
    for col in [out_name]:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=[out_name])
    if len(df_done) < 3:
        return

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           left=0.06, right=0.97, top=0.92, bottom=0.07,
                           hspace=0.40, wspace=0.30)

    cont_params = [p["name"] for p in cfg["parameters"] if p["type"] == "continuous"]
    rng_final   = np.random.default_rng(0)

    # P1 : Convergence annotée complète
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#FEFEFE")
    df_s = df_done.copy()
    df_s["_iter"] = pd.to_numeric(df_s.get("iteration_BO", pd.Series(dtype=float)),
                                   errors="coerce").fillna(0)
    df_s = df_s.sort_values("_iter")
    y    = df_s[out_name].values.astype(float)
    best = np.maximum.accumulate(y) if maximize else np.minimum.accumulate(y)
    prev_best = best[0]
    for i, (yi, bi) in enumerate(zip(y, best)):
        if bi > prev_best or i == 0:
            ax1.annotate(f"{bi:.1f}", xy=(i, bi), xytext=(3, 5),
                        textcoords="offset points", fontsize=6.5,
                        color=colors[2], fontweight="bold")
            prev_best = bi
    ax1.scatter(range(len(y)), y, s=25, alpha=0.6, c=colors[0], zorder=3)
    ax1.plot(range(len(best)), best, color=colors[2], lw=2.2, zorder=4)
    ax1.fill_between(range(len(best)), y.min(), best, alpha=0.1, color=colors[2])
    ax1.set_title("Convergence complète", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Expérience #", fontsize=8)
    ax1.set_ylabel(out_name, fontsize=8)
    ax1.grid(True, alpha=0.3, ls="--")
    ax1.tick_params(labelsize=7)

    # P2 : ARD
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#FEFEFE")
    try:
        ls = gp_display.kernel_.k2.length_scale
        if np.isscalar(ls): ls = np.full(len(param_names), ls)
        importance = 1.0 / (np.array(ls[:len(param_names)]) + 1e-9)
        importance /= importance.sum()
        sorted_i = np.argsort(importance)
        bar_c = [colors[i % len(colors)] for i in range(len(param_names))]
        ax2.barh([param_names[i] for i in sorted_i],
                 [importance[i] for i in sorted_i],
                 color=[bar_c[j] for j in range(len(param_names))],
                 alpha=0.85, edgecolor="white", height=0.55)
        ax2.set_title("Importance ARD (finale)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Importance relative", fontsize=8)
        ax2.grid(True, alpha=0.3, axis="x", ls="--")
        ax2.tick_params(labelsize=8)
    except Exception:
        ax2.text(0.5, 0.5, "ARD non dispo", ha="center", va="center",
                 transform=ax2.transAxes)

    # P3 : PDP grille tous les paramètres continus
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#FEFEFE")
    if len(cont_params) >= 1 and len(df_done) >= 5:
        for k, pname in enumerate(cont_params[:4]):
            try:
                grid, mu, sigma = compute_pdp(gp_display, encoder, cfg, df_done,
                                              pname, rng=rng_final)
                c = colors[k % len(colors)]
                ax3.plot(grid / (grid.max() or 1), mu / (mu.max() or 1),
                        label=pname[:18], color=c, lw=1.8)
            except Exception:
                pass
        ax3.set_title("PDP normalisé — variables continues", fontsize=10, fontweight="bold")
        ax3.set_xlabel("Valeur normalisée [0,1]", fontsize=8)
        ax3.set_ylabel("Y prédit normalisé", fontsize=8)
        ax3.legend(fontsize=7, framealpha=0.7)
        ax3.grid(True, alpha=0.3, ls="--")
        ax3.tick_params(labelsize=7)

    # P4 : Landscape GP top 2 variables
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#FEFEFE")
    if len(cont_params) >= 2 and len(df_done) >= 5:
        try:
            px, py = cont_params[0], cont_params[1]
            idx_x  = param_names.index(px)
            idx_y  = param_names.index(py)
            lo_x, hi_x = float(cfg["parameters"][idx_x]["min"]), float(cfg["parameters"][idx_x]["max"])
            lo_y, hi_y = float(cfg["parameters"][idx_y]["min"]), float(cfg["parameters"][idx_y]["max"])
            gx = np.linspace(0, 1, 40)
            gy = np.linspace(0, 1, 40)
            GX, GY = np.meshgrid(gx, gy)
            X_grid = np.zeros((GX.size, encoder.n_params))
            # Fixer les autres dims à leur moyenne
            X_mean = encoder.encode(
                df_done[param_names].values.astype(object)).mean(axis=0)
            for i in range(encoder.n_params):
                X_grid[:, i] = X_mean[i]
            # Trouver les dim_cursors pour px et py
            dc = 0
            dim_px = dim_py = 0
            for ii, p in enumerate(cfg["parameters"]):
                nd = encoder._encoded_dims[ii]
                if p["name"] == px: dim_px = dc
                if p["name"] == py: dim_py = dc
                dc += nd
            X_grid[:, dim_px] = GX.ravel()
            X_grid[:, dim_py] = GY.ravel()
            Z = gp_display.predict(X_grid).reshape(GX.shape)
            # Inverser normalisation
            bo = cfg["bo_settings"]
            if bo.get("boxcox", False):
                Y_ref = df_done[[out_name]].values.astype(float)
                _, scale, _ = fit_boxcox(scalarise(Y_ref, cfg), bo)
                if scale is not None: Z = invert_boxcox(Z, scale)
            im = ax4.contourf(GX*(hi_x-lo_x)+lo_x, GY*(hi_y-lo_y)+lo_y, Z,
                             levels=20, cmap="RdYlGn", alpha=0.85)
            plt.colorbar(im, ax=ax4, shrink=0.85, label=out_name, pad=0.02)
            xs = pd.to_numeric(df_done[px], errors="coerce").values
            ys = pd.to_numeric(df_done[py], errors="coerce").values
            ax4.scatter(xs, ys, c="white", s=30, zorder=5,
                       edgecolors="black", lw=0.5, alpha=0.9)
            ax4.set_xlabel(px, fontsize=8); ax4.set_ylabel(py, fontsize=8)
            ax4.set_title(f"Landscape GP — {px} × {py}", fontsize=10, fontweight="bold")
            ax4.tick_params(labelsize=7)
        except Exception:
            ax4.text(0.5, 0.5, "Landscape indisponible", ha="center", va="center",
                    transform=ax4.transAxes)

    # P5-P6 : PDP catégorielles physicochimiques
    physchem_info = encoder.physchem_info()
    cat_axes = [fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])]
    cat_items = list(physchem_info.items())[:2]  # max 2

    for ax_i, ax_cat in enumerate(cat_axes):
        ax_cat.set_facecolor("#FEFEFE")
        if ax_i >= len(cat_items) or len(df_done) < 5:
            ax_cat.text(0.5, 0.5, "PDP catégorielle non disponible" if ax_i==0 else "Ajouter descriptors_file dans le YAML",
                       ha="center", va="center",
                       transform=ax_cat.transAxes, fontsize=9, color="#888")
            continue
        pname, info = cat_items[ax_i]
        try:
            grid, mu_arr, _ = compute_pdp(gp_display, encoder, cfg, df_done,
                                           pname, rng=rng_final)
            if isinstance(grid, (list, np.ndarray)) and len(grid) > 0:
                if isinstance(grid[0], str):
                    bar_c = _colors_for_values(mu_arr)
                    ax_cat.bar(range(len(grid)), mu_arr, color=bar_c,
                              alpha=0.85, edgecolor="white")
                    ax_cat.set_xticks(range(len(grid)))
                    ax_cat.set_xticklabels(grid, rotation=35, ha="right", fontsize=7)
                    for j, cat in enumerate(grid):
                        cat_data = df_done[df_done[pname] == cat][out_name]
                        if len(cat_data) > 0:
                            ax_cat.scatter([j]*len(cat_data), cat_data.values,
                                          c="black", s=18, zorder=5, alpha=0.7)
                ax_cat.set_title(f"PDP — {pname}", fontsize=10, fontweight="bold")
                ax_cat.set_ylabel(f"{out_name} prédit", fontsize=8)
                ax_cat.grid(True, alpha=0.3, axis="y", ls="--")
                ax_cat.tick_params(labelsize=7)
        except Exception:
            ax_cat.text(0.5, 0.5, f"PDP {pname} indisponible",
                       ha="center", va="center", transform=ax_cat.transAxes)

    best_val = df_done[out_name].max() if maximize else df_done[out_name].min()
    fig.suptitle(
        f"{campaign}  |  RAPPORT FINAL  |  Meilleur : {best_val:.2f}%  |  "
        f"{len(df_done)} expériences  |  Iter BO : {iteration}",
        fontsize=12, fontweight="bold", y=0.97, color="#1A3A5C"
    )

    path = _unique_filename(
        os.path.join(os.path.dirname(_ts_path(campaign, "x", opts)),
                     f"{campaign}_FINAL_report.png"))
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK]   Rapport final      : {os.path.basename(path)}")



def plot_round(df, cfg, gp_display, encoder, iteration, opts=None):
    """
    Figure 2b style, sauvegardee sous {campaign}_iter_{iteration:02d}_convergence.png.
    Ne remplace jamais un fichier existant (incremente le suffixe si necessaire).

    Elements :
    - Ligne teal : meilleur score cumule (running max)
    - Cercles gris : experiences initiales (mesure reelle)
    - Cercles bordeaux : experiences BO (mesure reelle)
    - Cercles or ouverts : potentiel GP (prediction au moment de la suggestion)
    - Etoile : meilleur point absolu
    """
    output_names = [o["name"] for o in cfg["outputs"]]
    param_names  = [p["name"] for p in cfg["parameters"]]

    df_done = df[df["statut"].str.strip() == "fait"].copy()
    for col in output_names:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=output_names)
    if len(df_done) < 2:
        return

    Y      = df_done[output_names].values.astype(float)
    scores = scalarise(Y, cfg)
    cummax = np.maximum.accumulate(scores)
    n_exp  = len(scores)
    x_vals = np.arange(1, n_exp + 1)

    # Prediction GP actuelle sur tous les points
    X_raw  = df_done[param_names].values.astype(object)
    X_norm = encoder.encode(X_raw)
    gp_pred_all = gp_display.predict(X_norm)

    # Prediction stockee dans le CSV (au moment de la suggestion) si disponible
    has_stored_pred = "prediction_BO" in df_done.columns
    stored_pred = pd.to_numeric(
        df_done["prediction_BO"] if has_stored_pred else pd.Series(dtype=float),
        errors="coerce"
    ).values if has_stored_pred else np.full(n_exp, np.nan)

    # Identifier initiales vs BO
    bo_mask = pd.to_numeric(
        df_done["iteration_BO"].replace("", np.nan), errors="coerce"
    ).notna().values

    # Potentiel a afficher :
    # - BO experiments -> prediction stockee si dispo, sinon prediction GP actuelle
    # - Initial -> prediction GP actuelle (retroactive)
    potential = np.where(
        bo_mask & ~np.isnan(stored_pred),
        stored_pred,
        gp_pred_all
    )

    best_idx  = np.argmax(scores)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Potentiel GP (cercles ouverts, derriere)
    ax.scatter(x_vals[~bo_mask], potential[~bo_mask],
               facecolors="none", edgecolors=_COLORS["initial"],
               s=55, linewidths=1.2, zorder=2, alpha=0.7)
    ax.scatter(x_vals[bo_mask], potential[bo_mask],
               facecolors="none", edgecolors=_COLORS["potential"],
               s=55, linewidths=1.4, zorder=2, alpha=0.9,
               label="Potentiel GP (prediction)")

    # Valeurs mesurees
    ax.scatter(x_vals[~bo_mask], scores[~bo_mask],
               c=_COLORS["initial"], s=50, zorder=4, label="Experience initiale")
    ax.scatter(x_vals[bo_mask], scores[bo_mask],
               c=_COLORS["bo_dot"], s=50, zorder=4, label="Experience BO (mesure)")

    # Meilleur point
    ax.scatter(x_vals[best_idx], scores[best_idx],
               c=_COLORS["best_star"], s=160, marker="*", zorder=6,
               label=f"Meilleur ({scores[best_idx]:.3f})")

    # Courbe meilleur cumule
    ax.plot(x_vals, cummax, color=_COLORS["best_line"], lw=2.5, zorder=3,
            label="Meilleur cumule")

    # Separateur initial / BO
    n_init = int((~bo_mask).sum())
    if n_init > 0 and n_init < n_exp:
        ax.axvline(n_init + 0.5, color="black", lw=1, linestyle="--", alpha=0.4,
                   label="Debut BO")

    n_out = len(output_names)
    ylabel = output_names[0] if n_out == 1 else "Score scalarise"
    ax.set_xlabel("Numero d'experience")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Convergence – {cfg['campaign_name']}  (iteration BO #{iteration})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = _unique_filename(f"{cfg['campaign_name']}_iter_{iteration:02d}_convergence.png")
    _dpi = opts.get('dpi', 150) if opts else 150
    plt.savefig(fname, dpi=_dpi)
    plt.close()
    print(f"  [GRAPHIQUE] Convergence     : {fname}")


# =============================================================================
# 11.  PAYSAGE GP – style Figure 2c  (carte contour 2D + surface 3D)
# =============================================================================

def plot_gp_landscape(gp_display, encoder, cfg, df_done, iteration, opts=None):
    """
    Sauvegarde {campaign}_iter_{iteration:02d}_landscape.png.
    Panneau gauche  : carte de contour 2D (style Figure 2c)
    Panneau droite  : surface 3D du meme paysage
    Axes            : 2 parametres continus les plus influents (ARD length-scales)
    """
    output_names = [o["name"] for o in cfg["outputs"]]
    param_names  = [p["name"] for p in cfg["parameters"]]

    top2 = get_influential_params(gp_display, cfg, n=2)
    if len(top2) < 2:
        print("  [INFO] Pas assez de parametres continus pour la carte GP.")
        return
    p0_idx, p1_idx = top2[0], top2[1]
    p0 = cfg["parameters"][p0_idx]
    p1 = cfg["parameters"][p1_idx]

    # Point optimal connu (pour fixer les autres parametres)
    Y      = df_done[output_names].values.astype(float)
    scores = scalarise(Y, cfg)
    best_i = int(np.argmax(scores))
    X_best_raw  = df_done[param_names].values.astype(object)[best_i:best_i+1]
    X_best_norm = encoder.encode(X_best_raw)[0]

    # Grille 2D en espace normalise
    n_grid = 50
    p0_grid = np.linspace(0, 1, n_grid)
    p1_grid = np.linspace(0, 1, n_grid)
    P0n, P1n = np.meshgrid(p0_grid, p1_grid)

    n_pts  = n_grid * n_grid
    X_grid = np.tile(X_best_norm, (n_pts, 1))
    X_grid[:, p0_idx] = P0n.ravel()
    X_grid[:, p1_idx] = P1n.ravel()

    mu_grid = gp_display.predict(X_grid).reshape(n_grid, n_grid)

    # Axes en unites physiques
    lo0, hi0 = float(p0["min"]), float(p0["max"])
    lo1, hi1 = float(p1["min"]), float(p1["max"])
    P0_phys  = P0n * (hi0 - lo0) + lo0
    P1_phys  = P1n * (hi1 - lo1) + lo1

    # Points experimentaux
    X_norm_all = encoder.encode(df_done[param_names].values.astype(object))
    x0_pts = X_norm_all[:, p0_idx] * (hi0 - lo0) + lo0
    x1_pts = X_norm_all[:, p1_idx] * (hi1 - lo1) + lo1

    fig = plt.figure(figsize=(14, 5.5))
    fig.suptitle(
        f"Paysage GP – {cfg['campaign_name']}  (iteration #{iteration})\n"
        f"Axes : {p0['name']}  vs  {p1['name']}",
        fontsize=11, y=1.01
    )

    # ── Panneau gauche : contour 2D ──────────────────────────────────────────
    ax1 = fig.add_subplot(121)
    cf  = ax1.contourf(P0_phys, P1_phys, mu_grid, levels=15, cmap="viridis")
    ax1.contour(P0_phys, P1_phys, mu_grid, levels=15,
                colors="k", alpha=0.12, linewidths=0.4)
    cb = plt.colorbar(cf, ax=ax1)
    n_out = len(output_names)
    cb_label = output_names[0] if n_out == 1 else "Score scalarise"
    cb.set_label(f"Prediction GP\n({cb_label})", fontsize=8)

    sc = ax1.scatter(x0_pts, x1_pts, c=scores, cmap="viridis",
                     vmin=mu_grid.min(), vmax=mu_grid.max(),
                     edgecolors="k", linewidths=0.7, s=60, zorder=5)
    # Meilleur point
    ax1.scatter(x0_pts[best_i], x1_pts[best_i], c="gold", s=200, marker="*",
                edgecolors="k", linewidths=0.7, zorder=6,
                label=f"Meilleur ({scores[best_i]:.3f})")
    ax1.set_xlabel(p0["name"])
    ax1.set_ylabel(p1["name"])
    ax1.set_title("Carte de contour 2D", fontsize=10)
    ax1.legend(fontsize=8)

    # ── Panneau droite : surface 3D ──────────────────────────────────────────
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot_surface(P0_phys, P1_phys, mu_grid,
                     cmap="viridis", alpha=0.82, linewidth=0, antialiased=True)
    ax2.scatter(x0_pts, x1_pts, scores,
                c=scores, cmap="viridis", edgecolors="k", linewidths=0.5,
                s=40, zorder=5, depthshade=True)
    ax2.scatter([x0_pts[best_i]], [x1_pts[best_i]], [scores[best_i]],
                c="gold", s=150, marker="*", edgecolors="k", zorder=6)
    ax2.set_xlabel(p0["name"], labelpad=6, fontsize=8)
    ax2.set_ylabel(p1["name"], labelpad=6, fontsize=8)
    ax2.set_zlabel(cb_label, labelpad=6, fontsize=8)
    ax2.set_title("Surface 3D", fontsize=10)

    plt.tight_layout()
    fname = _unique_filename(f"{cfg['campaign_name']}_iter_{iteration:02d}_landscape.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [GRAPHIQUE] Paysage GP      : {fname}")


# =============================================================================
# 12.  DETECTION DE CONVERGENCE
# =============================================================================

def check_convergence(df, cfg, sat_warnings=None):
    """
    Detecte la convergence : le meilleur score n'a pas progresse
    sur les derniers `patience` rounds BO.

    Litterature : 5 rounds sans amelioration est standard en chimie
    (Shields et al. JACS 2021 ; Griffiths & Hernandez-Lobato 2020).
    Valeur par defaut : 5, configurable via convergence_patience dans bo_settings.

    sat_warnings : liste retournee par detect_boundary_saturation().
    Si des saturations de borne sont actives, la convergence est bloquee :
    l optimum est possiblement hors de la plage, pas encore atteint.

    Retourne (is_converged: bool, n_rounds_stagnant: int).
    """
    patience     = int(cfg["bo_settings"].get("convergence_patience", 5))
    output_names = [o["name"] for o in cfg["outputs"]]

    df_done = df[df["statut"].str.strip() == "fait"].copy()
    for col in output_names:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=output_names)

    # Combien de rounds BO completes ?
    n_bo = pd.to_numeric(
        df_done["iteration_BO"].replace("", np.nan), errors="coerce"
    ).notna().sum()

    if n_bo < patience:
        return False, int(n_bo)

    Y      = df_done[output_names].values.astype(float)
    scores = scalarise(Y, cfg)
    cummax = np.maximum.accumulate(scores)

    if len(cummax) < patience + 1:
        return False, int(n_bo)

    best_now    = cummax[-1]
    best_before = cummax[-(patience + 1)]
    # Amelioration negligeable = moins de 0.1% de la plage observee
    plage       = max(abs(scores).max() - abs(scores).min(), 1e-6)
    amelioration = (best_now - best_before) / plage

    if amelioration < 1e-3:
        # Bloquer la convergence si au moins une saturation de borne est active :
        # le GP pousse vers une limite → l optimum est probablement hors plage.
        if sat_warnings:
            sat_params = [w[0] for w in sat_warnings]
            print(f"  [CONVERGENCE BLOQUEE] Saturation active sur : "
                  f"{', '.join(sat_params)}. Patience reset.")
            return False, 0
        return True, patience
    return False, 0


# =============================================================================
# 13.  GRAPHIQUES FINAUX (apres convergence)
# =============================================================================

def run_benchmark(cfg, df_done, encoder, n_iter=10, n_rep=5, rng=None):
    """
    Compare EI, UCB, PI, Thompson, Random par simulation sur donnees existantes.

    Methode (simulation BO retrospective) :
    1. Un GP oracle est entraine sur TOUTES les donnees disponibles.
    2. Pour chaque fonction d'acquisition et chaque repetition :
       a. On part des donnees initiales uniquement.
       b. A chaque iteration simulee : la fonction suggere un point,
          le GP oracle fourni la 'vraie' valeur (sans aller au labo).
       c. On suit le meilleur cumule.
    3. On moyenne sur les repetitions -> courbe moyenne +/- ecart-type.

    L'optimum vrai = maximum de la prediction oracle sur une grille dense.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    output_names = [o["name"] for o in cfg["outputs"]]
    param_names  = [p["name"] for p in cfg["parameters"]]
    bo           = cfg["bo_settings"]
    kappa        = float(bo.get("ucb_kappa", 2.0))
    xi_val       = float(bo.get("xi", 0.01))
    n_restarts   = min(int(bo.get("n_restarts", 50)), 8)  # reduit pour la vitesse

    # Donnees initiales (iteration_BO vide)
    df_init = df_done[
        df_done["iteration_BO"].replace("", np.nan).isna() |
        (df_done["iteration_BO"] == "")
    ].copy() if "iteration_BO" in df_done.columns else df_done.copy()

    if len(df_init) < 3:
        print("  [AVERTISSEMENT] Pas assez de donnees initiales pour le benchmark (min 3).")
        return None, None

    # GP oracle entraine sur toutes les donnees
    X_all        = encoder.encode(df_done[param_names].values.astype(object))
    Y_all        = df_done[output_names].values.astype(float)
    y_scalar_all = scalarise(Y_all, cfg)
    oracle_gp    = build_gp(encoder.n_params, normalize_y=True)
    oracle_gp.fit(X_all, y_scalar_all)

    # Optimum theorique (max oracle sur grille aleatoire dense)
    X_dense      = rng.uniform(0, 1, (3000, encoder.n_params))
    true_optimum = float(oracle_gp.predict(X_dense).max())

    # Donnees initiales encodees
    X_init      = encoder.encode(df_init[param_names].values.astype(object))
    Y_init_raw  = df_init[output_names].values.astype(float)
    y_init      = scalarise(Y_init_raw, cfg)

    acq_configs = [
        ("EI",       _COLORS["EI"]),
        ("UCB",      _COLORS["UCB"]),
        ("PI",       _COLORS["PI"]),
        ("Thompson", _COLORS["Thompson"]),
        ("Random",   _COLORS["Random"]),
    ]
    results = {}

    for acq_name, color in acq_configs:
        print(f"    {acq_name:10s} : ", end="", flush=True)
        curves = []

        for rep in range(n_rep):
            rng_r   = np.random.default_rng(rep * 137 + 7)
            X_train = X_init.copy()
            y_train = y_init.copy()
            best_sf = [float(y_train.max())]

            for it in range(n_iter):
                # Benchmark : sklearn GP pour vitesse (ms vs ~10s ADAM)
                # Précision suffisante pour comparer les fonctions d acquisition
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import Matern, ConstantKernel
                _k = ConstantKernel(1.0) * Matern(
                    length_scale=np.ones(encoder.n_params),
                    length_scale_bounds=(1e-2, 5.0), nu=2.5)
                gp_b = GaussianProcessRegressor(
                    kernel=_k, alpha=0.5,
                    normalize_y=True, n_restarts_optimizer=3)
                gp_b.fit(X_train, y_train)
                y_best = float(y_train.max())

                if acq_name == "EI":
                    af = lambda x: acq_ei(x, gp_b, y_best, xi_val)
                elif acq_name == "UCB":
                    af = lambda x: acq_ucb(x, gp_b, kappa)
                elif acq_name == "PI":
                    af = lambda x: acq_pi(x, gp_b, y_best, xi_val)
                elif acq_name == "THOMPSON":
                    g  = rng_r.uniform(0, 1, (2000, encoder.n_params))
                    s  = acq_thompson(g, gp_b, rng_r)
                    af = lambda x, _g=g, _s=s: np.array([
                        _s[np.argmin(np.sum((_g - pt)**2, axis=1))]
                        for pt in (x if x.ndim == 2 else x.reshape(1, -1))
                    ])
                else:  # Random
                    af = None

                if af is None or acq_name == "Random":
                    x_next = rng_r.uniform(0, 1, encoder.n_params)
                else:
                    x_next = maximise_acquisition(af, encoder.n_params, n_restarts, rng_r)

                y_next  = float(oracle_gp.predict(x_next.reshape(1, -1))[0])
                X_train = np.vstack([X_train, x_next.reshape(1, -1)])
                y_train = np.append(y_train, y_next)
                best_sf.append(float(y_train.max()))

            curves.append(best_sf)
            print(".", end="", flush=True)

        curves = np.array(curves)
        results[acq_name] = {
            "mean" : curves.mean(axis=0),
            "std"  : curves.std(axis=0),
            "color": color,
        }
        print()

    return results, true_optimum


def plot_benchmark(results, true_optimum, cfg, n_rep):
    """
    Trace la figure style Figure 3 : meilleur score cumule moyen vs iteration simulee.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals  = np.arange(len(list(results.values())[0]["mean"]))

    for acq_name, data in results.items():
        mu    = data["mean"]
        sigma = data["std"]
        color = data["color"]
        ax.plot(x_vals, mu, color=color, lw=2.2, label=acq_name)
        ax.fill_between(x_vals, mu - sigma, mu + sigma, color=color, alpha=0.15)

    if true_optimum is not None:
        ax.axhline(true_optimum, color="black", linestyle="--", lw=1.5,
                   label="Optimum GP oracle")

    n_out  = len(cfg["outputs"])
    ylabel = cfg["outputs"][0]["name"] if n_out == 1 else "Score scalarise (moyen)"
    ax.set_xlabel("Iteration simulee")
    ax.set_ylabel(f"Meilleur {ylabel} obtenu")
    ax.set_title(
        f"Benchmark fonctions d'acquisition – {cfg['campaign_name']}\n"
        f"({n_rep} repetitions, oracle GP)"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = f"{cfg['campaign_name']}_benchmark.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [GRAPHIQUE] Benchmark       : {fname}")


# =============================================================================
# UTILITAIRE : nom de fichier non-ecrasant
# =============================================================================

def _unique_filename(base):
    """
    Si base existe deja, ajoute _2, _3... jusqu'a trouver un nom libre.
    Garantit que les graphiques des rounds precedents ne sont jamais ecrases.
    """
    if not os.path.exists(base):
        return base
    root, ext = os.path.splitext(base)
    i = 2
    while os.path.exists(f"{root}_{i}{ext}"):
        i += 1
    return f"{root}_{i}{ext}"


# =============================================================================
# 15.  POINT D'ENTREE PRINCIPAL
# =============================================================================

def _filter_redundant_suggestions(suggestions, gp_preds, gp_pred_stds, encoder,
                                   cfg, min_dist=0.08, stagnation_active=False):
    """
    Filtre les suggestions redondantes d'un batch pour économiser les expériences.

    Logique :
      - Si stagnation détectée : retourne tout le batch (diversité = objectif)
      - Sinon : si 2 suggestions sont trop proches (distance normalisée < min_dist),
        on ne garde que la première et on recommande à l'utilisateur de ne faire
        qu'une expérience ce round.

    Retourne (suggestions_filtrees, gp_preds_filtrees, gp_stds_filtrees, message)
    """
    if stagnation_active or len(suggestions) <= 1:
        return suggestions, gp_preds, gp_pred_stds, None

    from scipy.spatial.distance import pdist, squareform
    param_names = [p["name"] for p in cfg["parameters"]]

    # Encoder les suggestions dans l'espace [0,1]
    X_sug_raw = np.array([[s[n] for n in param_names] for s in suggestions],
                          dtype=object)
    try:
        X_sug_enc = encoder.encode(X_sug_raw)
    except Exception:
        return suggestions, gp_preds, gp_pred_stds, None

    # Calculer la matrice des distances
    dist_matrix = squareform(pdist(X_sug_enc))
    np.fill_diagonal(dist_matrix, np.inf)

    # Greedy : garder les points assez distants
    keep_mask = [True] * len(suggestions)
    for i in range(1, len(suggestions)):
        for j in range(i):
            if keep_mask[j] and dist_matrix[i, j] < min_dist:
                keep_mask[i] = False
                break

    n_kept = sum(keep_mask)
    if n_kept == len(suggestions):
        return suggestions, gp_preds, gp_pred_stds, None

    # Filtrer
    filtered_sugg  = [s for s, k in zip(suggestions, keep_mask)  if k]
    filtered_preds = [p for p, k in zip(gp_preds,    keep_mask)  if k]
    filtered_stds  = ([s for s, k in zip(gp_pred_stds, keep_mask) if k]
                      if gp_pred_stds else [])

    n_dropped = len(suggestions) - n_kept
    msg = (f"  [INFO] Batch intelligent : {n_dropped} suggestion(s) redondante(s) "
           f"retiree(s).\n"
           f"         Les points proposes par le GP etaient trop proches "
           f"(distance < {min_dist:.2f}).\n"
           f"         Economise {n_dropped} experience(s) ce round.")
    return filtered_sugg, filtered_preds, filtered_stds, msg



def _handle_final_mode(cfg, csv_path, encoder):
    """
    Mode --final : genere le rapport final haute resolution et l'animation.
    Utilise le GP fitte sur toutes les donnees actuelles.
    """
    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    bo           = cfg["bo_settings"]

    if not os.path.exists(csv_path):
        print(f"\n  [ERREUR] CSV introuvable : {csv_path}")
        return
    df = pd.read_csv(csv_path)
    df_done = df[df["statut"].astype(str).str.strip() == "fait"].copy()
    for col in output_names:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=output_names)

    if len(df_done) < 4:
        print(f"\n  [ERREUR] Pas assez de donnees ({len(df_done)} < 4) pour --final")
        return

    # Fit GP
    X_train = encoder.encode(df_done[param_names].values.astype(object))
    Y_train = df_done[output_names].values.astype(float)
    y_raw = scalarise(Y_train, cfg)
    y_mm, bc_lam, _ = fit_boxcox(y_raw, bo, verbose=False)
    gp_display = build_gp(encoder.n_params, normalize_y=(bc_lam is None), cfg=cfg, encoder=encoder)
    gp_display.fit(X_train, y_mm)

    # Iter max
    it_max = int(pd.to_numeric(df_done["iteration_BO"], errors="coerce").fillna(0).max())

    print(f"\n  Generation rapport final (it {it_max})...")
    opts = {"dpi": 200, "timestamp": False}
    try:
        plot_final_report(df, cfg, gp_display, encoder, it_max, opts)
    except Exception as e:
        print(f"  [WARN] Rapport final non genere : {e}")
    try:
        plot_trajectory_pca(df, cfg, gp_display, encoder, it_max, opts)
    except Exception as e:
        print(f"  [WARN] Dashboard PCA non genere : {e}")
    try:
        plot_convergence_animation(df, cfg, gp_display, encoder, opts)
    except Exception as e:
        print(f"  [WARN] Animation non generee : {e}")
    print(f"\n  Tous les rapports finaux ont ete generes dans :")
    print(f"    {os.path.dirname(os.path.abspath(csv_path))}")

def _handle_whatif_mode(cfg, csv_path, encoder, whatif_points, neighbors_mode=False):
    """
    Mode --what-if : prédit la valeur GP à un point hypothétique.
    Utilise le GP fitté sur les données actuelles (sans déclencher le workflow BO).

    Si neighbors_mode=True (v3.7) : affiche aussi les sensibilités locales
    (gradient GP par variable) et 5 voisins prometteurs dans un rayon.
    """
    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]
    bo           = cfg["bo_settings"]

    # Charger le CSV
    if not os.path.exists(csv_path):
        print(f"\n  [ERREUR] CSV introuvable : {csv_path}")
        return
    df = pd.read_csv(csv_path)

    df_done = df[df["statut"].astype(str).str.strip() == "fait"].copy()
    for col in output_names:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=output_names)

    if len(df_done) < 3:
        print(f"\n  [ERREUR] Pas assez de donnees completes ({len(df_done)} < 3)")
        return

    # Valider les points demandés
    valid_params = {p["name"] for p in cfg["parameters"]}
    unknown = set(whatif_points.keys()) - valid_params
    if unknown:
        print(f"\n  [ERREUR] Parametres inconnus : {unknown}")
        print(f"  Parametres valides : {valid_params}")
        return
    missing = valid_params - set(whatif_points.keys())
    if missing:
        print(f"\n  [ERREUR] Parametres manquants : {missing}")
        return

    # Construire le point
    try:
        point = {}
        for p in cfg["parameters"]:
            raw_val = whatif_points[p["name"]]
            if p["type"] == "continuous":
                point[p["name"]] = float(raw_val)
            else:
                cats = p.get("categories", [])
                if raw_val not in cats:
                    print(f"\n  [ERREUR] '{raw_val}' n'est pas dans {cats}")
                    return
                point[p["name"]] = raw_val
    except ValueError as e:
        print(f"\n  [ERREUR] Conversion de valeur : {e}")
        return

    # Fitter le GP
    X_train = encoder.encode(df_done[param_names].values.astype(object))
    Y_train = df_done[output_names].values.astype(float)
    y_raw = scalarise(Y_train, cfg)
    y_mm, bc_lam, _ = fit_boxcox(y_raw, bo, verbose=True)

    gp = build_gp(encoder.n_params, normalize_y=(bc_lam is None), cfg=cfg, encoder=encoder)
    gp.fit(X_train, y_mm)

    # Prédire
    X_test = encoder.encode(np.array([[point[n] for n in param_names]], dtype=object))
    mu_bc, sigma_bc = gp.predict(X_test, return_std=True)
    mu_real = invert_boxcox(mu_bc, bc_lam)
    mu_val = float(np.asarray(mu_real).ravel()[0])

    # sigma en espace réel
    if bc_lam is not None:
        # MinMax : sigma_real = sigma * y_scale (actuel)
        y_max = float(df_done[output_names[0]].max())
        y_scale = max(y_max * 1.5, 100.0)
        sigma_real = float(sigma_bc[0]) * y_scale
    else:
        sigma_real = float(sigma_bc[0])

    # Affichage
    print(f"\n  {'='*55}")
    print(f"  WHAT-IF : prediction GP au point hypothetique")
    print(f"  {'='*55}")
    print(f"\n  Point evalue :")
    for n in param_names:
        val = point[n]
        if isinstance(val, float):
            print(f"    {n:<30} : {val:.4f}")
        else:
            print(f"    {n:<30} : {val}")

    print(f"\n  Prediction GP :")
    print(f"    {output_names[0]:<30} : {mu_val:.2f} ± {sigma_real:.2f}")

    # Interpretation
    rel_std = sigma_real / max(abs(mu_val), 1e-6)
    if rel_std < 0.05:
        conf = "GP tres confiant (zone bien exploree)"
    elif rel_std < 0.15:
        conf = "GP confiant"
    elif rel_std < 0.30:
        conf = "GP incertain (peu de donnees proches)"
    else:
        conf = "GP explore l inconnu (tres incertain)"
    print(f"    Confiance                      : {conf}")

    # Distance au point le plus proche dans les données
    X_all_enc = encoder.encode(df_done[param_names].values.astype(object))
    dists = np.sqrt(((X_all_enc - X_test[0])**2).sum(axis=1))
    closest_idx = int(np.argmin(dists))
    closest_row = df_done.iloc[closest_idx]
    closest_y = float(closest_row[output_names[0]])
    print(f"\n  Point reel le plus proche :")
    print(f"    Distance normalisee           : {float(dists[closest_idx]):.3f}")
    print(f"    Y mesure                      : {closest_y:.2f}")
    for n in param_names:
        v = closest_row[n]
        try:
            v = f"{float(v):.4f}"
        except (ValueError, TypeError):
            v = str(v)
        print(f"    {n:<30} : {v}")

    # ── Mode --neighbors (v3.7) : sensibilités + voisins prometteurs ──────
    if neighbors_mode:
        print(f"\n  {'='*55}")
        print(f"  ANALYSE VOISINAGE (--neighbors)")
        print(f"  {'='*55}")

        # 1. Sensibilités locales via différences finies (gradient GP)
        print(f"\n  Sensibilites locales (gradient GP pour Y) :")
        eps = 0.03  # perturbation en espace normalise [0,1]
        base_mu = mu_val
        sensitivities = []
        for pi, p in enumerate(cfg["parameters"]):
            if p["type"] != "continuous":
                # Catégoriel : gradient pas défini, skip
                continue
            # Position de p dans espace encodé (1 dim par continu)
            dim = sum(encoder._encoded_dims[k] for k in range(pi))
            X_plus  = X_test.copy()
            X_minus = X_test.copy()
            X_plus[0, dim]  = min(X_plus[0, dim] + eps, 1.0)
            X_minus[0, dim] = max(X_minus[0, dim] - eps, 0.0)
            mu_plus, _  = gp.predict(X_plus,  return_std=True)
            mu_minus, _ = gp.predict(X_minus, return_std=True)
            mu_plus_real  = float(invert_boxcox(mu_plus,  bc_lam)[0])
            mu_minus_real = float(invert_boxcox(mu_minus, bc_lam)[0])
            grad = (mu_plus_real - mu_minus_real) / (2 * eps)
            sensitivities.append((p["name"], grad))

        # Trier par magnitude décroissante
        sensitivities.sort(key=lambda x: -abs(x[1]))
        for name, grad in sensitivities:
            arrow = "↑" if grad > 0 else "↓"
            print(f"    {name:<30} : {arrow} {grad:+.3f} / unit (norm)")

        # 2. Voisins prometteurs : grille dense autour du point + tri par mu
        print(f"\n  Voisins prometteurs dans un rayon 0.15 (normalise) :")
        n_cand = 500
        rng_local = np.random.default_rng(42)
        X_cand = np.tile(X_test[0], (n_cand, 1))
        # Perturbation aleatoire sur dims continues uniquement
        for pi, p in enumerate(cfg["parameters"]):
            if p["type"] != "continuous":
                continue
            dim = sum(encoder._encoded_dims[k] for k in range(pi))
            noise = rng_local.normal(0, 0.08, n_cand)
            X_cand[:, dim] = np.clip(X_cand[:, dim] + noise, 0.0, 1.0)
        # Filtrer par distance 0.15
        d_cand = np.sqrt(((X_cand - X_test[0])**2).sum(axis=1))
        mask = (d_cand > 0.02) & (d_cand < 0.15)
        X_cand = X_cand[mask]
        if len(X_cand) == 0:
            print("    (aucun voisin dans le rayon)")
        else:
            mu_cand, _ = gp.predict(X_cand, return_std=True)
            mu_cand_real = invert_boxcox(mu_cand, bc_lam)
            # Top 5 par mu
            maximize = cfg["outputs"][0].get("direction", "maximize") == "maximize"
            order = np.argsort(-mu_cand_real) if maximize else np.argsort(mu_cand_real)
            top_idx = order[:5]
            for rank, idx in enumerate(top_idx, 1):
                xn = X_cand[idx].reshape(1, -1)
                decoded = encoder.decode(xn)[0]
                mu_v = float(mu_cand_real[idx])
                delta = mu_v - mu_val
                sign = "+" if delta >= 0 else ""
                print(f"    #{rank}  Y≈{mu_v:.2f} ({sign}{delta:.2f})  ", end="")
                parts = []
                for n in param_names:
                    v = decoded[n]
                    try:
                        parts.append(f"{n}={float(v):.2f}")
                    except (ValueError, TypeError):
                        parts.append(f"{n}={v}")
                print("  ".join(parts))


def _generate_coverage_heatmap(df_done, cfg, encoder, base_dir, campaign,
                                iteration, rng=None):
    """
    Genere une heatmap PNG de coverage de l'espace de recherche (v3.8).

    Sauvegarde : {campaign}_iter_{N}_coverage.png
    Retourne : chemin du PNG ou None si echec.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    cov_data = autopilot_compute_coverage_grid(df_done, cfg, encoder,
                                                n_grid_dim=40, rng=rng)
    if cov_data is None:
        return None

    try:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=110)

        # Heatmap : faible distance = vert (bien explore), forte = rouge (zone d'ombre)
        # On affiche la distance directement, avec colormap qui va de vert a rouge
        im = ax.contourf(cov_data['X1'], cov_data['X2'], cov_data['Z'],
                          levels=20, cmap='RdYlGn_r')

        # Points observes
        X_obs = cov_data['X_obs_norm_2d']
        ax.scatter(X_obs[:, 0], X_obs[:, 1], c='black', s=40,
                   edgecolors='white', linewidths=1.0, zorder=10,
                   label=f"{len(X_obs)} obs")

        ax.set_xlabel(f"{cov_data['param_names'][0]} (normalise)")
        ax.set_ylabel(f"{cov_data['param_names'][1]} (normalise)")
        ax.set_title(f"Coverage map — Zones d'ombre en rouge (iter {iteration})")
        ax.legend(loc='upper right', fontsize=9)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Distance min au point observe le plus proche\n"
                       "(faible = explore, fort = zone d'ombre)", fontsize=9)

        # Annoter coverage %
        coverage_pct = float((cov_data['Z'] < 0.20).mean())
        ax.text(0.02, 0.98, f"Coverage = {coverage_pct:.0%}",
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

        cov_path = os.path.join(base_dir,
                                f"{campaign}_iter_{iteration:02d}_coverage.png")
        fig.tight_layout()
        fig.savefig(cov_path, dpi=110, bbox_inches='tight')
        plt.close(fig)
        return cov_path
    except Exception as e:
        try:
            plt.close('all')
        except Exception:
            pass
        return None


def _write_run_report_html(cfg, df_done, df_updated, suggestions, gp_preds,
                            iteration, output_names, param_names, csv_path,
                            gp_display=None, encoder=None, gp_pred_stds=None,
                            gp_pred_stds_split=None, shap_contribs=None,
                            autopilot_decision=None):
    """
    Écrit un rapport HTML autonome (parallèle au .txt) pour chaque round BO.
    Fichier : {campaign}_iter_{N:02d}_report.html

    Avantages sur le .txt :
      - Embarque le dashboard.png en base64 (consultable même sans les fichiers annexes)
      - Mise en forme pro (pour partage avec l'encadrant)
      - Navigation avec ancres internes
    """
    import base64
    from datetime import datetime

    campaign = cfg.get("campaign_name", "campaign")
    base_dir = os.path.dirname(os.path.abspath(csv_path))
    html_path = os.path.join(base_dir, f"{campaign}_iter_{iteration:02d}_report.html")

    # Unique : ne pas écraser
    suffix = 1
    while os.path.exists(html_path):
        html_path = os.path.join(base_dir,
                     f"{campaign}_iter_{iteration:02d}_report_{suffix}.html")
        suffix += 1

    # Charger le dashboard si disponible
    dashboard_path = os.path.join(base_dir, f"{campaign}_dashboard.png")
    dashboard_b64 = ""
    if os.path.exists(dashboard_path):
        try:
            with open(dashboard_path, "rb") as fh:
                dashboard_b64 = base64.b64encode(fh.read()).decode("ascii")
        except Exception:
            pass

    # Construire les données
    n_done    = len(df_done)
    n_sugg    = len(suggestions)
    maximize  = cfg["outputs"][0].get("direction", "maximize") == "maximize"
    best_val  = (df_done[output_names[0]].max() if maximize
                 else df_done[output_names[0]].min())

    # Hyperparamètres GP
    hp_html = ""
    if gp_display is not None and encoder is not None:
        try:
            ls = gp_display.kernel_.k2.length_scale
            rows = []
            dim_cursor = 0
            for pi, p in enumerate(cfg["parameters"]):
                nd = encoder._encoded_dims[pi]
                ls_p = ls[dim_cursor:dim_cursor+nd]
                importance = 1.0 / (float(ls_p.mean()) + 1e-9)
                rows.append(f"<tr><td>{p['name']}</td>"
                            f"<td>{float(ls_p.mean()):.3f}</td>"
                            f"<td>{importance:.2f}</td></tr>")
                dim_cursor += nd
            noise = float(gp_display.alpha)
            hp_html = f"""
    <h2>Hyperparamètres GP</h2>
    <table>
      <thead><tr><th>Paramètre</th><th>Lengthscale ARD</th><th>Importance relative</th></tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p><strong>Bruit estimé</strong> (GaussianLikelihood) : {noise:.5f}</p>
"""
        except Exception:
            pass

    # Suggestions avec std
    sugg_html_rows = []
    for i, (sug, pred) in enumerate(zip(suggestions, gp_preds), 1):
        lines_sug = [f"<h3>Suggestion {i}/{n_sugg}</h3>", "<table>"]
        for p_cfg in cfg["parameters"]:
            pname = p_cfg["name"]
            val   = sug[pname]
            if p_cfg["type"] == "continuous":
                lines_sug.append(f"<tr><td>{pname}</td><td>{float(val):.4f}</td></tr>")
            else:
                lines_sug.append(f"<tr><td>{pname}</td><td>{val}</td></tr>")
        if gp_pred_stds and (i-1) < len(gp_pred_stds) and gp_pred_stds[i-1] is not None:
            std_v = gp_pred_stds[i-1]
            lines_sug.append(f"<tr class='pred'><td><strong>Prédiction GP</strong></td>"
                             f"<td><strong>{pred:.2f} ± {std_v:.2f}</strong></td></tr>")
            # Decomposition ep/al (v3.6)
            if gp_pred_stds_split and (i-1) < len(gp_pred_stds_split) and gp_pred_stds_split[i-1] is not None:
                ep, al = gp_pred_stds_split[i-1]
                lines_sug.append(f"<tr><td style='padding-left:1.5em'>↳ épistémique</td>"
                                 f"<td><em>{ep:.2f}</em> <span style='color:#888'>(réductible)</span></td></tr>")
                lines_sug.append(f"<tr><td style='padding-left:1.5em'>↳ aléatoire</td>"
                                 f"<td><em>{al:.2f}</em> <span style='color:#888'>(bruit intrinsèque)</span></td></tr>")
        else:
            lines_sug.append(f"<tr class='pred'><td><strong>Prédiction GP</strong></td>"
                             f"<td><strong>{pred:.2f}</strong></td></tr>")
        lines_sug.append("</table>")

        # Contributions SHAP (v3.6)
        if shap_contribs is not None and (i-1) < len(shap_contribs):
            sc = shap_contribs[i-1]
            baseline = sc.get("baseline", 0.0)
            contribs = sc.get("contributions", {})
            if contribs:
                sorted_c = sorted(contribs.items(), key=lambda x: -abs(x[1]))
                lines_sug.append(f'<h4 style="margin-top:1em;color:#2176AE">'
                                 f'Pourquoi ce point ? <span style="font-weight:normal;color:#888">'
                                 f'(baseline {baseline:.2f})</span></h4>')
                lines_sug.append("<table style='width:auto'>")
                for pname, val in sorted_c:
                    color = "#2E7D32" if val >= 0 else "#C62828"
                    sign = "+" if val >= 0 else ""
                    lines_sug.append(f"<tr><td>{pname}</td>"
                                     f"<td style='color:{color};font-weight:600'>"
                                     f"{sign}{val:.2f}</td></tr>")
                lines_sug.append("</table>")

        sugg_html_rows.append("\n".join(lines_sug))

    dashboard_img = ""
    if dashboard_b64:
        dashboard_img = (f'<img src="data:image/png;base64,{dashboard_b64}" '
                         f'alt="Dashboard" style="width:100%;max-width:1200px;'
                         f'border:1px solid #ddd;border-radius:4px"/>')

    # === v3.8 : Coverage heatmap + autopilot status ====
    autopilot_html = ""
    coverage_img = ""
    if autopilot_decision and autopilot_decision.get('phase'):
        # Generation heatmap PNG
        try:
            cov_path = _generate_coverage_heatmap(
                df_done, cfg, encoder, base_dir, campaign, iteration)
            if cov_path and os.path.exists(cov_path):
                with open(cov_path, "rb") as fh:
                    cov_b64 = base64.b64encode(fh.read()).decode("ascii")
                coverage_img = (f'<img src="data:image/png;base64,{cov_b64}" '
                                f'alt="Coverage map" style="width:100%;max-width:900px;'
                                f'border:1px solid #ddd;border-radius:4px"/>')
        except Exception:
            coverage_img = ""

        # Bloc HTML autopilot
        ap_phase    = autopilot_decision.get('phase', '?')
        ap_progress = autopilot_decision.get('progress')
        ap_coverage = autopilot_decision.get('coverage')
        ap_target   = autopilot_decision.get('target')
        ap_acq      = autopilot_decision.get('acquisition', '?')
        ap_reason   = autopilot_decision.get('reason', '')

        # Couleur badge phase
        phase_colors = {'EARLY': '#FF6F00', 'MID': '#2176AE', 'LATE': '#2E7D32'}
        phase_color  = phase_colors.get(ap_phase, '#888')

        rows_ap = []
        rows_ap.append(f"<tr><td><strong>Phase</strong></td>"
                       f"<td><span style='background:{phase_color};color:white;"
                       f"padding:4px 12px;border-radius:4px;font-weight:600'>"
                       f"{ap_phase}</span></td></tr>")
        if ap_target is not None:
            rows_ap.append(f"<tr><td>Target</td><td>{ap_target:.4f}</td></tr>")
        if ap_progress is not None:
            color_p = '#2E7D32' if ap_progress >= 0.9 else ('#2176AE' if ap_progress >= 0.5 else '#FF6F00')
            rows_ap.append(f"<tr><td>Progress (best/target)</td>"
                           f"<td style='color:{color_p};font-weight:600'>{ap_progress:.1%}</td></tr>")
        if ap_coverage is not None:
            color_c = '#2E7D32' if ap_coverage >= 0.5 else ('#FF6F00' if ap_coverage < 0.3 else '#2176AE')
            rows_ap.append(f"<tr><td>Coverage espace</td>"
                           f"<td style='color:{color_c};font-weight:600'>{ap_coverage:.1%}</td></tr>")
        rows_ap.append(f"<tr><td>Acquisition choisie</td>"
                       f"<td><code>{ap_acq}</code></td></tr>")

        autopilot_html = f"""
<h2>Auto-Pilot Orchestrateur (v3.8)</h2>
<div style="background:#f8f9fa;border-left:4px solid {phase_color};padding:1em 1.5em;border-radius:4px">
  <table style="margin:0">{''.join(rows_ap)}</table>
  <p style="margin-top:1em;color:#555;font-style:italic">
    <strong>Decision :</strong> {ap_reason}
  </p>
</div>

<h3>Coverage map — Zones d'ombre</h3>
<p>Les <span style="color:#C62828;font-weight:600">zones rouges</span> indiquent les regions
peu explorees (potentielles zones d'ombre). Les <span style="color:#2E7D32;font-weight:600">zones vertes</span>
sont bien couvertes.</p>
{coverage_img or '<p><em>Heatmap non disponible (necessite >= 2 parametres continus et points observes).</em></p>'}
"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>{campaign} — Iteration {iteration}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1100px; margin: 2em auto; padding: 0 2em; color: #222;
         line-height: 1.55; background: #fafafa; }}
  h1 {{ color: #1A3A5C; border-bottom: 3px solid #2176AE; padding-bottom: .4em; }}
  h2 {{ color: #1A3A5C; margin-top: 2em; border-left: 4px solid #2176AE;
        padding-left: .6em; }}
  h3 {{ color: #2176AE; margin-top: 1.4em; }}
  table {{ border-collapse: collapse; margin: 0.8em 0; background: white; }}
  th, td {{ padding: .5em 1em; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f4f8; color: #1A3A5C; font-weight: 600; }}
  .pred td {{ background: #fff8e8; }}
  .metrics {{ display: flex; gap: 1.5em; flex-wrap: wrap; margin: 1em 0; }}
  .metric-box {{ background: white; padding: 1em 1.5em; border-radius: 6px;
                 border-left: 4px solid #2176AE; flex: 1; min-width: 200px; }}
  .metric-label {{ color: #666; font-size: .85em; text-transform: uppercase;
                   letter-spacing: .05em; }}
  .metric-value {{ font-size: 1.6em; font-weight: 600; color: #1A3A5C;
                   margin-top: .2em; }}
  footer {{ color: #999; font-size: .85em; margin-top: 3em; border-top: 1px solid #eee;
            padding-top: 1em; }}
</style>
</head>
<body>
<h1>{campaign} — Rapport d'itération {iteration}</h1>
<p><em>Généré le {timestamp}</em></p>

<h2>Synthèse</h2>
<div class="metrics">
  <div class="metric-box"><div class="metric-label">Expériences complétées</div>
    <div class="metric-value">{n_done}</div></div>
  <div class="metric-box"><div class="metric-label">Meilleur {output_names[0]}</div>
    <div class="metric-value">{best_val:.2f}</div></div>
  <div class="metric-box"><div class="metric-label">Suggestions ce round</div>
    <div class="metric-value">{n_sugg}</div></div>
</div>

{autopilot_html}

<h2>Dashboard visuel</h2>
{dashboard_img or '<p><em>Dashboard non disponible pour cette itération.</em></p>'}

<h2>Suggestions</h2>
{''.join(sugg_html_rows)}

{hp_html}

<h2>Étapes suivantes</h2>
<ol>
  <li>Réaliser les expériences aux conditions suggérées</li>
  <li>Saisir les résultats dans le CSV : <code>{os.path.basename(csv_path)}</code></li>
  <li>Changer <code>statut</code> → <code>fait</code></li>
  <li>Relancer : <code>python bo_chemistry.py</code></li>
</ol>

<footer>
  Rapport généré par bo_chemistry.py v3.5<br>
  CSV : {os.path.basename(csv_path)}
</footer>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"  [OK]   Rapport HTML  : {os.path.basename(html_path)}")


def main():
    """
    Point d entree principal. Dispatch les 7 modes selon les flags CLI :

    Modes (exclusifs, le premier trouve gagne) :
      --init-doe [--doe-n N]  : genere un plan LHS N points dans le CSV
      --status                : resume lecture seule du CSV (meilleur Y, convergence)
      --benchmark             : benchmark des acquisitions sur donnees actuelles
      --final                 : rapport final haute resolution + animation GIF (v3.6)
      --what-if K=V K=V ...   : prediction GP hypothetique sans run BO
      --explore               : force VARMAX ce round (exploration pure)  [v3.7]
      (defaut)                : cycle BO normal (fit GP, suggere, ecrit CSV/rapports)

    Tous les modes sauf --init-doe et --status requierent gpytorch installe.
    """

    # ── Parsing des arguments ─────────────────────────────────────────────────
    args           = sys.argv[1:]
    flags          = [a for a in args if a.startswith("--")]
    positional     = [a for a in args if not a.startswith("--")]
    argument       = positional[0] if positional else None
    benchmark_mode = "--benchmark" in flags
    status_mode    = "--status"    in flags
    doe_mode       = "--init-doe"  in flags
    final_mode     = "--final"     in flags
    whatif_mode    = "--what-if"   in flags
    explore_mode   = "--explore"   in flags  # v3.7 : force VARMAX ce round
    # --doe-n N : nombre de points LHS (optionnel)
    # Note : chercher dans args (pas flags) car la valeur N ne commence pas par --
    doe_n = None
    for k, a in enumerate(args):
        if a == "--doe-n" and k + 1 < len(args):
            try: doe_n = int(args[k + 1])
            except ValueError: pass

    # --what-if : collecter les assignations K=V depuis les args positional
    whatif_points = {}
    if whatif_mode:
        for arg in args:
            if "=" in arg and not arg.startswith("--"):
                try:
                    k_, v_ = arg.split("=", 1)
                    whatif_points[k_.strip()] = v_.strip()
                except ValueError:
                    pass

    config_path = trouver_config(argument)
    cfg         = load_config(config_path)
    csv_path    = get_csv_path(cfg)
    encoder     = SpaceEncoder(cfg)

    bo           = cfg["bo_settings"]
    n_suggest    = int(bo.get("n_suggestions", 1))
    seed         = int(bo.get("seed", 42))
    np.random.seed(seed)          # seed global pour reproductibilité totale
    # v3.7c : n_init adaptatif selon dimension
    # Regle : 3 pts par dim continue + 1 pt par dim categorielle, plancher 5.
    # Override possible via bo_settings.min_initial_points dans le YAML.
    n_cont = sum(1 for p in cfg["parameters"] if p["type"] == "continuous")
    n_cat  = sum(1 for p in cfg["parameters"] if p["type"] != "continuous")
    _adaptive_default = max(5, 3 * n_cont + n_cat)
    if "min_initial_points" in bo:
        min_pts = int(bo["min_initial_points"])
    else:
        min_pts = _adaptive_default
        if min_pts > max(3, 2 * encoder.n_params):
            print(f"  [INFO] n_init adapte a {min_pts} (3 x {n_cont} continu + "
                  f"{n_cat} cat). Override : 'min_initial_points' dans le YAML.")
    rng          = np.random.default_rng(seed)
    param_names  = [p["name"] for p in cfg["parameters"]]
    output_names = [o["name"] for o in cfg["outputs"]]

    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  BO Chimie  |  Campagne : {cfg['campaign_name']}")
    print(f"  Config     |  {os.path.basename(config_path)}")
    print(f"  Donnees    |  {csv_path}")
    if benchmark_mode:
        print(f"  Mode       |  BENCHMARK")
    if status_mode:
        print(f"  Mode       |  STATUS (lecture seule)")
    if doe_mode:
        print(f"  Mode       |  INIT-DOE (Latin Hypercube Sampling)")
    if final_mode:
        print(f"  Mode       |  FINAL (rapport final haute resolution)")
    if whatif_mode:
        print(f"  Mode       |  WHAT-IF (prediction GP a un point hypothetique)")
    if explore_mode:
        print(f"  Mode       |  EXPLORATION (VARMAX forcee pour ce round)")
    # v3.7 : indiquer si contraintes actives
    _cstr_info = cfg.get("constraints", []) or []
    if _cstr_info:
        print(f"  Contraintes|  {len(_cstr_info)} actives (feasibility)")
    print(f"{sep}")

    # ── Mode INIT-DOE : plan LHS ─────────────────────────────────────────────
    if doe_mode:
        n_pts = doe_n if doe_n else int(bo.get("doe_n_points",
                    max(2 * sum(1 for p in cfg["parameters"] if p["type"]=="continuous") + 1, 6)))
        run_init_doe(cfg, csv_path, n_pts, rng)
        print(f"{sep}\n")
        return

    # ── Mode WHAT-IF : prediction GP a un point hypothetique ────────────────
    if whatif_mode:
        if not whatif_points:
            print(f"\n  [ERREUR] --what-if requiert des assignations K=V")
            print(f"  Exemple : python bo_chemistry.py config.yaml --what-if T=65 t=90 eq=2.0")
            print(f"{sep}\n")
            return
        neighbors_flag = "--neighbors" in flags
        _handle_whatif_mode(cfg, csv_path, encoder, whatif_points,
                            neighbors_mode=neighbors_flag)
        print(f"{sep}\n")
        return

    # ── Mode FINAL (v3.6) : rapport final + animation convergence ───────────
    if final_mode:
        _handle_final_mode(cfg, csv_path, encoder)
        print(f"{sep}\n")
        return

    # ── Premier lancement : creation du CSV ───────────────────────────────────
    if not os.path.exists(csv_path):
        create_csv(cfg, csv_path)
        return

    # ── Chargement du CSV ─────────────────────────────────────────────────────
    df = load_csv(csv_path)

    # Compatibilite arriere : ajoute prediction_BO si absent
    if "prediction_BO" not in df.columns:
        df["prediction_BO"] = ""

    missing = [c for c in get_all_columns(cfg) if c not in df.columns]
    if missing:
        print(f"\n  [ERREUR] Colonnes manquantes dans {csv_path} : {missing}")
        sys.exit(1)

    if df.empty:
        print(f"\n  [INFO] CSV vide. Remplissez vos experiences initiales")
        print(f"  (minimum : {min_pts} lignes, statut = 'fait'), puis relancez.\n")
        return

    # ── Filtrage des experiences completes ────────────────────────────────────
    df_pending = df[df["statut"].str.strip() == "suggere"]
    if not df_pending.empty:
        if benchmark_mode or status_mode:
            # Benchmark et status sont en lecture seule : on ignore les suggestions
            # en attente et on travaille uniquement sur les lignes "fait".
            print(f"\n  [INFO] {len(df_pending)} suggestion(s) en attente ignorees "
                  f"(mode {'BENCHMARK' if benchmark_mode else 'STATUS'} = lecture seule).")
        else:
            print(f"\n  [ATTENTE] {len(df_pending)} suggestion(s) sans resultats :")
            for _, row in df_pending.iterrows():
                conds = "  ".join(f"{p}={row[p]}" for p in param_names)
                print(f"    -> {conds}")
            print(f"\n  -> Remplissez les sorties dans {csv_path}, changez statut -> fait, relancez.\n")
            return

    df_done = df[df["statut"].str.strip() == "fait"].copy()
    for col in output_names:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=output_names)

    if df_done.empty:
        print("\n  [INFO] Aucune experience complete (statut = 'fait') trouvee.")
        return

    # ── Ajuster l'index de df_done pour coherence avec df ────────────────────
    df_done = df_done.reset_index(drop=True)

    # ── Resume ────────────────────────────────────────────────────────────────
    print(f"\n  Experiences completes : {len(df_done)}")
    for out_name in output_names:
        vals     = df_done[out_name].values.astype(float)
        best_val = vals.max()
        best_row = df_done.loc[df_done[out_name].idxmax()]
        conds    = "  ".join(f"{p}={best_row[p]}" for p in param_names)
        print(f"  Meilleur {out_name} : {best_val:.2f}  ({conds})")

    # ── Detection outliers ────────────────────────────────────────────────────
    if len(df_done) >= 5:
        try:
            enc_tmp  = SpaceEncoder(cfg)
            outliers = detect_outliers(df_done, cfg, enc_tmp, bo)
            _warn_outliers(outliers, df_done, param_names, output_names[0])
        except Exception:
            pass

    # ── Saturation de borne ───────────────────────────────────────────────────
    sat = detect_boundary_saturation(df, cfg, bo)
    _warn_boundary(sat, cfg)

    # ── Multi-modalite ────────────────────────────────────────────────────────
    if len(df_done) >= 10:
        try:
            enc_mm = SpaceEncoder(cfg)
            mm_detected, mm_msg = detect_multimodality(df_done, cfg, enc_mm, bo)
            if mm_detected:
                print(mm_msg)
        except Exception:
            pass

    # ── Mode STATUS : resume lecture seule ──────────────────────────────────
    if status_mode:
        print(f"\n  Experiences completes : {len(df_done)}")
        for out_name in output_names:
            out_cfg  = next(o for o in cfg["outputs"] if o["name"] == out_name)
            maximize = out_cfg.get("direction", "maximize") == "maximize"
            vals     = df_done[out_name].values.astype(float)
            bval     = vals.max() if maximize else vals.min()
            bidx     = int(vals.argmax() if maximize else vals.argmin())
            brow     = df_done.iloc[bidx]
            conds    = "  ".join(f"{p}={brow[p]}" for p in param_names)
            print(f"  Meilleur {out_name} : {bval:.3f}  ({conds})")
        bo_iters_s = pd.to_numeric(
            df["iteration_BO"].replace("", np.nan), errors="coerce"
        ).dropna()
        n_rounds_s   = int(bo_iters_s.max()) if not bo_iters_s.empty else 0
        n_pending_s  = len(df[df["statut"].str.strip() == "suggere"])
        conv_s, n_stag_s = check_convergence(df, cfg)
        patience_s   = int(bo.get("convergence_patience", 5))
        print(f"\n  Rounds BO effectues    : {n_rounds_s}")
        print(f"  Suggestions en attente : {n_pending_s}")
        if conv_s:
            print(f"  Convergence            : DETECTEE ({n_stag_s} rounds sans amelioration)")
        else:
            print(f"  Progression            : {n_stag_s}/{patience_s} rounds sans amelioration")
        print(f"\n  -> Mode STATUS : lecture seule, aucun fichier modifie.\n")
        print(f"{sep}\n")
        return

    # ── Mode BENCHMARK ────────────────────────────────────────────────────────
    if benchmark_mode:
        n_iter = int(bo.get("benchmark_n_iter", 10))
        n_rep  = int(bo.get("benchmark_n_rep",  5))
        print(f"\n  Benchmark : {n_iter} iterations simulees, {n_rep} repetitions")
        print(f"  Fonctions : EI, UCB, PI, Thompson, Random")
        print(f"  Duree estimee : {n_iter * n_rep // 5}-{n_iter * n_rep // 2} minutes\n")
        results, true_opt = run_benchmark(cfg, df_done, encoder, n_iter, n_rep, rng)
        if results:
            plot_benchmark(results, true_opt, cfg, n_rep)
        print(f"\n{sep}\n")
        return

    # ── Avertissement donnees insuffisantes ───────────────────────────────────
    if len(df_done) < min_pts:
        print(f"\n  [AVERTISSEMENT] Seulement {len(df_done)} points (recommande : >= {min_pts}).")
        print("  Continuer quand meme ? (o/n) : ", end="")
        if input().strip().lower() != "o":
            return

    # ── GP d'affichage (fitté sur donnees reelles uniquement) ─────────────────
    X_disp = encoder.encode(df_done[param_names].values.astype(object))
    Y_disp = df_done[output_names].values.astype(float)
    mode   = bo.get("multi_objective_mode", "weighted_sum")
    if mode == "ehvi" and len(output_names) > 1:
        # Pour l'affichage, on utilise l'objectif principal (index 0)
        # converti en sens "maximiser" pour coherence avec le reste du code
        signs  = np.array([_sign(o) for o in cfg["outputs"]])
        y_disp = (Y_disp * signs)[:, 0]   # objectif 0, maximise
    else:
        y_disp = scalarise(Y_disp, cfg)
    # Box-Cox transform (si active — stabilise la variance du bruit)
    y_disp_bc, bc_lam_d, bc_shift_d = fit_boxcox(y_disp, bo)
    bc_active  = bo.get("boxcox", False)
    gp_display = build_gp(encoder.n_params, normalize_y=not bc_active, cfg=cfg, encoder=encoder)
    gp_display.fit(X_disp, y_disp_bc)

    # ── Override acquisition si --explore (v3.7) ────────────────────────────
    _orig_acq = bo.get("acquisition", "EI")
    if explore_mode:
        bo["acquisition"] = "VARMAX"
        print(f"\n  [EXPLORE] Acquisition forcee: VARMAX (orig: {_orig_acq})")

    # ── Calcul des suggestions ────────────────────────────────────────────────
    mode = bo.get("multi_objective_mode", "weighted_sum")
    acq  = bo.get("acquisition", "EI").upper()

    # ── Restaurer acquisition originale apres --explore (v3.7) ─────────────
    if explore_mode:
        bo["acquisition"] = _orig_acq

    # ── Detection de stagnation ───────────────────────────────────────────────
    import copy
    stag_detected, stag_params, stag_msg = detect_variable_stagnation(df, cfg, bo)
    drift_detected, drift_params, drift_msg = detect_directional_drift(df, cfg, bo)

    # ── Fix v3.7b : switch EI → UCB apres 2 rounds sans progression ─────────
    # Evite que EI reste en mode exploration quand le GP a suffisamment de signal.
    # Seuil : 2 rounds = premiere moitie de patience (5) sans nouveau meilleur.
    _acq_cfg = bo.get("acquisition", "EI").upper()
    try:
        # fillna(0) : les points initiaux (sans iteration_BO) comptent comme round 0
        _ibo = pd.to_numeric(df_done["iteration_BO"], errors="coerce").fillna(0)
        _n_bo = int(_ibo.max()) + 1 if len(_ibo) > 0 else 0
    except (ValueError, TypeError):
        _n_bo = 0
    _no_impr = int(bo.get("_rounds_no_impr", 0))  # compteur injecte ci-apres
    # Calculer rounds_no_impr depuis les donnees
    try:
        import pandas as _pd
        _cm    = _pd.to_numeric(df_done[output_names[0]], errors="coerce")
        _iters = _pd.to_numeric(df_done["iteration_BO"], errors="coerce").fillna(0)
        _imax  = _iters.max()
        if _pd.isna(_imax) or _imax < 1:
            _no_impr = 0
        else:
            _bests_per_round = [float(_cm[_iters <= i].max()) for i in range(int(_imax)+1)]
            _best_0  = _bests_per_round[0] if _bests_per_round else 0.0
            _no_impr = sum(1 for j in range(1, len(_bests_per_round))
                           if _bests_per_round[j] <= _best_0 + 0.5)
    except Exception:
        _no_impr = 0

    # ── v3.8 : AUTO-PILOT ORCHESTRATEUR ────────────────────────────────────
    # Decide acquisition + xi/kappa selon (phase, target, coverage, no_impr).
    # Coordonne en amont les triggers reactifs : si declenche, prend priorite
    # sauf en cas de stagnation/drift confirmee (problemes actuels prioritaires).
    autopilot_decision = autopilot_decide(df_done, cfg, bo, encoder,
                                          _no_impr_existing=_no_impr)
    _autopilot_active = autopilot_decision.get('override', False)
    if _autopilot_active and bo.get("autopilot_verbose", True):
        print(autopilot_format_status(autopilot_decision))

    _auto_ucb = (_acq_cfg == "EI"
                 and _no_impr >= 2
                 and _n_bo >= 3)


    # Hierarchie des overrides :
    #   1. stagnation/drift detectee -> probleme reel actuel, prioritaire
    #   2. autopilot v3.8           -> decision orchestrateur (phase-aware)
    #   3. auto-UCB v3.7b           -> trigger reactif legacy (fallback)
    if stag_detected:
        print(stag_msg)
        acq       = "THOMPSON"
        n_suggest = 3
        cfg_stag  = copy.deepcopy(cfg)
        cfg_stag["bo_settings"]["acquisition"]   = "Thompson"
        cfg_stag["bo_settings"]["n_suggestions"] = 3
    elif drift_detected:
        print(drift_msg)
        acq       = "UCB"
        n_suggest = 1
        cfg_stag  = copy.deepcopy(cfg)
        cfg_stag["bo_settings"]["acquisition"] = "UCB"
        cfg_stag["bo_settings"]["ucb_kappa"]   = 0.2
        cfg_stag["bo_settings"]["n_suggestions"] = 1
    elif _autopilot_active:
        # Orchestrateur v3.8 prend la decision
        ap_acq = autopilot_decision['acquisition']
        cfg_stag = copy.deepcopy(cfg)
        cfg_stag["bo_settings"]["acquisition"] = ap_acq
        if autopilot_decision.get('xi') is not None:
            cfg_stag["bo_settings"]["ei_xi"] = autopilot_decision['xi']
        if autopilot_decision.get('kappa') is not None:
            cfg_stag["bo_settings"]["ucb_kappa"] = autopilot_decision['kappa']
        acq = f"{ap_acq} [autopilot {autopilot_decision['phase']}]"
    elif _auto_ucb:
        # AUTO-UCB legacy v3.7b (fallback si autopilot off)
        print(f"  [AUTO-UCB] EI → UCB (kappa=1.5) apres {_no_impr} rounds sans progression")
        acq      = "UCB [auto]"
        cfg_stag = copy.deepcopy(cfg)
        cfg_stag["bo_settings"]["acquisition"] = "UCB"
        cfg_stag["bo_settings"]["ucb_kappa"]   = 1.5
    else:
        cfg_stag = cfg

    if mode == "ehvi" and len(output_names) > 1:
        acq_display = "MC-EHVI (grille 2000 pts)"
    else:
        if stag_detected:
            acq_display = acq + " [override stagnation]"
        elif drift_detected:
            acq_display = acq + " [override derive]"
        elif _autopilot_active:
            acq_display = acq  # autopilot le contient deja
        elif _auto_ucb:
            acq_display = acq
        else:
            acq_display = acq
    print(f"\n  Mode         : {mode}")
    print(f"  Acquisition  : {acq_display}")
    print(f"  Suggestions  : {n_suggest}")
    if mode == "ehvi" and len(output_names) > 1:
        n_obj_count = len(output_names)
        est = "5-15s" if n_obj_count == 2 else "30-90s"
        print(f"\n  Calcul en cours... (EHVI : {est} par suggestion)\n")
    else:
        print(f"\n  Calcul en cours... (quelques secondes)\n")

    suggestions, gp_preds = suggest_experiments(cfg_stag, df_done, encoder, n_suggest, rng)

    # ── Calcul des incertitudes GP (std absolue) pour chaque suggestion ───────
    # std_total = std_epistemique + std_aleatoire (decomposition GP v3.6)
    gp_pred_stds = []
    gp_pred_stds_split = []  # liste de (ep, al) pour rapport detaille
    try:
        for sug in suggestions:
            sug_raw = np.array([[sug[p["name"]] for p in cfg["parameters"]]], dtype=object)
            sug_enc = encoder.encode(sug_raw)
            # Decomposition ep/al
            _, sigma_ep, sigma_al = gp_display.predict(sug_enc, return_std_split=True)
            ep, al = float(sigma_ep[0]), float(sigma_al[0])
            total = float(np.sqrt(ep**2 + al**2))
            if bc_active:
                y_max_cur = float(pd.to_numeric(df_done[output_names[0]], errors='coerce').max())
                y_scale_cur = max(y_max_cur * 1.5, 100.0)
                ep    *= y_scale_cur
                al    *= y_scale_cur
                total *= y_scale_cur
            gp_pred_stds.append(round(total, 3))
            gp_pred_stds_split.append((round(ep, 3), round(al, 3)))
    except Exception:
        gp_pred_stds = [None] * len(suggestions)
        gp_pred_stds_split = [None] * len(suggestions)

    # ── SHAP manuel : contributions par variable (v3.6) ──────────────────────
    try:
        shap_contribs = _compute_shap_contributions(
            gp_display, encoder, cfg, df_done, suggestions
        )
    except Exception:
        shap_contribs = None

    # ── Batch intelligent : filtrer les suggestions redondantes ──────────────
    if bo.get("smart_batch", True) and n_suggest > 1:
        suggestions, gp_preds, gp_pred_stds, batch_msg = _filter_redundant_suggestions(
            suggestions, gp_preds, gp_pred_stds, encoder, cfg,
            min_dist=float(bo.get("smart_batch_min_dist", 0.12)),
            stagnation_active=stag_detected
        )
        if batch_msg:
            print(batch_msg)
            n_suggest = len(suggestions)

    # ── Numero d'iteration ────────────────────────────────────────────────────
    bo_iters  = pd.to_numeric(df["iteration_BO"].replace("", np.nan), errors="coerce").dropna()
    next_iter = int(bo_iters.max() + 1) if not bo_iters.empty else 1

    # ── Affichage et ecriture dans le CSV ─────────────────────────────────────
    _explain_suggestion(cfg, gp_display, encoder, suggestions, gp_preds)
    print(f"  -- Suggestions (iteration BO #{next_iter}) " + "-" * 28)
    new_rows = []
    for i, (sug, pred) in enumerate(zip(suggestions, gp_preds)):
        print(f"\n  Suggestion {i+1}/{n_suggest} :")
        row = {}
        for p_cfg in cfg["parameters"]:
            p_name = p_cfg["name"]
            val    = sug[p_name]
            print(f"    {p_name} : {float(val):.4f}" if p_cfg["type"] == "continuous"
                  else f"    {p_name} : {val}")
            row[p_name] = val
        for out_name in output_names:
            row[out_name] = ""
        row["statut"]        = "suggere"
        row["iteration_BO"]  = next_iter
        row["prediction_BO"] = round(pred, 5)  # deja en espace reel (inverti dans suggest_experiments)
        row["prediction_std_BO"] = gp_pred_stds[i] if i < len(gp_pred_stds) else None
        new_rows.append(row)

    df_updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_csv(df_updated, csv_path, cfg=cfg)

    # ── Graphiques du round ───────────────────────────────────────────────────
    opts = _fig_opts(cfg)
    plot_dashboard(df_updated, cfg, gp_display, encoder, next_iter, opts)
    # Dashboard secondaire v3.6 : PCA trajectoire + distribution
    try:
        plot_trajectory_pca(df_updated, cfg, gp_display, encoder, next_iter, opts)
    except Exception as e:
        print(f"  [WARN] Dashboard PCA non genere : {e}")
    # Snapshots per-itération uniquement si timestamp_outputs: true
    if opts.get("timestamp", False):
        plot_round(df_updated, cfg, gp_display, encoder, next_iter, opts)
        plot_gp_landscape(gp_display, encoder, cfg, df_done, next_iter, opts)

    # ── Rapport texte resume ─────────────────────────────────────────────────
    _write_run_report(
        cfg, df_done, df_updated, suggestions, gp_preds,
        next_iter, output_names, param_names, csv_path,
        gp_display=gp_display, encoder=encoder,
        gp_pred_stds=gp_pred_stds,
        gp_pred_stds_split=gp_pred_stds_split,
        shap_contribs=shap_contribs
    )
    # Rapport HTML autonome en parallele (plus riche pour archivage/partage)
    try:
        _write_run_report_html(
            cfg, df_done, df_updated, suggestions, gp_preds,
            next_iter, output_names, param_names, csv_path,
            gp_display=gp_display, encoder=encoder,
            gp_pred_stds=gp_pred_stds,
            gp_pred_stds_split=gp_pred_stds_split,
            shap_contribs=shap_contribs,
            autopilot_decision=autopilot_decision if _autopilot_active else None,
        )
    except Exception as e:
        print(f"  [WARN] Rapport HTML non genere : {e}")

    # ── Detection de convergence ──────────────────────────────────────────────
    converged, n_stagnant = check_convergence(df_updated, cfg, sat_warnings=sat)
    patience = int(bo.get("convergence_patience", 5))
    if converged:
        print(f"\n  [CONVERGENCE] Aucune amelioration depuis {n_stagnant} rounds BO.")
        plot_final_report(df_updated, cfg, gp_display, encoder, next_iter, opts)
        # Animation convergence (v3.6) : sur convergence automatique
        try:
            plot_convergence_animation(df_updated, cfg, gp_display, encoder, opts)
        except Exception as e:
            print(f"  [WARN] Animation non generee : {e}")
        print(f"\n  -> La campagne semble avoir converge.")
        print(f"     Si vous souhaitez continuer, augmentez 'convergence_patience'")
        print(f"     dans le YAML (valeur actuelle : {patience}).")
    else:
        print(f"\n  Rounds sans amelioration : {n_stagnant}/{patience}")

    print(f"\n  OK - Suggestions dans : {csv_path}")
    print(f"\n  -- Etapes suivantes " + "-" * 42)
    print(f"  1. Realisez les experiences aux conditions ci-dessus")
    print(f"  2. Remplissez les sorties dans {csv_path}")
    print(f"  3. Changez  statut  ->  fait")
    print(f"  4. Relancez :  python bo_chemistry.py")
    print(f"\n{sep}\n")




# =============================================================================
# 15c.  EXPLICATION DES SUGGESTIONS (macro BO)
# =============================================================================

def _explain_suggestion(cfg, gp_display, encoder, suggestions, gp_preds):
    """
    Explique pourquoi le GP a suggere ces conditions.
    Affiche l importance de chaque parametre (ARD length-scales)
    et le motif principal de chaque suggestion (exploration vs exploitation).
    """
    param_names = [p["name"] for p in cfg["parameters"]]
    cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]
    sep = "-" * 62

    print("\n  " + sep)
    print("  ANALYSE DU MODELE GP")
    print("  " + sep)

    # Importance via ARD length-scales
    try:
        ls_raw   = gp_display.kernel_.k2.length_scale
        imp_raw  = 1.0 / np.maximum(ls_raw, 1e-6)
        imp_norm = imp_raw / imp_raw.sum()
        print("\n  Importance des parametres (ARD length-scales) :")
        for i, p in enumerate(cont_params):
            if i >= len(imp_norm):
                break
            imp = float(imp_norm[i])
            bar = chr(0x2588) * max(1, int(imp * 28))
            tag = ("FORT   ***" if imp > 0.45 else
                   "MOYEN   **" if imp > 0.20 else "FAIBLE   *")
            print(f"    {p['name']:<28} {bar:<28} {imp:5.1%}  {tag}")
        best_p = cont_params[int(np.argmax(imp_norm[:len(cont_params)]))]["name"]
        print(f"\n  -> Parametre le plus influent : {best_p}")
    except Exception:
        print("  (length-scales non disponibles)")

    # Explication par suggestion
    print("\n  Pourquoi ces conditions ?")
    for i, (sug, pred) in enumerate(zip(suggestions, gp_preds), 1):
        x_raw  = np.array([[sug.get(p, 0.0) for p in param_names]], dtype=object)
        x_norm = encoder.encode(x_raw)
        try:
            mu, sigma = gp_display.predict(x_norm, return_std=True)
            mu_v, sig_v = float(mu[0]), float(sigma[0])
            ratio = sig_v / (abs(mu_v) + 1e-9)
            motif = ("exploration   (zone peu connue, sigma eleve)"
                     if ratio > 0.15 else
                     "exploitation  (zone prometteuse, prediction confiante)")
            print(f"\n  Suggestion {i}/{len(suggestions)} :")
            print(f"    Prediction GP (mu)    : {mu_v:.4f}")
            print(f"    Incertitude  (sigma)  : {sig_v:.4f}  (ratio={ratio:.2f})")
            print(f"    Motif dominant        : {motif}")
        except Exception:
            print(f"\n  Suggestion {i} : prediction non disponible")

    print("  " + sep)


# =============================================================================
# 15d.  DETECTION DE STAGNATION DE VARIABLE
# =============================================================================

def _get_recent_bo_window(df, patience):
    """
    Extrait les lignes BO des 'patience' derniers rounds.
    Helper partage entre detect_variable_stagnation et detect_directional_drift.
    Retourne (df_recent, last_iters) ou (None, None) si pas assez de rounds.
    """
    df_bo = df[pd.to_numeric(df["iteration_BO"], errors="coerce").notna()].copy()
    df_bo["iteration_BO"] = pd.to_numeric(df_bo["iteration_BO"], errors="coerce")
    if df_bo["iteration_BO"].nunique() < patience:
        return None, None
    last_iters = sorted(df_bo["iteration_BO"].unique())[-patience:]
    df_recent  = df_bo[df_bo["iteration_BO"].isin(last_iters)]
    return df_recent, last_iters


def detect_variable_stagnation(df, cfg, bo_settings):
    """
    Detecte si une ou plusieurs variables continues sont coincees sur la
    meme valeur depuis plusieurs rounds BO consecutifs.

    Principe :
      Pour chaque parametre continu, on calcule l'ecart-type normalise
      des valeurs suggerees sur les N derniers rounds BO.
      Si std < seuil pour un parametre → stagnation detectee.

    Parametres YAML :
      stagnation_detection    : true/false (defaut false)
      stagnation_patience     : nb de rounds consecutifs pour declarer
                                stagnation (defaut 3)
      stagnation_std_threshold: seuil ecart-type normalise [0,1]
                                0 = meme valeur exacte
                                0.05 = varie de moins de 5% de la plage
                                (defaut 0.05)

    Retourne :
      (detected: bool, stagnant_params: list[str], message: str)
    """
    if not bo_settings.get("stagnation_detection", False):
        return False, [], ""

    patience  = int(bo_settings.get("stagnation_patience",      3))
    threshold = float(bo_settings.get("stagnation_std_threshold", 0.05))

    param_names  = [p["name"] for p in cfg["parameters"]]
    cont_params  = [p for p in cfg["parameters"] if p["type"] == "continuous"]

    if len(cont_params) < 1:
        return False, [], ""

    df_recent, last_iters = _get_recent_bo_window(df, patience)
    if df_recent is None:
        return False, [], ""

    stagnant = []
    for p in cont_params:
        pname  = p["name"]
        pmin   = float(p["min"])
        pmax   = float(p["max"])
        plage  = pmax - pmin
        if plage < 1e-9:
            continue

        vals = pd.to_numeric(df_recent[pname], errors="coerce").dropna()
        if len(vals) < patience:
            continue

        std_norm = float(vals.std()) / plage  # normalise dans [0, 1]
        if std_norm < threshold:
            stagnant.append((pname, std_norm, float(vals.mean())))

    if not stagnant:
        return False, [], ""

    # Construire le message
    lines = [f"\n  {'='*62}"]
    lines.append(f"  [STAGNATION DETECTEE] sur {len(stagnant)} parametre(s)")
    lines.append(f"  {patience} derniers rounds  |  seuil std = {threshold:.1%}")
    lines.append(f"  {'-'*62}")
    for pname, std_norm, mean_val in stagnant:
        lines.append(f"    {pname:<30} std={std_norm:.3f}  moy={mean_val:.4f}")
    lines.append(f"  {'-'*62}")
    lines.append(f"  Action automatique :")
    lines.append(f"    Acquisition -> Thompson  (diversite naturelle par batch)")
    lines.append(f"    n_suggestions -> 3       (au moins 1 sortira du plateau)")
    lines.append(f"  Pour desactiver : stagnation_detection: false dans le YAML")
    lines.append(f"  {'='*62}")

    stagnant_names = [s[0] for s in stagnant]
    return True, stagnant_names, "\n".join(lines)



# =============================================================================
# 15e.  DETECTION DE DERIVE DIRECTIONNELLE
# =============================================================================
#
# Complement de la detection de stagnation.
# La stagnation detecte : une variable reste IMMOBILE (std faible).
# La derive detecte     : une variable DERIVE dans une direction
#                         (t augmente a chaque round, ou T diminue...)
#                         tout en continuant a progresser (pas de stagnation).
#
# Exemple observe :
#   iter8 : t=124  iter9 : t=128  iter10 : t=122
#   → le GP depasse l'optimum et continue de tirer vers t=130-140
#   → action : UCB kappa=0.2 (exploitation pure) pour ramener vers l'optimum
#
# Critere : sur les N derniers rounds BO,
#   - regression lineaire de la variable vs le numero d'iteration
#   - si |pente normalisee| > seuil ET score ne progresse plus → derive
#
# Parametres YAML :
#   drift_detection        : true/false (defaut false)
#   drift_patience         : nb de rounds (defaut 3)
#   drift_slope_threshold  : pente normalisee min pour declencher (defaut 0.10)
#
# Retourne :
#   (detected: bool, drifting_params: list[str], message: str)
# =============================================================================

def detect_directional_drift(df, cfg, bo_settings):
    """
    Detecte quand une variable derive de maniere monotone sans ameliorer le score.
    Declenchement : bascule sur UCB kappa=0.2 (exploitation pure).
    """
    if not bo_settings.get("drift_detection", False):
        return False, [], ""

    patience  = int(bo_settings.get("drift_patience",       3))
    slope_thr = float(bo_settings.get("drift_slope_threshold", 0.10))

    cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]
    output_names = [o["name"] for o in cfg["outputs"]]

    if len(cont_params) < 1:
        return False, [], ""

    df_recent, last_iters = _get_recent_bo_window(df, patience)
    if df_recent is None:
        return False, [], ""

    # Score moyen par iteration pour verifier si le score progresse
    df_done     = df[df["statut"].str.strip() == "fait"].copy()
    for col in output_names:
        df_done[col] = pd.to_numeric(df_done[col], errors="coerce")
    df_done = df_done.dropna(subset=output_names)
    scores_by_iter = {}
    for it in last_iters:
        df_it = df_done[pd.to_numeric(df_done["iteration_BO"], errors="coerce") == it]
        if len(df_it) > 0:
            vals_it = pd.to_numeric(df_it[output_names[0]], errors="coerce").dropna()
            if len(vals_it) > 0:
                out_cfg = cfg["outputs"][0]
                if out_cfg.get("direction","maximize") == "maximize":
                    scores_by_iter[it] = float(vals_it.max())
                else:
                    scores_by_iter[it] = float(vals_it.min())

    # Si le score progresse vraiment, pas de derive a corriger
    # Utilise directement la premiere sortie (maximise ou minimise)
    if len(scores_by_iter) >= 2:
        iters_sorted = sorted(scores_by_iter)
        s_first = scores_by_iter[iters_sorted[0]]
        s_last  = scores_by_iter[iters_sorted[-1]]
        out_cfg  = cfg["outputs"][0]
        maximize = out_cfg.get("direction", "maximize") == "maximize"
        if maximize:
            score_improving = s_last > s_first * 1.005
        else:
            score_improving = s_last < s_first * 0.995
        if score_improving:
            return False, [], ""

    # Calculer la pente de chaque variable sur les derniers rounds
    drifting = []
    for p in cont_params:
        pname = p["name"]
        pmin, pmax = float(p["min"]), float(p["max"])
        plage = pmax - pmin
        if plage < 1e-9:
            continue

        vals = pd.to_numeric(df_recent[pname], errors="coerce").dropna()
        iters_for_vals = pd.to_numeric(
            df_recent.loc[vals.index, "iteration_BO"], errors="coerce"
        ).dropna()

        if len(vals) < patience:
            continue

        # Regression lineaire iteration → valeur
        x = iters_for_vals.values
        y = vals.values
        if len(x) < 2:
            continue
        slope = np.polyfit(x, y, 1)[0]
        slope_norm = abs(slope) / plage  # normalise par la plage

        if slope_norm > slope_thr:
            direction = "augmente" if slope > 0 else "diminue"
            drifting.append((pname, slope_norm, direction, vals.mean()))

    if not drifting:
        return False, [], ""

    lines = ["\n  " + "=" * 62]
    lines.append(f"  [DERIVE DIRECTIONNELLE] sur {len(drifting)} parametre(s)")
    lines.append(f"  {patience} derniers rounds  |  seuil pente = {slope_thr:.0%}/plage")
    lines.append(f"  Score stagne malgre la derive -> depasse l'optimum")
    lines.append("  " + "-" * 62)
    for pname, slope_n, direction, mean_val in drifting:
        lines.append(f"    {pname:<30} {direction} ({slope_n:.3f}/round)")
    lines.append("  " + "-" * 62)
    lines.append("  Action automatique :")
    lines.append("    Acquisition -> UCB kappa=0.2  (exploitation pure de mu)")
    lines.append("    n_suggestions -> 1")
    lines.append("  Pour desactiver : drift_detection: false dans le YAML")
    lines.append("  " + "=" * 62)

    drifting_names = [d[0] for d in drifting]
    return True, drifting_names, "\n".join(lines)


# =============================================================================
# 15f.  DETECTION D'OUTLIERS GP
# =============================================================================
#
# Un outlier est une mesure experimentale qui s'ecarte fortement de la
# prediction du modele GP entraine sur TOUTES les autres donnees.
# Cause typique : faute de frappe, experience ralee, contamination.
# Un seul outlier peut biaiser le GP et corrompre toutes les suggestions.
#
# Methode Leave-One-Out :
#   Pour chaque mesure, on entraine un GP sans ce point, puis on predit
#   sa valeur. Si |valeur_reelle - prediction| > n_sigma * sigma_GP,
#   c'est un outlier potentiel.
#
# Parametres YAML :
#   outlier_detection : true/false (defaut false — couteux en temps)
#   outlier_n_sigma   : seuil en ecarts-types (defaut 3.0)
#
# =============================================================================

def detect_outliers(df_done, cfg, encoder, bo_settings):
    """
    Detection d'outliers par residus GP.
    Retourne une liste de (index, compound_id, valeur_reelle, prediction, zscore).
    """
    if not bo_settings.get("outlier_detection", False):
        return []

    n_sigma      = float(bo_settings.get("outlier_n_sigma", 3.0))
    output_names = [o["name"] for o in cfg["outputs"]]
    param_names  = [p["name"] for p in cfg["parameters"]]
    out_name     = output_names[0]   # premier objectif uniquement

    if len(df_done) < 5:
        return []   # pas assez de points pour LOO fiable

    X_all = encoder.encode(df_done[param_names].values.astype(object))
    Y_all = df_done[output_names].values.astype(float)
    y_all = scalarise(Y_all, cfg)

    outliers = []
    for i in range(len(df_done)):
        mask     = np.ones(len(df_done), dtype=bool); mask[i] = False
        X_train  = X_all[mask]; y_train = y_all[mask]
        X_test   = X_all[i:i+1]
        y_true   = float(y_all[i])

        gp_loo = build_gp(encoder.n_params, normalize_y=True)
        try:
            gp_loo.fit(X_train, y_train)
            mu, sigma = gp_loo.predict(X_test, return_std=True)
            mu_v, sig_v = float(mu[0]), float(sigma[0])
            if sig_v < 1e-9: continue
            z = abs(y_true - mu_v) / sig_v
            if z > n_sigma:
                outliers.append((i, float(df_done[out_name].iloc[i]), mu_v, z))
        except Exception:
            continue

    return outliers


def _warn_outliers(outliers, df_done, param_names, out_name):
    """Affiche les outliers detectes."""
    if not outliers:
        return
    sep62 = "=" * 62
    print(f"\n  {sep62}")
    print(f"  [OUTLIERS DETECTES] {len(outliers)} mesure(s) suspecte(s)")
    print(f"  Methode : residus GP Leave-One-Out  |  seuil : 3σ")
    print(f"  {'-'*62}")
    for idx, y_real, y_pred, z in outliers:
        row = df_done.iloc[idx]
        conds = "  ".join(f"{p}={row[p]:.4f}" if isinstance(row[p], float)
                          else f"{p}={row[p]}" for p in param_names)
        print(f"  Ligne {idx+1:3d} : {out_name}={y_real:.3f}  prediction={y_pred:.3f}"
              f"  z={z:.1f}σ")
        print(f"    Conditions : {conds}")
    print(f"  {'-'*62}")
    print(f"  -> Verifiez ces mesures. Si erreur, corrigez le CSV et relancez.")
    print(f"  -> Le GP peut etre biaise si ces valeurs sont incorrectes.")
    _sep62 = "=" * 62
    print(f"  {_sep62}")


# =============================================================================
# 15g.  DETECTION DE SATURATION DE BORNE
# =============================================================================
#
# Si le GP suggere systematiquement des valeurs proches d'une borne
# (min ou max) pendant plusieurs rounds, l'optimum est peut-etre
# en dehors de la plage definie. Le script avertit et suggere d'elargir.
#
# Critere : sur les N derniers rounds BO, une variable continue a
# toutes ses suggestions dans les 10% extremes de sa plage.
#
# =============================================================================

def detect_boundary_saturation(df, cfg, bo_settings):
    """
    Detecte si les suggestions se concentrent systematiquement
    pres d'une borne (min ou max). Retourne un message d'avertissement.
    """
    patience  = int(bo_settings.get("stagnation_patience", 3))
    edge_frac = 0.12   # 12% de la plage = zone "bord"

    cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]
    df_recent, last_iters = _get_recent_bo_window(df, patience)
    if df_recent is None:
        return []

    warnings_out = []
    for p in cont_params:
        pname = p["name"]
        pmin, pmax = float(p["min"]), float(p["max"])
        plage = pmax - pmin
        edge  = plage * edge_frac

        vals = pd.to_numeric(df_recent[pname], errors="coerce").dropna()
        if len(vals) < patience:
            continue

        near_min = (vals <= pmin + edge).all()
        near_max = (vals >= pmax - edge).all()

        if near_min:
            warnings_out.append((pname, "MINIMUM", pmin, vals.mean()))
        elif near_max:
            warnings_out.append((pname, "MAXIMUM", pmax, vals.mean()))

    return warnings_out


def _warn_boundary(sat_warnings, cfg):
    """Affiche les avertissements de saturation de borne."""
    if not sat_warnings:
        return
    sep62b = "=" * 62
    print(f"\n  {sep62b}")
    print(f"  [SATURATION DE BORNE] sur {len(sat_warnings)} parametre(s)")
    print(f"  Les suggestions sont systematiquement pres d'une borne.")
    print(f"  {'-'*62}")
    for pname, side, bound_val, mean_val in sat_warnings:
        print(f"  {pname:<30} -> borne {side} ({bound_val})")
        print(f"    Moyenne des suggestions : {mean_val:.4f}")
        if side == "MINIMUM":
            print(f"    -> Considerez de reduire 'min' dans le YAML.")
        else:
            print(f"    -> Considerez d'augmenter 'max' dans le YAML.")
    print(f"  {'-'*62}")
    print(f"  L'optimum est peut-etre en dehors de la plage definie.")
    _sep62 = "=" * 62
    print(f"  {_sep62}")



# =============================================================================
# 15g.  AUTO-PILOT ORCHESTRATEUR (v3.8)
# =============================================================================
#
# Pourquoi : les triggers reactifs existants (stagnation v2.5, drift v2.6,
# auto-UCB v3.7b, xi-tuning v3.7d) sont independants et purement reactifs.
# Aucun ne tient compte de la PHASE de la campagne (early vs late) ni
# de l'OBJECTIF utilisateur (target rendement).
#
# L'orchestrateur introduit une couche superieure qui :
#   1. Calcule la phase courante (EARLY/MID/LATE) selon avancement
#      vs target (option D : hybride distance-target + iter-relatif).
#   2. Calcule le coverage de l'espace (% zones explorees).
#   3. Choisit la strategie optimale en fonction de phase + coverage.
#   4. Coordonne les triggers reactifs : si l'orchestrateur a deja decide
#      d'une strategie agressive, on ne re-stack pas un override redondant.
#
# Difference avec auto-UCB v3.7b :
#   - v3.7b : "EI -> UCB apres 2 rounds sans progres" (1 trigger reactif)
#   - v3.8  : Choisit acquisition + xi/kappa basee sur (phase, target,
#             coverage, no_impr) coordonnes en amont.
#
# Parametres YAML (tous optionnels) :
#   autopilot              : true/false (defaut false)
#   autopilot_target       : float ou "auto" (defaut "auto")
#                            ex: 0.80 pour 80% rendement
#                            "auto" = 0.95 * best_observed initial * 1.20
#   autopilot_max_iter     : int ou null (defaut null)
#                            si renseigne, definit phase via iter-relatif
#                            sinon : phase basee sur target uniquement
#   autopilot_verbose      : bool (defaut true) — affiche decisions
#
# Phases (option D, hybride) :
#   EARLY : best/target < 0.5 OU iter < 30% max_iter
#           Action : UCB kappa=2.5 (exploration agressive)
#   MID   : 0.5 <= best/target < 0.9 ET 30% <= iter < 70%
#           Action : EI xi=0.05 (equilibre)
#   LATE  : best/target >= 0.9 OU iter >= 70%
#           Action : EI xi=0.005 (affinage)
#
# Note : la stagnation/drift reactives restent prioritaires si declenchees
#        (elles indiquent un probleme actuel a resoudre maintenant).
# =============================================================================

def autopilot_compute_target(df_done, cfg, bo_settings):
    """
    Calcule la target rendement pour l'orchestrateur.

    Si bo_settings["autopilot_target"] est numerique : renvoie tel quel.
    Sinon : auto-calibre = 1.20 * best_observed_initial.

    Retourne :
      target : float dans la meme echelle que les outputs (None si erreur).
    """
    target_cfg = bo_settings.get("autopilot_target", "auto")

    if isinstance(target_cfg, (int, float)):
        return float(target_cfg)

    # Auto-calibrage : 1.20 * best initial (points avant BO)
    output_names = [o["name"] for o in cfg["outputs"]]
    if not output_names:
        return None

    out_col = output_names[0]
    out_dir = cfg["outputs"][0].get("direction", "maximize")

    try:
        # Points initiaux (iteration_BO == 0 ou NaN)
        _ibo = pd.to_numeric(df_done["iteration_BO"], errors="coerce").fillna(0)
        df_init = df_done[_ibo == 0]
        if len(df_init) == 0:
            df_init = df_done  # fallback : tous les points

        vals_init = pd.to_numeric(df_init[out_col], errors="coerce").dropna()
        if len(vals_init) == 0:
            return None

        if out_dir == "maximize":
            best_init = float(vals_init.max())
            # Si best_init <= 0, utiliser le range
            if best_init <= 0:
                rng_init = float(vals_init.max() - vals_init.min()) + 1e-9
                return best_init + 1.20 * rng_init
            return best_init * 1.20
        else:
            best_init = float(vals_init.min())
            if best_init >= 0:
                rng_init = float(vals_init.max() - vals_init.min()) + 1e-9
                return best_init - 0.20 * rng_init
            return best_init * 0.80

    except Exception:
        return None


def autopilot_compute_progress(df_done, cfg, target):
    """
    Calcule l'avancement vers la target.

    Retourne :
      progress : float dans [0, 1+] = best_observed / target (ratio).
                 < 1 : pas encore atteint, > 1 : depasse.
    """
    if target is None:
        return None

    output_names = [o["name"] for o in cfg["outputs"]]
    out_col = output_names[0]
    out_dir = cfg["outputs"][0].get("direction", "maximize")

    try:
        vals = pd.to_numeric(df_done[out_col], errors="coerce").dropna()
        if len(vals) == 0:
            return None

        if out_dir == "maximize":
            best = float(vals.max())
            return best / target if abs(target) > 1e-9 else 0.0
        else:
            # Pour minimize : progress = 1 - (best / target_initial)
            # On suppose target < best_initial pour minimize
            best = float(vals.min())
            return target / best if abs(best) > 1e-9 else 0.0
    except Exception:
        return None


def autopilot_compute_coverage(df_done, cfg, encoder, n_grid=2000, rng=None):
    """
    Calcule le coverage de l'espace : % de zones explorees.

    Methode :
      - Genere une grille de n_grid points uniformes en [0, 1]^n_dims (espace
        normalise).
      - Pour chaque point, calcule la distance min aux points observes.
      - Coverage = fraction des points de grille avec distance < seuil.

    Seuil ADAPTATIF a la dimension (v3.9) :
      Un rayon fixe sous-estime gravement le coverage en haute dimension
      (malediction de la dimension : le volume d'une boule de rayon r decroit
      comme r^d). Pour qu'un point "couvre" une fraction de volume comparable
      quelle que soit d, on fixe le rayon tel que le volume couvert par point
      reste a peu pres constant : r = base^(1/sqrt(d)), borne dans [0.15, 0.45].
      Resultat : en 1-2D le seuil reste serre (~0.20), en haute dimension il
      s'elargit pour rester atteignable.

    Retourne :
      coverage : float dans [0, 1] = fraction de l'espace exploree.
    """
    from scipy.spatial.distance import cdist as _cdist

    if rng is None:
        rng = np.random.default_rng(42)

    try:
        param_names = [p["name"] for p in cfg["parameters"]]
        X_obs_raw = df_done[param_names].to_numpy(dtype=object)
        X_obs_norm = encoder.encode(X_obs_raw)

        if len(X_obs_norm) == 0:
            return 0.0

        n_dims = encoder.n_params
        X_grid = rng.uniform(0, 1, (n_grid, n_dims))

        D = _cdist(X_grid, X_obs_norm)
        d_min = D.min(axis=1)

        # Seuil adaptatif : 0.20 en basse dim, s'elargit avec d
        threshold = float(np.clip(0.20 * (n_dims ** 0.25), 0.15, 0.45))
        coverage = float((d_min < threshold).mean())
        return coverage
    except Exception:
        return None


def autopilot_compute_lengthscales(gp, encoder):
    """
    Extrait les lengthscales ARD du GP fitte (v3.9).

    Les lengthscales renseignent sur l'importance relative de chaque dimension :
    une lengthscale courte = la sortie varie vite selon cette dimension (a
    echantillonner finement) ; une lengthscale longue = dimension peu influente.

    Retourne un dict :
      'lengthscales'   : np.array des lengthscales par dimension (ou None)
      'relative_var'   : coefficient de variation des lengthscales (heterogeneite)
                         eleve = quelques dimensions dominent ; faible = uniforme
      'n_active'       : nb de dimensions "actives" (lengthscale < mediane*2)
    Retourne None si indisponible (GP non fourni ou pas de kernel ARD).
    """
    if gp is None:
        return None
    try:
        # Acces au modele GPyTorch sous-jacent
        model = getattr(gp, "_model", None)
        if model is None:
            return None
        # Chercher le kernel avec lengthscale (Matern ARD)
        ls = None
        covar = getattr(model, "covar_module", None)
        if covar is not None:
            base = getattr(covar, "base_kernel", covar)
            raw_ls = getattr(base, "lengthscale", None)
            if raw_ls is not None:
                ls = raw_ls.detach().cpu().numpy().flatten()
        if ls is None or len(ls) == 0:
            return None
        ls = np.asarray(ls, dtype=float)
        median = float(np.median(ls))
        cv = float(np.std(ls) / (np.mean(ls) + 1e-12))
        n_active = int((ls < median * 2.0).sum())
        return {
            "lengthscales": ls,
            "relative_var": cv,
            "n_active": n_active,
        }
    except Exception:
        return None


def autopilot_decide(df_done, cfg, bo_settings, encoder, _no_impr_existing=0,
                     gp=None):
    """
    Decide la strategie BO optimale pour ce round.

    v3.9 — Logique de phase a TROIS criteres (au lieu du seul progress) :
      Le passage en exploitation (LATE) ne se fait QUE si l'une des deux
      conditions est remplie :
        (A) bon resultat ET espace suffisamment explore
            (progress >= seuil ET coverage >= seuil_adaptatif(d))
        (B) stagnation confirmee ET assez de points
            (no_impr >= patience(d) ET n_done >= n_min(d))
      Sinon on reste en exploration (EARLY/MID). Rationale : un bon rendement
      ne justifie pas d'arreter d'explorer si l'espace est mal couvert ; c'est
      la stagnation (le GP n'apprend plus) qui signale qu'on peut exploiter.

      Les seuils scalent avec la dimension d (malediction de la dimension) :
        - coverage requis baisse quand d augmente (sinon inatteignable)
        - patience et n_min montent avec d (plus de dims = plus a explorer)

      Si un GP fitte est fourni, les lengthscales ARD affinent la decision
      (heterogeneite forte = quelques dims dominent = exploitation plus sure).

    Args:
        df_done : DataFrame des experiences faites
        cfg, bo_settings : configuration
        encoder : SpaceEncoder
        _no_impr_existing : compteur rounds sans progres
        gp : GPModel fitte (optionnel) — active l'analyse des lengthscales

    Returns:
        dict (voir cle 'reason' pour le detail de la decision)
    """
    result = {
        'phase': None, 'progress': None, 'coverage': None, 'target': None,
        'override': False, 'acquisition': None, 'xi': None, 'kappa': None,
        'reason': '', 'lengthscale_info': None,
    }

    if not bo_settings.get("autopilot", False):
        return result  # Desactive

    # 1. Calcul target / progress / coverage
    target = autopilot_compute_target(df_done, cfg, bo_settings)
    result['target'] = target
    progress = autopilot_compute_progress(df_done, cfg, target)
    result['progress'] = progress
    coverage = autopilot_compute_coverage(df_done, cfg, encoder)
    result['coverage'] = coverage

    # 1b. Lengthscales ARD (si GP fourni)
    ls_info = autopilot_compute_lengthscales(gp, encoder)
    result['lengthscale_info'] = ls_info

    # 2. Parametres scalant avec la dimension
    n_dims = encoder.n_params
    n_done = len(df_done)
    # coverage requis : 0.55 en 1D -> baisse en haute dim, plancher 0.25
    cov_required = float(np.clip(0.55 - 0.05 * (n_dims - 1), 0.25, 0.55))
    # patience stagnation : 2 + ceil(d/2), borne raisonnable
    patience = int(min(2 + (n_dims + 1) // 2, 8))
    # n_min avant d'autoriser LATE par stagnation : ~ 4*d
    n_min_late = max(8, 4 * n_dims)

    # 3. Determination de phase a TROIS criteres
    #    LATE (exploitation) seulement si :
    #      (A) progress eleve ET coverage suffisante, OU
    #      (B) stagnation confirmee ET assez de points
    progress_ok = (progress is not None and progress >= 0.90)
    coverage_ok = (coverage is not None and coverage >= cov_required)
    stagnating  = (_no_impr_existing >= patience and n_done >= n_min_late)
    can_exploit = (progress_ok and coverage_ok) or stagnating

    if can_exploit:
        phase = 'LATE'
        if stagnating and not (progress_ok and coverage_ok):
            late_reason = f"stagnation {_no_impr_existing} rounds (>= {patience})"
        else:
            late_reason = f"progress {progress:.0%} + coverage suffisante"
    else:
        early = (progress is None or progress < 0.5) and \
                (coverage is None or coverage < cov_required * 0.6)
        phase = 'EARLY' if early else 'MID'
        late_reason = ""

    result['phase'] = phase

    # 4. Strategie d'acquisition par phase
    # On reutilise progress/coverage/lengthscales pour affiner.
    cv_txt = f"{coverage:.0%}" if coverage is not None else "?"
    pr_txt = f"{progress:.0%}" if progress is not None else "?"
    # heterogeneite des lengthscales : si forte, quelques dims dominent
    ls_hetero = (ls_info is not None and ls_info.get("relative_var", 0) > 0.6)

    if phase == 'EARLY':
        # Exploration : espace mal couvert -> UCB agressif (kappa scale en dim)
        kappa_explore = float(np.clip(2.0 + 0.15 * n_dims, 2.0, 3.5))
        result['override'] = True
        result['acquisition'] = 'UCB'
        result['kappa'] = kappa_explore
        result['reason'] = (f"EARLY (progress {pr_txt}, coverage {cv_txt}) "
                            f"-> UCB kappa={kappa_explore:.1f} (exploration, d={n_dims})")

    elif phase == 'MID':
        if _no_impr_existing >= max(2, patience - 1):
            # Stagnation naissante -> relancer exploration
            result['override'] = True
            result['acquisition'] = 'UCB'
            result['kappa'] = 1.8
            result['reason'] = (f"MID + stagnation ({_no_impr_existing} rounds) "
                                f"-> UCB kappa=1.8 (relance exploration)")
        elif coverage is not None and coverage < cov_required:
            # Couverture encore insuffisante -> diversifier via Thompson
            result['override'] = True
            result['acquisition'] = 'EI-TS'
            result['reason'] = (f"MID + coverage {cv_txt} < requis "
                                f"{cov_required:.0%} -> EI-TS (diversification)")
        else:
            result['override'] = True
            result['acquisition'] = 'EI'
            result['xi'] = 0.03
            result['reason'] = f"MID (coverage {cv_txt} OK) -> EI xi=0.03 (equilibre)"

    else:  # LATE
        if coverage is not None and coverage < cov_required and not stagnating:
            # Bon resultat mais trous : combler avant d'exploiter a fond
            result['override'] = True
            result['acquisition'] = 'EI-TS'
            result['reason'] = (f"LATE ({late_reason}) mais coverage {cv_txt} "
                                f"faible -> EI-TS (combler zones d'ombre)")
        else:
            # Affinage : xi plus serre si lengthscales heterogenes (optimum net)
            xi_late = 0.003 if ls_hetero else 0.005
            result['override'] = True
            result['acquisition'] = 'EI'
            result['xi'] = xi_late
            ls_note = " (lengthscales heterogenes)" if ls_hetero else ""
            result['reason'] = (f"LATE ({late_reason}) -> EI xi={xi_late} "
                                f"(affinage strict){ls_note}")

    return result


def autopilot_format_status(decision):
    """Formate un message de statut autopilot pour le terminal."""
    if not decision.get('override'):
        return ""

    lines = [f"\n  {'='*62}"]
    lines.append(f"  [AUTO-PILOT] decision orchestrateur")
    lines.append(f"  {'-'*62}")

    # Stats
    phase = decision.get('phase', '?')
    progress = decision.get('progress')
    coverage = decision.get('coverage')
    target = decision.get('target')

    lines.append(f"    Phase        : {phase}")
    if target is not None:
        lines.append(f"    Target       : {target:.4f}")
    if progress is not None:
        lines.append(f"    Progress     : {progress:.1%} (best/target)")
    if coverage is not None:
        lines.append(f"    Coverage     : {coverage:.1%} (espace explore)")
    ls_info = decision.get('lengthscale_info')
    if ls_info is not None:
        lines.append(f"    Lengthscales : {ls_info['n_active']} dims actives, "
                     f"heterogeneite CV={ls_info['relative_var']:.2f}")

    lines.append(f"  {'-'*62}")
    lines.append(f"  Action :")
    lines.append(f"    {decision['reason']}")
    lines.append(f"  {'='*62}")

    return "\n".join(lines)


def autopilot_compute_coverage_grid(df_done, cfg, encoder, n_grid_dim=40,
                                     rng=None):
    """
    Calcule une grille 2D coverage pour visualisation HTML.

    Retourne :
      X1, X2 : meshgrid normalise [0, 1]
      Z      : array (n_grid_dim, n_grid_dim) avec valeurs distance min
               aux points observes (faible = bien explore)
      param_names : list[str] des 2 params choisis (les plus influents)
    """
    from scipy.spatial.distance import cdist as _cdist

    if rng is None:
        rng = np.random.default_rng(42)

    try:
        # Choix des 2 params : les 2 continus avec plus large plage
        # (heuristique simple ; en pratique les 2 plus influents seraient
        # via length-scales ARD mais ca demanderait gp en argument).
        cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]
        if len(cont_params) < 2:
            return None
        p1, p2 = cont_params[0], cont_params[1]

        param_names_all = [p["name"] for p in cfg["parameters"]]
        idx1 = param_names_all.index(p1["name"])
        idx2 = param_names_all.index(p2["name"])

        # Encoder points observes
        X_obs_raw = df_done[param_names_all].to_numpy(dtype=object)
        X_obs_norm = encoder.encode(X_obs_raw)
        if len(X_obs_norm) == 0:
            return None

        # Grille 2D
        x1 = np.linspace(0, 1, n_grid_dim)
        x2 = np.linspace(0, 1, n_grid_dim)
        X1, X2 = np.meshgrid(x1, x2)

        # Pour chaque point grille, on construit un X complet (autres params
        # = leur median observe).
        n_dims = encoder.n_params
        median_X = np.median(X_obs_norm, axis=0)

        X_full = np.tile(median_X, (n_grid_dim * n_grid_dim, 1))
        X_full[:, idx1] = X1.ravel()
        X_full[:, idx2] = X2.ravel()

        # Distance min aux observes
        D = _cdist(X_full, X_obs_norm).min(axis=1)
        Z = D.reshape(n_grid_dim, n_grid_dim)

        return {
            'X1': X1, 'X2': X2, 'Z': Z,
            'param_names': [p1["name"], p2["name"]],
            'X_obs_norm_2d': X_obs_norm[:, [idx1, idx2]],
        }
    except Exception:
        return None




# =============================================================================
# 15h.  DETECTION DE MULTI-MODALITE (CANDIDATS LOCAUX)
# =============================================================================
#
# Pourquoi : le script retourne normalement le MEILLEUR candidat de
# l'acquisition. Mais l'espace peut avoir plusieurs optima locaux
# (multi-modal) — certains chimiiquement interessants meme si inferieurs.
#
# Methode : conserver les TOP-K maxima de la fonction d'acquisition
# identifies lors de l'optimisation multi-start, en filtrant les doublons
# (distance normalisee > 0.15 entre candidats).
#
# Affichage : si plusieurs optima distincts sont detectes, le terminal
# les liste avec leurs valeurs GP. L'utilisateur peut choisir de tester
# un optimum alternatif en entrant manuellement ses conditions dans le CSV.
#
# Aucun parametre YAML requis — detection automatique silencieuse.
#
# =============================================================================

def find_modal_candidates(acq_func, encoder, n_restarts=20, n_top=3,
                          min_dist=0.15, rng=None):
    """
    Optimise la fonction d'acquisition depuis n_restarts points et retourne
    les TOP-K candidats distincts (distance normalisee > min_dist entre eux).

    Retourne : liste de (x_norm, acq_value) tries par valeur decroissante.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_params  = encoder.n_params
    all_candidates = []

    # Multi-start : essayer n_restarts points de depart aleatoires
    X_starts = rng.uniform(0, 1, (n_restarts, n_params))
    for x0 in X_starts:
        try:
            from scipy.optimize import minimize
            res = minimize(
                lambda x: -float(np.sum(acq_func(x.reshape(1, -1)))),
                x0,
                method="L-BFGS-B",
                bounds=[(0, 1)] * n_params,
                options={"maxiter": 50, "ftol": 1e-6}
            )
            if res.success or res.fun < 0:
                all_candidates.append((res.x.copy(), -res.fun))
        except Exception:
            # Fallback : evaluer sur grille grossiere
            acq_val = float(np.sum(acq_func(x0.reshape(1, -1))))
            all_candidates.append((x0, acq_val))

    if not all_candidates:
        return []

    # Trier par valeur decroissante
    all_candidates.sort(key=lambda c: c[1], reverse=True)

    # Filtrer les doublons (distance normalisee < min_dist)
    distinct = [all_candidates[0]]
    for x, val in all_candidates[1:]:
        dists = [np.linalg.norm(x - d[0]) for d in distinct]
        if min(dists) > min_dist:
            distinct.append((x, val))
        if len(distinct) >= n_top:
            break

    return distinct


def _report_modal_candidates(candidates, encoder, cfg, best_acq_val):
    """
    Affiche les candidats alternatifs si l'espace est multi-modal.
    Seulements affiche si plusieurs optima distincts sont detectes.
    """
    if len(candidates) < 2:
        return  # espace unimodal, rien a signaler

    param_names = [p["name"] for p in cfg["parameters"]]
    cont_params = [p for p in cfg["parameters"] if p["type"] == "continuous"]

    print(f"\n  {'─'*62}")
    print(f"  ESPACE MULTI-MODAL DETECTE — {len(candidates)} zones prometteuses")
    print(f"  (seule la meilleure est suggeree — les autres sont listees ici)")
    print(f"  {'─'*62}")

    for rank, (x_norm, val) in enumerate(candidates[1:], 2):
        # Decoder
        row = {}
        for j, p in enumerate(cont_params):
            if j < len(x_norm):
                lo, hi = float(p["min"]), float(p["max"])
                row[p["name"]] = round(lo + x_norm[j] * (hi - lo), 4)
        pct_of_best = 100 * val / best_acq_val if best_acq_val > 0 else 0
        print(f"  Optimum local #{rank} (acquisition = {val:.4f}, "
              f"{pct_of_best:.0f}% du meilleur) :")
        for pname, pval in row.items():
            print(f"    {pname:<30}: {pval}")
    print(f"  {'─'*62}")
    print(f"  Pour tester un optimum alternatif : ajoutez-le manuellement")
    print(f"  dans le CSV avec statut = 'fait' apres l'experience.")

# =============================================================================
# 15b.  RAPPORT TEXTE RESUME (genere apres chaque round BO)
# =============================================================================




def _compute_prior_mean(X_norm, cfg, encoder):
    """
    Calcule la moyenne a priori (prior mean) pour une grille de points.
    Utilise pour injecter des connaissances physico-chimiques dans le GP (v3.7).

    API YAML :
      prior_mean:
        type: "linear" | "threshold" | "none"
        params: ...

    Types supportes :
      "none"      : mu_prior = 0 (comportement par defaut, equivalent a pas de prior)
      "linear"    : mu_prior = intercept + sum(coeffs[name] * x_norm)
      "threshold" : fonction par morceaux sur une seule variable
                    mu = below si var < threshold_norm
                    mu = above si var >= threshold_norm

    Args :
      X_norm   : array (n, d) en espace encode [0,1]
      cfg      : config YAML complete
      encoder  : SpaceEncoder instance

    Retourne : array (n,) de moyennes a priori.
    """
    X_norm = np.atleast_2d(X_norm)
    n = len(X_norm)
    prior_cfg = cfg.get("prior_mean", {})
    if not prior_cfg or prior_cfg.get("type", "none") == "none":
        return np.zeros(n)

    ptype = prior_cfg.get("type", "none")
    params = prior_cfg.get("params", {})
    param_names = [p["name"] for p in cfg["parameters"]]

    if ptype == "linear":
        intercept = float(params.get("intercept", 0.0))
        coeffs    = params.get("coeffs", {})
        mu = np.full(n, intercept, dtype=float)
        for pname, coef in coeffs.items():
            if pname not in param_names:
                continue
            pi = param_names.index(pname)
            # Position dans espace encode (1re dim du parametre)
            dim = sum(encoder._encoded_dims[k] for k in range(pi))
            mu = mu + float(coef) * X_norm[:, dim]
        return mu

    if ptype == "threshold":
        var_name = params.get("variable")
        threshold_raw = float(params.get("threshold", 0.0))
        below = float(params.get("below", 0.0))
        above = float(params.get("above", 100.0))
        if var_name not in param_names:
            return np.zeros(n)
        pi = param_names.index(var_name)
        p_cfg = cfg["parameters"][pi]
        if p_cfg["type"] != "continuous":
            return np.zeros(n)
        # Convertir threshold en espace normalise
        lo, hi = float(p_cfg["min"]), float(p_cfg["max"])
        thr_norm = (threshold_raw - lo) / max(hi - lo, 1e-9)
        dim = sum(encoder._encoded_dims[k] for k in range(pi))
        mu = np.where(X_norm[:, dim] < thr_norm, below, above)
        return mu.astype(float)

    # Type inconnu
    return np.zeros(n)


def _apply_boundary_avoidance(acq_vals, X_norm, X_train_norm, y_train,
                               threshold_pct=0.33, radius=0.15, penalty=0.3):
    """
    Penalise les candidats proches de zones deja explorees avec mauvais rendement.
    Evite que le GP revienne sur des conditions connues pour etre mauvaises (v3.7b).

    threshold_pct : fraction du y_max en dessous de laquelle un point est "mauvais"
    radius        : rayon en espace normalise [0,1]
    penalty       : facteur multiplicatif sur acq_fn (0.3 = reduit de 70%)
    """
    if len(X_train_norm) == 0 or len(y_train) == 0:
        return acq_vals

    y_arr    = np.asarray(y_train, dtype=float)
    y_max    = y_arr.max()
    y_thresh = y_max * threshold_pct

    # Identifier les "mauvais" points
    bad_mask = y_arr < y_thresh
    if not bad_mask.any():
        return acq_vals

    X_bad = X_train_norm[bad_mask]

    # Pour chaque candidat, calculer distance min aux mauvais points
    from scipy.spatial.distance import cdist
    dists = cdist(X_norm, X_bad, metric="euclidean").min(axis=1)

    # Penaliser si trop proche
    close_mask = dists < radius
    result = acq_vals.copy().astype(float)
    result[close_mask] *= penalty
    return result

def check_feasibility(point_decoded, constraints):
    """
    Evalue les contraintes de faisabilite sur un point decode (dict param→val).

    Chaque contrainte est une chaine Python evaluee dans un scope restreint.
    Exemples YAML :
      constraints:
        - "solvent != 'H2O' or T < 100"
        - "eq > 1.0 or base == 'DBU'"
        - "not (T > 80 and t > 120)"

    Syntaxe : operateurs Python standard (and, or, not, ==, !=, <, >, <=, >=).
    Scope : les noms de parametres + constantes Python (True, False, None, math).

    Retourne : (feasible: bool, failed_constraint: str|None)
    """
    if not constraints:
        return True, None
    # Scope restreint : juste les parametres + fonctions math basiques
    import math
    safe_globals = {"__builtins__": {}, "math": math, "abs": abs,
                    "min": min, "max": max, "round": round,
                    "True": True, "False": False, "None": None}
    safe_locals = dict(point_decoded)
    for i, expr in enumerate(constraints):
        try:
            result = bool(eval(expr, safe_globals, safe_locals))
        except Exception as e:
            # Contrainte mal formee : on avertit et considere feasible pour ne pas bloquer
            print(f"  [WARN] Contrainte #{i+1} invalide : '{expr}' ({type(e).__name__})")
            continue
        if not result:
            return False, expr
    return True, None


# v3.7d : Constrained BO (cBO) — Gardner 2014 / Gelbart 2014
# ─────────────────────────────────────────────────────────────────────────────
def fit_constraint_gps(cfg, df_done, encoder):
    """
    Pour chaque contrainte 'soft' definie dans le YAML (cfg["soft_constraints"]),
    entraine un GP separe sur la sortie associee.

    Format YAML :
        soft_constraints:
          - name: "Stabilite thermique"
            output: "tar_formed"      # nom d'une colonne du CSV
            direction: "minimize"     # ou "maximize"
            threshold: 5              # contrainte respectee si <= 5

    Retourne : list[dict] avec keys: gp, direction, threshold, name, output
    """
    sc_list = cfg.get("soft_constraints", []) or []
    if not sc_list:
        return []

    param_names = [p["name"] for p in cfg["parameters"]]
    X_raw = df_done[param_names].values.astype(object)
    X_norm = encoder.encode(X_raw)

    constraint_models = []
    for sc in sc_list:
        out_col = sc.get("output")
        if out_col not in df_done.columns:
            continue
        y_c = pd.to_numeric(df_done[out_col], errors="coerce").values
        valid = ~np.isnan(y_c)
        if valid.sum() < 3:
            continue  # pas assez de donnees pour un GP
        gp_c = build_gp(encoder.n_params, normalize_y=True, cfg=cfg, encoder=encoder,
                        y_train=y_c[valid])
        gp_c.fit(X_norm[valid], y_c[valid])
        constraint_models.append({
            "gp":        gp_c,
            "direction": sc.get("direction", "maximize").lower(),
            "threshold": float(sc.get("threshold", 0.0)),
            "name":      sc.get("name", out_col),
            "output":    out_col,
        })
    return constraint_models


def constrained_acquisition(acq_fn, constraint_models):
    """
    Renvoie une nouvelle fonction d'acquisition multipliee par la probabilite
    que TOUTES les contraintes soient respectees, calculee depuis les GPs c.

    P(c respecte) = P(c <= threshold)  pour direction='minimize'
                   = P(c >= threshold) pour direction='maximize'

    EI_c(x) = EI(x) * prod_j P(c_j respecte | x)
    """
    if not constraint_models:
        return acq_fn

    from scipy.stats import norm as _norm

    def _constrained(X):
        Xa = X if X.ndim == 2 else X.reshape(1, -1)
        base_vals = acq_fn(Xa).ravel().astype(float)
        prob_feasible = np.ones_like(base_vals)
        for cm in constraint_models:
            mu, sigma = cm["gp"].predict(Xa, return_std=True)
            sigma = np.maximum(sigma, 1e-9)
            if cm["direction"] == "minimize":
                # P(c <= threshold) = Phi((threshold - mu) / sigma)
                p = _norm.cdf((cm["threshold"] - mu) / sigma)
            else:
                # P(c >= threshold) = 1 - Phi((threshold - mu) / sigma)
                p = 1.0 - _norm.cdf((cm["threshold"] - mu) / sigma)
            prob_feasible *= p
        return base_vals * prob_feasible

    return _constrained


def _compute_shap_contributions(gp_display, encoder, cfg, df_done, suggestions):
    """
    Calcule les contributions individuelles par variable pour chaque suggestion.
    Méthode : ablation simple (remplacer chaque variable par sa moyenne empirique).

    C'est une approximation de SHAP bon marché :
      contribution_i = pred(x) - pred(x avec x_i = baseline)

    Ne respecte pas l'additivité parfaite SHAP mais donne l'ordre de grandeur
    et le signe correct des influences.

    Retourne : list[dict {param_name: contribution}] (une entrée par suggestion)
               ou [] si pas assez de donnees pour calculer une baseline.
    """
    # Validation entree : besoin d'au moins 1 point pour baseline + GP fitte
    if len(df_done) == 0 or len(suggestions) == 0:
        return []
    if not hasattr(gp_display, "_model") or gp_display._model is None:
        # GP non fitté (ou stub sans predict utilisable)
        pass  # on continue, predict lèvera exception rattrapée plus bas

    param_names  = [p["name"] for p in cfg["parameters"]]

    # Baseline : moyenne (continus) / mode (catégoriels) des données observées
    X_done_raw = df_done[param_names].values.astype(object)
    baseline_raw = []
    for pi, p in enumerate(cfg["parameters"]):
        col = X_done_raw[:, pi]
        if p["type"] == "continuous":
            try:
                baseline_raw.append(float(np.nanmean(col.astype(float))))
            except Exception:
                baseline_raw.append((float(p["min"]) + float(p["max"])) / 2)
        else:
            from collections import Counter
            try:
                most_common = Counter([str(v) for v in col]).most_common(1)[0][0]
                baseline_raw.append(most_common)
            except Exception:
                baseline_raw.append(p.get("categories", ["?"])[0])
    baseline_raw = np.array(baseline_raw, dtype=object)

    results = []
    for sug in suggestions:
        # Prédiction au point complet
        sug_raw = np.array([[sug[n] for n in param_names]], dtype=object)
        sug_enc = encoder.encode(sug_raw)
        pred_full = float(gp_display.predict(sug_enc)[0])

        # Prédiction avec baseline (point nul)
        base_enc = encoder.encode(baseline_raw.reshape(1, -1))
        pred_base = float(gp_display.predict(base_enc)[0])

        # Ablation : pour chaque variable, remplacer par baseline
        contribs = {}
        for pi, pname in enumerate(param_names):
            ablated = sug_raw.copy()
            ablated[0, pi] = baseline_raw[pi]
            abl_enc = encoder.encode(ablated)
            pred_abl = float(gp_display.predict(abl_enc)[0])
            # contribution : effet de la variable = pred_full - pred_sans
            contribs[pname] = pred_full - pred_abl

        # Normaliser pour que la somme reflete (pred_full - pred_base) mieux
        total_effect = pred_full - pred_base
        sum_contribs = sum(contribs.values())
        if abs(sum_contribs) > 1e-6:
            scale = total_effect / sum_contribs
            contribs = {k: v * scale for k, v in contribs.items()}

        results.append({"baseline": pred_base, "contributions": contribs})
    return results

def _write_run_report(cfg, df_done, df_updated, suggestions, gp_preds,
                      iteration, output_names, param_names, csv_path,
                      gp_display=None, encoder=None, gp_pred_stds=None,
                      gp_pred_stds_split=None, shap_contribs=None):
    """
    Ecrit un rapport texte resume apres chaque round BO.
    Fichier nomme : {campaign}_iter_{N:02d}_report.txt
    Non-ecrasant  : incremente automatiquement si le fichier existe.

    Contenu :
      - Statistiques actuelles (meilleur par objectif)
      - Conditions suggerees avec prediction GP
      - Progression de la convergence
      - Etapes suivantes

    Utile pour le cahier de labo electronique et le suivi hebdomadaire.
    """
    import datetime

    campaign = cfg["campaign_name"]
    base     = f"{campaign}_iter_{iteration:02d}_report.txt"
    if os.path.exists(base):
        root, ext = os.path.splitext(base)
        i = 2
        while os.path.exists(f"{root}_{i}{ext}"):
            i += 1
        base = f"{root}_{i}{ext}"

    bo       = cfg["bo_settings"]
    patience = int(bo.get("convergence_patience", 5))
    conv, n_stag = check_convergence(df_updated, cfg)

    bo_iters = pd.to_numeric(
        df_updated["iteration_BO"].replace("", np.nan), errors="coerce"
    ).dropna()
    n_rounds = int(bo_iters.max()) if not bo_iters.empty else 0

    lines = []
    sep60 = "=" * 60

    lines += [
        sep60,
        f"  RAPPORT BO — {campaign}",
        f"  Iteration #{iteration}  |  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sep60,
        "",
        "STATISTIQUES ACTUELLES",
        "-" * 40,
        f"  Experiences dans le GP  : {len(df_done)}",
    ]

    for out_name in output_names:
        out_cfg  = next(o for o in cfg["outputs"] if o["name"] == out_name)
        maximize = out_cfg.get("direction", "maximize") == "maximize"
        vals     = df_done[out_name].values.astype(float)
        if maximize:
            bidx = int(vals.argmax()); bval = vals.max()
        else:
            bidx = int(vals.argmin()); bval = vals.min()
        brow  = df_done.iloc[bidx]
        conds = "  ".join(f"{p}={brow[p]}" for p in param_names)
        lines += [
            f"  Meilleur {out_name:<24}: {bval:.4f}",
            f"    Conditions            : {conds}",
        ]

    lines += [
        "",
        f"SUGGESTIONS (iteration #{iteration})",
        "-" * 40,
    ]
    for i, (sug, pred) in enumerate(zip(suggestions, gp_preds), 1):
        lines.append(f"  Suggestion {i}/{len(suggestions)} :")
        for p_cfg in cfg["parameters"]:
            pname = p_cfg["name"]
            val   = sug[pname]
            if p_cfg["type"] == "continuous":
                lines.append(f"    {pname:<30}: {float(val):.4f}")
            else:
                lines.append(f"    {pname:<30}: {val}")
        # Prediction avec incertitude GP (std absolue)
        if gp_pred_stds is not None and (i-1) < len(gp_pred_stds) and gp_pred_stds[i-1] is not None:
            std_val = gp_pred_stds[i-1]
            lines.append(f"    Prediction GP         : {pred:.4f} ± {std_val:.2f}")
            # Decomposition epistemique / aleatoire (v3.6)
            if gp_pred_stds_split is not None and (i-1) < len(gp_pred_stds_split) and gp_pred_stds_split[i-1] is not None:
                ep, al = gp_pred_stds_split[i-1]
                lines.append(f"      dont epistemique    : {ep:.2f} (reductible en mesurant ici)")
                lines.append(f"      dont aleatoire      : {al:.2f} (bruit intrinseque)")
            # Interpretation textuelle
            if std_val < pred * 0.05:
                conf = "GP tres confiant"
            elif std_val < pred * 0.15:
                conf = "GP confiant"
            elif std_val < pred * 0.30:
                conf = "GP incertain"
            else:
                conf = "GP explore l inconnu"
            lines.append(f"    Confiance             : {conf}")
        else:
            lines.append(f"    Prediction GP         : {pred:.4f}")

        # Contributions SHAP (v3.6) : pourquoi ce point ?
        if shap_contribs is not None and (i-1) < len(shap_contribs):
            sc = shap_contribs[i-1]
            baseline = sc.get("baseline", 0.0)
            contribs = sc.get("contributions", {})
            if contribs:
                sorted_c = sorted(contribs.items(), key=lambda x: -abs(x[1]))
                lines.append(f"    Pourquoi ce point ?   (baseline = {baseline:.2f})")
                for pname, val in sorted_c:
                    sign = "+" if val >= 0 else ""
                    lines.append(f"      {pname:<26} : {sign}{val:.2f}")

        lines.append("")

    lines += [
        "PROGRESSION",
        "-" * 40,
        f"  Rounds BO effectues     : {n_rounds}",
        f"  Stagnation              : {n_stag}/{patience} rounds",
        f"  Convergence             : {'DETECTEE' if conv else 'non atteinte'}",
        f"  Fichier de donnees      : {csv_path}",
        "",
        "HYPERPARAMETRES GP (iteration courante)",
        "-" * 40,
    ]

    # Ajouter les hyperparamètres GP si disponibles
    if gp_display is not None and encoder is not None:
        try:
            ls = gp_display.kernel_.k2.length_scale
            param_names_raw = [p["name"] for p in cfg["parameters"]]
            lines.append("  Lengthscales ARD (espace normalise [0,1]) :")
            dim_cursor = 0
            for pi, p in enumerate(cfg["parameters"]):
                nd = encoder._encoded_dims[pi]
                ls_p = ls[dim_cursor:dim_cursor+nd]
                importance = 1.0 / (float(ls_p.mean()) + 1e-9)
                lines.append(f"    {p['name']:<28} ls={float(ls_p.mean()):.3f}"
                             f"  importance_rel={importance:.2f}")
                dim_cursor += nd
            # Normaliser importances pour affichage
            try:
                importances = np.array([1.0/(ls[d:d+encoder._encoded_dims[pi]].mean()+1e-9)
                                        for pi,p in enumerate(cfg["parameters"])
                                        for d in [sum(encoder._encoded_dims[:pi])]])
                importances /= importances.sum()
                lines.append(f"  Parametre le plus important : "
                             f"{param_names_raw[np.argmax(importances)]}")
            except Exception:
                pass
            # Bruit GP (GPyTorch)
            try:
                noise = float(gp_display.alpha)
                lines.append(f"  Bruit estime (likelihood noise) : {noise:.5f}")
            except Exception:
                pass
            # Warping Kumaraswamy appris (v3.6)
            try:
                if hasattr(gp_display, '_warper') and gp_display._warper is not None:
                    a_arr, b_arr = gp_display._warper.get_params_numpy()
                    if a_arr is not None:
                        lines.append("  Warping input (Kumaraswamy a, b) :")
                        dim_cursor = 0
                        for pi, p in enumerate(cfg["parameters"]):
                            nd = encoder._encoded_dims[pi]
                            a_p = float(a_arr[dim_cursor:dim_cursor+nd].mean())
                            b_p = float(b_arr[dim_cursor:dim_cursor+nd].mean())
                            # Interpretation
                            if abs(a_p - 1) < 0.1 and abs(b_p - 1) < 0.1:
                                interp = "lineaire"
                            elif a_p > 1.3 and b_p < 0.8:
                                interp = "plateau haut"
                            elif a_p < 0.8 and b_p > 1.3:
                                interp = "plateau bas"
                            elif a_p > 1.2 and b_p > 1.2:
                                interp = "sigmoide"
                            else:
                                interp = "non-lineaire leger"
                            lines.append(f"    {p['name']:<26} a={a_p:.2f} b={b_p:.2f}  ({interp})")
                            dim_cursor += nd
            except Exception:
                pass
        except Exception:
            lines.append("  (hyperparametres non disponibles)")
        lines.append("")

    lines += [
        "ETAPES SUIVANTES",
        "-" * 40,
        f"  1. Realisez les experiences aux conditions ci-dessus",
        f"  2. Saisissez les resultats dans le CSV",
        f"  3. Changez statut -> fait",
        f"  4. Relancez : python bo_chemistry.py",
        "",
        sep60,
    ]

    with open(base, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  [OK]   Rapport texte : {base}")


# =============================================================================

if __name__ == "__main__":
    main()