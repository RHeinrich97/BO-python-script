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
    python bo_chemistry.py config.yaml --final        <- rapport final haute resolution

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

    Retourne un DataFrame CSV-ready avec les colonnes du CSV de campagne
    et statut = 'initial' (a changer en 'fait' apres les experiences).

    Parametres :
        cfg      : configuration chargee depuis le YAML
        n_points : nombre de points (defaut : 2 x nb_parametres_continus + 1)
        rng      : generateur aleatoire (reproductibilite)
    """
    if rng is None:
        rng = np.random.default_rng(cfg["bo_settings"].get("seed", 42))

    params       = cfg["parameters"]
    output_names = [o["name"] for o in cfg["outputs"]]
    cont_params  = [p for p in params if p["type"] == "continuous"]
    cat_params   = [p for p in params if p["type"] == "categorical"]

    n_cont = len(cont_params)
    if n_points is None:
        n_points = max(2 * n_cont + 1, 4)

    # ── LHS pour variables continues ─────────────────────────────────────────
    # Pour chaque variable, diviser [0,1] en n_points intervalles egaux,
    # tirer un point uniforme dans chaque intervalle, puis permuter.
    lhs_matrix = np.zeros((n_points, n_cont))
    for j in range(n_cont):
        # Points au centre des intervalles + bruit uniforme dans l'intervalle
        intervals = (np.arange(n_points) + rng.uniform(0, 1, n_points)) / n_points
        perm      = rng.permutation(n_points)
        lhs_matrix[:, j] = intervals[perm]

    # Denormaliser vers les vraies valeurs
    rows = []
    for i in range(n_points):
        row = {}
        for j, p in enumerate(cont_params):
            lo, hi     = float(p["min"]), float(p["max"])
            val        = lo + lhs_matrix[i, j] * (hi - lo)
            row[p["name"]] = round(val, 4)

        # Variables categorielles : round-robin permute
        for p in cat_params:
            cats  = p["categories"]
            idx   = int(rng.integers(0, len(cats)))
            row[p["name"]] = cats[idx]

        # Colonnes de sortie vides
        for o in output_names:
            row[o] = ""
        row["statut"]        = "initial"
        row["iteration_BO"]  = ""
        row["prediction_BO"] = ""
        rows.append(row)

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
    save_csv(df, csv_path)

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


def save_csv(df, csv_path):
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

        cat_count = 0
        for i, p in enumerate(self.params):
            if p["type"] == "continuous":
                self.continuous_idx.append(i)
                self.bounds_phys.append((float(p["min"]), float(p["max"])))
                self._encoded_dims.append(1)
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
                X_norm[:, dim_cursor] = (X_raw[:, param_idx].astype(float) - lo) / max(hi - lo, 1e-9)
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
                    k     = self.continuous_idx.index(param_idx)
                    lo, hi = self.bounds_phys[k]
                    val   = float(np.clip(X_norm[i, d], 0, 1)) * (hi - lo) + lo
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


def scalarise(Y, cfg, pareto_weights=None):
    outputs = cfg["outputs"]
    mode    = cfg["bo_settings"].get("multi_objective_mode", "weighted_sum")
    n_out   = len(outputs)
    Y_s = np.zeros_like(Y, dtype=float)
    for j, out in enumerate(outputs):
        Y_s[:, j] = Y[:, j] * _sign(out)
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


def feasible_best(Y, cfg, pareto_weights=None):
    scores = scalarise(Y, cfg, pareto_weights)
    return float(scores.max()) if scores.size > 0 else 0.0


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

class _ExactGPModel(gpytorch.models.ExactGP if _TORCH_OK else object):
    """Modele GP exact GPyTorch — Matern 5/2 ARD avec GammaPriors."""

    def __init__(self, X, y, likelihood, n_dims):
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

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


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
    _TRAIN_ITERS   = 150    # iterations ADAM par restart
    _N_RESTARTS    = 5      # restarts aléatoires pour éviter les minima locaux
    _LR            = 0.05   # learning rate ADAM
    _NOISE_FLOOR   = 1e-4   # bruit minimal (évite la dégénérescence)

    def __init__(self, n_dims, normalize_y=True):
        _require_torch()
        self.n_dims       = n_dims
        self.normalize_y  = normalize_y
        self._y_mean = 0.0
        self._y_std  = 1.0
        # kernel_ stub initial (avant fit)
        self.kernel_ = _KernelStub(np.ones(n_dims))
        self.alpha   = self._NOISE_FLOOR

    # ── Conversion numpy → torch ──────────────────────────────────────────────
    @staticmethod
    def _t(arr):
        return torch.tensor(np.array(arr, dtype=np.float32))

    # ── Construction du modèle GPyTorch ──────────────────────────────────────
    def _build(self, X_t, y_t):
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(self._NOISE_FLOOR)
        )
        model = _ExactGPModel(X_t, y_t, likelihood, self.n_dims)
        return model, likelihood

    # ── Entrainement ──────────────────────────────────────────────────────────
    def fit(self, X, y):
        X_np = np.array(X, dtype=np.float32)
        y_np = np.array(y, dtype=np.float32)

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

        for restart in range(self._N_RESTARTS + 1):
            model, likelihood = self._build(X_t, y_t)
            model.train(); likelihood.train()

            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(likelihood.parameters()),
                lr=self._LR
            )
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            torch.manual_seed(restart * 17 + 42)
            for _ in range(self._TRAIN_ITERS):
                optimizer.zero_grad()
                try:
                    loss = -mll(model(X_t), y_t)
                    loss.backward()
                    optimizer.step()
                except Exception:
                    break

            try:
                final_loss = float(loss.item())
            except Exception:
                final_loss = np.inf

            if final_loss < best_loss:
                best_loss  = final_loss
                best_state = {
                    "model":      {k: v.clone().detach() for k,v in model.state_dict().items()},
                    "likelihood": {k: v.clone().detach() for k,v in likelihood.state_dict().items()},
                }

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
    def predict(self, X, return_std=False, return_cov=False):
        X_t = self._t(np.array(X, dtype=np.float32))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._model(X_t))
            mu  = pred.mean.numpy().astype(np.float64)
            mu  = mu * self._y_std + self._y_mean

            if return_cov:
                # Covariance complète (pour Thompson Sampling)
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


def build_gp(n_dims, normalize_y=True, **kwargs):
    """
    Crée un GPModel GPyTorch.
    Interface identique à l ancienne fonction sklearn pour compatibilité.
    Les kwargs (alpha, etc.) sont ignorés — le GP gère son bruit en interne.
    """
    return GPModel(n_dims=n_dims, normalize_y=normalize_y)


# =============================================================================
# 5b.  BACKEND GPYTORCH (optionnel — gp_backend: "torch" dans le YAML)
# =============================================================================

def _try_import_torch():
    """Importe GPyTorch si disponible, retourne False sinon."""
    try:
        import torch
        import gpytorch
        return torch, gpytorch
    except ImportError:
        return None, None



# =============================================================================
# 6.  FONCTIONS D'ACQUISITION
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


def maximise_acquisition(acq_fn, n_params, n_restarts=20, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    unit_bounds = [(0.0, 1.0)] * n_params
    seed_val    = int(rng.integers(0, 2**31))
    de_result   = differential_evolution(
        func=lambda x: -float(acq_fn(x.reshape(1, -1)).item()),
        bounds=unit_bounds, seed=seed_val, maxiter=300, tol=1e-6, popsize=15, workers=1,
    )
    best_x, best_val = de_result.x, -de_result.fun
    for _ in range(n_restarts):
        x0  = rng.uniform(0.0, 1.0, size=n_params)
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
    kappa      = float(bo.get("ucb_kappa", 2.0))
    xi_val     = float(bo.get("xi", 0.01))
    n_restarts = int(bo.get("n_restarts", 20))
    mode       = bo.get("multi_objective_mode", "weighted_sum")

    X_raw        = df_done[param_names].values.astype(object)
    Y            = df_done[output_names].values.astype(float)
    X_norm_base  = encoder.encode(X_raw)

    suggestions     = []
    gp_predictions  = []   # NEW: prediction GP a chaque suggestion
    X_norm_train    = X_norm_base.copy()
    Y_scalar_train  = None
    y_phantom_list  = []

    for s in range(n_suggest):
        pareto_w = None
        if mode == "pareto" and len(output_names) > 1:
            pareto_w = rng.dirichlet(np.ones(len(output_names)))

        if s == 0:
            Y_scalar_raw   = scalarise(Y, cfg, pareto_w)
            Y_scalar_train, _bc_lam, _bc_off = fit_boxcox(Y_scalar_raw, bo, verbose=(s==0))
            _bc_train_max  = float(Y_scalar_train.max()) if _bc_lam is not None else 0.0
        else:
            Y_real         = scalarise(Y, cfg, pareto_w)
            Y_scalar_train = np.append(Y_real, y_phantom_list)

        y_best = float(Y_scalar_train.max())
        gp = build_gp(encoder.n_params, normalize_y=(_bc_lam is None))
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
                errors="coerce").dropna().max() or 1)
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

        else:
            raise ValueError(f"Acquisition inconnue : '{acq_name}'.")

        x_next_norm = maximise_acquisition(acq_fn, encoder.n_params, n_restarts, rng)
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

        # Maximise MC-EHVI
        print(f"    EHVI : calcul suggestion {s+1}/{n_suggest} "
              f"({'exact 2D' if n_obj == 2 else 'MC 3D+'})...", flush=True)
        x_next_norm = _maximise_ehvi_grid(gps, ref_point, Y_aug_signed, cfg, encoder, rng)
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

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor("#F8F9FA")
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97, top=0.93, bottom=0.07,
                            hspace=0.38, wspace=0.28)

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

def check_convergence(df, cfg):
    """
    Detecte la convergence : le meilleur score n'a pas progresse
    sur les derniers `patience` rounds BO.

    Litterature : 5 rounds sans amelioration est standard en chimie
    (Shields et al. JACS 2021 ; Griffiths & Hernandez-Lobato 2020).
    Valeur par defaut : 5, configurable via convergence_patience dans bo_settings.

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
        return True, patience
    return False, 0


# =============================================================================
# 13.  GRAPHIQUES FINAUX (apres convergence)
# =============================================================================

def plot_final(df, cfg, gp_display, encoder, iteration):
    """
    Genere les figures de publication a la convergence.
    Sauvegarde {campaign}_FINAL_convergence.png et {campaign}_FINAL_landscape.png.
    """
    print("\n  [CONVERGENCE DETECTEE] Generation des graphiques finaux...")

    # Reutilise les memes fonctions avec le prefixe FINAL
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

    X_norm_all  = encoder.encode(df_done[param_names].values.astype(object))
    gp_pred_all = gp_display.predict(X_norm_all)
    has_stored  = "prediction_BO" in df_done.columns
    stored_pred = pd.to_numeric(
        df_done["prediction_BO"] if has_stored else pd.Series(dtype=float),
        errors="coerce"
    ).values if has_stored else np.full(n_exp, np.nan)

    bo_mask = pd.to_numeric(
        df_done["iteration_BO"].replace("", np.nan), errors="coerce"
    ).notna().values
    potential = np.where(bo_mask & ~np.isnan(stored_pred), stored_pred, gp_pred_all)
    best_idx  = np.argmax(scores)
    n_init    = int((~bo_mask).sum())

    # ── Figure finale convergence ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.scatter(x_vals[~bo_mask], potential[~bo_mask],
               facecolors="none", edgecolors=_COLORS["initial"],
               s=60, linewidths=1.2, zorder=2, alpha=0.7)
    ax.scatter(x_vals[bo_mask], potential[bo_mask],
               facecolors="none", edgecolors=_COLORS["potential"],
               s=60, linewidths=1.5, zorder=2, alpha=0.9,
               label="Potentiel GP (prediction au moment de la suggestion)")
    ax.scatter(x_vals[~bo_mask], scores[~bo_mask],
               c=_COLORS["initial"], s=55, zorder=4, label="Experience initiale")
    ax.scatter(x_vals[bo_mask], scores[bo_mask],
               c=_COLORS["bo_dot"], s=55, zorder=4, label="Experience BO (mesure)")
    ax.scatter(x_vals[best_idx], scores[best_idx],
               c=_COLORS["best_star"], s=200, marker="*", zorder=6,
               label=f"Meilleur ({scores[best_idx]:.3f})")
    ax.plot(x_vals, cummax, color=_COLORS["best_line"], lw=2.5, zorder=3,
            label="Meilleur cumule")
    if 0 < n_init < n_exp:
        ax.axvline(n_init + 0.5, color="black", lw=1, linestyle="--", alpha=0.4,
                   label="Debut BO")
    n_out  = len(output_names)
    ylabel = output_names[0] if n_out == 1 else "Score scalarise"
    ax.set_xlabel("Numero d'experience")
    ax.set_ylabel(ylabel)
    ax.set_title(f"[FINAL] Convergence – {cfg['campaign_name']}", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_conv = f"{cfg['campaign_name']}_FINAL_convergence.png"
    plt.savefig(fname_conv, dpi=200)
    plt.close()
    print(f"  [GRAPHIQUE] FINAL convergence : {fname_conv}")

    # ── Figure finale paysage ─────────────────────────────────────────────────
    top2 = get_influential_params(gp_display, cfg, n=2)
    if len(top2) >= 2:
        p0_idx, p1_idx = top2[0], top2[1]
        p0 = cfg["parameters"][p0_idx]
        p1 = cfg["parameters"][p1_idx]
        best_i      = int(np.argmax(scores))
        X_best_raw  = df_done[param_names].values.astype(object)[best_i:best_i+1]
        X_best_norm = encoder.encode(X_best_raw)[0]

        n_grid = 60
        p0g = np.linspace(0, 1, n_grid)
        p1g = np.linspace(0, 1, n_grid)
        P0n, P1n = np.meshgrid(p0g, p1g)
        n_pts   = n_grid * n_grid
        X_grid  = np.tile(X_best_norm, (n_pts, 1))
        X_grid[:, p0_idx] = P0n.ravel()
        X_grid[:, p1_idx] = P1n.ravel()
        mu_grid = gp_display.predict(X_grid).reshape(n_grid, n_grid)

        lo0, hi0 = float(p0["min"]), float(p0["max"])
        lo1, hi1 = float(p1["min"]), float(p1["max"])
        P0_phys  = P0n * (hi0 - lo0) + lo0
        P1_phys  = P1n * (hi1 - lo1) + lo1

        x0_pts = X_norm_all[:, p0_idx] * (hi0 - lo0) + lo0
        x1_pts = X_norm_all[:, p1_idx] * (hi1 - lo1) + lo1

        fig = plt.figure(figsize=(14, 6))
        fig.suptitle(
            f"[FINAL] Paysage GP – {cfg['campaign_name']}\n"
            f"Axes : {p0['name']}  vs  {p1['name']}",
            fontsize=11
        )
        ax1 = fig.add_subplot(121)
        cf  = ax1.contourf(P0_phys, P1_phys, mu_grid, levels=20, cmap="viridis")
        ax1.contour(P0_phys, P1_phys, mu_grid, levels=20,
                    colors="k", alpha=0.1, linewidths=0.3)
        cb = plt.colorbar(cf, ax=ax1)
        cb.set_label(f"Prediction GP\n({ylabel})", fontsize=8)
        ax1.scatter(x0_pts, x1_pts, c=scores, cmap="viridis",
                    vmin=mu_grid.min(), vmax=mu_grid.max(),
                    edgecolors="k", linewidths=0.7, s=65, zorder=5)
        ax1.scatter(x0_pts[best_i], x1_pts[best_i], c="gold", s=220, marker="*",
                    edgecolors="k", linewidths=0.7, zorder=6,
                    label=f"Meilleur ({scores[best_i]:.3f})")
        ax1.set_xlabel(p0["name"]); ax1.set_ylabel(p1["name"])
        ax1.set_title("Carte de contour 2D"); ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_surface(P0_phys, P1_phys, mu_grid, cmap="viridis",
                         alpha=0.82, linewidth=0, antialiased=True)
        ax2.scatter(x0_pts, x1_pts, scores, c=scores, cmap="viridis",
                    edgecolors="k", linewidths=0.5, s=45, zorder=5, depthshade=True)
        ax2.scatter([x0_pts[best_i]], [x1_pts[best_i]], [scores[best_i]],
                    c="gold", s=160, marker="*", edgecolors="k", zorder=6)
        ax2.set_xlabel(p0["name"], fontsize=8, labelpad=6)
        ax2.set_ylabel(p1["name"], fontsize=8, labelpad=6)
        ax2.set_zlabel(ylabel,     fontsize=8, labelpad=6)
        ax2.set_title("Surface 3D")
        plt.tight_layout()
        fname_land = f"{cfg['campaign_name']}_FINAL_landscape.png"
        plt.savefig(fname_land, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  [GRAPHIQUE] FINAL paysage   : {fname_land}")


# =============================================================================
# 14.  BENCHMARK DES FONCTIONS D'ACQUISITION
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
    n_restarts   = min(int(bo.get("n_restarts", 20)), 8)  # reduit pour la vitesse

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
                gp_b   = build_gp(encoder.n_params, normalize_y=True)
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

def main():

    # ── Parsing des arguments ─────────────────────────────────────────────────
    args           = sys.argv[1:]
    flags          = [a for a in args if a.startswith("--")]
    positional     = [a for a in args if not a.startswith("--")]
    argument       = positional[0] if positional else None
    benchmark_mode = "--benchmark" in flags
    status_mode    = "--status"    in flags
    doe_mode       = "--init-doe"  in flags
    final_mode     = "--final"     in flags
    # --doe-n N : nombre de points LHS (optionnel)
    doe_n = None
    for k, fl in enumerate(flags):
        if fl == "--doe-n" and k + 1 < len(flags):
            try: doe_n = int(flags[k + 1])
            except ValueError: pass

    config_path = trouver_config(argument)
    cfg         = load_config(config_path)
    csv_path    = get_csv_path(cfg)
    encoder     = SpaceEncoder(cfg)

    bo           = cfg["bo_settings"]
    n_suggest    = int(bo.get("n_suggestions", 1))
    seed         = int(bo.get("seed", 42))
    min_pts      = int(bo.get("min_initial_points", max(3, 2 * encoder.n_params)))
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
    print(f"{sep}")

    # ── Mode INIT-DOE : plan LHS ─────────────────────────────────────────────
    if doe_mode:
        n_pts = doe_n if doe_n else int(bo.get("doe_n_points",
                    max(2 * sum(1 for p in cfg["parameters"] if p["type"]=="continuous") + 1, 6)))
        run_init_doe(cfg, csv_path, n_pts, rng)
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
    gp_display = build_gp(encoder.n_params, normalize_y=not bc_active)
    gp_display.fit(X_disp, y_disp_bc)

    # ── Calcul des suggestions ────────────────────────────────────────────────
    mode = bo.get("multi_objective_mode", "weighted_sum")
    acq  = bo.get("acquisition", "EI").upper()

    # ── Detection de stagnation ───────────────────────────────────────────────
    import copy
    stag_detected, stag_params, stag_msg = detect_variable_stagnation(df, cfg, bo)
    drift_detected, drift_params, drift_msg = detect_directional_drift(df, cfg, bo)

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
    else:
        cfg_stag = cfg

    if mode == "ehvi" and len(output_names) > 1:
        acq_display = "MC-EHVI (grille 2000 pts)"
    else:
        if stag_detected:
            acq_display = acq + " [override stagnation]"
        elif drift_detected:
            acq_display = acq + " [override derive]"
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
        new_rows.append(row)

    df_updated = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_csv(df_updated, csv_path)

    # ── Graphiques du round ───────────────────────────────────────────────────
    opts = _fig_opts(cfg)
    plot_dashboard(df_updated, cfg, gp_display, encoder, next_iter, opts)
    # Snapshots per-itération uniquement si timestamp_outputs: true
    if opts.get("timestamp", False):
        plot_round(df_updated, cfg, gp_display, encoder, next_iter, opts)
        plot_gp_landscape(gp_display, encoder, cfg, df_done, next_iter, opts)

    # ── Rapport texte resume ─────────────────────────────────────────────────
    _write_run_report(
        cfg, df_done, df_updated, suggestions, gp_preds,
        next_iter, output_names, param_names, csv_path
    )

    # ── Detection de convergence ──────────────────────────────────────────────
    converged, n_stagnant = check_convergence(df_updated, cfg)
    patience = int(bo.get("convergence_patience", 5))
    if converged:
        print(f"\n  [CONVERGENCE] Aucune amelioration depuis {n_stagnant} rounds BO.")
        plot_final_report(df_updated, cfg, gp_display, encoder, next_iter, opts)
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

def _write_run_report(cfg, df_done, df_updated, suggestions, gp_preds,
                      iteration, output_names, param_names, csv_path):
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
        lines.append(f"    Prediction GP         : {pred:.4f}")
        lines.append("")

    lines += [
        "PROGRESSION",
        "-" * 40,
        f"  Rounds BO effectues     : {n_rounds}",
        f"  Stagnation              : {n_stag}/{patience} rounds",
        f"  Convergence             : {'DETECTEE' if conv else 'non atteinte'}",
        f"  Fichier de donnees      : {csv_path}",
        "",
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