#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app_bo.py — Application desktop pour bo_chemistry.py (pywebview)
==================================================================

Lancement :
    python app_bo.py

Architecture :
    [Fenetre desktop native] contenant [HTML/CSS/JS custom]
           ↕ js_api (appels directs JavaScript -> Python)
    [Api class] -> bo_chemistry.py (import direct)

Persistance (identique au MVP precedent) :
    campaigns/
      <nom_campagne>/
        config.yaml
        results.csv

Tout est sauve sur disque immediatement. Fermer l'app ne perd rien.

Les logs du moteur BO (AUTO-PILOT, NOISE-AUTO, TuRBO, AUTO-XI...) sont
captures pendant le calcul et renvoyes a l'interface (badges + console brute).
"""

import os
import sys
import io
import json
import shutil
import re
import contextlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import webview

# Import direct du moteur BO
APP_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(APP_DIR))
import bo_chemistry as bo

CAMPAIGNS_DIR = APP_DIR / "campaigns"
CAMPAIGNS_DIR.mkdir(exist_ok=True)

ACQUISITIONS = ["EI", "UCB", "TEI", "EI-TS", "PI", "Thompson"]

# Bibliotheque de descripteurs physico-chimiques (emplacement fixe).
# L'utilisateur peut remplacer ce fichier par sa propre bibliotheque.
DESC_LIBRARY = APP_DIR / "examples" / "descripteurs_chimiques.xlsx"

# Variables continues classiques : (nom, unite, placeholder_min, placeholder_max)
CONTINUOUS_PRESETS = [
    {"name": "Temperature",        "unit": "C",      "ph_min": 25,   "ph_max": 150, "log_ok": False},
    {"name": "Temps (min)",        "unit": "",       "ph_min": 15,   "ph_max": 240, "log_ok": False},
    {"name": "Temps (h)",          "unit": "",       "ph_min": 0.5,  "ph_max": 24,  "log_ok": False},
    {"name": "Equivalents",        "unit": "eq",     "ph_min": 1.0,  "ph_max": 3.0, "log_ok": True},
    {"name": "Concentration",      "unit": "M",      "ph_min": 0.05, "ph_max": 1.0, "log_ok": True},
    {"name": "Loading catalyseur", "unit": "mol%",   "ph_min": 0.5,  "ph_max": 10,  "log_ok": True},
    {"name": "Pression",           "unit": "bar",    "ph_min": 1,    "ph_max": 50,  "log_ok": False},
    {"name": "Debit",              "unit": "mL/min", "ph_min": 0.1,  "ph_max": 5.0, "log_ok": True},
    {"name": "pH",                 "unit": "",       "ph_min": 1,    "ph_max": 14,  "log_ok": False},
    {"name": "Ratio reactifs (A/B)", "unit": "",     "ph_min": 0.5,  "ph_max": 2.0, "log_ok": True},
]

# Pour chaque feuille de la bibliotheque : les 5 descripteurs les plus
# globaux/rationnels (choisis pour bien estimer la majorite des cas).
SHEET_DEFAULT_DESCRIPTORS = {
    "Solvants": ["ε (25°C)", "μ (D)", "δP (MPa½)", "δH (MPa½)", "log P"],
    "Bases":    ["pKa (H₂O)", "Encombrement", "M (g/mol)"],
    "Ligands":  ["TEP (cm⁻¹)", "%Vbur (3.5Å)", "HOMO (eV)*", "LUMO (eV)*",
                 "Angle Tolman (°)"],
}

# Mapping nom de variable categorielle classique -> feuille bibliotheque
CATEGORICAL_PRESETS = {
    "Solvant": "Solvants",
    "Base":    "Bases",
    "Ligand":  "Ligands",
}

MAX_N_SUGGEST = 20


# =============================================================================
# UTILITAIRES (numpy/pandas -> JSON safe)
# =============================================================================
def _to_native(obj):
    """Convertit recursivement numpy/pandas vers types Python natifs JSON-safe."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if obj is pd.NaT:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def safe_campaign_name(name):
    """Valide et nettoie un nom de campagne. Empeche le path traversal.

    Retourne (nom_nettoye, erreur_ou_None).
    """
    if name is None:
        return None, "Nom requis"
    name = str(name).strip()
    if not name:
        return None, "Nom requis"
    if len(name) > 80:
        return None, "Nom trop long (max 80 caracteres)"
    # Interdire tout ce qui pourrait sortir du dossier campaigns/
    forbidden = set('/\\:*?"<>|')
    if any(c in forbidden for c in name):
        return None, "Caracteres interdits dans le nom : / \\ : * ? \" < > |"
    if name in (".", "..") or name.startswith("."):
        return None, "Nom invalide"
    # Verifier que le chemin resolu reste bien dans CAMPAIGNS_DIR
    resolved = (CAMPAIGNS_DIR / name).resolve()
    if CAMPAIGNS_DIR.resolve() not in resolved.parents:
        return None, "Nom invalide (chemin)"
    return name, None


def validate_config(cfg):
    """Valide une config BO. Retourne (ok, erreur_ou_None)."""
    params = cfg.get("parameters", [])
    if not params:
        return False, "Au moins un parametre requis"
    seen_names = set()
    for p in params:
        pname = p.get("name", "").strip()
        if not pname:
            return False, "Tout parametre doit avoir un nom"
        if pname in seen_names:
            return False, f"Nom de parametre duplique : '{pname}'"
        seen_names.add(pname)
        if p.get("type") == "continuous":
            if "min" not in p or "max" not in p:
                return False, f"Parametre '{pname}' : min et max requis"
            if float(p["min"]) >= float(p["max"]):
                return False, f"Parametre '{pname}' : min ({p['min']}) doit etre < max ({p['max']})"
        elif p.get("type") == "categorical":
            cats = p.get("categories", [])
            if len(cats) < 2:
                return False, f"Parametre '{pname}' : au moins 2 categories requises"
            if len(set(cats)) != len(cats):
                return False, f"Parametre '{pname}' : categories dupliquees"
            # Validation descripteurs si presents
            if p.get("descriptors_file"):
                dcols = p.get("descriptors_columns", [])
                if not dcols:
                    return False, (f"Parametre '{pname}' : descriptors_file fourni "
                                   f"mais aucune colonne selectionnee")
                if len(dcols) > 8:
                    return False, (f"Parametre '{pname}' : trop de descripteurs "
                                   f"({len(dcols)}), max 8 recommande pour eviter "
                                   f"un GP sous-determine")
        else:
            return False, f"Parametre '{pname}' : type inconnu '{p.get('type')}'"
    outs = cfg.get("outputs", [])
    if not outs:
        return False, "Au moins un objectif requis"
    seen_out = set()
    for o in outs:
        oname = o.get("name", "").strip()
        if not oname:
            return False, "Tout objectif doit avoir un nom"
        if oname in seen_out:
            return False, f"Objectif duplique : '{oname}'"
        seen_out.add(oname)
        if o.get("direction") not in ("maximize", "minimize"):
            return False, f"Objectif '{oname}' : direction doit etre maximize ou minimize"
        # Poids : seulement en multi-objectif, doit etre > 0
        if len(outs) > 1:
            w = o.get("weight", 1.0)
            try:
                if float(w) <= 0:
                    return False, f"Objectif '{oname}' : poids doit etre > 0"
            except (TypeError, ValueError):
                return False, f"Objectif '{oname}' : poids invalide"
    # Collision noms parametres / outputs
    out_names = {o["name"] for o in outs}
    if seen_names & out_names:
        return False, f"Un parametre et un objectif ont le meme nom : {seen_names & out_names}"

    # Validation des contraintes hard (expressions evaluables sur les parametres)
    hard = cfg.get("constraints", []) or []
    if hard:
        import math as _math
        # point fictif au centre des bornes pour tester l'evaluabilite
        test_point = {}
        for p in cfg["parameters"]:
            if p["type"] == "continuous":
                test_point[p["name"]] = (float(p["min"]) + float(p["max"])) / 2
            else:
                test_point[p["name"]] = p["categories"][0]
        safe_g = {"__builtins__": {}, "math": _math, "abs": abs,
                  "min": min, "max": max, "round": round,
                  "True": True, "False": False, "None": None}
        for expr in hard:
            if not str(expr).strip():
                return False, "Contrainte vide"
            try:
                eval(str(expr), safe_g, dict(test_point))
            except Exception as e:
                return False, (f"Contrainte invalide : '{expr}' "
                               f"({type(e).__name__}). Verifie les noms de parametres.")

    # Validation des contraintes soft (doivent pointer vers une sortie existante)
    soft = cfg.get("soft_constraints", []) or []
    for sc in soft:
        oc = sc.get("output")
        if not oc:
            return False, "Contrainte soft sans 'output'"
        if oc not in out_names:
            return False, (f"Contrainte soft : l'objectif '{oc}' n'existe pas. "
                           f"Objectifs disponibles : {', '.join(out_names)}")
        if sc.get("direction") not in ("maximize", "minimize"):
            return False, f"Contrainte soft '{oc}' : direction maximize ou minimize requise"
        try:
            float(sc.get("threshold"))
        except (TypeError, ValueError):
            return False, f"Contrainte soft '{oc}' : seuil numerique requis"

    return True, None


def campaign_paths(name):
    cdir = CAMPAIGNS_DIR / name
    return cdir, cdir / "config.yaml", cdir / "results.csv"


def load_cfg(name):
    _, ypath, _ = campaign_paths(name)
    with open(ypath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_cfg(name, cfg):
    cdir, ypath, _ = campaign_paths(name)
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = ypath.with_suffix(".yaml.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    tmp.replace(ypath)


def csv_columns(cfg):
    return ([p["name"] for p in cfg["parameters"]]
            + [o["name"] for o in cfg["outputs"]]
            + ["statut", "iteration_BO"])


def load_df(name, cfg):
    _, _, cpath = campaign_paths(name)
    cols = csv_columns(cfg)
    if cpath.exists():
        df = pd.read_csv(cpath)
        for c in cols:
            if c not in df.columns:
                df[c] = np.nan
        return df[cols]
    return pd.DataFrame(columns=cols)


def save_df(name, df):
    cdir, _, cpath = campaign_paths(name)
    cdir.mkdir(parents=True, exist_ok=True)
    tmp = cpath.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(cpath)


def get_done(df, cfg):
    if df.empty:
        return df
    out_cols = [o["name"] for o in cfg["outputs"]]
    d = df.copy()
    for c in out_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    # Une ligne est "faite" si TOUTES les sorties sont remplies
    mask = d[out_cols].notna().all(axis=1)
    return d[mask]


# =============================================================================
# PARSING DES LOGS MOTEUR (pour badges)
# =============================================================================
LOG_PATTERNS = [
    # (regex, type_badge, format_label)
    (r"\[AUTO-PILOT\]",            "autopilot",  "Autopilot"),
    (r"Phase\s*:\s*(\w+)",         "phase",      "Phase {0}"),
    (r"Progress\s*:\s*([\d.]+%)",  "progress",   "Progress {0}"),
    (r"Coverage\s*:\s*([\d.]+%)",  "coverage",   "Coverage {0}"),
    (r"\[NOISE-AUTO\].*noise_prior=(\w+)", "noise", "Noise prior auto: {0}"),
    (r"\[AUTO-XI\].*xi ([\d.]+) -> ([\d.]+)", "xi", "Auto-xi: {0} → {1}"),
    (r"\[AUTO-UCB\]",              "ucb",        "Switch EI → UCB"),
    (r"\[TuRBO\] init L=([\d.]+)", "turbo",      "TuRBO L={0}"),
    (r"\[TuRBO\].*restart",        "turbo",      "TuRBO restart"),
    (r"\[STAGNATION\]",            "stagnation", "Stagnation detectee"),
    (r"\[DRIFT\]",                 "drift",      "Drift detecte"),
    (r"\[AUTO\].*HEDGE",           "hedge",      "HEDGE active"),
    (r"\[CONVERGENCE\]",           "convergence","Convergence atteinte"),
    (r"Action\s*:\s*$",            None,         None),  # ignore
]


def parse_log_badges(log_text):
    """Extrait des badges structures depuis le log brut du moteur."""
    badges = []
    seen = set()
    for line in log_text.splitlines():
        for pattern, btype, label_fmt in LOG_PATTERNS:
            if btype is None:
                continue
            m = re.search(pattern, line)
            if m:
                label = label_fmt.format(*m.groups()) if m.groups() else label_fmt
                key = (btype, label)
                if key not in seen:
                    seen.add(key)
                    badges.append({"type": btype, "label": label})
                break
    return badges


# =============================================================================
# API EXPOSEE AU JAVASCRIPT
# =============================================================================
class Api:
    """Chaque methode publique est appelable depuis JS :
    window.pywebview.api.method_name(args).then(result => ...)
    """

    # ── Presets pour le formulaire de creation ──────────────────────────────
    def get_presets(self):
        """Retourne les presets continus + categoriels disponibles selon la
        bibliotheque de descripteurs presente."""
        cat_presets = []
        library_ok = DESC_LIBRARY.exists()
        if library_ok:
            try:
                xl = pd.ExcelFile(DESC_LIBRARY)
                sheets = xl.sheet_names
                for label, sheet in CATEGORICAL_PRESETS.items():
                    if sheet in sheets:
                        cat_presets.append({"label": label, "sheet": sheet})
            except Exception:
                library_ok = False
        return _to_native({
            "continuous": CONTINUOUS_PRESETS,
            "categorical": cat_presets,
            "library_available": library_ok,
            "library_name": DESC_LIBRARY.name,
        })

    def get_library_sheet(self, sheet_name):
        """Retourne les composes (lignes) et descripteurs par defaut d'une
        feuille de la bibliotheque, pour la liste cochable."""
        try:
            if not DESC_LIBRARY.exists():
                return {"ok": False, "error": "Bibliotheque de descripteurs absente"}
            df = pd.read_excel(DESC_LIBRARY, sheet_name=sheet_name, header=1)
            if "Nom" not in df.columns:
                return {"ok": False, "error": f"Feuille '{sheet_name}' sans colonne Nom"}
            df = df[df["Nom"].notna()]
            names = [str(n).strip() for n in df["Nom"].tolist() if str(n).strip()]
            default_cols = SHEET_DEFAULT_DESCRIPTORS.get(sheet_name, [])
            available = [c for c in default_cols if c in df.columns]
            return _to_native({
                "ok": True,
                "sheet": sheet_name,
                "compounds": names,
                "descriptor_columns": available,
            })
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Campagnes ────────────────────────────────────────────────────────────
    def list_campaigns(self):
        out = []
        for d in sorted(CAMPAIGNS_DIR.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists():
                try:
                    cfg = load_cfg(d.name)
                    df = load_df(d.name, cfg)
                    done = get_done(df, cfg)
                    out_col = cfg["outputs"][0]["name"]
                    direction = cfg["outputs"][0].get("direction", "maximize")
                    n = len(done)
                    best = None
                    if n > 0:
                        best = float(done[out_col].max() if direction == "maximize"
                                      else done[out_col].min())
                    out.append({"name": d.name, "n_done": n, "best": best,
                                "out_col": out_col})
                except Exception:
                    out.append({"name": d.name, "n_done": 0, "best": None,
                                "out_col": "?"})
        return _to_native(out)

    def create_campaign(self, payload_json):
        """payload : {name, parameters, output, bo}"""
        try:
            p = json.loads(payload_json)
            name, err = safe_campaign_name(p.get("name"))
            if err:
                return {"ok": False, "error": err}
            if (CAMPAIGNS_DIR / name).exists():
                return {"ok": False, "error": f"Campagne '{name}' existe deja"}

            params = p["parameters"]
            # Resoudre descriptors_file vers le chemin absolu de la bibliotheque
            for prm in params:
                if prm.get("use_library_descriptors"):
                    prm["descriptors_file"] = str(DESC_LIBRARY)
                    prm.pop("use_library_descriptors", None)

            # Objectifs : accepte 'objectives' (pluriel) ou 'output' (singulier, compat)
            if "objectives" in p and p["objectives"]:
                outputs = p["objectives"]
            else:
                outputs = [p["output"]]

            is_multi = len(outputs) > 1
            bo_settings = {
                "n_suggestions": 1,
                "acquisition": "EI",
                "autopilot": bool(p.get("autopilot", True)),
                "autopilot_target": float(p.get("target", 95)),
                "autopilot_max_iter": 30,
            }
            if is_multi:
                bo_settings["multi_objective_mode"] = "ehvi"
                bo_settings["mo_acquisition"] = "ehvi"
                bo_settings["autopilot"] = False  # autopilot mono-obj only

            cfg = {
                "campaign_name": name,
                "parameters": params,
                "outputs": outputs,
                "noise_prior": p.get("noise_prior", "low"),
                "bo_settings": bo_settings,
            }
            # Contraintes (optionnelles)
            if p.get("constraints"):
                cfg["constraints"] = p["constraints"]
            if p.get("soft_constraints"):
                cfg["soft_constraints"] = p["soft_constraints"]
            ok, verr = validate_config(cfg)
            if not ok:
                return {"ok": False, "error": verr}

            save_cfg(name, cfg)
            save_df(name, load_df(name, cfg))
            return {"ok": True, "name": name}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def import_yaml(self, yaml_text):
        try:
            cfg = yaml.safe_load(yaml_text)
            if not isinstance(cfg, dict):
                return {"ok": False, "error": "YAML invalide (pas un dictionnaire)"}
            # Cles vraiment requises
            for key in ["campaign_name", "parameters", "outputs"]:
                if key not in cfg:
                    return {"ok": False, "error": f"Cle manquante : {key}"}
            # bo_settings optionnel : defaut raisonnable si absent
            if "bo_settings" not in cfg or not cfg["bo_settings"]:
                is_multi = len(cfg["outputs"]) > 1
                cfg["bo_settings"] = {
                    "n_suggestions": 1,
                    "acquisition": "EI",
                    "autopilot": not is_multi,
                    "autopilot_target": 95,
                    "autopilot_max_iter": 30,
                }
                if is_multi:
                    cfg["bo_settings"]["multi_objective_mode"] = "ehvi"
                    cfg["bo_settings"]["mo_acquisition"] = "ehvi"
            name, err = safe_campaign_name(cfg["campaign_name"])
            if err:
                return {"ok": False, "error": f"Nom de campagne : {err}"}
            # Resoudre les descriptors_file relatifs par rapport au dossier de l'app
            for prm in cfg.get("parameters", []):
                dfile = prm.get("descriptors_file")
                if dfile and not os.path.isabs(dfile):
                    candidate = (APP_DIR / dfile).resolve()
                    if candidate.exists():
                        prm["descriptors_file"] = str(candidate)
                    elif DESC_LIBRARY.exists():
                        # repli sur la bibliotheque par defaut
                        prm["descriptors_file"] = str(DESC_LIBRARY)
            ok, verr = validate_config(cfg)
            if not ok:
                return {"ok": False, "error": verr}
            if (CAMPAIGNS_DIR / name).exists():
                return {"ok": False, "error": f"Campagne '{name}' existe deja"}
            cfg["campaign_name"] = name
            save_cfg(name, cfg)
            save_df(name, load_df(name, cfg))
            return {"ok": True, "name": name}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete_campaign(self, name):
        cleaned, err = safe_campaign_name(name)
        if err:
            return {"ok": False, "error": err}
        cdir, _, _ = campaign_paths(cleaned)
        if not cdir.exists():
            return {"ok": True}  # deja absent
        try:
            # onerror : gere les fichiers en lecture seule (frequent sous Windows)
            def _on_rm_error(func, path, exc_info):
                import stat
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            shutil.rmtree(cdir, onerror=_on_rm_error)
            # Verifier que la suppression a reellement abouti
            if cdir.exists():
                return {"ok": False,
                        "error": (f"Impossible de supprimer '{cleaned}' : un fichier "
                                  f"est peut-etre ouvert ailleurs (Excel, explorateur). "
                                  f"Ferme-le et reessaie.")}
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": f"Echec suppression : {e}"}

    def rename_campaign(self, old_name, new_name):
        """Renomme une campagne (dossier + champ campaign_name du YAML)."""
        try:
            old_clean, err = safe_campaign_name(old_name)
            if err:
                return {"ok": False, "error": err}
            new_clean, err = safe_campaign_name(new_name)
            if err:
                return {"ok": False, "error": err}
            old_dir, _, _ = campaign_paths(old_clean)
            new_dir, _, _ = campaign_paths(new_clean)
            if not old_dir.exists():
                return {"ok": False, "error": "Campagne introuvable"}
            if new_clean == old_clean:
                return {"ok": True, "name": new_clean}  # rien a faire
            if new_dir.exists():
                return {"ok": False, "error": f"'{new_clean}' existe deja"}
            # Renommer le dossier puis mettre a jour le YAML
            old_dir.rename(new_dir)
            cfg = load_cfg(new_clean)
            cfg["campaign_name"] = new_clean
            save_cfg(new_clean, cfg)
            return {"ok": True, "name": new_clean}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Donnees campagne ─────────────────────────────────────────────────────
    def get_campaign(self, name):
        """Retourne config + data + etat autopilot."""
        try:
            cfg = load_cfg(name)
            df = load_df(name, cfg)
            done = get_done(df, cfg)
            outputs = cfg["outputs"]
            is_multi = len(outputs) > 1
            out_col = outputs[0]["name"]
            direction = outputs[0].get("direction", "maximize")

            # "best" par objectif
            objectives = []
            for o in outputs:
                oc = o["name"]
                odir = o.get("direction", "maximize")
                b = None
                if len(done) > 0:
                    b = float(done[oc].max() if odir == "maximize"
                              else done[oc].min())
                objectives.append({"name": oc, "direction": odir,
                                    "best": b, "weight": o.get("weight", 1.0)})

            best = objectives[0]["best"]

            # Etat autopilot (mono-objectif seulement ; en multi, EHVI gere)
            ap = None
            if cfg["bo_settings"].get("autopilot") and len(done) > 0 and not is_multi:
                try:
                    encoder = bo.SpaceEncoder(cfg)
                    with contextlib.redirect_stdout(io.StringIO()):
                        ap_raw = bo.autopilot_decide(done, cfg,
                                                       cfg["bo_settings"], encoder)
                    ap = {"phase": ap_raw.get("phase"),
                          "progress": ap_raw.get("progress"),
                          "coverage": ap_raw.get("coverage"),
                          "target": ap_raw.get("target")}
                except Exception:
                    ap = None

            rows = df.replace({np.nan: None}).to_dict(orient="records")
            return _to_native({
                "ok": True,
                "cfg": cfg,
                "rows": rows,
                "n_done": len(done),
                "best": best,
                "out_col": out_col,
                "direction": direction,
                "is_multi": is_multi,
                "objectives": objectives,
                "autopilot_state": ap,
                "acquisitions": ACQUISITIONS,
            })
        except Exception as e:
            return {"ok": False, "error": f"{e}\n{traceback.format_exc()}"}

    def save_results(self, name, rows_json):
        """Sauve les resultats edites. Marque statut=fait si output rempli."""
        try:
            cfg = load_cfg(name)
            rows = json.loads(rows_json)
            df = pd.DataFrame(rows)
            # Garantir colonnes
            for c in csv_columns(cfg):
                if c not in df.columns:
                    df[c] = np.nan
            df = df[csv_columns(cfg)]
            out_col = cfg["outputs"][0]["name"]
            for idx in df.index:
                v = pd.to_numeric(df.at[idx, out_col], errors="coerce")
                if pd.notna(v):
                    df.at[idx, "statut"] = "fait"
            save_df(name, df)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def add_manual_row(self, name, row_json):
        """Ajoute une ligne manuelle (prior). Valide contre la config."""
        try:
            cfg = load_cfg(name)
            df = load_df(name, cfg)
            row = json.loads(row_json)

            # Validation des valeurs de parametres
            for p in cfg["parameters"]:
                pname = p["name"]
                if pname not in row:
                    return {"ok": False, "error": f"Valeur manquante pour '{pname}'"}
                val = row[pname]
                if p["type"] == "continuous":
                    try:
                        fv = float(val)
                    except (TypeError, ValueError):
                        return {"ok": False,
                                "error": f"'{pname}' doit etre numerique"}
                    if fv < p["min"] or fv > p["max"]:
                        return {"ok": False,
                                "error": f"'{pname}' = {fv} hors bornes "
                                         f"[{p['min']}, {p['max']}]"}
                elif p["type"] == "categorical":
                    if str(val) not in [str(c) for c in p["categories"]]:
                        return {"ok": False,
                                "error": f"'{pname}' : '{val}' n'est pas une "
                                         f"categorie valide ({', '.join(map(str, p['categories']))})"}

            row["statut"] = "fait"
            row["iteration_BO"] = np.nan
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            save_df(name, df)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def set_mode(self, name, mode_auto):
        """Persiste le mode auto/manuel dans le YAML des le toggle."""
        try:
            cfg = load_cfg(name)
            cfg["bo_settings"]["autopilot"] = bool(mode_auto)
            save_cfg(name, cfg)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ── Suggestion BO (avec capture des logs) ───────────────────────────────
    def suggest(self, name, options_json):
        """Lance une suggestion. options : {n_suggest, mode_auto, acquisition,
        xi, kappa, force}. Retourne suggestions + logs + badges.

        Option B : si des suggestions non remplies existent deja, on bloque
        et on retourne {ok:False, pending:N} sauf si force=True.
        """
        try:
            opts = json.loads(options_json)
            cfg = load_cfg(name)
            df = load_df(name, cfg)
            is_multi = len(cfg["outputs"]) > 1

            # ── Persistance du mode auto/manuel dans le YAML ──
            mode_auto = bool(opts.get("mode_auto", True))
            if cfg["bo_settings"].get("autopilot", True) != mode_auto:
                cfg["bo_settings"]["autopilot"] = mode_auto
                if not mode_auto:
                    cfg["bo_settings"]["acquisition"] = opts.get("acquisition", "EI")
                save_cfg(name, cfg)

            # ── Option B : bloquer si suggestions en attente ──
            pending = df[df["statut"] == "suggere"] if not df.empty else df
            n_pending = len(pending)
            if n_pending > 0 and not opts.get("force", False):
                return {"ok": False, "pending": n_pending,
                        "error": f"{n_pending} suggestion(s) en attente de resultat. "
                                 f"Remplis-les (ou utilise 'Remplacer') avant d'en "
                                 f"generer de nouvelles."}

            # Si force=True : supprimer les suggestions non remplies
            if opts.get("force", False) and n_pending > 0:
                df = df[df["statut"] != "suggere"].reset_index(drop=True)
                save_df(name, df)

            done = get_done(df, cfg)

            # ── Clamp n_suggest ──
            n_suggest = int(opts.get("n_suggest", 1))
            n_suggest = max(1, min(n_suggest, MAX_N_SUGGEST))

            cfg_run = json.loads(json.dumps(cfg, default=str))
            bo_s = cfg_run["bo_settings"]
            bo_s["n_suggestions"] = n_suggest

            if mode_auto:
                bo_s["autopilot"] = True
                bo_s["autopilot_verbose"] = True
            else:
                bo_s["autopilot"] = False
                bo_s["acquisition"] = opts.get("acquisition", "EI")
                bo_s["xi"] = float(opts.get("xi", 0.01))
                bo_s["ucb_kappa"] = float(opts.get("kappa", 2.0))

            rng = np.random.default_rng()
            encoder = bo.SpaceEncoder(cfg_run)

            # Capture stdout du moteur
            log_buffer = io.StringIO()

            n_cont = sum(1 for p in cfg["parameters"] if p["type"] == "continuous")
            n_cat = sum(1 for p in cfg["parameters"] if p["type"] == "categorical")
            # Scaling dimensionnel aligne sur generate_lhs_design (v3.9) :
            # tient compte des continus ET du nb max de niveaux categoriels.
            n_cat_max = max([len(p["categories"]) for p in cfg["parameters"]
                             if p["type"] == "categorical"], default=0)
            _base = max(2 * (n_cont + 1), 3 * n_cont, 5)
            if n_cat_max > 0:
                _base = max(_base, n_cat_max, int(np.ceil(1.5 * n_cat_max)))
            _default_init = int(min(_base, 24))
            min_init = bo_s.get("min_initial_points", _default_init)

            with contextlib.redirect_stdout(log_buffer):
                if len(done) < min_init:
                    n_needed = min_init - len(done)
                    print(f"[INIT] {len(done)} exp < min_initial={min_init} "
                          f"-> generation LHS de {n_needed} points")
                    df_lhs = bo.generate_lhs_design(cfg_run, n_points=n_needed,
                                                      rng=rng)
                    suggestions = [
                        {p["name"]: r[p["name"]] for p in cfg["parameters"]}
                        for _, r in df_lhs.iterrows()
                    ]
                    is_init = True
                elif is_multi:
                    # Multi-objectif -> EHVI / qNEHVI
                    bo_s["multi_objective_mode"] = "ehvi"
                    if "mo_acquisition" not in bo_s:
                        bo_s["mo_acquisition"] = "qnehvi" if n_suggest >= 2 else "ehvi"
                    print(f"[MULTI-OBJ] {len(cfg['outputs'])} objectifs -> "
                          f"{bo_s['mo_acquisition'].upper()}")
                    suggestions, _info = bo.suggest_experiments_ehvi(
                        cfg_run, done, encoder, n_suggest, rng)
                    is_init = False
                else:
                    suggestions, _preds = bo.suggest_experiments(
                        cfg_run, done, encoder, n_suggest, rng)
                    is_init = False

            log_text = log_buffer.getvalue()
            badges = parse_log_badges(log_text)
            if is_init:
                badges.insert(0, {"type": "init",
                                   "label": f"Init LHS ({len(suggestions)} points)"})

            # Ajouter au CSV
            existing = pd.to_numeric(df["iteration_BO"], errors="coerce")
            next_iter = (int(existing.max()) + 1) if existing.notna().any() else 1
            new_rows = []
            for s in suggestions:
                row = dict(s)
                for o in cfg["outputs"]:
                    row[o["name"]] = np.nan
                row["statut"] = "suggere"
                row["iteration_BO"] = np.nan if is_init else next_iter
                new_rows.append(row)
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            save_df(name, df)

            return _to_native({
                "ok": True,
                "n_added": len(suggestions),
                "iteration": None if is_init else next_iter,
                "is_init": is_init,
                "log": log_text,
                "badges": badges,
            })
        except Exception as e:
            # Message lisible selon le type d'erreur courant
            msg = str(e)
            friendly = "Erreur pendant le calcul."
            if "singular" in msg.lower() or "cholesky" in msg.lower() or "positive definite" in msg.lower():
                friendly = ("Le modele GP n'a pas pu s'ajuster (matrice singuliere). "
                            "Cela arrive souvent avec trop peu de donnees ou des valeurs "
                            "quasi identiques. Ajoute quelques experiences plus variees.")
            elif "out of memory" in msg.lower() or "alloc" in msg.lower():
                friendly = "Memoire insuffisante pour ce calcul. Reduis n_suggestions."
            elif "convergence" in msg.lower():
                friendly = ("L'optimiseur n'a pas converge. Reessaie, ou ajoute des "
                            "donnees si le probleme persiste.")
            else:
                friendly = f"Erreur pendant le calcul : {msg}"
            return {"ok": False,
                    "error": friendly,
                    "log": traceback.format_exc()}

    # ── Graphes ──────────────────────────────────────────────────────────────
    def get_convergence(self, name):
        try:
            cfg = load_cfg(name)
            df = load_df(name, cfg)
            done = get_done(df, cfg)
            out_col = cfg["outputs"][0]["name"]
            direction = cfg["outputs"][0].get("direction", "maximize")
            if len(done) < 1:
                return {"ok": True, "empty": True}
            vals = done[out_col].values.astype(float)
            best = (np.maximum.accumulate(vals) if direction == "maximize"
                    else np.minimum.accumulate(vals))
            return _to_native({
                "ok": True, "empty": False,
                "x": list(range(1, len(vals) + 1)),
                "raw": vals.tolist(),
                "best": best.tolist(),
                "out_col": out_col,
            })
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_acq_1d(self, name, options_json):
        """Acquisition 1D le long d'un parametre continu."""
        try:
            opts = json.loads(options_json)
            param_name = opts["param"]
            acquisition = opts.get("acquisition", "EI").upper()
            xi = float(opts.get("xi", 0.01))
            kappa = float(opts.get("kappa", 2.0))

            cfg = load_cfg(name)
            df = load_df(name, cfg)
            done = get_done(df, cfg)
            out_col = cfg["outputs"][0]["name"]
            direction = cfg["outputs"][0].get("direction", "maximize")
            param_names = [p["name"] for p in cfg["parameters"]]

            if len(done) < 3:
                return {"ok": True, "empty": True,
                        "msg": "Au moins 3 experiences requises"}

            p_obj = next((p for p in cfg["parameters"]
                          if p["name"] == param_name), None)
            if p_obj is None or p_obj["type"] != "continuous":
                return {"ok": False, "error": "Parametre continu requis"}

            encoder = bo.SpaceEncoder(cfg)
            d = done.reset_index(drop=True)
            best_idx = (d[out_col].idxmax() if direction == "maximize"
                        else d[out_col].idxmin())
            best_row = d.loc[best_idx]

            grid = np.linspace(p_obj["min"], p_obj["max"], 200)
            rows = []
            for g in grid:
                cond = {pn: best_row[pn] for pn in param_names}
                cond[param_name] = g
                rows.append(cond)
            df_grid = pd.DataFrame(rows)

            with contextlib.redirect_stdout(io.StringIO()):
                X_train = encoder.encode(d[param_names].to_numpy(dtype=object))
                y_train = d[out_col].values.astype(float)
                if direction == "minimize":
                    y_train = -y_train
                gp = bo.build_gp(encoder.n_params, normalize_y=True, cfg=cfg,
                                  encoder=encoder, y_train=y_train)
                gp.fit(X_train, y_train)
                y_best = float(y_train.max())
                X_grid = encoder.encode(
                    df_grid[param_names].to_numpy(dtype=object))

                if acquisition == "EI":
                    acq = bo.acq_ei(X_grid, gp, y_best, xi=xi)
                elif acquisition == "UCB":
                    acq = bo.acq_ucb(X_grid, gp, kappa=kappa)
                elif acquisition == "TEI":
                    acq = bo.acq_tei(X_grid, gp, y_best, xi=xi)
                elif acquisition == "PI":
                    acq = bo.acq_pi(X_grid, gp, y_best, xi=xi)
                else:
                    acq = bo.acq_ei(X_grid, gp, y_best, xi=xi)

                mu, sigma = gp.predict(X_grid, return_std=True)
                if direction == "minimize":
                    mu = -mu

            return _to_native({
                "ok": True, "empty": False,
                "grid": grid.tolist(),
                "acq": np.asarray(acq).tolist(),
                "mu": np.asarray(mu).tolist(),
                "sigma": np.asarray(sigma).tolist(),
                "obs_x": d[param_name].values.tolist(),
                "obs_y": d[out_col].values.tolist(),
                "argmax_x": float(grid[int(np.argmax(acq))]),
                "param": param_name,
                "out_col": out_col,
                "acquisition": acquisition,
            })
        except Exception as e:
            return {"ok": False, "error": f"{e}\n{traceback.format_exc()}"}

    def get_cont_params(self, name):
        try:
            cfg = load_cfg(name)
            return _to_native([p["name"] for p in cfg["parameters"]
                                if p["type"] == "continuous"])
        except Exception:
            return []

    def get_pareto(self, name):
        """Front de Pareto pour les 2 premiers objectifs (multi-objectif)."""
        try:
            cfg = load_cfg(name)
            outputs = cfg["outputs"]
            if len(outputs) < 2:
                return {"ok": False, "error": "Pareto necessite >= 2 objectifs"}
            df = load_df(name, cfg)
            done = get_done(df, cfg)
            if len(done) < 1:
                return {"ok": True, "empty": True}

            o1, o2 = outputs[0], outputs[1]
            n1, n2 = o1["name"], o2["name"]
            d1, d2 = o1.get("direction", "maximize"), o2.get("direction", "maximize")
            x = done[n1].values.astype(float)
            y = done[n2].values.astype(float)

            # Determiner les points non-domines
            def dominates(i, j):
                # i domine j si i >= j sur les deux (selon direction) et > sur au moins un
                better1 = (x[i] >= x[j]) if d1 == "maximize" else (x[i] <= x[j])
                better2 = (y[i] >= y[j]) if d2 == "maximize" else (y[i] <= y[j])
                strict1 = (x[i] > x[j]) if d1 == "maximize" else (x[i] < x[j])
                strict2 = (y[i] > y[j]) if d2 == "maximize" else (y[i] < y[j])
                return better1 and better2 and (strict1 or strict2)

            n = len(x)
            is_pareto = [True] * n
            for i in range(n):
                for j in range(n):
                    if i != j and dominates(j, i):
                        is_pareto[i] = False
                        break

            return _to_native({
                "ok": True, "empty": False,
                "x": x.tolist(), "y": y.tolist(),
                "is_pareto": is_pareto,
                "name_x": n1, "name_y": n2,
                "dir_x": d1, "dir_y": d2,
            })
        except Exception as e:
            return {"ok": False, "error": f"{e}\n{traceback.format_exc()}"}

    def export_csv(self, name):
        """Retourne le contenu CSV de la campagne (pour telechargement)."""
        try:
            cfg = load_cfg(name)
            df = load_df(name, cfg)
            return {"ok": True, "csv": df.to_csv(index=False),
                    "filename": f"{name}_results.csv"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def delete_row(self, name, row_index):
        """Supprime une ligne du tableau par son index."""
        try:
            cfg = load_cfg(name)
            df = load_df(name, cfg)
            idx = int(row_index)
            if idx < 0 or idx >= len(df):
                return {"ok": False, "error": "Index hors limites"}
            df = df.drop(df.index[idx]).reset_index(drop=True)
            save_df(name, df)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# =============================================================================
# MAIN
# =============================================================================
def main():
    api = Api()
    html_path = (APP_DIR / "static" / "index.html").resolve()

    if not html_path.exists():
        print(f"ERREUR : fichier introuvable : {html_path}")
        print(f"Verifie que le dossier 'static/' avec index.html et plotly.min.js")
        print(f"est bien a cote de app_bo.py.")
        sys.exit(1)

    # Construire une URL file:// absolue, robuste sur Windows/Mac/Linux.
    # Path.as_uri() gere correctement les espaces, accents et backslashes Windows.
    file_url = html_path.as_uri()

    window = webview.create_window(
        "BO Chemistry",
        url=file_url,
        js_api=api,
        width=1380,
        height=880,
        min_size=(1100, 700),
        background_color="#0a0a0b",
    )
    webview.start(debug=False)


if __name__ == "__main__":
    main()
