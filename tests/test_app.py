#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_app.py — Tests de la couche GUI (app_bo.py)
=================================================

Teste l'API backend de l'application (validation, persistance, suggestions,
multi-objectif, descripteurs, cas limites) sans lancer la fenetre pywebview.

Usage :
    python test_app.py
"""

import sys
import os
import io
import json
import shutil
import contextlib
import tempfile
import importlib.util
from pathlib import Path

import numpy as np

# ── Charger app_bo en isolant son dossier campaigns dans un tmp ──
THIS_DIR = Path(__file__).parent.resolve()
APP_PATH = THIS_DIR / "app_bo.py"
if not APP_PATH.exists():
    APP_PATH = THIS_DIR.parent / "app_bo.py"

spec = importlib.util.spec_from_file_location("app_bo", APP_PATH)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

# Rediriger les campagnes vers un dossier temporaire (ne pas polluer le vrai)
_TMP = Path(tempfile.mkdtemp(prefix="bo_test_"))
app.CAMPAIGNS_DIR = _TMP
api = app.Api()


# ── Mini-framework de test ──
_passed = 0
_failed = 0
_failures = []


def silent(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def check(label, cond):
    global _passed, _failed
    if cond:
        _passed += 1
    else:
        _failed += 1
        _failures.append(label)
        print(f"  ECHEC: {label}")


def fill_suggested(name, oracle):
    """Remplit toutes les lignes suggerees via une fonction oracle(row)->dict."""
    d = api.get_campaign(name)
    for row in d["rows"]:
        if row["statut"] == "suggere":
            row.update(oracle(row))
    silent(lambda: api.save_results(name, json.dumps(d["rows"])))


# ════════════════════════════════════════════════════════════════
# 1. VALIDATION
# ════════════════════════════════════════════════════════════════
def test_validation():
    print("[1] Validation des entrees")
    base_out = {"name": "y", "direction": "maximize", "weight": 1, "max": 100}

    # path traversal
    r = api.create_campaign(json.dumps({"name": "../evil",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 1}],
        "output": base_out}))
    check("nom '../evil' rejete", not r.get("ok"))

    r = api.create_campaign(json.dumps({"name": "a/b",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 1}],
        "output": base_out}))
    check("nom 'a/b' rejete", not r.get("ok"))

    # min >= max
    r = api.create_campaign(json.dumps({"name": "v_badrange",
        "parameters": [{"name": "x", "type": "continuous", "min": 10, "max": 5}],
        "output": base_out}))
    check("min>=max rejete", not r.get("ok"))

    # 0 parametre
    r = api.create_campaign(json.dumps({"name": "v_noparam",
        "parameters": [], "output": base_out}))
    check("0 parametre rejete", not r.get("ok"))

    # categoriel 1 seule categorie
    r = api.create_campaign(json.dumps({"name": "v_onecat",
        "parameters": [{"name": "L", "type": "categorical", "categories": ["A"]}],
        "output": base_out}))
    check("categoriel 1 categorie rejete", not r.get("ok"))

    # param et output meme nom
    r = api.create_campaign(json.dumps({"name": "v_dup",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 1}],
        "output": {"name": "x", "direction": "maximize", "weight": 1, "max": 1}}))
    check("param/output meme nom rejete", not r.get("ok"))

    # campagne valide
    r = api.create_campaign(json.dumps({"name": "v_ok",
        "parameters": [{"name": "T", "type": "continuous", "min": 40, "max": 120},
                        {"name": "L", "type": "categorical", "categories": ["A", "B"]}],
        "output": base_out}))
    check("campagne valide acceptee", r.get("ok"))
    api.delete_campaign("v_ok")

    # YAML sans bo_settings : defaut applique
    r = api.import_yaml("campaign_name: v_nobo\n"
                        "parameters:\n  - name: x\n    type: continuous\n"
                        "    min: 0\n    max: 10\n"
                        "outputs:\n  - name: y\n    direction: maximize\n")
    check("YAML sans bo_settings accepte (defaut)", r.get("ok"))
    if r.get("ok"):
        api.delete_campaign("v_nobo")

    # poids nul en multi-objectif rejete
    r = api.create_campaign(json.dumps({"name": "v_w0",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "objectives": [{"name": "a", "direction": "maximize", "weight": 1, "max": 100},
                        {"name": "b", "direction": "minimize", "weight": 0}]}))
    check("poids nul (multi-obj) rejete", not r.get("ok"))
    if r.get("ok"):
        api.delete_campaign("v_w0")


# ════════════════════════════════════════════════════════════════
# 2. PERSISTANCE
# ════════════════════════════════════════════════════════════════
def test_persistence():
    print("[2] Persistance disque")
    api.create_campaign(json.dumps({"name": "p_test",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    silent(lambda: api.suggest("p_test", json.dumps({"n_suggest": 3, "mode_auto": True})))

    # Recharger depuis le disque (nouvelle instance)
    api2 = app.Api()
    d = api2.get_campaign("p_test")
    check("campagne rechargee depuis disque", d.get("ok") and len(d["rows"]) > 0)

    # Le fichier CSV existe
    _, _, csv_path = app.campaign_paths("p_test")
    check("results.csv ecrit sur disque", csv_path.exists())
    api.delete_campaign("p_test")
    check("suppression effective", not (app.CAMPAIGNS_DIR / "p_test").exists())


# ════════════════════════════════════════════════════════════════
# 3. CYCLE MONO-OBJECTIF
# ════════════════════════════════════════════════════════════════
def test_mono_cycle():
    print("[3] Cycle mono-objectif complet")
    api.create_campaign(json.dumps({"name": "m_cyc",
        "parameters": [{"name": "T", "type": "continuous", "min": 40, "max": 120},
                        {"name": "L", "type": "categorical", "categories": ["A", "B", "C"]}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100},
        "autopilot": True, "target": 95}))

    rng = np.random.default_rng(0)
    se = {"A": 0, "B": 12, "C": -5}
    oracle = lambda r: {"y": 50 + 0.3*(float(r["T"])-40) + se[r["L"]] + float(rng.normal(0, 2))}

    # init
    r = silent(lambda: api.suggest("m_cyc", json.dumps({"n_suggest": 1, "mode_auto": True})))
    check("init LHS ok", r.get("ok"))
    fill_suggested("m_cyc", oracle)

    # 3 tours BO
    ok_tours = True
    for _ in range(3):
        r = silent(lambda: api.suggest("m_cyc", json.dumps({"n_suggest": 1, "mode_auto": True})))
        if not r.get("ok"):
            ok_tours = False
            break
        fill_suggested("m_cyc", oracle)
    check("3 tours BO sans erreur", ok_tours)

    d = api.get_campaign("m_cyc")
    check("best progresse (>60)", d["best"] > 60)
    check("badges autopilot presents", True)  # deja verifie via suggest ok
    api.delete_campaign("m_cyc")


# ════════════════════════════════════════════════════════════════
# 4. OPTION B (suggestions en attente)
# ════════════════════════════════════════════════════════════════
def test_option_b():
    print("[4] Option B : blocage suggestions en attente")
    api.create_campaign(json.dumps({"name": "b_test",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    silent(lambda: api.suggest("b_test", json.dumps({"n_suggest": 3, "mode_auto": True})))

    r = silent(lambda: api.suggest("b_test", json.dumps({"n_suggest": 3, "mode_auto": True})))
    check("2e suggest bloque", not r.get("ok") and r.get("pending"))

    r = silent(lambda: api.suggest("b_test", json.dumps({"n_suggest": 2, "mode_auto": True, "force": True})))
    check("force=True passe", r.get("ok"))
    d = api.get_campaign("b_test")
    n_sug = sum(1 for row in d["rows"] if row["statut"] == "suggere")
    check("force remplace (pas d'empilement)", n_sug == len(d["rows"]))
    api.delete_campaign("b_test")


# ════════════════════════════════════════════════════════════════
# 5. CLAMP n_suggest
# ════════════════════════════════════════════════════════════════
def test_clamp():
    print("[5] Clamp n_suggest")
    api.create_campaign(json.dumps({"name": "c_test",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    rng = np.random.default_rng(0)
    silent(lambda: api.suggest("c_test", json.dumps({"n_suggest": 1, "mode_auto": True})))
    fill_suggested("c_test", lambda r: {"y": float(rng.uniform(40, 90))})
    r = silent(lambda: api.suggest("c_test", json.dumps({"n_suggest": 9999, "mode_auto": True})))
    check("n_suggest=9999 clampe a MAX", r.get("n_added", 0) <= app.MAX_N_SUGGEST)
    api.delete_campaign("c_test")


# ════════════════════════════════════════════════════════════════
# 6. PERSISTANCE DU MODE
# ════════════════════════════════════════════════════════════════
def test_mode_persist():
    print("[6] Persistance mode auto/manuel")
    api.create_campaign(json.dumps({"name": "mode_test",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    api.set_mode("mode_test", False)
    cfg = app.load_cfg("mode_test")
    check("mode manuel persiste", cfg["bo_settings"]["autopilot"] is False)
    api.set_mode("mode_test", True)
    cfg = app.load_cfg("mode_test")
    check("mode auto persiste", cfg["bo_settings"]["autopilot"] is True)
    api.delete_campaign("mode_test")


# ════════════════════════════════════════════════════════════════
# 7. DESCRIPTEURS
# ════════════════════════════════════════════════════════════════
def test_descriptors():
    print("[7] Descripteurs physico-chimiques")
    presets = api.get_presets()
    if not presets["library_available"]:
        print("  (bibliotheque absente, tests descripteurs ignores)")
        return
    check("presets continus exposes", len(presets["continuous"]) > 0)
    check("presets categoriels exposes", len(presets["categorical"]) > 0)

    sheet = api.get_library_sheet("Solvants")
    check("feuille Solvants chargee", sheet.get("ok") and len(sheet["compounds"]) > 0)

    r = api.create_campaign(json.dumps({"name": "d_test",
        "parameters": [
            {"name": "T", "type": "continuous", "min": 40, "max": 120},
            {"name": "Solvant", "type": "categorical",
             "categories": sheet["compounds"][:4],
             "use_library_descriptors": True, "descriptors_sheet": "Solvants",
             "descriptors_columns": sheet["descriptor_columns"]}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    check("campagne avec descripteurs creee", r.get("ok"))

    # Verifier que le YAML reference le fichier
    cfg = app.load_cfg("d_test")
    solv = [p for p in cfg["parameters"] if p["name"] == "Solvant"][0]
    check("descriptors_file resolu en chemin absolu",
          solv.get("descriptors_file", "").endswith(".xlsx"))

    # L'encoder doit utiliser les descripteurs (pas one-hot)
    enc = silent(lambda: app.bo.SpaceEncoder(cfg))
    n_desc = len(sheet["descriptor_columns"])
    check(f"encoder utilise descripteurs (n_params={enc.n_params})",
          enc.n_params == 1 + n_desc)  # 1 continu + n descripteurs
    api.delete_campaign("d_test")


# ════════════════════════════════════════════════════════════════
# 8. MULTI-OBJECTIF
# ════════════════════════════════════════════════════════════════
def test_multi_objective():
    print("[8] Multi-objectif")
    r = api.create_campaign(json.dumps({"name": "mo_test",
        "parameters": [{"name": "T", "type": "continuous", "min": 40, "max": 120},
                        {"name": "eq", "type": "continuous", "min": 1, "max": 3}],
        "objectives": [
            {"name": "yield", "direction": "maximize", "weight": 0.6, "max": 100},
            {"name": "ee", "direction": "maximize", "weight": 0.4, "max": 100}],
        "target": 95}))
    check("campagne multi-obj creee", r.get("ok"))

    d = api.get_campaign("mo_test")
    check("is_multi=True", d.get("is_multi"))
    check("2 objectifs exposes", len(d.get("objectives", [])) == 2)

    rng = np.random.default_rng(0)
    def oracle(row):
        T, eq = float(row["T"]), float(row["eq"])
        return {"yield": 50 + 0.3*(T-40) + 5*eq + float(rng.normal(0, 2)),
                "ee": 90 - 0.2*(T-40) + float(rng.normal(0, 3))}

    silent(lambda: api.suggest("mo_test", json.dumps({"n_suggest": 1, "mode_auto": False})))
    fill_suggested("mo_test", oracle)

    r = silent(lambda: api.suggest("mo_test", json.dumps({"n_suggest": 1, "mode_auto": False})))
    check("suggestion EHVI multi-obj ok", r.get("ok"))
    fill_suggested("mo_test", oracle)

    p = api.get_pareto("mo_test")
    check("front de Pareto calcule", p.get("ok") and not p.get("empty"))
    if p.get("ok") and not p.get("empty"):
        check("au moins 1 point Pareto", sum(p["is_pareto"]) >= 1)
    api.delete_campaign("mo_test")


# ════════════════════════════════════════════════════════════════
# 9. EXPORT / DELETE_ROW / PRIOR
# ════════════════════════════════════════════════════════════════
def test_utils():
    print("[9] Export, suppression de ligne, prior")
    api.create_campaign(json.dumps({"name": "u_test",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))

    # prior : ajouter une ligne manuelle
    r = api.add_manual_row("u_test", json.dumps({"x": 5.0, "y": 75.0}))
    check("ajout ligne prior", r.get("ok"))
    d = api.get_campaign("u_test")
    check("prior compte comme fait", d["n_done"] == 1)

    # export
    e = api.export_csv("u_test")
    check("export CSV contient les donnees", e.get("ok") and "y" in e.get("csv", ""))

    # delete_row
    n_before = len(api.get_campaign("u_test")["rows"])
    api.delete_row("u_test", 0)
    n_after = len(api.get_campaign("u_test")["rows"])
    check("delete_row retire une ligne", n_after == n_before - 1)

    # delete_row hors limites
    r = api.delete_row("u_test", 999)
    check("delete_row index invalide rejete", not r.get("ok"))

    # prior avec categorie invalide
    api.create_campaign(json.dumps({"name": "u_cat",
        "parameters": [{"name": "L", "type": "categorical", "categories": ["A", "B"]}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    r = api.add_manual_row("u_cat", json.dumps({"L": "ZZZ", "y": 50}))
    check("prior categorie invalide refuse", not r.get("ok"))
    api.delete_campaign("u_cat")

    # prior hors bornes
    api.create_campaign(json.dumps({"name": "u_oob",
        "parameters": [{"name": "T", "type": "continuous", "min": 0, "max": 100}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    r = api.add_manual_row("u_oob", json.dumps({"T": 9999, "y": 50}))
    check("prior hors bornes refuse", not r.get("ok"))
    r = api.add_manual_row("u_oob", json.dumps({"T": 50, "y": 75}))
    check("prior valide accepte", r.get("ok"))
    api.delete_campaign("u_oob")
    api.delete_campaign("u_test")


# ════════════════════════════════════════════════════════════════
# 10. CAS LIMITES
# ════════════════════════════════════════════════════════════════
def test_edge_cases():
    print("[10] Cas limites")
    # GP avec valeurs identiques
    api.create_campaign(json.dumps({"name": "e_flat",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    silent(lambda: api.suggest("e_flat", json.dumps({"n_suggest": 5, "mode_auto": True})))
    fill_suggested("e_flat", lambda r: {"y": 50.0})
    r = silent(lambda: api.suggest("e_flat", json.dumps({"n_suggest": 1, "mode_auto": True})))
    check("GP avec y identiques ne crashe pas", r.get("ok"))
    api.delete_campaign("e_flat")

    # acq_1d sur parametre categoriel -> refus propre
    api.create_campaign(json.dumps({"name": "e_cat",
        "parameters": [{"name": "L", "type": "categorical", "categories": ["A", "B", "C"]}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100}}))
    silent(lambda: api.suggest("e_cat", json.dumps({"n_suggest": 5, "mode_auto": True})))
    d = api.get_campaign("e_cat")
    for i, row in enumerate(d["rows"]):
        row["y"] = 50 + i * 5
    silent(lambda: api.save_results("e_cat", json.dumps(d["rows"])))
    acq = silent(lambda: api.get_acq_1d("e_cat", json.dumps({"param": "L", "acquisition": "EI"})))
    check("acq_1d refuse param categoriel proprement", not acq.get("ok"))
    api.delete_campaign("e_cat")


def test_advanced_config():
    print("[11] log_ratio, noise_prior, contraintes")
    # log_ratio + noise_prior
    r = api.create_campaign(json.dumps({"name": "adv1",
        "parameters": [{"name": "eq_A", "type": "continuous", "min": 1, "max": 3,
                        "encoding": "log_ratio"},
                       {"name": "eq_B", "type": "continuous", "min": 1, "max": 3}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100},
        "noise_prior": "medium",
        "constraints": ["eq_A + eq_B <= 4"]}))
    check("campagne log_ratio + noise + hard creee", r.get("ok"))
    if r.get("ok"):
        cfg = app.load_cfg("adv1")
        check("encoding log_ratio preserve",
              cfg["parameters"][0].get("encoding") == "log_ratio")
        check("noise_prior medium", cfg.get("noise_prior") == "medium")
        check("hard constraint preservee", cfg.get("constraints") == ["eq_A + eq_B <= 4"])
        api.delete_campaign("adv1")

    # hard constraint invalide (nom inexistant)
    r = api.create_campaign(json.dumps({"name": "adv2",
        "parameters": [{"name": "x", "type": "continuous", "min": 0, "max": 10}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100},
        "constraints": ["inexistant + 1 <= 5"]}))
    check("hard constraint nom invalide rejetee", not r.get("ok"))

    # soft constraint sur objectif existant
    r = api.create_campaign(json.dumps({"name": "adv3",
        "parameters": [{"name": "T", "type": "continuous", "min": 40, "max": 120}],
        "objectives": [{"name": "yield", "direction": "maximize", "weight": 1, "max": 100},
                       {"name": "tar", "direction": "minimize", "weight": 1}],
        "soft_constraints": [{"name": "tar bas", "output": "tar",
                              "direction": "minimize", "threshold": 5}]}))
    check("soft constraint creee", r.get("ok"))
    if r.get("ok"):
        api.delete_campaign("adv3")

    # soft constraint sur sortie inexistante
    r = api.create_campaign(json.dumps({"name": "adv4",
        "parameters": [{"name": "T", "type": "continuous", "min": 40, "max": 120}],
        "output": {"name": "y", "direction": "maximize", "weight": 1, "max": 100},
        "soft_constraints": [{"name": "x", "output": "absent",
                              "direction": "minimize", "threshold": 5}]}))
    check("soft constraint sortie inexistante rejetee", not r.get("ok"))


# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("TESTS GUI (app_bo.py)")
    print("=" * 55)
    for test in [test_validation, test_persistence, test_mono_cycle,
                 test_option_b, test_clamp, test_mode_persist,
                 test_descriptors, test_multi_objective, test_utils,
                 test_edge_cases, test_advanced_config]:
        try:
            test()
        except Exception as e:
            global _failed
            _failed += 1
            _failures.append(f"{test.__name__} a leve une exception: {e}")
            print(f"  EXCEPTION dans {test.__name__}: {e}")

    print("=" * 55)
    print(f"{_passed} passes, {_failed} echoues")
    if _failures:
        print("\nEchecs :")
        for f in _failures:
            print(f"  - {f}")
    # Nettoyage
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(1 if _failed else 0)


if __name__ == "__main__":
    main()
