# BO Chemistry - Bayesian Optimization for Experimental Chemistry

Moteur d'optimisation bayésienne conçu pour les campagnes d'expériences en
chimie de synthèse (couplages, catalyse, photochimie), avec une interface
desktop pour piloter les campagnes sans ligne de commande.

L'objectif : suggérer les conditions expérimentales les plus prometteuses à
tester ensuite, à partir des résultats déjà obtenus, pour atteindre un
rendement (ou tout autre objectif) cible en un minimum d'expériences.

> Projet développé dans le cadre de travaux de recherche en chimie organique
> de synthèse. Le moteur d'optimisation est le cœur du projet ; l'interface
> desktop est une surcouche pour l'usage quotidien au laboratoire.

<!-- Remplacer par une vraie capture d'ecran de l'app -->
<!-- ![Interface](docs/screenshot.png) -->

## Ce que fait le projet

- **Optimisation bayésienne mono- et multi-objectif** sur des espaces de
  paramètres mixtes (variables continues comme température/temps, et
  catégorielles comme ligand/solvant/base).
- **Autopilot adaptatif** : orchestration automatique de la stratégie
  d'exploration/exploitation selon l'avancement de la campagne (phases
  EARLY / MID / LATE).
- **Descripteurs physico-chimiques** : encodage des variables catégorielles
  via des propriétés moléculaires (RDKit / xTB / Mordred) plutôt qu'un
  one-hot, permettant au modèle de généraliser entre composés similaires.
- **Contraintes expérimentales** : contraintes connues a priori sur les
  paramètres (somme d'équivalents bornée, ratio entre réactifs, exclusion
  d'une combinaison infaisable) et contraintes sur les résultats mesurés
  (limiter un sous-produit tout en l'optimisant). L'acquisition ne propose
  que des points faisables, et le plan initial les respecte également.
- **Échelle logarithmique** par variable, pour les grandeurs dont l'effet est
  log-linéaire (concentration, équivalents, charge catalytique, ratio).
- **Fiabilité des mesures réglable** : le prior de bruit du GP s'ajuste selon
  la précision analytique déclarée, ce qui évite que le modèle lisse un vrai
  pic en le prenant pour du bruit.
- **Plan initial dimension-aware** : le nombre de points du design LHS tient
  compte des variables continues et du nombre de niveaux catégoriels, avec un
  critère maximin pour la couverture.
- **Interface desktop** : création de campagnes, saisie des résultats,
  suggestions, visualisation de la convergence et de la fonction
  d'acquisition, récapitulatif du coût expérimental avant lancement, le tout
  hors-ligne.

## Stack technique

| Domaine | Outils |
|---|---|
| Modèle | Gaussian Process (GPyTorch / PyTorch), noyau Matérn ARD |
| Acquisitions | EI, UCB, PI, Thompson, TEI, EI-TS, EHVI / qNEHVI (multi-objectif) |
| Méthodes avancées | TuRBO (trust region), Box-Cox warping, prior mean injection, input warping Kumaraswamy |
| Contraintes | Contraintes connues a priori (acquisition contrainte) et contraintes sur sorties mesurées (un GP par contrainte) |
| Descripteurs | RDKit, xTB, Mordred, morfeus |
| Interface | pywebview (desktop natif), HTML/CSS/JS, Plotly.js |
| Data | NumPy, pandas, scikit-learn, SciPy |

## Aperçu technique du moteur

Le moteur (`bo_chemistry.py`) implémente une boucle d'optimisation bayésienne
complète :

1. **Encodage** d'un espace de paramètres mixtes (continu + catégoriel) vers
   un espace numérique normalisé, avec support optionnel de descripteurs
   physico-chimiques pour les catégoriels.
2. **Modèle de substitution** : Gaussian Process GPyTorch avec bruit appris,
   priors Gamma, noyau Matérn 5/2 ARD (ou noyau composite cont x cat). Les
   priors de bruit sont calibrés sur `y` standardisé : un prior réglé pour
   l'échelle brute conduit le GP à surestimer le bruit et à écraser les
   meilleurs points (voir changelog v3.9).
3. **Fonction d'acquisition** sélectionnée selon le contexte (autopilot) ou
   choisie manuellement, optimisée par recherche multi-start.
4. **Stratégies de robustesse** : détection de stagnation, de dérive,
   d'outliers ; trust region adaptative ; filtrage anti-duplication des
   batchs.

L'autopilot calcule à chaque itération un `progress` (best / target) et une
`coverage` (fraction de l'espace explorée), et bascule la stratégie
d'acquisition en conséquence.

### Contraintes

Deux familles sont gérées, selon la distinction établie par Hickman, Aldeghi,
Häse et Aspuru-Guzik (*Bayesian Optimization with Known Experimental and
Design Constraints for Chemistry Applications*, Digital Discovery 2022, 1,
732) :

- **Contraintes connues a priori** (`constraints` dans le YAML) : des
  expressions booléennes sur les paramètres, évaluées avant toute expérience.
  L'optimisation de la fonction d'acquisition est restreinte au domaine
  faisable, et le design initial est filtré de la même façon. Elles encodent
  les limites de l'équipement ou du procédé, et plus largement la
  connaissance experte du chimiste.
- **Contraintes sur sorties mesurées** (`soft_constraints`) : un GP par
  contrainte modélise la probabilité de satisfaction, pondérant l'acquisition
  (approche Gardner / Gelbart). Utile quand la quantité contrainte n'est
  connue qu'après l'expérience, par exemple un sous-produit à maintenir sous
  un seuil.

## Validation

Le moteur a été validé sur des jeux de données HTE publiés (Reizman-Jensen
Suzuki, Doyle Buchwald-Hartwig) en émulation, ainsi que sur des fonctions de
benchmark synthétiques. Sur le couplage Doyle (4 dimensions catégorielles),
l'optimiseur avec descripteurs RDKit atteint ~96 % de rendement en 30
expériences, contre ~88 % pour un échantillonnage aléatoire.

> Note : les datasets ne sont pas redistribués dans ce dépôt. Voir
> `scripts/download_datasets.py` pour les récupérer depuis leurs sources.

## Installation

```bash
git clone <url-du-repo>
cd bo-chemistry
pip install -r requirements.txt

# Plotly.js (interface) - telecharge la librairie de graphes
python scripts/setup_assets.py
```

## Lancement

**Interface desktop :**
```bash
python app_bo.py
```

**Moteur en ligne de commande :**
```bash
python bo_chemistry.py ma_campagne.yaml
```

## Structure du dépôt

```
.
├── bo_chemistry.py          # moteur d'optimisation bayesienne
├── app_bo.py                # backend de l'interface desktop
├── static/
│   └── index.html           # interface (HTML/CSS/JS)
├── tests/                   # suite de tests (47 tests)
├── scripts/                 # utilitaires (descripteurs, setup)
├── docs/                    # documentation et guide
└── requirements.txt
```

## Tests

```bash
cd tests
python run_tests.py
```

## Limites connues

- Le bénéfice de l'optimisation bayésienne dépend du problème : sur des
  espaces petits ou des réactions à fort gradient évident, un plan
  d'expériences classique peut suffire. L'approche est la plus utile sur des
  espaces mixtes avec interactions non triviales et un coût expérimental élevé.
- Les descripteurs RDKit seuls capturent une partie du signal ; les
  descripteurs électroniques (xTB) apportent un gain mesurable mais
  nécessitent une installation plus lourde.
- Le moteur est un fichier monolithique conséquent ; une refonte modulaire
  est envisagée (voir `docs/architecture.md`).

## Licence

MIT - voir [LICENSE](LICENSE).
