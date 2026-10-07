# Campagne fictive pour tester l'application

Ce document te donne **deux campagnes prêtes à tester**, avec un « oracle »
caché qui joue le rôle du laboratoire : quand l'app te suggère des conditions,
tu cherches la valeur correspondante ici et tu la saisis. Ça te permet de faire
tourner une vraie boucle d'optimisation sans manip réelle.

L'oracle simule une **réaction de couplage de Suzuki–Miyaura** avec un optimum
réaliste (température modérée-haute, ligand riche et encombré, solvant polaire
aprotique). Il y a du bruit (±2-3 %) comme dans une vraie mesure.

---

## CAMPAGNE 1 — Mono-objectif avec descripteurs

**But** : maximiser le rendement. Tester l'autopilot + les descripteurs de solvant.

### Création (via le formulaire de l'app)

1. « Nouvelle campagne », nom : `suzuki_test`
2. **Paramètre 1** — choisir « Température » dans le menu :
   - min `40`, max `110`
3. **Paramètre 2** — « + variable continue », choisir « Loading catalyseur » :
   - min `1`, max `8`
4. **Paramètre 3** — « + variable catégorielle », choisir « Solvant (bibliothèque) » :
   - cocher uniquement : **DMF, THF, MeCN, 1,4-Dioxane, Toluène** (ou ce qui est dispo)
   - les descripteurs se rattachent automatiquement
5. **Objectif** : Rendement (%), maximize, max 100
6. Target autopilot : 95
7. Créer.

### L'oracle (rendement en %)

Formule cachée (ne triche pas trop) :

```
rendement = 30
          + 0.45 × (T − 40)          [température, optimum vers le haut]
          + 9 × ln(loading)           [loading, rendement décroissant marginal]
          + effet_solvant
          − pénalité_haute_temp       [si T > 95, dégradation]

effet_solvant :  DMF +14 | MeCN +10 | 1,4-Dioxane +6 | THF +4 | Toluène −6
pénalité :       si T > 95 : −(T − 95) × 1.2
plafond à 98 %, bruit ±2.5 %
```

**Quelques points pré-calculés** (arrondis, avec un bruit déjà inclus) pour t'aider
au début si tu ne veux pas calculer à la main :

| T (°C) | Loading | Solvant      | Rendement |
|--------|---------|--------------|-----------|
| 75     | 4       | DMF          | 71        |
| 60     | 2       | THF          | 48        |
| 90     | 6       | MeCN         | 77        |
| 100    | 5       | DMF          | 79        |
| 50     | 3       | Toluène      | 33        |
| 85     | 5       | 1,4-Dioxane  | 68        |
| 95     | 7       | DMF          | 88        |
| 105    | 6       | DMF          | 84        |
| 70     | 3       | MeCN         | 62        |
| 92     | 6       | DMF          | 85        |

Pour une condition non listée, estime avec la formule (ou invente une valeur
plausible cohérente avec la tendance — le but est de tester l'app, pas la
rigueur de l'oracle). **L'optimum réel est autour de T = 92-95, loading 6-7,
solvant DMF, soit ~88-90 %.**

### Déroulé du test

1. Clique « Suggérer » → l'app génère un plan initial LHS (plusieurs points).
2. Pour chaque ligne suggérée, calcule/lis le rendement et saisis-le.
3. « Sauvegarder les résultats ».
4. « Suggérer » à nouveau → cette fois le BO entraîne le GP et propose une
   condition. Regarde les badges (phase, progress) et le log.
5. Saisis le rendement, sauvegarde, recommence 5-6 fois.
6. Observe la **courbe de convergence** monter, et l'**acquisition 1D** sur
   la température.

Tu devrais voir le BO converger vers T élevé + DMF + loading haut.

---

## CAMPAGNE 2 — Multi-objectif (rendement + coût)

**But** : tester le mode multi-objectif et le front de Pareto. Ici deux
objectifs s'opposent : pousser le rendement coûte cher (loading élevé, ligand
précieux).

### Création

1. « Nouvelle campagne », nom : `suzuki_multiobj`
2. **Paramètre 1** — Température : min `40`, max `110`
3. **Paramètre 2** — Loading catalyseur : min `1`, max `8`
4. **Objectif 1** : Rendement (%), maximize, max 100, poids 0.6
5. « + objectif » → **Objectif 2** : choisir « Coût » dans la liste,
   minimize, poids 0.4 (laisse max vide)
6. Créer.

### Les oracles

```
rendement = 30 + 0.45×(T−40) + 9×ln(loading) − (T>95 ? (T−95)×1.2 : 0)
            plafond 98, bruit ±2.5

cout      = 5 + 1.8 × loading + 0.04 × T
            (le loading et la température font monter le coût)
```

**Points pré-calculés** :

| T  | Loading | Rendement | Coût  |
|----|---------|-----------|-------|
| 60 | 2       | 52        | 11.0  |
| 90 | 6       | 80        | 19.4  |
| 75 | 4       | 68        | 15.2  |
| 100| 7       | 84        | 21.6  |
| 50 | 2       | 44        | 10.6  |
| 95 | 5       | 80        | 17.8  |
| 80 | 3       | 64        | 13.6  |
| 105| 8       | 82        | 23.6  |

### Déroulé

Même boucle que la campagne 1, mais :
- Tu saisis **deux valeurs** par ligne (rendement ET coût).
- Le mode affiché est « EHVI / qNEHVI » (pas d'autopilot en multi-objectif).
- Coche le graphe **Front de Pareto** : tu verras les points non-dominés
  (compromis optimaux entre rendement et coût) en rouge, les autres en gris.

Le front de Pareto va montrer le compromis : pas moyen d'avoir le rendement
max ET le coût min en même temps.

---

## Ce que tu testes au passage

- Création via formulaire (continu preset + catégoriel à descripteurs)
- Plan initial LHS puis vraies suggestions BO
- Saisie + sauvegarde, persistance (ferme et rouvre l'app : tout est là)
- Badges autopilot + console moteur
- Option B : essaie de cliquer « Suggérer » deux fois sans remplir → message
- Ajout d'une ligne « prior », suppression d'une ligne, export CSV
- Multi-objectif + front de Pareto
- Courbe de convergence + acquisition 1D

Si quelque chose accroche (visuel, comportement), note-le.
