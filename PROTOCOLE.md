# Protocole expérimental — à reproduire en local

Ce document décrit, pour chaque algorithme de l'article, ce qui est testé, comment, avec quels paramètres, et ce que vous devez obtenir. Toutes les commandes se lancent depuis la racine du dépôt.

## 0. Installation (5 min)

```bash
git clone <URL_DU_DEPOT> && cd <depot>
python3 -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
make install                                            # ou : pip install -r requirements.txt
make test                                               # 16 tests, < 30 s
```
Attendu : `Ran 16 tests ... OK`. Si un test échoue, ne lancez pas les expériences : signalez le message d'erreur.

Données : la cohorte de l'article est fournie (`data/synthea_t2dm_cohort/`, 172 patients). Pour la régénérer, voir le README (Synthea, graines `-s 2027 -cs 2027`).

## 1. Algorithme 1 — le pipeline, étape par étape

**Objectif.** Vérifier que chaque étape produit ce que l'article décrit, sur un patient, avant de passer à la cohorte.

```bash
make demo
# équivalent à :
python3 scripts/demo_algo1.py --patient 38cbc124 \
   --question "Quels sont les traitements antidiabétiques en cours ?" \
   --answer "La patiente est traitée par metformine et sitagliptine. Elle présente par ailleurs une anémie." --save
```

| Étape | Ce qui est affiché | Valeur attendue pour ce patient | Test automatique |
|---|---|---|---|
| 1. BuildDG | nombre de nœuds et d'arcs par type | un nœud par ligne des tables, consultations chaînées par `NEXT` | `T1_BuildDG` |
| 2. DGsumm | séries, épisodes, problèmes, indicateurs, avec thème et provenance | 34 nœuds (DG : 2 075 nœuds) | `T2_DGsumm` (valeurs calculées = valeurs recalculées sur les tables brutes) |
| 3. RDF-GraphSyn / HERSE | triplets et nœuds anonymes avant / après | 678 → 613 triplets, 78 → 0 nœuds anonymes | `T3_RDF_HERSE` |
| 4. Link + BuildContext | concepts liés, score `rel × w_UP` de chaque élément, contexte retenu | 3 éléments, 81 jetons | `T4_BuildContext` (budget respecté, preuve présente, repli) |
| 5. QueryLLM | réponse du modèle (si `--base-url`) | — | — |
| 6. VerifyAndExplain | étiquette et témoins de chaque assertion, statut | metformine supportée, sitagliptine contredite, anémie supportée par le dossier → `KO` | `T5_VerifyAndExplain` |

`--save` écrit `out/<patient>_gstar.ttl` (G*_RDF en Turtle, lisible dans Protégé ou GraphDB) et `out/<patient>_context.txt`.

Variantes utiles :
```bash
python3 scripts/demo_algo1.py --patient 0 --question "Quelle est la dernière kaliémie mesurée ?"    # repli vers DG
python3 scripts/demo_algo1.py --patient 0 --question "..." --profile endocrinologue --budget 300      # effet du profil
python3 scripts/demo_algo1.py --patient 0 --question "..." --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct
```

## 2. Algorithme 2 — le vérifieur

**Hypothèses testées.**
- H1 : le vérifieur binaire (version atelier) rejette des réponses correctes.
- H2 : le monde ouvert laisse passer les médicaments et pathologies inventés.
- H3 : le monde fermé appliqué au seul résumé rejette des faits vrais que le résumé a écartés (circularité).
- H4 : la vérification hiérarchique (V4) détecte les erreurs sans ces fausses alertes.

```bash
python3 scripts/demo_algo2.py --patient 0 --perturb Q4     # P0, P1, P6 sur la question Q4, variantes V1 à V4
python3 scripts/demo_algo2.py --patient 38cbc124 --answer "..."
```
Attendu sur `--patient 0 --perturb Q4` : P1 (liraglutide inventé) → V2 `PARTIEL`, V1/V3/V4 `KO` ; P6 (gingivite, vraie mais hors résumé) → V1 `KO`, V2/V3 `PARTIEL`, V4 `OK`.

## 3. Expériences E1–E4 sur la cohorte (sans LLM, ~2 min)

```bash
make check        # = run_symbolic.py + analyze.py + check_results.py
```
Attendu : `248/248 valeurs conformes à l'article`. Les tableaux LaTeX sont réécrits dans `paper/generated/` et l'article les inclut directement.

| Exp. | Question posée | Unité d'analyse | Conditions | Métrique | Sortie |
|---|---|---|---|---|---|
| E1 | Quelle compression apportent DGsumm et HERSE, à quel coût ? | patient (172) | — | nœuds, triplets, nœuds anonymes, jetons ; temps médian par étape | `tab_sizes.tex` |
| E2 | Le contexte contient-il la preuve de la réponse ? | question (1 479 du périmètre, 486 hors périmètre) | 7 conditions × budgets 250, 500, 1 000, 2 000, 4 000 | % de contextes suffisants ; jetons utilisés | `tab_coverage.tex`, `tab_perq.tex` |
| E3 | Le profil oriente-t-il le résumé ? | patient × profil | 5 conditions × budgets 300, 600 | % des faits clés du profil présents | `tab_profile.tex` |
| E4 | Le vérifieur sépare-t-il réponses justes et erronées ? | réponse (P0–P6) ; patients répartis 50/50 validation/test | V1–V4 | détection par type, fausses alertes P0 et P6, F1, précision KO | `tab_verifier.tex` |

Paramètres fixés (tous dans le code, aucun ajusté sur le test) : seuil de pertinence τ = 0,5 ; tolérances HbA1c 0,15 point, DFG 1 mL/min, LDL 1 mg/dL, dates 30 jours ; tendance « stable » si |pente| ≤ 0,1 point/an ; seuils θ du vérifieur choisis sur la moitié validation dans {1 ; 0,9 ; 0,75 ; 0,67 ; 0,5} × {0,5 ; 0,34 ; 0,25 ; 0} ; graine 2027.

## 4. Expérience E5 — pipeline complet avec un LLM

**Hypothèses.**
- H5 : à budget égal, notre contexte donne une exactitude supérieure au texte intégral et au RAG textuel.
- H6 : le taux d'assertions non supportées et de réponses dangereuses est plus bas avec notre contexte.
- H7 : sans repli, les questions hors périmètre échouent ; sans HERSE, l'exactitude baisse à petit budget.
- H8 : parmi les réponses jugées `OK` par le vérifieur, la part d'erreurs (colonne OK→err) est faible.

**Mise en place (machine avec 8 Go de RAM au moins ; GPU conseillé).**
```bash
# 1. installer Ollama : https://ollama.com/download
ollama pull qwen2.5:7b-instruct          # modèle principal
ollama pull llama3.1:8b                  # second modèle (robustesse), facultatif
# 2. test de la chaîne sans modèle, puis premier essai court
make e5-dry
make e5 MODEL=qwen2.5:7b-instruct N=5    # ~455 appels, quelques minutes
# 3. campagne complète (≈ 4 500 appels ; interruptible, la reprise est automatique)
make e5 MODEL=qwen2.5:7b-instruct N=50
make e5 MODEL=llama3.1:8b N=50           # facultatif
```

**Paramètres.** 50 patients tirés dans la moitié test (graine 2028) ; 13 questions par patient ; 7 conditions ; budget 2 000 jetons (8 000 pour le texte intégral) ; température 0, graine 2027, 200 jetons de sortie max ; même prompt pour toutes les conditions (`ehrkg/prompt.py`).

**Métriques** (`experiments/score_llm.py`, règles déterministes testées par `tests/test_scorer.py`) : exactitude ; abstention ; part d'assertions non supportées (V4) ; réponses dangereuses (médicament ou pathologie contredits) ; réponses erronées jugées OK ; part de réponses entièrement couvertes par G_exp ; jetons du contexte.

**Statistiques.** Intervalle de confiance à 95 % de l'exactitude par bootstrap sur les patients (2 000 tirages) ; test de McNemar exact apparié entre notre approche et chaque autre condition (mêmes patients, mêmes questions).

**Sorties.** `paper/generated/tab_llm.tex` et `tab_llm_stats.tex`, inclus automatiquement dans la section Résultats ; `paper/generated/summary_llm.json`.

**À relire à la main** : tirez 50 réponses au hasard dans `results/llm_*.jsonl` et vérifiez que le score automatique est juste. Rapportez l'accord dans l'article (une ligne suffit).

## 5. Ce que la CI GitHub vérifie

Le workflow `.github/workflows/ci.yml` lance, à chaque push : `make test`, `make check` (les chiffres E1–E4 doivent correspondre à l'article) et `make e5-dry`. Un déclenchement manuel (onglet *Actions* → *reproduction* → *Run workflow*, case `llm_smoke`) exécute aussi E5 sur 3 patients avec `qwen2.5:0.5b` pour vérifier la chaîne de bout en bout ; ces chiffres ne sont pas ceux de l'article.

## 6. Publier le dépôt

```bash
gh repo create egc2027-ehr-neurosymbolique --public --source . --push   # ou créer le dépôt sur github.com puis git push
```
Mettez ensuite l'URL dans `paper/macros.tex` (`\depot`).
