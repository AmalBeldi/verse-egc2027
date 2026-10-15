# Synthèse neuro-symbolique de DME — code et expériences (EGC 2027)

**Démarrage rapide :** `make install && make test && make demo && make check` — le protocole complet, étape par étape, est dans [PROTOCOLE.md](PROTOCOLE.md).

Pipeline : Synthea CSV → `BuildDG` → `DGsumm` (+ profil UP) → `RDF-GraphSyn` → `HERSE` → `BuildContext` (+ repli DG) → LLM → `VerifyAndExplain` (3 classes, hiérarchique).

## Dépendances
Python ≥ 3.10 : pandas, numpy, networkx, matplotlib, requests. Java ≥ 17 pour Synthea. Aucun téléchargement de modèle n'est nécessaire pour E1–E4.

## 1. Données (reproductibles)
La cohorte utilisée dans l'article (172 patients, tables filtrées) est fournie dans `data/synthea_t2dm_cohort/` ; toutes les commandes l'utilisent par défaut. Pour la régénérer :
```bash
curl -L -o synthea.jar https://github.com/synthetichealth/synthea/releases/download/master-branch-latest/synthea-with-dependencies.jar
java -Xmx5g -jar synthea.jar -s 2027 -cs 2027 -p 1500 -a 30-85 \
  --exporter.csv.export=true --exporter.fhir.export=false \
  --exporter.hospital.fhir.export=false --exporter.practitioner.fhir.export=false \
  --exporter.baseDirectory=./out Massachusetts
```
Remarque : la branche `master` de Synthea évolue. Notez le hash de version (`java -jar synthea.jar --version` ou la date de la release) et publiez le CSV utilisé avec le code pour garantir la reproductibilité exacte.

## 2. Expériences sans LLM (E1–E4, ~2 min)
```bash
python3 experiments/run_symbolic.py                     # écrit results/E*.json (--csv pour une autre cohorte)
python3 experiments/analyze.py results paper/generated  # tableaux LaTeX + summary.json
python3 tests_scorer.py                                 # test du score d'exactitude
```

## 3. Expérience E5 avec un LLM
Toute API compatible OpenAI convient (Ollama, vLLM, llama.cpp, LM Studio, API hébergée).
```bash
ollama pull qwen2.5:7b-instruct
python3 experiments/run_llm.py --base-url http://localhost:11434/v1 \
   --model qwen2.5:7b-instruct --n-patients 50
python3 experiments/score_llm.py results/llm_qwen2.5_7b-instruct.jsonl
```
50 patients × 13 questions × 7 conditions ≈ 4 500 appels (reprise automatique en cas d'interruption ; `--n-patients 20` pour un premier essai). `score_llm.py` écrit `paper/generated/tab_llm.tex`, inséré automatiquement dans l'article. `--dry-run` teste la plomberie sans modèle et ne produit jamais de tableau.

## Organisation
- `ehrkg/data.py` chargement Synthea, identifiants de faits stables
- `ehrkg/questions.py` questions et réponses de référence calculées sur les tables brutes (indépendant du pipeline)
- `ehrkg/dg.py`, `dgsumm.py`, `rdf.py` (RDF-GraphSyn + HERSE), `context.py` (conditions, BM25, Pack, repli)
- `ehrkg/verify.py` extraction d'assertions, bases de connaissances résumé / dossier, vérifieurs V1–V4
- `ehrkg/perturb.py` injection d'hallucinations P1–P6
- `scripts/demo_algo1.py`, `scripts/demo_algo2.py` traces pas à pas des algorithmes 1 et 2
- `tests/` tests unitaires (algorithmes, score E5) ; `experiments/check_results.py` compare une exécution aux chiffres de l'article
- `paper/sections/` une source LaTeX par section ; `paper/generated/` tableaux produits par le code
- `paper/` article (`article_preview.tex` compile localement ; `article_rnti.tex` pour la classe RNTI)

DGsumm et HERSE sont ici des réimplémentations, pour la tâche T2DM, des principes publiés ; remplacer par les implémentations de la thèse si besoin.

## Nouveautés (version EGC 2027)

- **Baseline PPR à la HippoRAG** (`KG-PPR`) : PageRank personnalisé sur le graphe complet, amorcé par les concepts de la question. Elle est incluse dans E2 et E5.
- **Groupes de questions** : factuelles (preuve brute), agrégation (preuve brute ou indicateur calculé), hors périmètre (`E2_groups` dans `paper/generated/summary.json`).
- **E5** : exactitude par groupe, et accord automatique de V4 avec le score déterministe sur toutes les réponses.
- **Données réelles (optionnel)** : téléchargez MIMIC-IV Clinical Database Demo 2.2 (accès libre, https://physionet.org/content/mimic-iv-demo/2.2/), puis lancez `make mimic MIMIC=<dossier>`.
