PY ?= python3
MODEL ?= qwen2.5:7b-instruct
BASE_URL ?= http://localhost:11434/v1
N ?= 50

.PHONY: install test demo e1-e4 check e5 e5-dry paper all

install:          ## dépendances Python
	$(PY) -m pip install -r requirements.txt

test:             ## tests unitaires des algorithmes 1 et 2 et du score
	$(PY) -m unittest discover -s tests -v

demo:             ## trace de l algorithme 1 puis comparaison V1-V4 (algorithme 2)
	$(PY) scripts/demo_algo1.py --patient 38cbc124 --question "Quels sont les traitements antidiabétiques en cours ?" --answer "La patiente est traitée par metformine et sitagliptine. Elle présente par ailleurs une anémie." --save
	$(PY) scripts/demo_algo2.py --patient 0 --perturb Q4

e1-e4:            ## expériences sans LLM (~2 min) + tableaux LaTeX
	$(PY) experiments/run_symbolic.py
	$(PY) experiments/analyze.py results paper/generated

check: e1-e4      ## vérifie que les chiffres obtenus sont ceux de l article
	$(PY) experiments/check_results.py

e5-dry:           ## teste la chaîne E5 sans modèle (aucun tableau produit)
	$(PY) experiments/run_llm.py --dry-run --n-patients 2 --out out/dry.jsonl

e5:               ## E5 avec un LLM : make e5 MODEL=qwen2.5:7b-instruct N=50
	$(PY) experiments/run_llm.py --base-url $(BASE_URL) --model $(MODEL) --n-patients $(N)
	$(PY) experiments/score_llm.py results/llm_$(subst :,_,$(subst /,_,$(MODEL))).jsonl

paper:            ## compile la prévisualisation de l article
	cd paper && latexmk -pdf -interaction=nonstopmode article_preview.tex

all: test check paper

JUDGE ?= llama3.1:8b
ANS = results/llm_$(subst :,_,$(subst /,_,$(MODEL)))

judge:            ## LLM juge sur les réponses E5 : make judge MODEL=... JUDGE=...
	$(PY) experiments/judge_llm.py $(ANS).jsonl --base-url $(BASE_URL) --model $(JUDGE)

annotate:         ## feuille d annotation humaine (200 réponses, à l aveugle)
	$(PY) experiments/make_annotation.py $(ANS).jsonl --n 200

verif-humain:     ## V3 / V4 / LLM juge contre les annotations humaines
	$(PY) experiments/score_verifiers.py results/annotation_$(subst :,_,$(subst /,_,$(MODEL))).csv $(ANS).judge_$(subst :,_,$(subst /,_,$(JUDGE))).jsonl

MIMIC ?= ../mimic-iv-clinical-database-demo-2.2
mimic:            ## E1, E2, E4 sur MIMIC-IV demo : make mimic MIMIC=<dossier téléchargé>
	$(PY) scripts/mimic_to_synthea.py --mimic $(MIMIC) --out data/mimic_demo_t2dm
	$(PY) experiments/run_symbolic.py --csv data/mimic_demo_t2dm --out results_mimic
	$(PY) experiments/analyze.py results_mimic paper/generated_mimic
