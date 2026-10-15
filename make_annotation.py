"""Prépare la feuille d'annotation humaine des réponses E5 (échantillon stratifié par condition).

  python3 experiments/make_annotation.py results/llm_qwen2.5_7b-instruct.jsonl --n 200
écrit results/annotation_<modele>.csv. L'annotateur remplit la colonne `humain` :
  OK = toutes les affirmations de la réponse sont vraies d'après la référence ;
  KO = au moins une affirmation est fausse ou inventée.
Les étiquettes des vérifieurs ne figurent PAS dans la feuille (annotation à l'aveugle).
"""
import argparse, csv, json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.pipeline import prepare
from ehrkg import context as C

ap = argparse.ArgumentParser()
ap.add_argument("answers"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--csv", default="data/synthea_t2dm_cohort"); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.answers)]
rows = [r for r in rows if r["answer"] and "non disponible" not in r["answer"].lower()]
by = defaultdict(list)
for r in rows: by[r["cond"]].append(r)
rng = random.Random(a.seed); k = max(1, a.n // max(1, len(by)))
sample = [x for c in sorted(by) for x in rng.sample(by[c], min(k, len(by[c])))]
rng.shuffle(sample)
REC = load_cohort(a.csv)
base = os.path.basename(a.answers).replace(".jsonl", "")
base = base[4:] if base.startswith("llm_") else base
out = os.path.join(os.path.dirname(a.answers), f"annotation_{base}.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "pid", "qid", "cond", "question", "reponse", "reference_calculee", "faits_du_dossier", "humain", "commentaire"])
    for i, r in enumerate(sample):
        P = prepare(REC[r["pid"]])
        ctx, _ = C.ctx_kg_dg(P["dg_items"], r["question"], 800)
        w.writerow([i, r["pid"], r["qid"], r["cond"], r["question"], r["answer"],
                    json.dumps(r["gold"], ensure_ascii=False), C.to_text(ctx), "", ""])
print("écrit", out, len(sample), "réponses")
