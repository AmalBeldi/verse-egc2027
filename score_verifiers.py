"""Compare, sur les annotations humaines, le vérifieur hiérarchique (V4), sa version
limitée au résumé (V3) et le LLM juge. Écrit paper/generated/tab_verif_humain.tex.

  python3 experiments/score_verifiers.py results/annotation_qwen2.5_7b-instruct.csv \
      results/llm_qwen2.5_7b-instruct.judge_llama3.1_8b.jsonl
"""
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.pipeline import prepare
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, decide
from ehrkg.concepts import generic_med_label

ann = [r for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")) if r["humain"].strip().upper() in ("OK", "KO")]
judge = {}
if len(sys.argv) > 2:
    for x in map(json.loads, open(sys.argv[2])):
        judge[(x["pid"], x["qid"], x["cond"])] = x["label"]
        jname = x["judge"]
REC = load_cohort()
X = Extractor(extra_meds=[m for m in sorted({generic_med_label(x) for r in REC.values() for x in r["medications"].DESCRIPTION}) if len(m) > 3])
pred = {"V3 (résumé seul)": [], "V4 (hiérarchique)": [], "LLM juge": []}
gold = []
for r in ann:
    P = prepare(REC[r["pid"]]); KS, KR = SummaryKB(P["rdf_star"]), RecordKB(P["G"])
    cl = X.extract(r["reponse"])
    gold.append(r["humain"].strip().upper() == "KO")
    for V, name in (("V3", "V3 (résumé seul)"), ("V4", "V4 (hiérarchique)")):
        pred[name].append(decide(verify(cl, KS, KR, V))[0] != "OK")
    j = judge.get((r["pid"], r["qid"], r["cond"]))
    pred["LLM juge"].append(None if j is None else j == "KO")

def metrics(g, p):
    pairs = [(a, b) for a, b in zip(g, p) if b is not None]
    if not pairs: return None
    g, p = map(np.array, zip(*pairs))
    tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum()); tn = int((~g & ~p).sum())
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    po = (tp + tn) / len(g); pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / len(g) ** 2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    return dict(n=len(g), P=100 * P, R=100 * R, F1=100 * F, kappa=kappa)

L = ["\\begin{tabular}{lrrrrr}", "\\hline", "Vérifieur & $n$ & Préc. KO & Rappel KO & F1 & $\\kappa$ \\\\", "\\hline"]
S = {}
for name, p in pred.items():
    m = metrics(gold, p)
    if m:
        S[name] = m
        L.append(f"{name} & {m['n']} & {m['P']:.1f} & {m['R']:.1f} & {m['F1']:.1f} & {m['kappa']:.2f} \\\\")
L += ["\\hline", "\\end{tabular}"]
print(f"{len(ann)} réponses annotées, {sum(gold)} erronées selon l'annotateur")
print("\n".join(L))
os.makedirs("paper/generated", exist_ok=True)
open("paper/generated/tab_verif_humain.tex", "w").write("\n".join(L))
json.dump(S, open("paper/generated/summary_verif_humain.json", "w"), indent=1)
