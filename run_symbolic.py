"""Experiments that do not need a language model.

E1  sizes and running time of each symbolic step (DG, GS, G_RDF, G*_RDF)
E2  evidence sufficiency of the context vs token budget, per condition and question
E3  profile-oriented consultation summary: key-fact recall with / without UP
E4  graph-guided verifier on injected hallucinations (V1..V4), thresholds tuned
    on a validation split of patients, reported on the test split
Usage: python3 experiments/run_symbolic.py --csv <synthea csv dir>
"""
import argparse, json, os, random, sys, statistics as st, time
from collections import defaultdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.questions import build_questions, profile_keys
from ehrkg.pipeline import prepare, contexts, CONDITIONS
from ehrkg import context as C, rdf as R
from ehrkg.verify import Extractor, SummaryKB, RecordKB, verify, decide, SUP, CON, UNV
from ehrkg.perturb import perturbations
from ehrkg.concepts import med_ingredient, generic_med_label

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--out", default="results")
ap.add_argument("--budgets", default="250,500,1000,2000,4000")
ap.add_argument("--seed", type=int, default=2027)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
budgets = [int(x) for x in a.budgets.split(",")]
REC = load_cohort(a.csv)
pids = sorted(REC)
rng = random.Random(a.seed)
val = set(rng.sample(pids, len(pids) // 2))

E1 = defaultdict(list); E2 = []; E3 = []; E4 = []
extra = sorted({generic_med_label(x) for r in REC.values() for x in r["medications"].DESCRIPTION})
X = Extractor(extra_meds=[m for m in extra if len(m) > 3])
t0 = time.time()
for k, pid in enumerate(pids):
    r = REC[pid]
    P = prepare(r, "generaliste")
    G, GS = P["G"], P["GS"]
    # ---- E1
    E1["DG_nodes"].append(G.number_of_nodes()); E1["DG_edges"].append(G.number_of_edges())
    E1["GS_nodes"].append(GS.number_of_nodes()); E1["GS_edges"].append(GS.number_of_edges())
    s1, s2 = R.stats(P["rdf"]), R.stats(P["rdf_star"])
    E1["RDF_triples"].append(s1["triples"]); E1["RDF_blank"].append(s1["blank"])
    E1["RDFstar_triples"].append(s2["triples"]); E1["RDFstar_blank"].append(s2["blank"])
    E1["text_tokens"].append(sum(x["ntok"] for x in P["notes"]))
    E1["dg_lin_tokens"].append(sum(x["ntok"] for x in P["dg_items"]))
    E1["gs_lin_tokens"].append(sum(x["ntok"] for x in P["items"]))
    E1["gs_lin_tokens_noherse"].append(sum(x["ntok_raw"] for x in P["items"]))
    for kk, v in P["times"].items():
        E1["t_" + kk].append(v)
    # ---- E2
    Q = build_questions(r)
    for q in Q:
        if not q["evidence"]:
            continue
        for B in budgets:
            for cond, (ctx, used) in contexts(P, q["question"], B).items():
                cov = C.covered(ctx)
                suf, rec = C.sufficient(q["evidence"], cov)
                sraw, _ = C.sufficient(C.raw_evidence(q["evidence"]), cov)
                E2.append(dict(pid=pid, qid=q["qid"], cond=cond, budget=B, sufficient=suf, recall=rec,
                               sufficient_raw=sraw, used=used))
    # ---- E3 (consultation summary, two profiles)
    keys = profile_keys(r)
    for prof in ("generaliste", "endocrinologue"):
        Pp = P if prof == "generaliste" else prepare(r, prof)
        for B in (150, 300, 600):
            ctxs = contexts(Pp, None, B, conds=["TEXT-FULL", "TEXT-RAG", "KG-DG", "KG-PPR", "OURS", "OURS-noUP"])
            for cond, (ctx, used) in ctxs.items():
                cov = C.covered(ctx)
                for target in ("generaliste", "endocrinologue"):
                    kset = keys[target]
                    if kset:
                        E3.append(dict(pid=pid, profile=prof, target=target, cond=cond, budget=B,
                                       recall=len(kset & cov) / len(kset), used=used))
    # ---- E4 (verifier)
    KS, KR = SummaryKB(P["rdf_star"]), RecordKB(G)
    for q in Q:
        if q["qid"].startswith("QX"):
            continue
        for ptype, text in perturbations(q, r, rng):
            claims = X.extract(text)
            row = dict(pid=pid, split="val" if pid in val else "test", qid=q["qid"], ptype=ptype,
                       text=text, n_claims=len(claims))
            for V in ("V1", "V2", "V3", "V4"):
                res = verify(claims, KS, KR, V)
                row[V] = [lab for _, lab, _ in res]
                row[V + "_gexp"] = sum(len(w) for _, _, w in res)
            E4.append(row)
    if k % 20 == 0:
        print(f"{k+1}/{len(pids)} patients  {time.time()-t0:.0f}s", flush=True)

json.dump(dict(E1=E1), open(f"{a.out}/E1.json", "w"))
json.dump(E2, open(f"{a.out}/E2.json", "w"))
json.dump(E3, open(f"{a.out}/E3.json", "w"))
json.dump(E4, open(f"{a.out}/E4.json", "w"), ensure_ascii=False)
print("done", time.time() - t0)
