"""Aggregate results/*.json into LaTeX tables, a figure and a summary JSON."""
import json, os, sys, statistics as st, itertools
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "paper/generated"
os.makedirs(OUT, exist_ok=True)
E1 = json.load(open(f"{R}/E1.json"))["E1"]
E2 = json.load(open(f"{R}/E2.json"))
E3 = json.load(open(f"{R}/E3.json"))
E4 = json.load(open(f"{R}/E4.json"))
S = {}

def med(x): return float(np.median(x))
def iqr(x): return np.percentile(x, 25), np.percentile(x, 75)

# ---------------- E1
rows = [("DG (n\\oe uds / arcs)", "DG_nodes", "DG_edges"), ("GS (n\\oe uds / arcs)", "GS_nodes", "GS_edges"),
        ("$G_{RDF}$ (triplets / n\\oe uds anonymes)", "RDF_triples", "RDF_blank"),
        ("$G^*_{RDF}$ (triplets / n\\oe uds anonymes)", "RDFstar_triples", "RDFstar_blank")]
S["E1"] = {k: dict(median=med(v), q1=iqr(v)[0], q3=iqr(v)[1], max=max(v)) for k, v in E1.items()}
L = ["\\begin{tabular}{lrr}", "\\hline", "Structure & médiane & [Q1 ; Q3] \\\\", "\\hline"]
for lab, a, b in rows:
    L.append(f"{lab} & {med(E1[a]):.0f} / {med(E1[b]):.0f} & [{iqr(E1[a])[0]:.0f} ; {iqr(E1[a])[1]:.0f}] / [{iqr(E1[b])[0]:.0f} ; {iqr(E1[b])[1]:.0f}] \\\\")
L.append("\\hline")
for lab, k in [("Notes concaténées (jetons)", "text_tokens"), ("Faits de DG linéarisés (jetons)", "dg_lin_tokens"),
               ("$G_{RDF}$ linéarisé (jetons)", "gs_lin_tokens_noherse"), ("$G^*_{RDF}$ linéarisé (jetons)", "gs_lin_tokens")]:
    L.append(f"{lab} & {med(E1[k]):.0f} & [{iqr(E1[k])[0]:.0f} ; {iqr(E1[k])[1]:.0f}] \\\\")
L += ["\\hline", "\\end{tabular}"]
open(f"{OUT}/tab_sizes.tex", "w").write("\n".join(L))
tsum = {k[2:]: 1000 * med(v) for k, v in E1.items() if k.startswith("t_")}
S["E1_time_ms"] = tsum

# ---------------- E2
NAMES = {"TEXT-FULL": "Texte intégral (tronqué)", "TEXT-RAG": "RAG textuel (BM25)", "KG-DG": "RAG sur DG (sans DGsumm)",
         "KG-PPR": "PPR sur DG (à la HippoRAG)", "OURS": "Notre approche", "OURS-GSonly": "-- sans repli DG", "OURS-noHERSE": "-- sans HERSE",
         "OURS-noUP": "-- sans UP"}
conds = ["TEXT-FULL", "TEXT-RAG", "KG-DG", "KG-PPR", "OURS", "OURS-GSonly", "OURS-noHERSE", "OURS-noUP"]
budgets = sorted({x["budget"] for x in E2})
agg = defaultdict(list); used = defaultdict(list)
for x in E2:
    sc = "hors" if x["qid"].startswith("QX") else "in"
    agg[(x["cond"], x["budget"], sc)].append(x["sufficient"])
    agg[(x["cond"], x["budget"], x["qid"])].append(x["sufficient"])
    used[(x["cond"], x["budget"], sc)].append(x["used"])
    grp = "hors" if sc == "hors" else ("agg" if x["qid"] in ("Q3", "Q10") else "fact")
    used[(x["cond"], x["budget"], "G:" + grp)].append(x["used"])
    agg[(x["cond"], x["budget"], "G:" + grp)].append(x["sufficient"])
    if grp == "agg":
        agg[(x["cond"], x["budget"], "G:aggraw")].append(bool(x.get("sufficient_raw")))
S["E2"] = {f"{c}@{b}@{sc}": 100 * np.mean(agg[(c, b, sc)]) for c in conds for b in budgets for sc in ("in", "hors")}
S["E2_used_median"] = {f"{c}@{b}@{sc}": med(used[(c, b, sc)]) for c in conds for b in budgets for sc in ("in", "hors")}
S["E2_groups"] = {f"{c}@{b}@{g}": 100 * np.mean(agg[(c, b, "G:" + g)]) for c in conds for b in budgets
                  for g in ("fact", "agg", "aggraw", "hors")}
S["E2_used_groups"] = {f"{c}@{b}@{g}": med(used[(c, b, "G:" + g)]) for c in conds for b in budgets
                       for g in ("fact", "agg", "hors")}
S["E2_n_items"] = {sc: len(agg[("OURS", budgets[0], sc)]) for sc in ("in", "hors")}
L = ["\\begin{tabular}{l" + "r" * len(budgets) + "rr}", "\\hline",
     "& \\multicolumn{" + str(len(budgets)) + "}{c}{Questions du périmètre (Q1--Q10)} & \\multicolumn{2}{c}{Hors périmètre} \\\\",
     "Condition & " + " & ".join(f"$B={b}$" for b in budgets) + " & $B=500$ & jetons \\\\", "\\hline"]
for c in conds:
    vals = [100 * np.mean(agg[(c, b, "in")]) for b in budgets]
    best = [max(100 * np.mean(agg[(cc, b, "in")]) for cc in conds) for b in budgets]
    cells = [(f"\\textbf{{{v:.1f}}}" if abs(v - bb) < 1e-9 else f"{v:.1f}") for v, bb in zip(vals, best)]
    cells.append(f"{100*np.mean(agg[(c, 500, 'hors')]):.1f}")
    cells.append(f"{med(used[(c, 500, 'in')]):.0f}")
    L.append(f"{NAMES[c]} & " + " & ".join(cells) + " \\\\")
L += ["\\hline", "\\end{tabular}"]
open(f"{OUT}/tab_coverage.tex", "w").write("\n".join(L))
# per question at B=500
qids = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "QX1", "QX2", "QX3", "QX4", "QX5", "QX6"]
B0 = 500
S["E2_perq"] = {f"{c}@{q}": 100 * np.mean(agg[(c, B0, q)]) for c in conds for q in qids if agg[(c, B0, q)]}
L = ["\\begin{tabular}{l" + "r" * len(qids) + "}", "\\hline", "Condition & " + " & ".join(qids) + " \\\\", "\\hline"]
for c in ["TEXT-FULL", "TEXT-RAG", "KG-DG", "KG-PPR", "OURS-GSonly", "OURS"]:
    L.append(f"{NAMES[c]} & " + " & ".join(f"{100*np.mean(agg[(c,B0,q)]):.0f}" for q in qids) + " \\\\")
L += ["\\hline", "\\end{tabular}"]
open(f"{OUT}/tab_perq.tex", "w").write("\n".join(L))
# figure
plt.figure(figsize=(5.2, 3.0))
sty = {"TEXT-FULL": ("#999999", ":", "o"), "TEXT-RAG": ("#d08c2a", "--", "s"), "KG-DG": ("#2a7ab0", "-.", "^"),
       "OURS": ("#1a1a1a", "-", "D"), "OURS-noHERSE": ("#6b6b6b", "-", "x"), "OURS-noUP": ("#b0b0b0", "-", "+")}
sty["OURS-GSonly"] = ("#555555", "--", "v"); sty["KG-PPR"] = ("#5a9e4b", "-.", "P")
for c in conds:
    col, ls, mk = sty[c]
    plt.plot(budgets, [100 * np.mean(agg[(c, b, "in")]) for b in budgets], ls, color=col, marker=mk, label=NAMES[c], lw=1.4, ms=4)
plt.xscale("log"); plt.xticks(budgets, [str(b) for b in budgets])
plt.xlabel("Budget de contexte (jetons)"); plt.ylabel("Contextes suffisants (%)")
plt.ylim(0, 102); plt.grid(alpha=0.3); plt.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_coverage.pdf"); plt.savefig(f"{OUT}/fig_coverage.png", dpi=160)

# ---------------- E3
a3 = defaultdict(list)
for x in E3:
    a3[(x["profile"], x["target"], x["cond"], x["budget"])].append(x["recall"])
S["E3"] = {"|".join(map(str, k)): 100 * np.mean(v) for k, v in a3.items()}
L = ["\\begin{tabular}{lrrrr}", "\\hline",
     "& \\multicolumn{2}{c}{Profil généraliste} & \\multicolumn{2}{c}{Profil endocrinologue} \\\\",
     "Condition & $B=300$ & $B=600$ & $B=300$ & $B=600$ \\\\", "\\hline"]
for c in ["TEXT-FULL", "TEXT-RAG", "KG-DG", "KG-PPR", "OURS-noUP", "OURS"]:
    cells = []
    for p in ("generaliste", "endocrinologue"):
        for b in (300, 600):
            cells.append(f"{100*np.mean(a3[(p, p, c, b)]):.1f}")
    L.append(f"{NAMES[c] if c != 'OURS' else 'Notre approche (UP du profil)'} & " + " & ".join(cells) + " \\\\")
cells = []
for p, o in (("generaliste", "endocrinologue"), ("endocrinologue", "generaliste")):
    for b in (300, 600):
        cells.append(f"{100*np.mean(a3[(o, p, 'OURS', b)]):.1f}")
L.append("Notre approche (UP de l'autre profil) & " + " & ".join(cells) + " \\\\")
L += ["\\hline", "\\end{tabular}"]
open(f"{OUT}/tab_profile.tex", "w").write("\n".join(L))

# ---------------- E4
ERR = {"P1", "P2", "P3", "P4", "P5"}
def status(labels, th_ok, th_p):
    if not labels: return "KO"
    sup = sum(1 for l in labels if l == "supporte") / len(labels)
    if any(l == "contredit" for l in labels) or sup < th_p: return "KO"
    if sup < th_ok: return "PARTIEL"
    return "OK"
grid = [(o, p) for o in (1.0, 0.9, 0.75, 0.67, 0.5) for p in (0.5, 0.34, 0.25, 0.0) if p <= o]
def f1(rows, V, th):
    tp = sum(1 for r in rows if r["ptype"] in ERR and status(r[V], *th) != "OK")
    fp = sum(1 for r in rows if r["ptype"] not in ERR and status(r[V], *th) != "OK")
    fn = sum(1 for r in rows if r["ptype"] in ERR and status(r[V], *th) == "OK")
    p = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
    return (2 * p * rc / (p + rc) if p + rc else 0), p, rc
val = [r for r in E4 if r["split"] == "val"]; test = [r for r in E4 if r["split"] == "test"]
S["E4_counts"] = {pt: sum(1 for r in test if r["ptype"] == pt) for pt in ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]}
S["E4_extraction_P0_noclaim"] = sum(1 for r in E4 if r["ptype"] == "P0" and r["n_claims"] == 0)
VN = {"V1": "V1 binaire, $G^*_{RDF}$", "V2": "V2 3 classes, monde ouvert", "V3": "V3 3 classes, CWA local",
      "V4": "V4 3 classes, hiérarchique $G^*_{RDF}\\!\\rightarrow\\!DG$"}
L = ["\\begin{tabular}{lrrrrrrrrr}", "\\hline",
     "Vérifieur & P1 & P2 & P3 & P4 & P5 & FA(P0) & FA(P6) & F1 & Préc. KO \\\\", "\\hline"]
S["E4"] = {}
for V in ("V1", "V2", "V3", "V4"):
    th = max(grid, key=lambda t: f1(val, V, t)[0])
    F, p, rc = f1(test, V, th)
    det = {pt: 100 * np.mean([status(r[V], *th) != "OK" for r in test if r["ptype"] == pt]) for pt in ["P1", "P2", "P3", "P4", "P5"]}
    fa0 = 100 * np.mean([status(r[V], *th) != "OK" for r in test if r["ptype"] == "P0"])
    fa6 = 100 * np.mean([status(r[V], *th) != "OK" for r in test if r["ptype"] == "P6"])
    ko = [r for r in test if status(r[V], *th) == "KO"]
    kop = 100 * np.mean([r["ptype"] in ERR for r in ko]) if ko else 0
    gexp = np.mean([r[V + "_gexp"] for r in test if r["ptype"] == "P0"])
    S["E4"][V] = dict(th=th, F1=F, P=p, R=rc, det=det, fa_P0=fa0, fa_P6=fa6, ko_precision=kop, gexp_mean=gexp)
    L.append(f"{VN[V]} & " + " & ".join(f"{det[pt]:.1f}" for pt in ["P1", "P2", "P3", "P4", "P5"])
             + f" & {fa0:.1f} & {fa6:.1f} & {100*F:.1f} & {kop:.1f} \\\\")
L += ["\\hline", "\\end{tabular}"]
open(f"{OUT}/tab_verifier.tex", "w").write("\n".join(L))
# claim label distribution on correct answers (P0+P6) for V2/V3/V4
S["E4_labels_correct"] = {}
for V in ("V2", "V3", "V4"):
    c = defaultdict(int)
    for r in test:
        if r["ptype"] in ("P0", "P6"):
            for l in r[V]: c[l] += 1
    S["E4_labels_correct"][V] = dict(c)
json.dump(S, open(f"{OUT}/summary.json", "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps(S, indent=1, ensure_ascii=False, default=float))
