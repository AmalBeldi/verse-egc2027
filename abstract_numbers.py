"""Affiche, à partir des résultats, chaque chiffre cité dans le résumé de l'article.

  python3 scripts/abstract_numbers.py                 # chiffres E1-E4 (déjà dans le résumé)
  python3 scripts/abstract_numbers.py --model qwen2.5_7b-instruct   # + phrase E5 remplie

Le nom du modèle est celui du fichier results/llm_<modele>.jsonl.
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default="paper/generated/summary.json")
ap.add_argument("--llm", default="paper/generated/summary_llm.json")
ap.add_argument("--model", default=None)
a = ap.parse_args()

S = json.load(open(a.summary))
E1 = S["E1"]
print("=== Chiffres E1-E4 cités dans le résumé ===")
print("patients                         : 172")
print(f"jetons du dossier (médiane)      : {E1['text_tokens']['median']:.0f}   -> « environ 20 000 »")
G, U = S["E2_groups"], S["E2_used_groups"]
print(f"questions factuelles couvertes   : {G['OURS@250@fact']:.1f} % (B=250) -> « toutes les questions factuelles »")
print(f"jetons utilisés (médiane, fact.) : {U['OURS@500@fact']:.0f}   -> « 258 »")
print(f"RAG textuel, fact., B=4000       : {G['TEXT-RAG@4000@fact']:.1f} % -> « 79 % »")
print(f"RAG sur DG / PPR, fact., B=250   : {G['KG-DG@250@fact']:.1f} % / {G['KG-PPR@250@fact']:.1f} % -> « 90 % » / « 62 % »")
red = 1 - E1["gs_lin_tokens"]["median"] / E1["gs_lin_tokens_noherse"]["median"]
print(f"réduction HERSE (jetons, médianes): {100*red:.1f} %  -> « 46 % »")
print(f"fausses alertes P6, V3 / V4      : {S['E4']['V3']['fa_P6']:.1f} % / {S['E4']['V4']['fa_P6']:.1f} % -> « 48 % », supprimées")
print(f"détection P1-P5, V4              : {min(S['E4']['V4']['det'].values()):.1f} % minimum")

if a.model:
    if not os.path.exists(a.llm):
        sys.exit(f"{a.llm} absent : lancez d'abord make e5 MODEL=...")
    L = json.load(open(a.llm))
    o, t = L.get(f"{a.model}|OURS"), L.get(f"{a.model}|TEXT-RAG")
    if not o or not t:
        sys.exit(f"modèle {a.model} introuvable ; disponibles : {sorted({k.split('|')[0] for k in L})}")
    print("\n=== E5 ===")
    for c in ("TEXT-FULL", "TEXT-RAG", "KG-DG", "KG-PPR", "OURS", "OURS-GSonly", "OURS-noHERSE", "OURS-noUP"):
        v = L.get(f"{a.model}|{c}")
        if v:
            extra = f"  IC95 [{v['ci'][0]:.1f} ; {v['ci'][1]:.1f}]  p={v['p']:.2g}" if "p" in v else ""
            grp = "  ".join(f"{g} {v[f'exact_{g}']:5.1f}" for g in ("fact", "agg", "hors") if f"exact_{g}" in v)
            print(f"{c:<12} exact {v['exact']:5.1f} %  [{grp}]  halluc {v['halluc']:5.1f} %  unsafe {v['unsafe']:4.1f} %  "
                  f"OK->err {v['ok_wrong']:4.1f} %  n={v['n']}{extra}")
    va = L.get(f"{a.model}|verif_auto")
    if va:
        print(f"accord V4 / règles sur {va['n']} réponses : précision {va['precision']:.1f} %  rappel {va['recall']:.1f} %")
    better = o["exact"] > t["exact"] and t.get("p", 1) < 0.05
    fewer = o["halluc"] < t["halluc"]
    print("\nPhrase proposée :")
    if better and fewer:
        print(f"FR : Avec {a.model}, notre contexte porte l'exactitude des réponses à {o['exact']:.0f} %, contre "
              f"{t['exact']:.0f} % pour le RAG textuel, et réduit la part d'assertions non étayées de "
              f"{t['halluc']:.0f} % à {o['halluc']:.0f} %.")
        print(f"EN : With {a.model}, our context raises answer accuracy to {o['exact']:.0f}%, versus "
              f"{t['exact']:.0f}% for text RAG, and reduces unsupported claims from {t['halluc']:.0f}% to {o['halluc']:.0f}%.")
    else:
        print("ATTENTION : le gain n'est pas net (exactitude non supérieure, p >= 0,05 ou hallucinations non réduites).")
        print("Ne reprenez pas la phrase telle quelle : décrivez le résultat observé (voir la conversation).")
