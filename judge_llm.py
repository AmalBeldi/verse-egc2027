"""Baseline neuronale de vérification : un LLM juge chaque réponse produite en E5.

Le juge reçoit les faits du dossier liés à la question (mêmes faits que le RAG sur DG,
2 000 jetons) et la réponse ; il répond OK (tout est étayé) ou KO (au moins une
affirmation fausse ou non étayée). On compare ensuite ce juge au vérifieur
hiérarchique sur les annotations humaines (score_verifiers.py).

  python3 experiments/judge_llm.py results/llm_qwen2.5_7b-instruct.jsonl \
      --base-url http://localhost:11434/v1 --model llama3.1:8b
"""
import argparse, json, os, re, sys, time
import requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ehrkg.data import load_cohort
from ehrkg.pipeline import prepare
from ehrkg import context as C

PROMPT = ("Vous vérifiez la réponse d'un assistant à une question sur le dossier d'un patient.\n"
          "Faits extraits du dossier :\n{facts}\n\nQuestion : {q}\nRéponse : {a}\n\n"
          "La réponse contient-elle au moins une affirmation fausse ou non étayée par ces faits ? "
          "Répondez uniquement par OK (toutes les affirmations sont étayées) ou KO.")

ap = argparse.ArgumentParser()
ap.add_argument("answers")
ap.add_argument("--csv", default="data/synthea_t2dm_cohort")
ap.add_argument("--base-url", default="http://localhost:11434/v1")
ap.add_argument("--model", default="llama3.1:8b")
ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
ap.add_argument("--budget", type=int, default=2000)
ap.add_argument("--out", default=None)
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

REC = load_cohort(a.csv)
rows = [json.loads(l) for l in open(a.answers)]
tag = "DRYRUN" if a.dry_run else a.model.replace("/", "_").replace(":", "_")
out = a.out or a.answers.replace(".jsonl", f".judge_{tag}.jsonl")
done = set()
if os.path.exists(out):
    done = {(x["pid"], x["qid"], x["cond"]) for x in map(json.loads, open(out))}
f = open(out, "a"); P = {}
for r in rows:
    if r["answer"] is None or (r["pid"], r["qid"], r["cond"]) in done:
        continue
    if r["pid"] not in P:
        P = {r["pid"]: prepare(REC[r["pid"]])}
    ctx, _ = C.ctx_kg_dg(P[r["pid"]]["dg_items"], r["question"], a.budget)
    msg = PROMPT.format(facts=C.to_text(ctx), q=r["question"], a=r["answer"])
    if a.dry_run:
        raw = "OK"
    else:
        raw = None
        for attempt in range(3):
            try:
                resp = requests.post(a.base_url.rstrip("/") + "/chat/completions", timeout=300,
                                     headers={"Authorization": f"Bearer {a.api_key}"} if a.api_key else {},
                                     json=dict(model=a.model, temperature=0, seed=2027, max_tokens=5,
                                               messages=[{"role": "user", "content": msg}]))
                resp.raise_for_status(); raw = resp.json()["choices"][0]["message"]["content"]; break
            except Exception as e:
                print("retry", e); time.sleep(5)
    lab = None if raw is None else ("KO" if re.search(r"\bKO\b", raw.upper()) else "OK" if "OK" in raw.upper() else None)
    f.write(json.dumps(dict(pid=r["pid"], qid=r["qid"], cond=r["cond"], judge=tag, raw=raw, label=lab)) + "\n")
    f.flush()
print("written", out)
