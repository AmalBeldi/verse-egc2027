"""Convertit le module `hosp` de MIMIC-IV (ou MIMIC-IV Clinical Database Demo) au format
CSV de Synthea attendu par ehrkg.data.load_cohort, pour rejouer E1, E2 et E4 sur des
dossiers réels sans modifier le pipeline.

Données : MIMIC-IV Clinical Database Demo 2.2 (100 patients, accès libre, licence ODbL)
  https://physionet.org/content/mimic-iv-demo/2.2/
  python3 scripts/mimic_to_synthea.py --mimic <dossier>/mimic-iv-clinical-database-demo-2.2 \
      --out data/mimic_demo_t2dm
  python3 experiments/run_symbolic.py --csv data/mimic_demo_t2dm --out results_mimic
  python3 experiments/analyze.py results_mimic paper/generated_mimic

Correspondances (documentées dans l'article) :
  - consultations : admissions (hospitalisations) + journées de mesures ambulatoires (omr)
    + journées d'analyses hors hospitalisation ;
  - diagnostics : diagnoses_icd, datés par l'admission ; diabète de type 2 = ICD-10 E11*,
    ICD-9 250.x0 / 250.x2 ; complications, HTA, dyslipidémie, MRC et anémie projetées sur les
    codes SNOMED CT utilisés par le pipeline ; les autres diagnostics gardent leur code ICD ;
  - analyses : labevents (itemid -> LOINC) ; le DFG est calculé à partir de la créatinine
    (CKD-EPI 2021, sans coefficient ethnique) car MIMIC ne le fournit pas en valeur numérique ;
  - mesures ambulatoires : omr (pression artérielle, poids, IMC) ;
  - prescriptions : prescriptions (libellé du médicament conservé tel quel).
Les patients décédés sont conservés (la date de référence est la dernière consultation).
"""
import argparse, os, re
import numpy as np
import pandas as pd

T2DM = "44054006"
LAB = {  # itemid -> (LOINC, description de type Synthea, unité)
    50852: ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    50931: ("2339-0", "Glucose", "mg/dL"),
    50912: ("38483-4", "Creatinine", "mg/dL"),
    50905: ("18262-6", "Low Density Lipoprotein Cholesterol", "mg/dL"),
    50906: ("18262-6", "Low Density Lipoprotein Cholesterol", "mg/dL"),
    50907: ("2093-3", "Total Cholesterol", "mg/dL"),
    50904: ("2085-9", "High Density Lipoprotein Cholesterol", "mg/dL"),
    51000: ("2571-8", "Triglycerides", "mg/dL"),
    50971: ("6298-4", "Potassium", "mmol/L"),
    50983: ("2947-0", "Sodium", "mmol/L"),
    51222: ("718-7", "Hemoglobin [Mass/volume] in Blood", "g/dL"),
}
EGFR = ("33914-3", "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] in Serum, Plasma or Blood by Creatinine-based formula (CKD-EPI)", "mL/min/{1.73_m2}")

def icd_map(code, ver):
    """(SNOMED, description) pour les diagnostics utiles à la tâche, sinon None."""
    c = str(code).strip().upper()
    if ver == 10:
        if c.startswith("E11"):
            s = c[3:]
            if s.startswith("2"): return "127013003", "Disorder of kidney due to diabetes mellitus (disorder)"
            if s.startswith(("35",)): return "1501000119109", "Proliferative diabetic retinopathy due to type II diabetes mellitus (disorder)"
            if s.startswith("3"): return "1551000119108", "Nonproliferative diabetic retinopathy due to type 2 diabetes mellitus (disorder)"
            if s.startswith("4"): return "368581000119106", "Neuropathy due to type 2 diabetes mellitus (disorder)"
            return T2DM, "Diabetes mellitus type 2 (disorder)"
        if c.startswith("I10") or c.startswith("I12") or c.startswith("I13"): return "59621000", "Essential hypertension (disorder)"
        if c.startswith("E785") or c.startswith("E784") or c.startswith("E782"): return "55822004", "Hyperlipidemia (disorder)"
        if c.startswith("N18"):
            st = {"1": ("431855005", "1"), "2": ("431856006", "2"), "3": ("433144002", "3"),
                  "4": ("431857002", "4"), "5": ("433146000", "5")}.get(c[3:4])
            if st: return st[0], f"Chronic kidney disease stage {st[1]} (disorder)"
            return "709044004", "Chronic kidney disease (disorder)"
        if c.startswith("D649") or c.startswith("D63"): return "271737000", "Anemia (disorder)"
    else:
        if c.startswith("250") and len(c) >= 5 and c[4] in "02":
            k = c[3]
            if k == "4": return "127013003", "Disorder of kidney due to diabetes mellitus (disorder)"
            if k == "5": return "1551000119108", "Nonproliferative diabetic retinopathy due to type 2 diabetes mellitus (disorder)"
            if k == "6": return "368581000119106", "Neuropathy due to type 2 diabetes mellitus (disorder)"
            return T2DM, "Diabetes mellitus type 2 (disorder)"
        if c.startswith("401") or c.startswith("403") or c.startswith("404"): return "59621000", "Essential hypertension (disorder)"
        if c.startswith("2724") or c.startswith("2720") or c.startswith("2722"): return "55822004", "Hyperlipidemia (disorder)"
        if c.startswith("585"):
            st = {"1": ("431855005", "1"), "2": ("431856006", "2"), "3": ("433144002", "3"),
                  "4": ("431857002", "4"), "5": ("433146000", "5")}.get(c[3:4])
            if st: return st[0], f"Chronic kidney disease stage {st[1]} (disorder)"
            return "709044004", "Chronic kidney disease (disorder)"
        if c.startswith("2859") or c.startswith("2852"): return "271737000", "Anemia (disorder)"
    return None

def ckd_epi_2021(scr, age, female):
    k, a = (0.7, -0.241) if female else (0.9, -0.302)
    return 142 * min(scr / k, 1) ** a * max(scr / k, 1) ** -1.200 * 0.9938 ** age * (1.012 if female else 1)

def rd(path, name, **kw):
    for ext in (".csv.gz", ".csv"):
        p = os.path.join(path, name + ext)
        if os.path.exists(p):
            return pd.read_csv(p, **kw)
    raise FileNotFoundError(os.path.join(path, name + ".csv.gz"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic", required=True, help="dossier racine (contient hosp/)")
    ap.add_argument("--out", default="data/mimic_demo_t2dm")
    a = ap.parse_args()
    H = os.path.join(a.mimic, "hosp")
    os.makedirs(a.out, exist_ok=True)

    pats = rd(H, "patients")
    adm = rd(H, "admissions", parse_dates=["admittime"])
    dx = rd(H, "diagnoses_icd", dtype={"icd_code": str})
    try:
        dicd = rd(H, "d_icd_diagnoses", dtype={"icd_code": str})
        dlab = {(r.icd_code, int(r.icd_version)): r.long_title for r in dicd.itertuples()}
    except FileNotFoundError:
        dlab = {}

    # ---- cohorte : au moins un code de diabète de type 2
    dx["map"] = [icd_map(c, int(v)) for c, v in zip(dx.icd_code, dx.icd_version)]
    t2 = set(dx.loc[dx["map"].map(lambda m: m is not None and m[0] == T2DM), "subject_id"])
    # une complication diabétique ICD-10 E11.x implique aussi le diabète de type 2
    t2 |= set(dx.loc[dx.icd_code.str.upper().str.startswith("E11") & (dx.icd_version == 10), "subject_id"])
    print(f"{len(pats)} patients, {len(t2)} avec un diabète de type 2")
    pats = pats[pats.subject_id.isin(t2)]
    adm = adm[adm.subject_id.isin(t2)]
    sid = lambda s: f"mimic-{int(s)}"
    hdate = dict(zip(adm.hadm_id, adm.admittime.dt.strftime("%Y-%m-%d")))

    # ---- patients
    P = pd.DataFrame(dict(Id=pats.subject_id.map(sid),
                          BIRTHDATE=(pats.anchor_year - pats.anchor_age).astype(int).astype(str) + "-01-01",
                          DEATHDATE="", GENDER=pats.gender))
    age = dict(zip(pats.subject_id, pats.anchor_age)); ay = dict(zip(pats.subject_id, pats.anchor_year))
    fem = dict(zip(pats.subject_id, pats.gender == "F"))

    enc = {}  # Id -> row
    for r in adm.itertuples():
        enc[str(r.hadm_id)] = dict(Id=str(r.hadm_id), START=r.admittime.strftime("%Y-%m-%d"), PATIENT=sid(r.subject_id),
                                   ENCOUNTERCLASS="inpatient", DESCRIPTION=f"Hospital admission ({r.admission_type})",
                                   REASONDESCRIPTION="")
    def day_enc(s, d, kind):
        k = f"{kind}-{int(s)}-{d}"
        if k not in enc:
            enc[k] = dict(Id=k, START=d, PATIENT=sid(s), ENCOUNTERCLASS="ambulatory",
                          DESCRIPTION="Outpatient measurements" if kind == "omr" else "Outpatient laboratory tests",
                          REASONDESCRIPTION="")
        return k

    # ---- diagnostics
    dx = dx[dx.subject_id.isin(t2) & dx.hadm_id.isin(hdate)]
    C = []
    for r in dx.itertuples():
        m = r.map
        code, desc = m if m else (f"ICD{int(r.icd_version)}-{r.icd_code}",
                                  dlab.get((r.icd_code, int(r.icd_version)), f"ICD-{int(r.icd_version)} {r.icd_code}"))
        C.append(dict(START=hdate[r.hadm_id], STOP="", PATIENT=sid(r.subject_id), ENCOUNTER=str(r.hadm_id),
                      CODE=code, DESCRIPTION=desc))
    C = pd.DataFrame(C)

    # ---- analyses
    O = []
    lab = rd(H, "labevents", usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum"])
    lab = lab[lab.subject_id.isin(t2) & lab.itemid.isin(LAB) & lab.valuenum.notna()]
    for r in lab.itertuples():
        d = str(r.charttime)[:10]
        e = str(int(r.hadm_id)) if r.hadm_id == r.hadm_id and str(int(r.hadm_id)) in enc else day_enc(r.subject_id, d, "lab")
        code, desc, unit = LAB[r.itemid]
        O.append(dict(DATE=d, PATIENT=sid(r.subject_id), ENCOUNTER=e, CODE=code, DESCRIPTION=desc,
                      VALUE=f"{r.valuenum:.2f}", UNITS=unit))
        if r.itemid == 50912 and r.valuenum > 0:   # DFG calculé (CKD-EPI 2021)
            ag = age[r.subject_id] + int(d[:4]) - ay[r.subject_id]
            O.append(dict(DATE=d, PATIENT=sid(r.subject_id), ENCOUNTER=e, CODE=EGFR[0], DESCRIPTION=EGFR[1],
                          VALUE=f"{ckd_epi_2021(r.valuenum, ag, fem[r.subject_id]):.1f}", UNITS=EGFR[2]))
    # ---- mesures ambulatoires
    try:
        omr = rd(H, "omr", dtype=str)
        omr = omr[omr.subject_id.astype(int).isin(t2)]
        for r in omr.itertuples():
            s, d, n, v = int(r.subject_id), r.chartdate[:10], r.result_name, r.result_value
            e = day_enc(s, d, "omr")
            base = dict(DATE=d, PATIENT=sid(s), ENCOUNTER=e)
            if n.startswith("Blood Pressure") and re.fullmatch(r"\d+/\d+", v or ""):
                sy, di = v.split("/")
                O.append(dict(base, CODE="8480-6", DESCRIPTION="Systolic Blood Pressure", VALUE=sy, UNITS="mm[Hg]"))
                O.append(dict(base, CODE="8462-4", DESCRIPTION="Diastolic Blood Pressure", VALUE=di, UNITS="mm[Hg]"))
            elif n.startswith("BMI"):
                O.append(dict(base, CODE="39156-5", DESCRIPTION="Body mass index (BMI) [Ratio]", VALUE=v, UNITS="kg/m2"))
            elif n.startswith("Weight"):
                try:
                    kg = float(v) * 0.45359237
                    O.append(dict(base, CODE="29463-7", DESCRIPTION="Body Weight", VALUE=f"{kg:.1f}", UNITS="kg"))
                except ValueError:
                    pass
    except FileNotFoundError:
        print("omr absent : pas de mesures ambulatoires")
    O = pd.DataFrame(O)

    # ---- prescriptions
    rx = rd(H, "prescriptions", dtype={"ndc": str})
    rx = rx[rx.subject_id.isin(t2) & rx.drug.notna() & rx.starttime.notna()]
    M = pd.DataFrame(dict(START=rx.starttime.str[:10], STOP=rx.stoptime.fillna("").str[:10], PATIENT=rx.subject_id.map(sid),
                          ENCOUNTER=rx.hadm_id.map(lambda h: str(int(h)) if h == h else ""),
                          CODE=rx.ndc.fillna("").where(rx.ndc.fillna("0") != "0", rx.drug),
                          DESCRIPTION=rx.drug, REASONDESCRIPTION=""))
    # ---- actes
    try:
        pr = rd(H, "procedures_icd", dtype={"icd_code": str})
        pr = pr[pr.subject_id.isin(t2)]
        Pr = pd.DataFrame(dict(START=pr.chartdate.str[:10], PATIENT=pr.subject_id.map(sid),
                               ENCOUNTER=pr.hadm_id.astype(str), CODE="ICDPCS-" + pr.icd_code,
                               DESCRIPTION="ICD procedure " + pr.icd_code))
    except FileNotFoundError:
        Pr = pd.DataFrame(columns=["START", "PATIENT", "ENCOUNTER", "CODE", "DESCRIPTION"])

    E = pd.DataFrame(list(enc.values()))
    for name, df in [("patients", P), ("encounters", E), ("conditions", C), ("observations", O),
                     ("medications", M), ("procedures", Pr)]:
        df.to_csv(os.path.join(a.out, name + ".csv"), index=False)
        print(f"  {name:13s} {len(df):7d} lignes")
    with open(os.path.join(a.out, "README.txt"), "w") as f:
        f.write("Dérivé de MIMIC-IV (PhysioNet) par scripts/mimic_to_synthea.py. "
                "Respecter la licence des données sources ; ne pas redistribuer si la licence l'interdit.\n")

if __name__ == "__main__":
    main()
