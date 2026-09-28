from src import io_utils, config as C

linkage = io_utils.load_df(C.SILVER_DIR, "patient_linkage")
tele    = io_utils.load_df(C.SILVER_DIR, "teleconsultations")
lab     = io_utils.load_df(C.SILVER_DIR, "lab_tests")

def attach(df, pid_col, sys):
    subset = linkage[linkage.source_system == sys][["source_patient_id", "patient_id"]].drop_duplicates("source_patient_id")
    return df.merge(subset.rename(columns={"source_patient_id": pid_col}), on=pid_col, how="left")

tele2 = attach(tele, "tele_source_patient_id", "teleconsultations")
lab2  = attach(lab,  "lab_source_patient_id",  "lab_tests")

print("tele patient_id null%:", round(tele2.patient_id.isna().mean()*100, 2))
print("lab  patient_id null%:", round(lab2.patient_id.isna().mean()*100, 2))

overlap = set(tele2.patient_id.dropna()) & set(lab2.patient_id.dropna())
print("patients appearing in BOTH tele & lab (post-linkage):", len(overlap), "/", tele2.patient_id.nunique())

merged = tele2.merge(lab2, on="patient_id", how="inner")
print("teleconsults with >=1 same-patient lab row (ANY date, no window filter):",
      merged.teleconsult_id.nunique(), "/", len(tele2))