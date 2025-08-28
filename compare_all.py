# compare_all.py
import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt

FORMANTS_CSV = ".dist/features_formants.csv"
MFCC_CSV     = ".dist/features_mfcc.csv"
OUTDIR       = Path(".dist/compare_out")
(OUTDIR/"cfms").mkdir(parents=True, exist_ok=True)
(OUTDIR/"reports").mkdir(parents=True, exist_ok=True)

df_f = pd.read_csv(FORMANTS_CSV)  
df_m = pd.read_csv(MFCC_CSV)     

for d in (df_f, df_m):
    if "vowel" in d:
        d = d[~d["vowel"].isin(["error", "unknown"])]
    d.dropna(axis=1, how="all", inplace=True)

base_cols = [c for c in ["path","vowel","type"] if c in df_f.columns and c in df_m.columns]
if "path" not in base_cols:
    raise SystemExit("لازم العمود 'path' موجود عند الطرفين للمزج.")

cand_f = [c for c in df_f.columns if c.lower() in ["f1","f2"] or c.lower().endswith(("f1","f2"))]
if len(cand_f) < 2:

    f_like = [c for c in df_f.columns if "F1" in c.upper() or "F2" in c.upper()]
    cand_f = f_like[:2]
assert len(cand_f) >= 2, " error features_formants.csv"

meta_m = [c for c in ["path","vowel","type","frames"] if c in df_m.columns]
mfcc_numeric_cols = df_m.select_dtypes(include=[np.number]).columns.tolist()
mfcc_feat_cols = [c for c in mfcc_numeric_cols if c not in ["frames"]]

F_form = df_f[base_cols + cand_f].dropna().copy()
F_mfcc = df_m[base_cols + mfcc_feat_cols].dropna().copy()

F_mix = pd.merge(F_form, F_mfcc, on=base_cols, how="inner", suffixes=("_form","_mfcc"))

def to_Xy(df, feat_cols):
    X = df[feat_cols].to_numpy(dtype=float)
    y = df["vowel"].to_numpy()
    return X, y, df

SETS = {
    "formants": (F_form, cand_f),
    "mfcc":     (F_mfcc, mfcc_feat_cols),
    "mixed":    (F_mix,  cand_f + mfcc_feat_cols),
}

def crossval_nearest_prototype(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y, dtype=object)
    for tr, te in skf.split(X, y):
        Xtr, ytr = X[tr], y[tr]
        Xte = X[te]
        classes = np.unique(ytr)
        centroids = {c: Xtr[ytr==c].mean(axis=0) for c in classes}
        C = np.stack([centroids[c] for c in classes], axis=0)  
        XX = np.sum(Xte**2, axis=1, keepdims=True)         
        CC = np.sum(C**2, axis=1, keepdims=True).T        
        sims = XX + CC - 2*np.dot(Xte, C.T)               
        idx = np.argmin(sims, axis=1)
        y_pred[te] = classes[idx]
    return y_pred

def get_models():
    return {
        "SVM(RBF)": Pipeline([("sc", StandardScaler()), ("clf", SVC(kernel="rbf", C=10, gamma="scale"))]),
        "RandomForest": Pipeline([("clf", RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1))]),
    }

def crossval_predict_model(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_pred = np.empty_like(y, dtype=object)
    for tr, te in skf.split(X, y):
        model.fit(X[tr], y[tr])
        y_pred[te] = model.predict(X[te])
    return y_pred

def plot_cm(cm, labels, title, out_png):
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close(fig)

summary_rows = []
for set_name, (DF, feat_cols) in SETS.items():
    X, y, D = to_Xy(DF, feat_cols)
    classes = np.unique(y)

    y_pred_np = crossval_nearest_prototype(X, y, n_splits=5, seed=42)
    acc_np = accuracy_score(y, y_pred_np)
    f1m_np = f1_score(y, y_pred_np, average="macro")
    rep_np = classification_report(y, y_pred_np, digits=3, output_dict=True)
    cm_np  = confusion_matrix(y, y_pred_np, labels=classes)
    summary_rows.append([set_name, "No", "Nearest-Prototype", acc_np, f1m_np])

    plot_cm(cm_np, classes, f"{set_name} — No ML (Nearest-Prototype)", OUTDIR/"cfms"/f"cm_{set_name}_NoML.png")
    with open(OUTDIR/"reports"/f"report_{set_name}_NoML.json","w",encoding="utf-8") as f:
        json.dump({"accuracy":acc_np,"macro_f1":f1m_np,"report":rep_np}, f, ensure_ascii=False, indent=2)

    # (ب) مع ML — SVM, RF
    for mname, model in get_models().items():
        y_pred = crossval_predict_model(model, X, y, n_splits=5, seed=42)
        acc = accuracy_score(y, y_pred)
        f1m = f1_score(y, y_pred, average="macro")
        rep = classification_report(y, y_pred, digits=3, output_dict=True)
        cm  = confusion_matrix(y, y_pred, labels=classes)
        summary_rows.append([set_name, "Yes", mname, acc, f1m])

        tag = mname.replace("(","_").replace(")","")
        plot_cm(cm, classes, f"{set_name} — {mname}", OUTDIR/"cfms"/f"cm_{set_name}_{tag}.png")
        with open(OUTDIR/"reports"/f"report_{set_name}_{tag}.json","w",encoding="utf-8") as f:
            json.dump({"accuracy":acc,"macro_f1":f1m,"report":rep}, f, ensure_ascii=False, indent=2)

df_sum = pd.DataFrame(summary_rows, columns=["Features","ML?","Method","Accuracy","MacroF1"])
df_sum.sort_values(["Features","ML?","Method"], inplace=True)
df_sum.to_csv(OUTDIR/"compare_summary.csv", index=False, encoding="utf-8")

print("\n=== Comparison Summary ===")
print(df_sum.to_string(index=False))
print(f"\nSaved summary to {OUTDIR/'compare_summary.csv'}")
print(f"Confusion matrices in {OUTDIR/'cfms'} and reports in {OUTDIR/'reports'}")

