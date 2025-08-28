import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

import matplotlib.pyplot as plt


CSV = ".dist/features_mfcc.csv"
OUTDIR = Path(".dist/mfcc_models")
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

if "vowel" in df.columns:
    df = df[~df["vowel"].isin(["error", "unknown"])]

meta_cols = [c for c in ["path", "vowel", "type", "frames"] if c in df.columns]
X = df.select_dtypes(include=[np.number]).copy()
X = X.to_numpy(dtype=np.float32)

y = df["vowel"].to_numpy()
classes = np.unique(y)

print("X shape:", X.shape)
assert X.shape[1] > 0, " error in  features_mfcc.csv"

models = {
    "SVM(RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=10, gamma="scale"))
    ]),
    "RandomForest": Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  
        ("clf", RandomForestClassifier(n_estimators=300, random_state=42))
    ]),
    "LogReg": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, multi_class="auto"))
    ]),
    "KNN-5": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ]),
    "NearestCentroid": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", NearestCentroid())
    ]),
}

def plot_cm(cm, classes, title, path_png):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.tight_layout()
    plt.savefig(path_png, dpi=180)
    plt.close(fig)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_reports = {}

for name, model in models.items():
    print(f"[CV] {name} ...")
    y_pred = cross_val_predict(model, X, y, cv=skf, n_jobs=-1)

    acc = accuracy_score(y, y_pred)
    f1m = f1_score(y, y_pred, average="macro")
    rep = classification_report(y, y_pred, digits=3, output_dict=True)
    cm = confusion_matrix(y, y_pred, labels=classes)

    all_reports[name] = {"accuracy": float(acc), "macro_f1": float(f1m), "report": rep}

    plot_cm(cm, classes, f"{name} — MFCC", OUTDIR / f"cm_{name.replace('(','_').replace(')','')}.png")

    with open(OUTDIR / f"report_{name}.json", "w", encoding="utf-8") as f:
        json.dump(all_reports[name], f, ensure_ascii=False, indent=2)

summary = {m: {"accuracy": round(r["accuracy"], 4), "macro_f1": round(r["macro_f1"], 4)}
           for m, r in all_reports.items()}
print("MFCC CV summary:", summary)

with open(OUTDIR / "summary.txt", "w", encoding="utf-8") as f:
    for k, v in summary.items():
        f.write(f"{k}: acc={v['accuracy']}, macroF1={v['macro_f1']}\n")

print(f"Saved results to {OUTDIR}")
