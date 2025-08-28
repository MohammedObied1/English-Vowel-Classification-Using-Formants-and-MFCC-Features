# features_mfcc_13.py
import os, re, glob
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = "data_clean/raw"          # نفس مجلد التنظيف
OUT_CSV   = ".dist/features_mfcc.csv" # سيستبدل الملف السابق بنفس الاسم
OUT_X_NPY = ".dist/X_mfcc.npy"
OUT_Y_NPY = ".dist/y_labels.npy"
HEAT_DIR  = ".dist/mfcc13_heatmaps"

os.makedirs(".dist", exist_ok=True)
os.makedirs(HEAT_DIR, exist_ok=True)

# خريطة أسماء المقاطع من لاحقة الملف
VOWEL_MAP = {
    "ii":"heed","il":"hid","ee":"hayed","el":"head","al":"had",
    "aw":"hawed","aa":"hod","oo":"hoed","ul":"hood","uu":"whod",
    "uh":"hud","er":"herd"
}
TAIL_RE = re.compile(r"([a-z]{2})\.wav$", re.IGNORECASE)

def infer_vowel(path):
    m = TAIL_RE.search(os.path.basename(path).lower())
    if not m: return "unknown"
    return VOWEL_MAP.get(m.group(1), m.group(1))

def infer_type(path):
    n = os.path.basename(path).lower()
    if n.startswith("n"): return "natural"
    if n.startswith("s"): return "synthesized"
    return "unknown"

def load_audio(path, sr_target=16000):
    """قراءة الصوت، تحويل لمونو، إعادةSampling لـ16k، وقص الصمت من الأطراف"""
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != sr_target:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr_target)
            sr = sr_target
    except Exception:
        y, sr = librosa.load(path, sr=sr_target, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) == 0:
        y = np.zeros(int(0.2*sr), dtype=np.float32)
    return y.astype(np.float32), sr

def mfcc_13_feat(y, sr, n_mfcc=13):
    """إرجاع:
       - feat_full: مصفوفة (39, T) = [13 MFCC; 13 delta; 13 delta2]
       - vec: متجه ثابت (78,) = mean & std لكل القنوات الـ39
       - mfcc_base: مصفوفة (13, T) للـ heatmap
    """
    hop = int(0.010 * sr)   # 10 ms
    win = int(0.025 * sr)   # 25 ms
    n_fft = 1
    while n_fft < win: n_fft *= 2  # أقرب قوة 2

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop, win_length=win)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat_full = np.vstack([mfcc, d1, d2])           # (39, T)

    mean = np.mean(feat_full, axis=1)
    std  = np.std(feat_full, axis=1, ddof=1)
    vec = np.concatenate([mean, std], axis=0)       # (78,)

    return feat_full, vec, mfcc

# جمع الملفات
files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**", "*.wav"), recursive=True))
if not files:
    raise SystemExit(f"لا يوجد ملفات داخل {AUDIO_DIR}")

rows = []
X_list, y_list = [], []
heat_by_vowel = {}     # لتجميع الـ MFCC الأساسي (13, T) لكل vowel

for p in tqdm(files, desc="MFCC(13+Δ+ΔΔ)"):
    try:
        y, sr = load_audio(p, sr_target=16000)
        feat_full, vec, mfcc_base = mfcc_13_feat(y, sr, n_mfcc=13)

        vowel = infer_vowel(p)
        vtype = infer_type(p)
        rows.append({"path": p, "vowel": vowel, "type": vtype,
                     "frames": feat_full.shape[1]})
        X_list.append(vec)
        y_list.append(vowel)

        heat_by_vowel.setdefault(vowel, []).append(mfcc_base)  # (13, T)
    except Exception as e:
        rows.append({"path": p, "vowel": "error", "type": "unknown",
                     "frames": 0, "error": str(e)})

# حفظ CSV + NPY
col_mean = [f"f{i+1}_mean" for i in range(39)]
col_std  = [f"f{i+1}_std"  for i in range(39)]
df_meta  = pd.DataFrame(rows)
df_feat  = pd.DataFrame(np.array(X_list, dtype=np.float32),
                        columns=col_mean + col_std)
df = pd.concat([df_meta[["path","vowel","type","frames"]].reset_index(drop=True),
                df_feat], axis=1)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
np.save(OUT_X_NPY, np.array(X_list, dtype=np.float32))
np.save(OUT_Y_NPY, np.array(y_list))
print(f"Saved {OUT_CSV}\nSaved {OUT_X_NPY}\nSaved {OUT_Y_NPY}")

# Heatmaps بــ 13 MFCC لكل vowel (مع توحيد الطول الزمني)
try:
    import matplotlib.pyplot as plt
    from librosa.util import fix_length

    for v, arrs in heat_by_vowel.items():
        # arrs: قائمة (13, T)
        target_len = int(np.median([a.shape[1] for a in arrs]))
        target_len = max(target_len, 5)
        norm = [fix_length(a, size=target_len, axis=1) for a in arrs]  # (13, L)
        M = np.mean(np.stack(norm, axis=0), axis=0)                     # (13, L)

        plt.figure(figsize=(9, 5.2))
        plt.imshow(M, aspect="auto", origin="lower")
        plt.title(f"Avg MFCC Heatmap - {v}")
        plt.xlabel("Frames"); plt.ylabel("Coeff (1..13)")
        plt.tight_layout()
        plt.savefig(os.path.join(HEAT_DIR, f"{v}.png"), dpi=160)
        plt.close()

    print(f"Saved heatmaps to {HEAT_DIR}")
except Exception as e:
    print("تعذر حفظ الـ heatmaps:", e)
