# features_mfcc.py
import os, re, glob, warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

AUDIO_DIR = "data_clean/raw"          # نفس مجلد التنظيف
OUT_CSV   = ".dist/features_mfcc.csv"
OUT_X_NPY = ".dist/X_mfcc.npy"
OUT_Y_NPY = ".dist/y_labels.npy"
HEAT_DIR  = ".dist/mfcc_heatmaps"     # اختياري لرسم متوسط كل vowel
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
    # نستخدم soundfile لأنه أسرع/أدق، ولو فيه مشكلة نرجع لـ librosa
    try:
        y, sr = sf.read(path, always_2d=False)
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != sr_target:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=sr_target)
            sr = sr_target
    except Exception:
        y, sr = librosa.load(path, sr=sr_target, mono=True)
    # نزيل الصمت الخفيف من الأطراف
    y, _ = librosa.effects.trim(y, top_db=25)
    if len(y) == 0:
        y = np.zeros(int(0.2*sr))  # أمان
    return y.astype(np.float32), sr

def mfcc_features(y, sr, n_mfcc=20):
    # نافذة 25ms وهوب 10ms: n_fft≈400 عند 16k، نستخدم 512
    hop = int(0.010 * sr)   # 10 ms
    win = int(0.025 * sr)   # 25 ms
    n_fft = 1
    while n_fft < win: n_fft *= 2   # أقرب قوة لـ2 (512)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                n_fft=n_fft, hop_length=hop, win_length=win)
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)
    feat = np.vstack([mfcc, d1, d2])  # شكل (60, T)
    # إحصائيات إطار-زمن → متجه ثابت الطول
    mean = np.mean(feat, axis=1)
    std  = np.std(feat, axis=1, ddof=1)
    vec = np.concatenate([mean, std], axis=0)  # (120,)
    return feat, vec

files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**", "*.wav"), recursive=True))
if not files:
    raise SystemExit(f"لا يوجد ملفات داخل {AUDIO_DIR}")

rows = []
X_list, y_list = [], []
heat_by_vowel = {}  # لتجميع متوسط MFCC لكل vowel

for p in tqdm(files, desc="MFCC"):
    try:
        y, sr = load_audio(p, sr_target=16000)
        feat, vec = mfcc_features(y, sr, n_mfcc=20)
        vowel = infer_vowel(p)
        vtype = infer_type(p)
        rows.append({"path":p, "vowel":vowel, "type":vtype, "frames":feat.shape[1]})
        X_list.append(vec)
        y_list.append(vowel)
        # تجميع للـ heatmap المتوسط
        if vowel not in heat_by_vowel:
            heat_by_vowel[vowel] = []
        heat_by_vowel[vowel].append(feat[:20])  # فقط الـ MFCC الأساس (بدون دلتا)
    except Exception as e:
        rows.append({"path":p, "vowel":"error", "type":"unknown", "frames":0, "error":str(e)})

# حفظ CSV للميّزات الإحصائية
col_mean = [f"c{i+1}_mean" for i in range(60)]
col_std  = [f"c{i+1}_std"  for i in range(60)]
df_meta = pd.DataFrame(rows)
df_feat = pd.DataFrame(np.array(X_list), columns=col_mean+col_std)
df = pd.concat([df_meta[["path","vowel","type","frames"]].reset_index(drop=True), df_feat], axis=1)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
np.save(OUT_X_NPY, np.array(X_list, dtype=np.float32))
np.save(OUT_Y_NPY, np.array(y_list))
print(f"Saved {OUT_CSV}\nSaved {OUT_X_NPY}\nSaved {OUT_Y_NPY}")

# (اختياري) رسم Heatmap متوسط لكل vowel مع توحيد الطول الزمني
try:
    import matplotlib.pyplot as plt
    from librosa.util import fix_length

    for v, arrs in heat_by_vowel.items():
        # arrs: قائمة مصفوفات شكلها (20, T) لكل ملف
        # اختر طولًا موحّدًا (الوسيط غالبًا أفضل من الأعظم)
        target_len = int(np.median([a.shape[1] for a in arrs]))
        if target_len < 5:  # أمان
            target_len = max(5, max(a.shape[1] for a in arrs))

        # قص/إكمال لكل مصفوفة إلى الطول الموحد
        norm = [fix_length(a, size=target_len, axis=1) for a in arrs]  # (20, target_len) لكل عنصر
        M = np.mean(np.stack(norm, axis=0), axis=0)  # (20, target_len) متوسط عبر الملفات

        plt.figure(figsize=(4,3))
        plt.imshow(M, aspect="auto", origin="lower")
        plt.title(f"Avg MFCC Heatmap - {v}")
        plt.xlabel("Frames"); plt.ylabel("Coeff (1..20)")
        plt.tight_layout()
        plt.savefig(os.path.join(HEAT_DIR, f"{v}.png"), dpi=180)
        plt.close()

    print(f"Saved heatmaps to {HEAT_DIR}")
except Exception as e:
    print("تعذر حفظ الـ heatmaps:", e)
