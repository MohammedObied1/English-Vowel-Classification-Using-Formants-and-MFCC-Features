# features_formants.py
import os, re, glob, warnings
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

# جرّب استخدام Praat (أفضل)، ولو مش متوفر نستخدم LPC
try:
    import parselmouth
    HAS_PRAAT = True
except Exception:
    HAS_PRAAT = False
    warnings.warn("parselmouth غير متوفر. سنستخدم LPC كخطة بديلة.")

AUDIO_DIR = "data_clean/raw"          # مكان الملفات بعد التنظيف
OUT_CSV   = ".dist/features_formants.csv"
OUT_PNG   = ".dist/formant_space.png"

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

def infer_type(path):  # natural / synthesized
    n = os.path.basename(path).lower()
    if n.startswith("n"): return "natural"
    if n.startswith("s"): return "synthesized"
    return "unknown"

def get_formants_praat(path, fmax=5500):
    snd = parselmouth.Sound(path)
    formant = parselmouth.praat.call(snd, "To Formant (burg)", 0, 5, fmax, 0.025, 50)
    times = np.linspace(snd.xmin, snd.xmax, 20)
    F1, F2 = [], []
    for t in times:
        f1 = parselmouth.praat.call(formant, "Get value at time", 1, t, "Hertz", "Linear")
        f2 = parselmouth.praat.call(formant, "Get value at time", 2, t, "Hertz", "Linear")
        if f1 and f2 and f1>90 and f2>90:
            F1.append(f1); F2.append(f2)
    if not F1 or not F2: return np.nan, np.nan
    return float(np.nanmedian(F1)), float(np.nanmedian(F2))

def get_formants_lpc(path, sr_target=16000, order=18, fmax=5500):
    y, sr = sf.read(path, always_2d=False)
    if y.ndim>1: y = np.mean(y, axis=1)
    if sr != sr_target:
        y = librosa.resample(y.astype(np.float32), sr, sr_target)
        sr = sr_target
    y = librosa.util.normalize(y)
    # نافذة من الوسط ~250ms
    win = int(0.25*sr); start = max(0, (len(y)-win)//2)
    yw = y[start:start+win] if win<len(y) else y
    yw = yw * np.hamming(len(yw))
    a = librosa.lpc(yw, order=order)
    roots = np.roots(a); roots = roots[np.imag(roots)>=0]
    ang = np.arctan2(np.imag(roots), np.real(roots))
    freqs = ang * (sr/(2*np.pi))
    freqs = np.array([f for f in freqs if 90 < f < fmax]); freqs.sort()
    if len(freqs)<2: return np.nan, np.nan
    return float(freqs[0]), float(freqs[1])

def extract_file(path):
    if HAS_PRAAT:
        f1, f2 = get_formants_praat(path)
        if not (np.isnan(f1) or np.isnan(f2)):
            return f1, f2
    return get_formants_lpc(path)

# --------- استخراج ---------
os.makedirs(".dist", exist_ok=True)
rows = []
files = sorted(glob.glob(os.path.join(AUDIO_DIR, "**", "*.wav"), recursive=True))
if not files:
    raise SystemExit(f"لا يوجد ملفات داخل {AUDIO_DIR}")

for p in tqdm(files, desc="Formants"):
    try:
        f1, f2 = extract_file(p)
        dur = librosa.get_duration(path=p)
        rows.append({"path":p, "vowel":infer_vowel(p), "type":infer_type(p),
                     "duration":dur, "F1":f1, "F2":f2})
    except Exception as e:
        rows.append({"path":p, "vowel":"error", "type":"unknown",
                     "duration":np.nan, "F1":np.nan, "F2":np.nan, "error":str(e)})

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"Saved {OUT_CSV}")

# --------- رسم F2 مقابل F1 ---------
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    ok = df.dropna(subset=["F1","F2"])
    # نقيّم اللون حسب الـ vowel
    for v, sub in ok.groupby("vowel"):
        plt.scatter(sub["F2"], sub["F1"], s=20, label=v, alpha=0.7)
    plt.gca().invert_xaxis()  # متعارف: فونسبيس F2 يقل يساراً
    plt.gca().invert_yaxis()  # اختياري: لفونسبيس تقليدي
    plt.xlabel("F2 (Hz)"); plt.ylabel("F1 (Hz)")
    plt.title("Vowel Space (F2 vs F1)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print(f"Saved {OUT_PNG}")
except Exception as e:
    print("تعذر الرسم:", e)
