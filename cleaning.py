import os
import librosa
import soundfile as sf

# مجلد البيانات الأصلية
input_dir = "data"
# مجلد البيانات بعد المعالجة
output_dir = "data_clean"
os.makedirs(output_dir, exist_ok=True)

# فلتر Pre-emphasis
def pre_emphasis(signal, alpha=0.97):
    return signal[1:] - alpha * signal[:-1]

# المرور على كل الملفات الصوتية
for root, dirs, files in os.walk(input_dir):
    for file in files:
        if file.lower().endswith(".wav"):
            in_path = os.path.join(root, file)
            print(f"Processing: {in_path}")

            # قراءة الملف وتحويله لـ 16kHz
            y, sr = librosa.load(in_path, sr=16000)

            # تطبيق pre-emphasis
            y = pre_emphasis(y)

            # إزالة الصمت من البداية والنهاية
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)

            # تحديد مسار الحفظ الجديد مع الحفاظ على البنية
            relative_path = os.path.relpath(in_path, input_dir)
            out_path = os.path.join(output_dir, relative_path)

            # إنشاء المجلد الفرعي إذا ما كان موجود
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            # حفظ الملف
            sf.write(out_path, y_trimmed, 16000)

print("✅ تم تجهيز جميع الملفات وحفظها في:", output_dir)
