"""Generator notebook 04-06 untuk project churn Telco."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(r"C:\Users\53\Documents\data sains\machine learning\project ML Supervised")
NB = ROOT / "notebooks"

SETUP = '''from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 4)

BASE = Path.cwd()
if not (BASE / "data").exists():
    BASE = BASE.parent
print("root project:", BASE)'''


def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return n


def md(t):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t):
    return nbf.v4.new_code_cell(t.strip("\n"))


# ---------------------------------------------------------------- 04
n04 = nb([
md("""
# 04 — EDA: Mencari Jawaban "Kenapa Mereka Pergi?"

**Kenapa notebook ini ada?** Notebook 01–03 menyiapkan data. Notebook ini bertanya
pada datanya. Tujuannya bukan membuat banyak grafik, tapi menjawab empat pertanyaan
yang akan dipakai di presentasi:

1. Kapan pelanggan paling sering pergi? (`tenure`)
2. Apa yang membedakan yang pergi dan bertahan? (`Contract`)
3. Apakah harga benar-benar penyebabnya? (`MonthlyCharges`)
4. Layanan mana yang paling berkaitan dengan churn?

Setiap grafik di sini punya satu kalimat kesimpulan. Kalau sebuah grafik tidak
menghasilkan kesimpulan, grafik itu dibuang.
"""),
md("""
## 0. Persiapan
"""),
code(SETUP + '''

df = pd.read_csv(BASE / "data" / "clean" / "churn_clean.csv")
print("data:", df.shape)
print("churn rate:", f"{df['Churn'].mean():.2%}")'''),
md("""
## 1. Kapan pelanggan pergi?

`tenure` (masa berlangganan dalam bulan) kita kelompokkan agar polanya terlihat.
Median tenure adalah 29 bulan, jadi pembagian di bawah masuk akal.
"""),
code('''df["kelompok_tenure"] = pd.cut(
    df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
    labels=["0-6 bln", "7-12 bln", "13-24 bln", "25-48 bln", "49-72 bln"],
)

t = df.groupby("kelompok_tenure", observed=True)["Churn"].agg(
    jumlah="size", churn="sum", rate="mean")
t["rate"] = (t["rate"] * 100).round(1)
t["porsi_churn"] = (t["churn"] / t["churn"].sum() * 100).round(1)
t'''),
code('''fig, ax = plt.subplots(figsize=(8, 4))
warna = ["#dd8452" if r > 30 else "#4c72b0" for r in t["rate"]]
bars = ax.bar(t.index.astype(str), t["rate"], color=warna)
ax.axhline(df["Churn"].mean() * 100, color="gray", linestyle="--", linewidth=1.2,
           label=f"rata-rata keseluruhan ({df['Churn'].mean()*100:.1f}%)")
for b, r in zip(bars, t["rate"]):
    ax.annotate(f"{r:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10)
ax.set_title("Churn rate menurut masa berlangganan")
ax.set_xlabel("masa berlangganan")
ax.set_ylabel("churn rate (%)")
ax.set_ylim(0, 62)
ax.legend()
plt.tight_layout()
plt.show()'''),
md("""
**Temuan #1 — setengah pelanggan baru pergi dalam 6 bulan.**

| Kelompok | Churn rate | Porsi dari seluruh churn |
|---|---|---|
| **0–6 bulan** | **52,9%** | **41,9%** |
| 7–12 bulan | 35,9% | 13,5% |
| 13–24 bulan | 28,7% | 15,7% |
| 25–48 bulan | 20,4% | 17,4% |
| 49–72 bulan | 9,5% | 11,4% |

Dua angka ini menjawab pertanyaan berbeda dan keduanya penting:

- **Rate 52,9%** → kalau kamu pelanggan baru, risikonya sangat tinggi (per orang)
- **Porsi 41,9%** → kelompok ini penyumbang churn terbesar (secara volume)

Pelanggan yang bertahan 4 tahun ke atas hanya 9,5% yang pergi. Artinya masalahnya
**bukan di semua pelanggan**, tapi terkonsentrasi di masa awal.
"""),
md("""
## 2. Apa yang membedakan yang pergi dan bertahan?
"""),
code('''t = df.groupby("Contract")["Churn"].agg(jumlah="size", churn="sum", rate="mean")
t["rate"] = (t["rate"] * 100).round(1)
t["porsi_churn"] = (t["churn"] / t["churn"].sum() * 100).round(1)
t = t.sort_values("rate", ascending=False)
t'''),
code('''fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

a1.bar(t.index, t["rate"], color=["#dd8452", "#4c72b0", "#4c72b0"])
for i, v in enumerate(t["rate"]):
    a1.annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom", fontsize=10)
a1.set_title("Churn rate per tipe kontrak")
a1.set_ylabel("churn rate (%)")
a1.set_ylim(0, 50)

a2.bar(t.index, t["porsi_churn"], color=["#dd8452", "#4c72b0", "#4c72b0"])
for i, v in enumerate(t["porsi_churn"]):
    a2.annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom", fontsize=10)
a2.set_title("Sumbangan tiap kontrak ke total churn")
a2.set_ylabel("porsi churn (%)")
a2.set_ylim(0, 100)

plt.tight_layout()
plt.show()'''),
md("""
**Temuan #2 — kontrak bulanan adalah sumber masalahnya, bukan cuma kelompok berisiko.**

- Churn rate: bulanan **42,7%** vs 1 tahun 11,3% vs 2 tahun **2,8%**
- Porsi churn: bulanan **88,6%**

Kontrak bulanan hanya 55% dari pelanggan, tapi menyumbang 88,6% dari semua yang
pergi. Ini kalimat yang layak masuk slide: *"bukan kelompok berisiko tinggi saja —
ini sumber masalahnya."*
"""),
md("""
## 3. Apakah benar penyebabnya harga?

Ini pertanyaan yang paling sering dijawab salah. Intuisinya: makin mahal, makin
banyak yang pergi. Mari lihat datanya.
"""),
code('''banding = df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().round(2)
banding.index = ["Bertahan (0)", "Churn (1)"]
banding'''),
code('''fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=a1, palette=["#4c72b0", "#dd8452"])
a1.set_xticks([0, 1])
a1.set_xticklabels(["Bertahan", "Churn"])
a1.set_title("Sebaran tagihan bulanan")
a1.set_xlabel("")

rata = df.groupby("Churn")[["tenure", "MonthlyCharges"]].mean()
x = np.arange(2)
a2.bar(x - 0.2, rata["MonthlyCharges"], 0.4, label="Tagihan bulanan ($)", color="#dd8452")
a2.bar(x + 0.2, rata["tenure"], 0.4, label="Tenure (bulan)", color="#4c72b0")
a2.set_xticks(x)
a2.set_xticklabels(["Bertahan", "Churn"])
a2.set_title("Rata-rata tagihan vs masa berlangganan")
a2.legend()
plt.tight_layout()
plt.show()'''),
md("""
**Temuan #3 — yang pergi justru membayar LEBIH MAHAL.**

| | Tenure rata-rata | Tagihan bulanan rata-rata |
|---|---|---|
| Bertahan | 37,6 bulan | **$61,27** |
| Churn | 18,0 bulan | **$74,44** |

Ini membalik dugaan umum. Pelanggan yang pergi membayar ~$13 lebih mahal per bulan.
Jadi **churn bukan soal harga murah**, tapi soal **nilai yang dirasa sebanding
dengan harga**. Yang bayar mahal (biasanya fiber optic) merasa tidak dapat imbalan
setara, lalu pergi.

Perhatikan juga: mereka yang pergi punya tenure jauh lebih pendek (18 vs 37 bulan).
Jadi dua faktor ini saling terkait, bukan berdiri sendiri.
"""),
md("""
## 4. Layanan mana yang paling berkaitan?
"""),
code('''fig, axes = plt.subplots(2, 2, figsize=(12, 7))
pasangan = [
    ("PaymentMethod", "Metode bayar"),
    ("InternetService", "Layanan internet"),
    ("TechSupport", "Dukungan teknis"),
    ("OnlineSecurity", "Keamanan online"),
]

for ax, (kolom, judul) in zip(axes.ravel(), pasangan):
    r = df.groupby(kolom)["Churn"].mean().mul(100).sort_values(ascending=False)
    ax.barh(r.index.astype(str), r.values, color="#dd8452")
    for i, v in enumerate(r.values):
        ax.annotate(f"{v:.1f}%", (v, i), va="center", ha="left", fontsize=9)
    ax.set_title(f"Churn rate: {judul}")
    ax.set_xlabel("churn rate (%)")
    ax.set_xlim(0, 55)
    ax.invert_yaxis()

plt.tight_layout()
plt.show()'''),
md("""
**Temuan #4 — ada tiga penanda risiko yang sangat jelas.**

| Faktor | Kelompok berisiko | Churn rate |
|---|---|---|
| Metode bayar | Electronic check | **45,3%** |
| Layanan internet | Fiber optic | **41,9%** |
| Dukungan teknis | Tidak punya | **41,6%** |
| Keamanan online | Tidak punya | **41,8%** |

Pembandingnya: pelanggan dengan autodebit ~15–17%, DSL 19,0%, punya tech support
15,2%. Selisihnya bukan sedikit — hampir **tiga kali lipat**.

Satu hal yang perlu hati-hati: ini **korelasi, bukan sebab-akibat**. Kita tidak
bisa bilang "tidak punya tech support menyebabkan churn". Bisa jadi sebaliknya:
yang sudah niat pergi memang tidak memperpanjang layanan tambahan.
"""),
md("""
## 5. Ringkasan EDA

| # | Temuan | Angka kunci |
|---|---|---|
| 1 | Churn terkonsentrasi di masa awal | 52,9% churn di 6 bulan pertama; 41,9% dari seluruh churn |
| 2 | Kontrak bulanan = sumber masalah | rate 42,7%, porsi **88,6%** |
| 3 | Bukan soal harga | churn bayar $74,44 vs bertahan $61,27 |
| 4 | Tiga penanda risiko | electronic check 45,3%, fiber 41,9%, tanpa tech support 41,6% |

**Yang belum dijawab:** kalau semua faktor ini digabung, seberapa akurat kita bisa
menebak **per orang**? Itu pekerjaan notebook 05.
"""),
md("""
## 6. Panduan storytelling (dipakai di notebook 06)

Aturan yang dipegang: **satu temuan = satu kalimat**, dan setiap kalimat harus
punya angka pendukung.

| Temuan | Kalimat untuk presentasi |
|---|---|
| #1 | "Setengah pelanggan baru pergi dalam 6 bulan pertama." |
| #2 | "Kontrak bulanan cuma 55% pelanggan, tapi menyumbang 88,6% churn." |
| #3 | "Yang pergi justru yang bayar lebih mahal — jadi ini soal nilai, bukan harga." |
| #4 | "Tanpa tech support, risiko churn hampir tiga kali lipat." |

Hindari kalimat tanpa angka ("banyak pelanggan yang churn") dan hindari angka tanpa
makna ("churn rate 42,7%"). Selalu pasangkan keduanya.
"""),
])

# ---------------------------------------------------------------- 05
n05 = nb([
md("""
# 05 — Modeling & Evaluasi

**Kenapa notebook ini ada?** EDA menjawab "kelompok mana yang berisiko". Notebook
ini menjawab pertanyaan yang lebih sulit: **bisa atau tidak kita menebak per orang?**

**Urutan kerja yang dipegang:**

```
baseline naif  →  3 model + cross-validation  →  tuning  →  threshold  →  evaluasi akhir
```

Aturan besi: data uji **hanya** disentuh sekali, di bagian paling akhir.
"""),
md("""
## 0. Persiapan
"""),
code(SETUP + '''
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     cross_val_predict, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)

FEAT = BASE / "data" / "clean" / "churn_features.joblib"
data = joblib.load(FEAT)

X_train, X_test = data["X_train"], data["X_test"]
X_train_sc, X_test_sc = data["X_train_scaled"], data["X_test_scaled"]
y_train, y_test = data["y_train"], data["y_test"]

print("train:", X_train.shape, "| test:", X_test.shape)
print("churn di train:", int(y_train.sum()), f"({y_train.mean():.2%})")
print("churn di test :", int(y_test.sum()), f"({y_test.mean():.2%})")'''),
md("""
## 1. Baseline naif — bukti bahwa accuracy menipu

Sebelum memuji model apa pun, kita perlu pembanding paling bodoh: **selalu menjawab
"tidak churn"**. Kalau model kita tidak jauh lebih baik dari ini, model kita tidak
berguna.
"""),
code('''dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
pred_dummy = dummy.predict(X_test)

print("tebakan yang dipakai model:", dummy.predict(X_test)[:5])
print()
print(f"accuracy : {accuracy_score(y_test, pred_dummy):.3f}")
print(f"recall   : {recall_score(y_test, pred_dummy):.3f}")
print(f"precision: {precision_score(y_test, pred_dummy, zero_division=0):.3f}")'''),
md("""
Model yang **tidak belajar apa pun** mendapat akurasi **73,5%** — persis proporsi
pelanggan yang bertahan. Tapi `recall`-nya **0**: tidak satu pun pelanggan yang
akan pergi berhasil ditangkap.

Inilah alasan project ini memakai **recall** sebagai metrik utama:

- Salah bilang "aman" padahal akan pergi → **pelanggan hilang** (mahal)
- Salah bilang "akan pergi" padahal aman → hanya **kelebihan budget** kampanye (murah)

Kesalahan pertama jauh lebih mahal, jadi kita optimalkan recall.
"""),
md("""
## 2. Tiga model, dibandingkan dengan cross-validation

Satu kali split bisa kebetulan "gampang" atau "susah". Cross-validation 5-fold
membagi data train menjadi 5 bagian, bergantian jadi data uji, lalu dirata-rata.
Hasilnya jauh lebih jujur.

Catatan teknis yang penting:

- **KNN** dan **Logistic Regression** memakai data **yang sudah di-scaling**
- **Random Forest** memakai data **tanpa scaling** (tidak butuh)
- LogReg dan RF memakai `class_weight="balanced"` agar tidak mengabaikan kelas minoritas
"""),
code('''cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

kandidat = {
    "Dummy (tebak mayoritas)": (DummyClassifier(strategy="most_frequent"), X_train),
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"), X_train_sc),
    "KNN (k=15)": (KNeighborsClassifier(n_neighbors=15), X_train_sc),
    "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=42,
                                             class_weight="balanced", n_jobs=-1), X_train),
}

hasil = []
for nama, (model, Xd) in kandidat.items():
    baris = {"model": nama}
    for metrik in ["recall", "f1", "roc_auc", "accuracy"]:
        skor = cross_val_score(model, Xd, y_train, cv=cv, scoring=metrik, n_jobs=1).mean()
        baris[metrik] = round(float(skor), 3)
    hasil.append(baris)
    print(baris)

tabel_cv = pd.DataFrame(hasil).set_index("model")
tabel_cv'''),
md("""
**Hasil cross-validation (data train):**

| Model | Recall | F1 | AUC | Accuracy |
|---|---|---|---|---|
| Dummy | 0,000 | 0,000 | 0,500 | 0,735 |
| **Logistic Regression** | **0,801** | **0,629** | **0,846** | 0,749 |
| KNN (k=15) | 0,559 | 0,577 | 0,821 | 0,783 |
| Random Forest | 0,649 | 0,609 | 0,828 | 0,779 |

Dua hal yang menarik di sini:

1. **KNN paling akurat (0,783) tapi recall-nya terburuk di antara tiga model (0,559).**
   Ini contoh sempurna kenapa accuracy tidak bisa dipakai sendirian.
2. **Logistic Regression mengalahkan Random Forest** di semua metrik utama. Model
   paling sederhana justru yang terbaik — dan ini wajar untuk data yang sebagian
   besar fiturnya biner hasil one-hot.

KNN lemah karena *curse of dimensionality*: 30 fitur membuat konsep "tetangga
terdekat" jadi kurang bermakna.
"""),
md("""
## 3. Tuning hyperparameter Random Forest

Kita tetap mencoba memperbaiki Random Forest. Kalau setelah tuning ia bisa
menyamai LogReg, kita punya dua kandidat yang layak dibandingkan di data uji.

Grid kecil saja — tujuannya mencari peningkatan nyata, bukan mengejar angka.
"""),
code('''grid = {
    "n_estimators": [200, 400],
    "max_depth": [8, 12, None],
    "min_samples_leaf": [1, 5, 10],
}

gs = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1),
    grid, cv=cv, scoring="recall", n_jobs=1,
)
gs.fit(X_train, y_train)

print("parameter terbaik:", gs.best_params_)
print("CV recall        :", round(gs.best_score_, 3))
print("kombinasi diuji  :", len(gs.cv_results_["params"]))'''),
md("""
Tuning menaikkan CV recall Random Forest dari **0,649** menjadi **0,795** — hampir
menyamai LogReg (0,801). Parameter terpilih: `max_depth=8`, `min_samples_leaf=10`,
`n_estimators=400`.

`max_depth=8` dan `min_samples_leaf=10` membuat pohon **dangkal dan konservatif**.
Itu masuk akal: pohon yang terlalu dalam menghafal data train dan kehilangan
kemampuan generalisasi.
"""),
md("""
## 4. Threshold — 0,5 tidak sakral

Model mengeluarkan **peluang** (0–1). Secara default, peluang ≥ 0,5 dianggap churn.
Tapi ambang itu bisa digeser sesuai kapasitas tim marketing.

Cara yang jujur memilih threshold: pakai prediksi **out-of-fold** dari data train
(bukan data uji!), lalu cari threshold yang memenuhi syarat bisnis.

Syarat yang dipakai: **precision minimal 0,50** (separuh dari yang dihubungi harus
benar-benar churn, agar budget tidak habis), lalu **recall dimaksimalkan**.
"""),
code('''rf = gs.best_estimator_

# prediksi out-of-fold: setiap baris diprediksi oleh model yang TIDAK melihatnya
oof = cross_val_predict(rf, X_train, y_train, cv=cv,
                        method="predict_proba", n_jobs=1)[:, 1]

baris = []
for t in np.arange(0.20, 0.66, 0.05):
    pred = (oof >= t).astype(int)
    baris.append({
        "threshold": round(float(t), 2),
        "precision": round(precision_score(y_train, pred, zero_division=0), 3),
        "recall": round(recall_score(y_train, pred), 3),
        "f1": round(f1_score(y_train, pred), 3),
    })

tabel_threshold = pd.DataFrame(baris)
tabel_threshold'''),
code('''layak = tabel_threshold[tabel_threshold["precision"] >= 0.50]
pilih = layak.loc[layak["recall"].idxmax()]
threshold = float(pilih["threshold"])

print("threshold terpilih :", threshold)
print("precision (train)  :", pilih["precision"])
print("recall (train)     :", pilih["recall"])'''),
md("""
Threshold terpilih: **0,50** — dengan precision 0,524 dan recall 0,795 di data train.

Artinya pada dataset ini, menggeser threshold **tidak memberi keuntungan** yang
sepadan. Kita tetap menampilkan tabelnya karena keputusan ini harus dibuktikan,
bukan diasumsikan. Kalau nanti kapasitas kampanye berubah (misal hanya sanggup
menghubungi 200 orang/bulan), threshold bisa digeser ke 0,55–0,60.
"""),
md("""
## 5. Evaluasi akhir di data uji — satu kali saja
"""),
code('''proba = rf.predict_proba(X_test)[:, 1]
pred = (proba >= threshold).astype(int)

cm = confusion_matrix(y_test, pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print()
print(classification_report(y_test, pred, target_names=["Bertahan", "Churn"], digits=3))
print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")'''),
code('''fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a1,
            xticklabels=["Prediksi: bertahan", "Prediksi: churn"],
            yticklabels=["Aktual: bertahan", "Aktual: churn"])
a1.set_title(f"Confusion matrix (threshold {threshold})")

fpr, tpr, _ = roc_curve(y_test, proba)
a2.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
a2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="tebakan acak (AUC 0,5)")
a2.set_xlabel("False Positive Rate")
a2.set_ylabel("True Positive Rate")
a2.set_title("ROC Curve")
a2.legend()

plt.tight_layout()
plt.show()'''),
md("""
**Hasil di data uji (1.409 pelanggan, 374 di antaranya benar-benar churn):**

| Metrik | Nilai | Artinya |
|---|---|---|
| Accuracy | 0,754 | 75,4% tebakan benar (hampir sama dengan dummy, jadi bukan ukuran utama) |
| **Recall** | **0,802** | **80,2% pelanggan yang akan pergi berhasil ditangkap** |
| Precision | 0,524 | Dari yang ditandai, 52,4% benar-benar pergi |
| F1 | 0,634 | keseimbangan recall & precision |
| ROC-AUC | 0,845 | kemampuan membedakan dua kelas tergolong baik |

Dari confusion matrix:

- **TP = 300** → 300 pelanggan berisiko berhasil diidentifikasi
- **FN = 74** → 74 pelanggan akan pergi tanpa terdeteksi
- **FP = 272** → 272 pelanggan aman ikut dihubungi (biaya kampanye)
- **TN = 763** → 763 pelanggan aman dan tidak diganggu

**Cara membaca untuk orang non-teknis:** dari 374 pelanggan yang benar-benar akan
pergi, model menangkap **300 orang**. Tanpa model, kita tidak punya cara memilih
siapa yang harus dihubungi lebih dulu.
"""),
md("""
## 6. Perbandingan final di data uji

Cross-validation bagus untuk memilih model, tapi keputusan akhir harus memakai
data uji yang belum pernah dilihat model.
"""),
code('''pembanding = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced"), X_train_sc, X_test_sc),
    "KNN (k=15)": (KNeighborsClassifier(n_neighbors=15), X_train_sc, X_test_sc),
    "Random Forest (tuned)": (rf, X_train, X_test),
}

baris = []
for nama, (m, a, b) in pembanding.items():
    if nama != "Random Forest (tuned)":
        m.fit(a, y_train)
    p = m.predict_proba(b)[:, 1]
    pr = (p >= 0.5).astype(int)
    baris.append({
        "model": nama,
        "recall": round(recall_score(y_test, pr), 3),
        "precision": round(precision_score(y_test, pr), 3),
        "f1": round(f1_score(y_test, pr), 3),
        "auc": round(roc_auc_score(y_test, p), 3),
        "akurasi": round(accuracy_score(y_test, pr), 3),
    })

pd.DataFrame(baris).set_index("model")'''),
md("""
| Model | Recall | Precision | F1 | AUC | Accuracy |
|---|---|---|---|---|---|
| Logistic Regression | 0,786 | 0,507 | 0,616 | 0,841 | 0,740 |
| KNN (k=15) | 0,535 | 0,570 | 0,552 | 0,808 | 0,769 |
| **Random Forest (tuned)** | **0,802** | 0,524 | **0,634** | **0,845** | 0,754 |

**Model final: Random Forest hasil tuning.**

Alasannya:
1. Recall tertinggi di data uji (**0,802**) — metrik utama project ini
2. F1 dan AUC juga tertinggi
3. Memberi **feature importance**, yang dibutuhkan untuk rekomendasi bisnis

**Kejujuran yang perlu disampaikan:** LogReg menang di cross-validation (0,801 vs
0,795) dan jauh lebih sederhana. Selisih di data uji hanya 1,6 poin recall — secara
statistik tipis. Kalau mentor bertanya *"kenapa tidak LogReg saja?"*, jawaban yang
jujur: *"keduanya layak; kami memilih RF karena unggul tipis di data uji dan
feature importance-nya lebih mudah diterjemahkan jadi rekomendasi."*
"""),
md("""
## 7. Fitur mana yang paling menentukan?
"""),
code('''importance = pd.Series(rf.feature_importances_, index=data["features"]).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
importance.head(10).sort_values().plot(kind="barh", ax=ax, color="#4c72b0")
ax.set_title("10 fitur paling berpengaruh (Random Forest)")
ax.set_xlabel("tingkat kepentingan")
plt.tight_layout()
plt.show()

importance.head(10).round(4)'''),
md("""
| Peringkat | Fitur | Kepentingan |
|---|---|---|
| 1 | `tenure` | 0,177 |
| 2 | `Contract_Two year` | 0,144 |
| 3 | `TotalCharges` | 0,116 |
| 4 | `InternetService_Fiber optic` | 0,095 |
| 5 | `PaymentMethod_Electronic check` | 0,066 |
| 6 | `MonthlyCharges` | 0,066 |
| 7 | `Contract_One year` | 0,048 |
| 8 | `OnlineSecurity_Yes` | 0,032 |
| 9 | `TechSupport_Yes` | 0,030 |

Tiga teratas semuanya berbicara tentang **lama berlangganan dan komitmen** — bukan
harga. Ini konsisten dengan temuan EDA dan memperkuat cerita utama project.
"""),
code('''# arah pengaruh: koefisien Logistic Regression (model yang lebih mudah dibaca)
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_train_sc, y_train)
koef = pd.Series(lr.coef_[0], index=data["features"]).sort_values()

print("Menurunkan risiko churn (koefisien negatif):")
print(koef.head(5).round(3).to_string())
print()
print("Menaikkan risiko churn (koefisien positif):")
print(koef.tail(5).round(3).to_string())'''),
md("""
Feature importance hanya bilang **seberapa penting**, bukan **arah pengaruhnya**.
Untuk arah, kita lihat koefisien LogReg:

**Menurunkan risiko:** `tenure`, `Contract_Two year`, `Contract_One year`,
`OnlineSecurity_Yes`, `Dependents_Yes`
**Menaikkan risiko:** `InternetService_Fiber optic`, `TotalCharges`,
`PaymentMethod_Electronic check`

Satu keanehan yang jujur perlu disebut: `MonthlyCharges` berkoefisien **negatif**
(setelah memperhitungkan `TotalCharges`). Ini gejala **multikolinearitas** —
`TotalCharges` ≈ `tenure` × `MonthlyCharges`, jadi ketiganya saling terkait. Model
Random Forest tidak terganggu, tapi interpretasi koefisien satuan harus hati-hati.
Ini masuk daftar limitasi.
"""),
md("""
## 8. Pembanding paling penting: aturan sederhana tanpa ML

Ini bagian yang membuat project ini berbeda. Sebelum memakai model, kita uji
**aturan manual** yang bisa dibuat siapa pun tanpa machine learning:

> "Tandai pelanggan yang kontraknya bulanan, tenure ≤ 12 bulan, dan tidak punya
> tech support."
"""),
code('''# X_test sudah ter-encode, jadi profil asli diambil ulang lewat indeksnya
profil = pd.read_csv(BASE / "data" / "clean" / "churn_clean.csv").loc[X_test.index]

aturan = (
    (profil["Contract"] == "Month-to-month")
    & (profil["tenure"] <= 12)
    & (profil["TechSupport"] == "No")
)

tp_rule = int((aturan & (y_test == 1)).sum())
print("pelanggan ditandai aturan :", int(aturan.sum()))
print("benar-benar churn         :", tp_rule)
print("precision                 :", round(tp_rule / max(int(aturan.sum()), 1), 3))
print("recall                    :", round(tp_rule / y_test.sum(), 3))
print()
print("--- model final ---")
print("pelanggan ditandai model  :", int(pred.sum()))
print("benar-benar churn         :", tp)
print("precision                 :", round(tp / max(int(pred.sum()), 1), 3))
print("recall                    :", round(tp / y_test.sum(), 3))'''),
md("""
| Pendekatan | Ditandai | Churn tertangkap | Precision | Recall |
|---|---|---|---|---|
| Aturan sederhana | 270 | 167 | **0,619** | 0,447 |
| **Model Random Forest** | 572 | **300** | 0,524 | **0,802** |

**Ini inti argumen project:** model menangkap **300** pelanggan berisiko, aturan
manual hanya **167** — selisih **133 orang** yang akan hilang tanpa model.

Tapi jujur: precision model lebih rendah (0,524 vs 0,619), artinya lebih banyak
"false alarm". Trade-off-nya: **model lebih baik dalam tidak melewatkan orang**,
aturan lebih baik dalam tidak mengganggu orang yang salah.

Untuk kasus churn, melewatkan orang lebih mahal → model menang. Kalau biaya
kampanye per orang sangat tinggi, aturan sederhana bisa jadi pilihan rasional.
Mengetahui kapan model **tidak** lebih baik adalah bagian dari pemahaman.
"""),
md("""
## 9. Simpan model

Satu file berisi model + threshold + metrik + daftar fitur, supaya dashboard bisa
memakainya tanpa mengulang training.
"""),
code('''MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "churn_model.joblib"

joblib.dump({
    "model": rf,
    "threshold": threshold,
    "features": data["features"],
    "encoder": data["encoder"],
    "scaler": data["scaler"],
    "metrics": {
        "recall": round(float(recall_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred)), 3),
        "f1": round(float(f1_score(y_test, pred)), 3),
        "auc": round(float(roc_auc_score(y_test, proba)), 3),
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
    },
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    "importance": importance.round(4).to_dict(),
    "best_params": gs.best_params_,
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "rule_baseline": {"flagged": int(aturan.sum()), "tp": tp_rule},
}, MODEL_PATH)

print("disimpan:", MODEL_PATH)
print("ukuran  :", round(MODEL_PATH.stat().st_size / 1024, 1), "KB")'''),
code('''cek = joblib.load(MODEL_PATH)
print("isi:", list(cek.keys()))
print()
print("threshold:", cek["threshold"])
print("metrics  :", cek["metrics"])
assert cek["threshold"] == threshold
assert len(cek["features"]) == 30
print()
print("verifikasi lulus")'''),
md("""
## 10. Kesimpulan & limitasi

**Yang berhasil:**

1. Model menangkap **80,2%** pelanggan yang akan pergi (recall), dengan AUC 0,845.
2. Model mengalahkan aturan sederhana: **300 vs 167** pelanggan tertangkap.
3. Tiga fitur terpenting (`tenure`, `Contract`, `TotalCharges`) mendukung cerita EDA:
   masalahnya **komitmen**, bukan harga.

**Limitasi yang harus disebut jujur:**

| Limitasi | Dampak |
|---|---|
| Data satu perusahaan, satu waktu | Belum tentu berlaku untuk provider lain |
| Hanya 26,5% churn | Kelas minoritas cukup kecil; precision masih 52,4% |
| 22 profil identik | Potensi kebocoran sangat kecil (0,3%) |
| `TotalCharges` ≈ `tenure` × `MonthlyCharges` | Multikolinearitas, koefisien satuan sulit ditafsirkan |
| Tidak ada data kompetitor/harga pasar | Tidak bisa menjelaskan **alasan** pelanggan pergi |
| Korelasi ≠ sebab-akibat | Tidak bisa bilang "tech support mencegah churn" |

**Yang belum dikerjakan (bisa jadi pengembangan):**

- Uji model pada data tahun berbeda untuk melihat penurunan performa
- Tambahkan biaya kampanye untuk menghitung *expected value* per pelanggan
- Coba XGBoost / LightGBM sebagai pembanding ensemble
"""),
])

# ---------------------------------------------------------------- 06
n06 = nb([
md("""
# 06 — Storytelling: Cara Menjelaskan Project Ini dalam 8–10 Menit

**Kenapa notebook ini ada?** Pekerjaan data tidak selesai saat model jadi — selesai
ketika orang lain paham dan mau memakainya.

Notebook ini bukan analisis baru. Isinya **cara menceritakan** hasil notebook 01–05,
dengan pola yang sama seperti project sebelumnya:

> ## Hook → Angka → Makna → Rekomendasi
>
> 1. **Hook** — kalimat pembuka yang membuat orang mau mendengarkan
> 2. **Angka** — 1–2 bukti, bukan seluruh tabel
> 3. **Makna** — terjemahan angka ke bahasa manusia
> 4. **Rekomendasi** — apa yang bisa dilakukan pembaca

**Aturan emasnya:** setiap angka yang disebut harus bisa ditunjukkan sumbernya di
notebook 04 atau 05.
"""),
md("""
## 0. Persiapan
"""),
code(SETUP + '''

df = pd.read_csv(BASE / "data" / "clean" / "churn_clean.csv")
df["kelompok_tenure"] = pd.cut(
    df["tenure"], bins=[-1, 6, 12, 24, 48, 72],
    labels=["0-6 bln", "7-12 bln", "13-24 bln", "25-48 bln", "49-72 bln"],
)
print("data:", df.shape)'''),
md("""
## 1. Cerita #1 — "Setengah pelanggan baru pergi dalam 6 bulan"

**Hook:** "Kalau Anda punya 100 pelanggan baru hari ini, tiga bulan lagi tinggal 70.
Enam bulan lagi, sebagian dari mereka sudah pindah ke kompetitor."

**Angka:** pelanggan dengan masa berlangganan 0–6 bulan punya churn rate **52,9%** —
dua kali lipat rata-rata keseluruhan (26,5%). Kelompok ini menyumbang **41,9%** dari
seluruh churn, padahal hanya 21% dari pelanggan.

**Makna:** masalahnya bukan pelanggan lama yang bosan, tapi **pelanggan baru yang
tidak pernah merasa betah**. Masa kritisnya ada di awal, bukan di akhir.

**Rekomendasi:** program onboarding 90 hari — kontak proaktif di minggu 1, 4, dan 12.
"""),
code('''t = df.groupby("kelompok_tenure", observed=True)["Churn"].agg(jumlah="size", rate="mean")
t["rate"] = (t["rate"] * 100).round(1)
t["porsi"] = (df.groupby("kelompok_tenure", observed=True)["Churn"].sum()
              / df["Churn"].sum() * 100).round(1)

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(t.index.astype(str), t["rate"], color=["#dd8452"] + ["#4c72b0"] * 4)
for b, v in zip(bars, t["rate"]):
    ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center", va="bottom", fontsize=10)
ax.axhline(df["Churn"].mean() * 100, color="gray", ls="--", lw=1.2,
           label=f"rata-rata {df['Churn'].mean()*100:.1f}%")
ax.set_title("Churn rate menurut masa berlangganan")
ax.set_ylabel("churn rate (%)")
ax.set_ylim(0, 62)
ax.legend()
plt.tight_layout()
plt.show()
t'''),
md("""
## 2. Cerita #2 — "Yang pergi justru yang bayar lebih mahal"

**Hook:** "Kalau ditanya kenapa pelanggan pergi, jawaban paling umum adalah 'kemahalan'.
Datanya bilang sebaliknya."

**Angka:** pelanggan yang pergi membayar rata-rata **$74,44/bulan**, sedangkan yang
bertahan **$61,27** — selisih 21% **lebih mahal**. Ditambah lagi, **88,6%** dari
seluruh churn berasal dari kontrak bulanan (churn rate 42,7% vs 2,8% untuk kontrak
2 tahun).

**Makna:** ini bukan masalah **harga**, tapi masalah **komitmen dan nilai**. Pelanggan
mau bayar mahal asal merasa dapat imbalan setara. Yang tidak merasa, pergi — dan
biasanya mereka yang terkunci tanpa kontrak.

**Rekomendasi:** jangan turunkan harga. Dorong **perpanjangan kontrak** untuk
pelanggan fiber optic yang masih bulanan — insentif 1 tahun lebih murah daripada
kehilangan pelanggan $74/bulan.
"""),
code('''fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

rata = df.groupby("Churn")["MonthlyCharges"].mean()
a1.bar(["Bertahan", "Churn"], rata.values, color=["#4c72b0", "#dd8452"])
for i, v in enumerate(rata.values):
    a1.annotate(f"${v:.2f}", (i, v), ha="center", va="bottom", fontsize=11)
a1.set_title("Rata-rata tagihan bulanan")
a1.set_ylabel("dolar per bulan")
a1.set_ylim(0, 85)

t = df.groupby("Contract")["Churn"].mean().mul(100).sort_values(ascending=False)
a2.bar(t.index, t.values, color=["#dd8452", "#4c72b0", "#4c72b0"])
for i, v in enumerate(t.values):
    a2.annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom", fontsize=10)
a2.set_title("Churn rate per tipe kontrak")
a2.set_ylabel("churn rate (%)")
a2.set_ylim(0, 50)

plt.tight_layout()
plt.show()'''),
md("""
## 3. Cerita #3 — "Model mengalahkan aturan sederhana, tapi tidak sempurna"

**Hook:** "Sebelum percaya machine learning, saya uji dulu: apa bisa ditebak pakai
aturan biasa? Karena kalau iya, kita tidak butuh ML."

**Angka:** aturan sederhana — *kontrak bulanan + tenure ≤ 12 bulan + tanpa tech
support* — menandai 270 pelanggan dan menangkap **167** dari 374 yang benar-benar
churn (recall 44,7%). Model Random Forest menangkap **300** (recall 80,2%) dengan
AUC 0,845.

**Makna:** model menangkap **133 pelanggan lebih banyak**. Tapi model juga
menghasilkan lebih banyak alarm palsu (precision 52,4% vs 61,9%). Jadi model ini
layak dipakai **sebagai daftar prioritas**, bukan sebagai keputusan otomatis.

**Rekomendasi:** pakai skor model untuk mengurutkan pelanggan, lalu hubungi dari
skor tertinggi sampai kapasitas tim habis. Jangan pakai model untuk menolak layanan
ke pelanggan.
"""),
code('''import joblib
from sklearn.metrics import confusion_matrix

model_data = joblib.load(BASE / "models" / "churn_model.joblib")
m = model_data["metrics"]
c = model_data["confusion"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))

cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a1,
            xticklabels=["Prediksi: bertahan", "Prediksi: churn"],
            yticklabels=["Aktual: bertahan", "Aktual: churn"])
a1.set_title("Model: 300 dari 374 churn tertangkap")

a2.bar(["Aturan sederhana", "Model ML"], [167, 300], color=["#4c72b0", "#dd8452"])
for i, v in enumerate([167, 300]):
    a2.annotate(f"{v} pelanggan", (i, v), ha="center", va="bottom", fontsize=11)
a2.set_title("Churn yang berhasil ditangkap (dari 374)")
a2.set_ylabel("jumlah pelanggan")
a2.set_ylim(0, 360)

plt.tight_layout()
plt.show()
print("metrik model:", m)'''),
md("""
## 4. Naskah presentasi 8–10 menit

Bacakan ini sambil membuka notebook 01–05. Angka dalam kurung menunjukkan waktu.

---

**(0:00–0:40) Pembuka — masalah**

> "Perusahaan ini punya 7.043 pelanggan, dan 26,5% di antaranya sudah berhenti
> berlangganan. Mempertahankan pelanggan jauh lebih murah daripada mencari yang
> baru, jadi pertanyaan saya sederhana: **bisa tidak kita tahu siapa yang akan pergi
> sebelum dia benar-benar pergi?**"

---

**(0:40–1:40) Data & tantangan**

> "Datanya 21 kolom per pelanggan: profil, layanan, kontrak, tagihan. Yang menarik,
> tidak ada nilai kosong sama sekali — hanya satu kolom, `TotalCharges`, yang
> tersimpan sebagai teks dengan 11 nilai kosong. Semuanya pelanggan baru dengan
> tenure nol, jadi saya isi nol karena mereka memang belum pernah ditagih.
>
> Tantangan utamanya: kelasnya tidak seimbang, 26,5% versus 73,5%. Artinya model
> yang selalu menjawab 'tidak churn' sudah dapat akurasi 73,5% tanpa belajar
> apa pun. **Jadi saya tidak memakai accuracy** — saya memakai recall, karena
> kesalahan yang mahal adalah bilang 'aman' padahal pelanggan akan pergi."

---

**(1:40–3:30) Tiga temuan EDA**

> "Temuan pertama, **masalahnya ada di awal**. Pelanggan dengan masa berlangganan
> di bawah 6 bulan punya churn rate 52,9% — dua kali rata-rata. Kelompok ini
> menyumbang 41,9% dari seluruh churn.
>
> Temuan kedua, **kontrak bulanan adalah sumber masalahnya**. Churn rate-nya 42,7%,
> sementara kontrak 2 tahun hanya 2,8%. Kontrak bulanan cuma 55% pelanggan, tapi
> menyumbang 88,6% dari semua yang pergi.
>
> Temuan ketiga yang paling mengejutkan: **yang pergi justru membayar lebih mahal**.
> $74,44 versus $61,27. Jadi ini bukan soal harga, tapi soal nilai dan komitmen."

---

**(3:30–4:30) Metode**

> "Saya bandingkan tiga model: Logistic Regression, KNN, dan Random Forest, dengan
> cross-validation 5-fold. Data dibagi 80–20 dengan stratifikasi, dan scaling hanya
> dilakukan pada data latih supaya tidak terjadi data leakage. Saya juga menambahkan
> baseline naif dan satu aturan manual sebagai pembanding."

---

**(4:30–6:30) Hasil**

> "Di data uji, Random Forest hasil tuning memberi recall **80,2%** dengan AUC 0,845.
> Dari 374 pelanggan yang benar-benar akan pergi, model menangkap **300 orang**.
>
> Yang penting saya sampaikan jujur: precision-nya hanya 52,4%. Artinya dari yang
> ditandai model, separuhnya adalah alarm palsu. Untuk kasus ini saya menerima
> trade-off itu, karena melewatkan pelanggan yang akan pergi lebih mahal daripada
> menghubungi orang yang ternyata aman.
>
> Saya juga membandingkan dengan aturan sederhana tanpa ML. Aturan itu hanya
> menangkap 167 orang. Model menangkap 133 orang lebih banyak."

---

**(6:30–8:00) Insight & rekomendasi**

> "Fitur paling berpengaruh adalah tenure, tipe kontrak, dan total tagihan — semuanya
> soal **lama berlangganan dan komitmen**, bukan harga.
>
> Tiga rekomendasi konkret:
> Satu, program onboarding 90 hari untuk pelanggan baru, karena 6 bulan pertama
> adalah masa paling rawan.
> Dua, dorong perpanjangan kontrak untuk pelanggan fiber optic yang masih bulanan.
> Tiga, pakai skor model untuk mengurutkan prioritas kontak marketing — bukan untuk
> menolak layanan."

---

**(8:00–9:00) Limitasi & penutup**

> "Saya perlu jujur soal batasannya. Data ini dari satu perusahaan pada satu periode,
> jadi belum tentu berlaku di tempat lain. Model ini menangkap **korelasi**, bukan
> sebab-akibat — saya tidak bisa bilang 'tech support mencegah churn'. Dan precision
> 52% berarti kampanye tetap akan menyentuh sebagian pelanggan yang sebenarnya aman.
>
> Jadi kesimpulan saya: model ini bukan peramal nasib, tapi **daftar prioritas** yang
> jauh lebih baik daripada menebak atau menghubungi semua orang."

---

## 5. Versi 30 detik (untuk README atau kalau waktu dipotong)

> "Kami menganalisis 7.043 pelanggan telco dengan churn 26,5%. Tiga temuan:
> **pertama**, setengah pelanggan baru pergi dalam 6 bulan; **kedua**, kontrak
> bulanan menyumbang 88,6% churn padahal hanya 55% pelanggan; **ketiga**, yang pergi
> justru membayar lebih mahal — jadi ini soal komitmen, bukan harga. Model Random
> Forest menangkap **80,2%** pelanggan yang akan pergi, mengalahkan aturan sederhana
> yang hanya menangkap 44,7%. Semua angka bisa dicek di notebook 04 dan 05."
"""),
md("""
## 6. Setiap cerita punya "rumah" yang bisa dicek

Supaya audiens bisa memverifikasi sendiri:

| Cerita | Bukti di notebook | Bagian |
|---|---|---|
| #1 Churn di 6 bulan pertama | 04 | bagian 1 |
| #2 Bukan harga, tapi komitmen | 04 | bagian 2 & 3 |
| #3 Model vs aturan sederhana | 05 | bagian 5 & 8 |
| Fitur paling berpengaruh | 05 | bagian 7 |
| Keputusan metrik (recall) | 05 | bagian 1 |

**Aturan emas storytelling data:** setiap angka yang kamu sebut di presentasi harus
bisa diklik orang lain sampai ke sumbernya.
"""),
md("""
## 7. Antisipasi pertanyaan mentor

Siapkan jawaban ini — semuanya sudah ada dasarnya di notebook.

| Pertanyaan | Jawaban singkat |
|---|---|
| **Kenapa recall, bukan accuracy?** | Model yang selalu jawab "tidak churn" sudah dapat akurasi 73,5% tanpa belajar. Recall mengukur hal yang benar-benar penting: berapa pelanggan berisiko yang tertangkap. |
| **Kenapa tidak pakai SMOTE?** | Kelasnya tidak ekstrem (26,5%) dan `class_weight="balanced"` sudah menangani. SMOTE membuat data sintetis yang tidak ada di dunia nyata. |
| **Kenapa Random Forest, bukan LogReg?** | Unggul tipis di data uji (recall 0,802 vs 0,786). Tapi jujur: LogReg menang di CV dan lebih sederhana — keduanya pilihan sah. |
| **Apa itu data leakage?** | Informasi data uji bocor ke proses training. Dicegah dengan split dulu, baru `fit` encoder & scaler hanya di data latih. |
| **Kenapa threshold 0,5?** | Diuji dari 0,20–0,65 dengan prediksi out-of-fold. 0,50 yang terbaik dengan syarat precision ≥ 0,50. |
| **Kenapa tidak XGBoost?** | Bisa, dan mungkin lebih akurat. Tapi project ini menekankan pemahaman alur, dan RF sudah cukup dengan feature importance yang mudah dijelaskan. |
| **Precision cuma 52%, apa tidak buruk?** | Untuk penggunaan sebagai daftar prioritas, tidak. Yang penting model tidak melewatkan orang berisiko. Kalau budget kampanye terbatas, threshold dinaikkan ke 0,60 → precision 0,596, recall 0,677. |
| **Apakah ini sebab-akibat?** | Tidak. Ini korelasi. Tidak ada eksperimen (A/B test) yang membuktikan tech support mencegah churn. |

## 8. Lima langkah storytelling yang bisa dipakai lagi

1. **Tentukan 1 pesan utama** — bukan 10 temuan. Di sini: *"churn itu masalah
   komitmen di masa awal, bukan harga."*
2. **Susun dengan pola Hook → Angka → Makna → Rekomendasi.**
3. **Pilih 1–2 grafik terbaik** sebagai bukti, bukan semua grafik.
4. **Sebutkan keterbatasan dengan jujur** — justru menaikkan kredibilitas.
5. **Tunjukkan sumbernya** supaya audiens bisa memeriksa sendiri.

> "Data membuat angka terdengar masuk akal. Storytelling membuat angka terdengar
> masuk akal untuk dijalankan."
"""),
])

for name, notebook in [("04_eda_churn.ipynb", n04),
                       ("05_modeling_evaluasi.ipynb", n05),
                       ("06_storytelling.ipynb", n06)]:
    nbf.write(notebook, str(NB / name))
    print("dibuat:", name)
