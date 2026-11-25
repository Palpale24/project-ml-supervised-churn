"""Generator notebook 01-03 untuk project churn Telco."""
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

# notebook dijalankan dari folder notebooks/, jadi naik satu level ke root project
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


# ---------------------------------------------------------------- 01
n01 = nb([
md("""
# 01 — Data Understanding

**Kenapa notebook ini ada?** Sebelum menyentuh model, kita harus tahu bentuk datanya.
Semua keputusan berikutnya (metrik apa, model apa, preprocessing apa) bergantung pada
jawaban lima pertanyaan di bawah. Melewatkan tahap ini adalah penyebab nomor satu
project machine learning yang hasilnya tidak bisa dijelaskan.

**Lima pertanyaan yang harus dijawab:**

1. Berapa baris dan kolom? Apa saja isinya?
2. Tipe datanya apa? Mana angka, mana teks?
3. Ada data kotor (kosong, duplikat, tipe salah)?
4. Targetnya apa, dan seimbang atau tidak?
5. Fitur mana yang kelihatannya berhubungan dengan target?
"""),
md("""
## 0. Persiapan
"""),
code(SETUP + '''

RAW = BASE / "data" / "raw" / "telco_churn.csv"
df = pd.read_csv(RAW)
print("file:", RAW.name)'''),
md("""
## 1. Bentuk data
"""),
code('''print("jumlah baris :", df.shape[0])
print("jumlah kolom :", df.shape[1])
print()
print("Daftar kolom:")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:2d}. {c}")'''),
code('''df.head()'''),
md("""
Setiap baris = **satu pelanggan**. Kolom terakhir (`Churn`) adalah jawaban yang mau
diprediksi. Sisanya adalah "pertanyaan" yang boleh dipakai model.

Satu kolom yang perlu dicurigai sejak awal: `customerID` — isinya kode unik per
pelanggan, bukan sifat pelanggan. Nanti kolom ini dibuang.
"""),
md("""
## 2. Tipe data
"""),
code('''ringkas = df.dtypes.astype(str).value_counts()
print("Ringkasan tipe data:")
print(ringkas.to_string())
print()
print("Kolom numerik :", df.select_dtypes(include=np.number).columns.tolist())
print()
print("Kolom teks    :", df.select_dtypes(include="str").columns.tolist())'''),
md("""
Ternyata hanya **4 kolom** yang berupa angka (`SeniorCitizen`, `tenure`,
`MonthlyCharges`, `TotalCharges`), sisanya teks. Ini penting: model scikit-learn
**tidak bisa** menerima teks, jadi 15 kolom teks harus diubah jadi angka pada
notebook 03.
"""),
md("""
## 3. Kualitas data — cari masalahnya sekarang
"""),
code('''print("Jumlah nilai kosong (NaN) per kolom:")
kosong = df.isna().sum()
print("  total:", int(kosong.sum()))
print()
print("Baris duplikat penuh :", int(df.duplicated().sum()))
print("customerID unik      :", df["customerID"].nunique(), "dari", len(df), "baris")'''),
code('''# Kolom TotalCharges terlihat angka, tapi cek tipenya
print("tipe TotalCharges:", df["TotalCharges"].dtype)
print("contoh 5 nilai  :", df["TotalCharges"].head().tolist())

# cari yang isinya kosong (spasi, bukan NaN)
blank = df["TotalCharges"].astype(str).str.strip() == ""
print("\\nTotalCharges yang kosong (isi spasi):", int(blank.sum()))
print("tenure dari baris tersebut:", sorted(df.loc[blank, "tenure"].unique().tolist()))
df.loc[blank, ["tenure", "MonthlyCharges", "TotalCharges"]].head()'''),
md("""
**Tiga masalah ditemukan:**

| # | Masalah | Dampak kalau dibiarkan |
|---|---|---|
| 1 | `TotalCharges` bertipe **teks**, berisi 11 nilai kosong (spasi) | Error saat scaling/model |
| 2 | `customerID` unik untuk setiap baris | Model menghafal ID, bukan belajar pola |
| 3 | `Churn` masih `Yes`/`No` | Target harus angka 1/0 |

Kabar baiknya: **tidak ada NaN asli dan tidak ada duplikat**. Jadi cleaning-nya
ringan, tidak perlu imputasi rumit.

11 baris kosong itu semuanya punya `tenure = 0` — artinya pelanggan baru daftar
dan belum pernah ditagih. Masuk akal kalau `TotalCharges`-nya kosong.
"""),
md("""
## 4. Target: seimbang atau tidak?
"""),
code('''print(df["Churn"].value_counts().to_string())
print()
print((df["Churn"].value_counts(normalize=True) * 100).round(2).to_string(), "%")'''),
code('''ax = sns.countplot(data=df, x="Churn", palette=["#4c72b0", "#dd8452"])
ax.set_title("Jumlah pelanggan: bertahan vs churn")
ax.set_xlabel("")
ax.set_ylabel("jumlah pelanggan")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=11)
plt.tight_layout()
plt.show()'''),
md("""
**26,5% churn vs 73,5% bertahan.** Ini disebut **class imbalance** (kelas tidak
seimbang), walaupun masih tergolong ringan.

Kenapa ini penting? Karena model yang selalu menjawab "tidak churn" akan dapat
**akurasi 73,5%** tanpa belajar apa pun. Artinya angka akurasi di dataset ini
menipu. Itu sebabnya project ini memakai **recall** sebagai metrik utama — akan
dibahas di notebook 05.
"""),
md("""
## 5. Statistik kolom numerik
"""),
code('''df[["SeniorCitizen", "tenure", "MonthlyCharges"]].describe().round(2)'''),
code('''print("Rentang nilai:")
print(f"  tenure         : {df['tenure'].min()} - {df['tenure'].max()} bulan (median {df['tenure'].median():.0f})")
tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"  MonthlyCharges : {df['MonthlyCharges'].min():.2f} - {df['MonthlyCharges'].max():.2f} (median {df['MonthlyCharges'].median():.2f})")
print(f"  TotalCharges   : {tc.min():.2f} - {tc.max():.2f}")'''),
md("""
Tidak ada outlier ekstrem. `tenure` 0–72 bulan (6 tahun), `MonthlyCharges`
18–119 dolar. Skalanya cukup jauh berbeda (`tenure` puluhan, `MonthlyCharges`
puluhan, `TotalCharges` ribuan) → nanti KNN dan Logistic Regression perlu
**scaling**, sedangkan Random Forest tidak.
"""),
md("""
## 6. Isi tiap kolom teks
"""),
code('''for c in df.select_dtypes(include="str").columns:
    if c in ("customerID", "TotalCharges"):
        continue
    nilai = df[c].value_counts().to_dict()
    print(f"{c:18s} ({df[c].nunique()} nilai) -> {nilai}")'''),
md("""
Perhatikan pola yang muncul: banyak kolom punya nilai **`"No internet service"`**
selain `"No"`. Keduanya sebenarnya berarti "tidak punya layanan itu", tapi ditulis
beda karena pelanggannya tidak berlangganan internet sama sekali.

Kita **biarkan apa adanya** (tidak digabung) supaya informasi "tidak punya internet"
tetap bisa dipakai model. Keputusan ini dicatat di README bagian *Keputusan penting*.
"""),
md("""
## 7. Ringkasan notebook 01

| Fakta | Nilai |
|---|---|
| Bentuk data | 7.043 pelanggan × 21 kolom |
| Kolom numerik | 4 (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) |
| Kolom teks | 15 (setelah `customerID` dan `Churn` dipisahkan) |
| Nilai kosong asli | 0 |
| `TotalCharges` bermasalah | 11 baris berisi spasi, semuanya `tenure = 0` |
| Duplikat | 0 |
| Churn | 1.869 (26,54%) |
| Tenure | 0–72 bulan, median 29 |
| Tagihan bulanan | $18,25–$118,75, median $70,35 |

**Yang belum terjawab:** *kenapa* pelanggan pergi? Notebook 01 hanya memastikan
datanya layak dipakai. Notebook 02 merapikan, notebook 04 mencari jawabannya.
"""),
])

# ---------------------------------------------------------------- 02
n02 = nb([
md("""
# 02 — Data Cleaning

**Kenapa notebook ini ada?** Notebook 01 menemukan tiga masalah. Notebook ini
memperbaikinya **satu per satu**, dengan alasan tertulis di tiap keputusan.

Prinsip yang dipegang: **cleaning hanya memperbaiki yang salah, bukan mengubah data
jadi "cantik"**. Data yang dibuang atau diisi harus bisa dipertanggungjawabkan.
"""),
md("""
## 0. Persiapan
"""),
code(SETUP + '''

RAW = BASE / "data" / "raw" / "telco_churn.csv"
CLEAN = BASE / "data" / "clean" / "churn_clean.csv"
CLEAN.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW)
print("data mentah:", df.shape)'''),
md("""
## 1. Masalah #1 — `TotalCharges` bertipe teks dan punya 11 nilai kosong

Konversi paksa ke angka dengan `errors="coerce"`: nilai yang tidak bisa dibaca
angka akan menjadi `NaN`, sehingga kita bisa melihat persis berapa yang rusak.
"""),
code('''df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("jumlah NaN setelah konversi:", int(df["TotalCharges"].isna().sum()))
print()
print("Profil 11 baris tersebut:")
df[df["TotalCharges"].isna()][["tenure", "MonthlyCharges", "TotalCharges", "Contract"]].head(11)'''),
md("""
**Keputusan: isi dengan 0, bukan median.**

Alasannya: 11 baris itu semuanya `tenure = 0` — pelanggan yang baru mendaftar dan
belum pernah ditagih. Total yang pernah dibayar memang **nol**, bukan "rata-rata
pelanggan lain". Mengisi median justru menciptakan data palsu.

Alternatif lain adalah membuang 11 baris itu (0,16% data). Itu juga sah, tapi kita
kehilangan informasi tentang pelanggan paling baru — padahal kelompok itu justru
yang paling berisiko churn.
"""),
code('''df["TotalCharges"] = df["TotalCharges"].fillna(0)
print("NaN tersisa:", int(df["TotalCharges"].isna().sum()))
print("TotalCharges minimum:", df["TotalCharges"].min())'''),
md("""
## 2. Masalah #2 — `customerID` bukan fitur

`customerID` unik untuk 7.043 baris. Kalau dibiarkan, model bisa "menghafal" ID
tertentu sebagai penanda churn. Itu overfitting yang tidak berguna: pelanggan baru
selalu punya ID baru, jadi pola itu tidak bisa dipakai.
"""),
code('''print("sebelum drop:", df.shape)
df = df.drop(columns=["customerID"])
print("sesudah drop:", df.shape)'''),
md("""
## 3. Masalah #3 — target masih teks

Model klasifikasi butuh target berupa angka. `Yes` → 1, `No` → 0.
"""),
code('''df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df["Churn"].value_counts().to_string())
print()
print("proporsi churn: {:.2%}".format(df["Churn"].mean()))'''),
md("""
## 4. Efek samping: 22 duplikat muncul setelah `customerID` dibuang

Sebelum `customerID` dibuang, tidak ada satu pun baris duplikat. Setelah dibuang,
muncul duplikat — karena dua pelanggan **berbeda** bisa punya profil yang sama
persis: gender sama, tenure sama, kontrak sama, tagihan sama.

Ini bukan data rusak. Ini konsekuensi wajar dari membuang kolom identitas.
"""),
code('''duplikat = df[df.duplicated(keep=False)]
print("jumlah baris duplikat:", int(df.duplicated().sum()))
print("baris yang terlibat  :", len(duplikat))
print()
print(duplikat.sort_values(["tenure", "MonthlyCharges"]).head(6)
      [["gender", "tenure", "Contract", "MonthlyCharges", "Churn"]].to_string())'''),
code('''# apakah ada kelompok duplikat yang targetnya berbeda? (itu tanda data rusak)
kelompok = df.groupby(list(df.columns), dropna=False)["Churn"].nunique()
print("kelompok duplikat dengan target berbeda:", int((kelompok > 1).sum()))'''),
md("""
**Keputusan: duplikat dibiarkan, tidak dibuang.**

Alasannya: 22 baris itu adalah **22 pelanggan sungguhan** yang kebetulan punya
profil identik. Kalau dibuang, kita menghapus pelanggan nyata dari data dan
membuat proporsi churn menjadi tidak akurat. Target mereka juga seragam, jadi
tidak ada kontradiksi yang perlu diselesaikan.

Catat satu hal jujur: karena ada 22 profil identik, sebagian kecil data uji bisa
"mirip" dengan data latih. Porsinya 0,3% dari total — terlalu kecil untuk
mengubah kesimpulan, tapi tetap dicatat sebagai limitasi di README.
"""),
md("""
## 5. Verifikasi hasil cleaning

Jangan percaya "sepertinya sudah bersih". Uji dengan `assert` supaya notebook
berhenti dengan error kalau ada yang terlewat.

Perhatikan: yang diuji **bukan** "tidak ada duplikat", karena duplikat di atas
memang sengaja dibiarkan. Yang diuji adalah hal-hal yang kalau salah akan
merusak model.
"""),
code('''assert df.isna().sum().sum() == 0, "masih ada nilai kosong"
assert "customerID" not in df.columns, "customerID belum dibuang"
assert set(df["Churn"].unique()) == {0, 1}, "target bukan 0/1"
assert pd.api.types.is_numeric_dtype(df["TotalCharges"]), "TotalCharges masih teks"
assert len(df) == 7043, "jumlah baris berubah — ada data yang hilang"
print("semua pemeriksaan lulus")
print("bentuk akhir:", df.shape)'''),
md("""
## 6. Simpan hasil cleaning

`data/clean/churn_clean.csv` adalah satu-satunya sumber untuk notebook berikutnya.
Aturan main project ini: **notebook 03 ke atas tidak boleh membaca data mentah**.
Kalau logika cleaning berubah, cukup jalankan ulang notebook ini.
"""),
code('''df.to_csv(CLEAN, index=False)
print("disimpan ke:", CLEAN)
print("ukuran:", round(CLEAN.stat().st_size / 1024, 1), "KB")'''),
md("""
## 7. Sebelum vs sesudah

| Aspek | Sebelum | Sesudah |
|---|---|---|
| Bentuk | 7.043 × 21 | 7.043 × 20 |
| `TotalCharges` | teks, 11 kosong | angka, 0 kosong |
| `customerID` | ada | dibuang |
| `Churn` | `Yes`/`No` | `1`/`0` |
| Duplikat | 0 | 22 (sengaja dibiarkan) |
| Baris hilang | — | **0** (tidak ada yang dibuang) |

Perhatikan: **tidak ada satu baris pun yang hilang**. Cleaning di sini memperbaiki
tipe data dan membuang kolom yang salah, bukan memangkas data.
"""),
])

# ---------------------------------------------------------------- 03
n03 = nb([
md("""
# 03 — Feature Engineering

**Kenapa notebook ini ada?** Model scikit-learn hanya mengerti angka. Notebook ini
mengubah 15 kolom teks menjadi angka, membagi data latih/uji, dan menyiapkan
artefak yang dipakai notebook 05 serta dashboard.

**Tiga hal yang dikerjakan:**

1. Pisahkan fitur (`X`) dan target (`y`)
2. Bagi data latih/uji **sebelum** encoding dan scaling
3. Simpan hasilnya sebagai `.joblib`
"""),
md("""
## 0. Persiapan
"""),
code(SETUP + '''
import joblib

CLEAN = BASE / "data" / "clean" / "churn_clean.csv"
FEAT = BASE / "data" / "clean" / "churn_features.joblib"

df = pd.read_csv(CLEAN)
print("data bersih:", df.shape)'''),
md("""
## 1. Pisahkan fitur dan target
"""),
code('''X = df.drop(columns=["Churn"])
y = df["Churn"]

kolom_kategori = X.select_dtypes(include="str").columns.tolist()
kolom_numerik = X.select_dtypes(include=np.number).columns.tolist()

print("Kolom kategorikal (butuh encoding):", len(kolom_kategori))
print(kolom_kategori)
print()
print("Kolom numerik (dipakai langsung):", kolom_numerik)
print()
print("X:", X.shape, "| y:", y.shape)'''),
md("""
## 2. Bagi data latih dan uji — **sebelum** encoding dan scaling

Ini urutan yang tidak boleh dibalik. Kalau kita meng-encode dan men-scale seluruh
data dulu baru membagi, informasi dari data uji ikut "bocor" ke proses training.
Akibatnya nilai evaluasi terlihat bagus tapi palsu. Istilahnya **data leakage**.

- `test_size=0.2` → 20% data disimpan untuk ujian akhir
- `stratify=y` → proporsi churn di train dan test dijaga sama (26,5%)
- `random_state=42` → hasil bisa direproduksi siapa pun
"""),
code('''from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("train:", X_train.shape, "| churn:", f"{y_train.mean():.2%}", f"({int(y_train.sum())} orang)")
print("test :", X_test.shape, "| churn:", f"{y_test.mean():.2%}", f"({int(y_test.sum())} orang)")'''),
md("""
## 3. Encoding — ubah teks jadi angka

Dipakai **One-Hot Encoding**: setiap kategori jadi kolom 0/1 sendiri.

Kenapa bukan Label Encoding (`Month-to-month` → 2, `Two year` → 0)? Karena Label
Encoding menciptakan urutan yang tidak ada artinya. Model akan menganggap
"Two year lebih besar dari One year", padahal itu cuma nama.

`drop="first"` membuang satu kolom per kategori untuk menghindari **dummy trap**
(kolom yang saling melengkapi sempurna, mengganggu Logistic Regression).
"""),
code('''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

encoder = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), kolom_kategori),
    ],
    remainder="passthrough",   # kolom numerik dibiarkan apa adanya
)

X_train_enc = encoder.fit_transform(X_train)   # fit HANYA di train
X_test_enc = encoder.transform(X_test)         # test cuma transform

fitur = [f.replace("cat__", "").replace("remainder__", "")
         for f in encoder.get_feature_names_out()]

X_train_enc = pd.DataFrame(X_train_enc, columns=fitur, index=X_train.index)
X_test_enc = pd.DataFrame(X_test_enc, columns=fitur, index=X_test.index)

print("sebelum encoding:", X_train.shape[1], "kolom")
print("sesudah encoding:", X_train_enc.shape[1], "kolom")
print()
print("contoh kolom baru:")
for f in fitur[:8]:
    print("  -", f)'''),
md("""
19 kolom asli menjadi **30 fitur**. Penambahan ini normal: satu kolom teks dengan
3 kategori menjadi 2 kolom 0/1.
"""),
md("""
## 4. Scaling — hanya untuk model yang butuh

`StandardScaler` menyamakan skala fitur (rata-rata 0, simpangan baku 1). Ini
**wajib** untuk KNN (karena menghitung jarak) dan Logistic Regression (agar
koefisiennya adil), tapi **tidak perlu** untuk Random Forest (berbasis pemisahan
nilai, bukan jarak).

Sama seperti encoding: `fit` hanya di train, lalu dipakai untuk `transform` test.
"""),
code('''from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_enc), columns=fitur, index=X_train.index)
X_test_sc = pd.DataFrame(scaler.transform(X_test_enc), columns=fitur, index=X_test.index)

print("sebelum scaling (3 fitur pertama):")
print(X_train_enc[fitur[:3]].head(3).round(2).to_string())
print()
print("sesudah scaling:")
print(X_train_sc[fitur[:3]].head(3).round(2).to_string())'''),
md("""
## 5. Simpan artefak

Satu file `.joblib` berisi semua yang dibutuhkan notebook 05 dan dashboard,
sehingga tidak ada yang perlu mengulang cleaning dari nol.
"""),
code('''FEAT.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "X_train": X_train_enc, "X_test": X_test_enc,
    "X_train_scaled": X_train_sc, "X_test_scaled": X_test_sc,
    "y_train": y_train, "y_test": y_test,
    "features": fitur,
    "kolom_kategori": kolom_kategori,
    "encoder": encoder, "scaler": scaler,
}, FEAT)

print("disimpan:", FEAT)
print("ukuran  :", round(FEAT.stat().st_size / 1024, 1), "KB")'''),
md("""
## 6. Verifikasi — muat ulang dan bandingkan

Kalau hasil muat ulang berbeda dengan aslinya, berarti ada yang salah saat menyimpan.
"""),
code('''cek = joblib.load(FEAT)
print("isi artefak:", list(cek.keys()))
print()
print("bentuk X_train      :", cek["X_train"].shape)
print("bentuk X_train_scaled:", cek["X_train_scaled"].shape)
print("bentuk X_test       :", cek["X_test"].shape)
print("jumlah baris y_train:", len(cek["y_train"]))
assert cek["X_train"].shape[0] == len(cek["y_train"])
assert list(cek["features"]) == fitur
print()
print("verifikasi lulus")'''),
md("""
## 7. Ringkasan notebook 03

| Hal | Nilai |
|---|---|
| Data latih | 5.634 pelanggan (1.496 churn) |
| Data uji | 1.409 pelanggan (374 churn) |
| Kolom asli | 19 |
| Fitur setelah encoding | 30 |
| Encoding | One-Hot, `drop="first"`, fit di train |
| Scaling | StandardScaler, fit di train |
| Artefak | `data/clean/churn_features.joblib` |

Urutan yang dipegang sepanjang project:

```
split  →  encoding (fit train)  →  scaling (fit train)  →  model
```

Kalau urutan ini dibalik, semua angka evaluasi di notebook 05 tidak bisa dipercaya.
"""),
])

for name, notebook in [("01_data_understanding.ipynb", n01),
                       ("02_data_cleaning.ipynb", n02),
                       ("03_feature_engineering.ipynb", n03)]:
    nbf.write(notebook, str(NB / name))
    print("dibuat:", name)
