# Churn Telco — Mini Project Machine Learning (Supervised)

https://churn-telco-palpale24.streamlit.app/

Prediksi pelanggan yang akan berhenti berlangganan (**churn**) memakai dataset
Telco Customer Churn: 7.043 pelanggan, 21 kolom, churn rate 26,5%.

Project ini menekankan **alur berpikir**, bukan mengejar akurasi: setiap keputusan
punya alasan tertulis, setiap angka di presentasi bisa ditelusuri ke notebook.

## Alur pengerjaan

| # | Notebook | Isi |
|---|---|---|
| 1 | `notebooks/01_data_understanding.ipynb` | Bentuk data, tipe kolom, masalah kualitas (11 `TotalCharges` kosong, `customerID`, target `Yes`/`No`), class imbalance |
| 2 | `notebooks/02_data_cleaning.ipynb` | Perbaiki `TotalCharges`, buang `customerID`, target → 1/0, jelaskan 22 duplikat yang muncul, simpan `churn_clean.csv` |
| 3 | `notebooks/03_feature_engineering.ipynb` | Split **sebelum** encoding & scaling, One-Hot + StandardScaler (fit di train saja), simpan `churn_features.joblib` |
| 4 | `notebooks/04_eda_churn.ipynb` | 4 temuan: churn di 6 bulan pertama, kontrak bulanan, "bukan harga", tiga penanda risiko |
| 5 | `notebooks/05_modeling_evaluasi.ipynb` | Baseline naif → LogReg/KNN/RF → tuning → threshold → evaluasi → dibandingkan dengan aturan sederhana → simpan model |
| 6 | `notebooks/06_storytelling.ipynb` | 3 cerita utuh (Hook → Angka → Makna → Rekomendasi) + **naskah presentasi 8–10 menit** + antisipasi pertanyaan |
| 7 | `app/app.py` | Dashboard Streamlit (opsional): coba prediksi, evaluasi model, lihat data |

## Struktur

```
project ML Supervised/
├── data/
│   ├── raw/telco_churn.csv                  # data asli (read-only)
│   └── clean/
│       ├── churn_clean.csv                  # output notebook 02
│       └── churn_features.joblib            # output notebook 03 (X/y + encoder + scaler)
├── models/
│   └── churn_model.joblib                   # output notebook 05 (model + metrik + threshold)
├── notebooks/                               # 6 notebook berurutan, sudah dieksekusi
├── app/app.py                               # dashboard (hanya baca artefak)
├── build/                                   # generator notebook & artefak kerja
├── docs/plan.md                             # keputusan & catatan verifikasi
└── requirements.txt
```

## Cara menjalankan

```bash
# 1. Pastikan interpreter Python 3.13 (bukan 3.15)
python -c "import pandas, sklearn, streamlit; print('siap')"

# 2. Jalankan notebook 01 -> 06 berurutan dari folder notebooks/
#    Notebook 03 membuat churn_features.joblib
#    Notebook 05 membuat models/churn_model.joblib

# 3. Dashboard (opsional), dari folder root project
streamlit run app/app.py
```

Dashboard membaca `models/churn_model.joblib`. Kalau file itu belum ada, app
menampilkan instruksi untuk menjalankan notebook 05 lebih dulu.

## Hasil utama (data uji 1.409 pelanggan, 374 churn)

| Metrik | Nilai |
|---|---|
| **Recall** | **0,802** — 300 dari 374 churn tertangkap |
| Precision | 0,524 |
| F1 | 0,634 |
| ROC-AUC | 0,845 |
| Accuracy | 0,754 (hampir sama dengan baseline naif → bukan ukuran utama) |

**Pembanding aturan sederhana** (kontrak bulanan + tenure ≤ 12 + tanpa tech support):
hanya menangkap 167 pelanggan (recall 0,447). Model menangkap **133 lebih banyak**.

**Fitur paling berpengaruh:** `tenure` (0,177), `Contract_Two year` (0,144),
`TotalCharges` (0,116), `InternetService_Fiber optic` (0,095),
`PaymentMethod_Electronic check` (0,066).

## Ringkasan insight (hasil notebook 04)

1. **Setengah pelanggan baru pergi dalam 6 bulan** — churn rate 52,9%, menyumbang
   41,9% dari seluruh churn.
2. **Kontrak bulanan = sumber masalah** — rate 42,7%, porsi **88,6%** dari total churn.
3. **Bukan soal harga** — yang pergi justru membayar lebih mahal ($74,44 vs $61,27).
4. **Tiga penanda risiko** — electronic check (45,3%), fiber optic (41,9%),
   tanpa tech support (41,6%).

## Keputusan penting (kenapa begitu)

- **Recall, bukan accuracy.** Baseline "selalu tidak churn" sudah dapat akurasi 73,5%
  dengan recall 0. Metrik yang benar adalah recall, karena melewatkan pelanggan
  berisiko lebih mahal daripada menghubungi yang ternyata aman.
- **`class_weight="balanced"`, bukan SMOTE.** Kelas tidak ekstrem (26,5%) dan SMOTE
  membuat data sintetis yang tidak ada di dunia nyata.
- **Split dulu, baru encoding & scaling.** `fit` hanya di data latih. Ini mencegah
  data leakage — kesalahan paling umum di project ML.
- **Tuning hanya di data train, data uji disentuh sekali.** Threshold dipilih dari
  prediksi out-of-fold, bukan dari data uji.
- **Duplikat setelah `customerID` dibuang dibiarkan (22 baris).** Itu pelanggan
  berbeda dengan profil identik; membuangnya berarti menghapus pelanggan nyata.
- **`TotalCharges` kosong diisi 0, bukan median.** 11 baris itu semuanya `tenure = 0`
  (belum pernah ditagih), jadi 0 adalah nilai yang benar.
- **Model final: Random Forest hasil tuning.** Recall tertinggi di data uji (0,802),
  F1 dan AUC juga tertinggi. Jujur dicatat: Logistic Regression menang di
  cross-validation (0,801 vs 0,795) dan lebih sederhana — keduanya pilihan sah.
- **`data/clean` dan `models` adalah kontrak tunggal.** Dashboard tidak pernah
  membaca data mentah atau melakukan training.

## Keterbatasan

- Data satu perusahaan pada satu periode — belum tentu berlaku di provider lain.
- Model menangkap **korelasi, bukan sebab-akibat**. Tidak ada A/B test.
- Precision 52,4%: separuh yang ditandai model ternyata aman.
- `TotalCharges ≈ tenure × MonthlyCharges` → multikolinearitas; koefisien satuan
  (Logistic Regression) tidak bisa ditafsirkan sendiri-sendiri.
- 22 profil identik (0,3%) → potensi kebocoran sangat kecil antara train dan test.
- Tidak ada data kompetitor, harga pasar, atau alasan pelanggan berhenti.

## Teknologi

pandas 3 · numpy · matplotlib · seaborn (notebook) · scikit-learn (model) ·
joblib (artefak) · **streamlit** (dashboard)
