# Plan: Mini Project Machine Learning — Churn Telco

Dokumen ini menyimpan konteks, keputusan yang disepakati, dan catatan verifikasi.
Status: **selesai dikerjakan dan diverifikasi**.

## Konteks

Tugas mini project dari mentor setelah materi Supervised Learning (KNN, Logistic
Regression, Decision Tree, Random Forest, Model Evaluation & Tuning). Diminta
dikerjakan di rumah lalu didiskusikan di kelas, **dengan penjelasan/presentasi**.

Keputusan yang disepakati bersama user:

1. **Dataset: Telco Customer Churn** (Kaggle, `blastchar`) — bukan dataset lokal yang
   sudah ada (heart/housing/Iris), supaya bukan mengulang materi.
2. **Presentasi langsung dari notebook di VS Code** (bukan PPT).
3. **Durasi 8–10 menit.**
4. **Streamlit tidak dipresentasikan.** User merasa tidak siap menjelaskannya, jadi
   dashboard hanya artefak pendukung dan tidak ditampilkan saat maju.
5. Struktur & gaya mengikuti project sebelumnya (`project spotify`): folder
   `raw`/`clean`, notebook berurutan, pola **Hook → Angka → Makna → Rekomendasi**,
   README dengan bagian "Keputusan penting" dan "Keterbatasan".

## Strategi yang dipakai

Mulai dari panggung, bukan dari kode:

```
3 kalimat yang mau diingat  ->  bukti yang dibutuhkan  ->  data mendukung?  ->  kode
```

Tiga kalimat awal (hipotesis, lalu diuji ke data):

1. Churn itu masalah **waktu**, bukan harga — 6 bulan pertama paling menentukan.
2. Kontrak bulanan adalah pintu keluar; perpanjangan kontrak lebih efektif daripada diskon.
3. Model cukup akurat untuk jadi **daftar prioritas**, bukan ramalan nasib.

Ketiganya **terbukti** setelah profiling data.

## Struktur akhir

```
project ML Supervised/
├── data/
│   ├── raw/telco_churn.csv
│   └── clean/{churn_clean.csv, churn_features.joblib}
├── models/churn_model.joblib
├── notebooks/01..06 (sudah dieksekusi, ada output & grafik)
├── app/app.py + .streamlit/config.toml
├── build/{gen_part1.py, gen_part2.py, metrics.json}
├── docs/plan.md
├── README.md
└── requirements.txt
```

## Keputusan teknis

| Keputusan | Alasan |
|---|---|
| Metrik utama **recall** | Baseline naif dapat akurasi 73,5% dengan recall 0; FN lebih mahal daripada FP |
| `class_weight="balanced"` | Menangani imbalance tanpa data sintetis (SMOTE ditolak) |
| Split sebelum encoding/scaling | Mencegah data leakage |
| `OneHotEncoder(drop="first")` | Menghindari dummy trap untuk Logistic Regression |
| RF tanpa scaling, KNN/LogReg dengan scaling | Sesuai sifat algoritmanya |
| Threshold dipilih dari prediksi **out-of-fold** | Data uji tidak boleh dipakai untuk keputusan apa pun |
| Simpan `churn_features.joblib` dan `churn_model.joblib` | Notebook 06 & dashboard tidak perlu mengulang training |

## Temuan yang mengubah rencana

**1. 22 duplikat muncul setelah `customerID` dibuang.**
Ditemukan oleh `assert` di notebook 02 saat eksekusi pertama. Setelah diperiksa,
semua kelompok duplikat punya target sama → itu pelanggan berbeda dengan profil
identik, bukan data rusak. Keputusan: **dibiarkan**, dicatat sebagai limitasi.

**2. `TotalCharges` kosong tepat pada `tenure = 0`.**
Semua 11 baris kosong adalah pelanggan baru yang belum pernah ditagih → diisi 0,
bukan median. Ini membuat klaim "isi 0" punya dasar, bukan asumsi.

**3. Logistic Regression hampir mengalahkan Random Forest.**
CV recall LogReg 0,801 vs RF 0,649 (sebelum tuning) → setelah tuning RF 0,795.
Di data uji RF unggul tipis (0,802 vs 0,786). Keputusan: RF dipilih karena unggul
tipis di data uji **dan** memberi feature importance; nuansa ini ditulis jujur di
notebook 05 dan di README.

## BUG yang ditemukan & diperbaiki saat verifikasi

**Bug:** notebook 05 gagal dengan `KeyError: 'Contract'` di bagian pembanding
aturan sederhana.

**Akar masalah:** `X_test` yang dimuat dari `churn_features.joblib` sudah berupa
matriks **ter-encode** (30 fitur one-hot), bukan kolom asli. Kode aturan masih
memakai nama kolom mentah (`Contract`, `TechSupport`).

**Perbaikan:** profil asli diambil ulang dari `data/clean/churn_clean.csv` memakai
indeks `X_test.index` (indeks dipertahankan saat split), lalu aturan dihitung dari
kolom asli. Ini juga lebih mudah dibaca daripada menalar kombinasi kolom one-hot.

## Angka kunci (untuk reproduksi)

| Metrik | Nilai |
|---|---|
| Baris × kolom | 7.043 × 21 |
| Churn | 1.869 (26,54%) |
| Train / test | 5.634 / 1.409 |
| Fitur setelah encoding | 30 |
| CV recall: Dummy / LogReg / KNN / RF | 0,000 / 0,801 / 0,559 / 0,649 |
| RF setelah tuning | depth 8, leaf 10, n=400; CV recall 0,795 |
| Threshold terpilih | 0,50 |
| Test: recall / precision / F1 / AUC / accuracy | 0,802 / 0,524 / 0,634 / 0,845 / 0,754 |
| Confusion matrix (TN, FP, FN, TP) | 763, 272, 74, 300 |
| Aturan sederhana (test) | 270 ditandai, 167 TP, precision 0,619, recall 0,447 |
| Churn 6 bulan pertama | 52,9% (porsi 41,9%) |
| Churn kontrak bulanan | 42,7% (porsi 88,6%) |
| Tagihan: churn vs bertahan | $74,44 vs $61,27 |

## Verifikasi (sudah dijalankan)

1. Notebook 01–06 dieksekusi berurutan dengan `jupyter nbconvert --execute`,
   **tanpa error**, dan seluruh grafik tersimpan di dalam notebook.
2. Assert kunci lulus: tidak ada NaN, `customerID` tidak ada, target 0/1,
   `TotalCharges` numerik, jumlah baris tetap 7.043 (tidak ada data hilang).
3. Artefak dapat dimuat ulang: `churn_features.joblib` (30 fitur, 5.634 baris train)
   dan `churn_model.joblib` (threshold, metrik, importance).
4. Dashboard diuji headless dengan `streamlit.testing.v1.AppTest`: 3 tab tampil,
   **0 exception**, tombol prediksi bekerja (input default → 84,5% risiko TINGGI),
   metrik dan confusion matrix tampil.
5. Peringatan `use_container_width` dihapus (diganti `width="stretch"`), dan
   peringatan nama fitur Random Forest dihilangkan dengan membungkus hasil
   encoder menjadi DataFrame.

## Opsi lanjutan (belum dikerjakan)

- [ ] XGBoost / LightGBM sebagai pembanding ensemble
- [ ] Perhitungan *expected value* kampanye (biaya kontak vs nilai pelanggan)
- [ ] Uji stabilitas model pada potongan data per tahun
- [ ] Deploy dashboard ke Streamlit Community Cloud untuk portofolio
