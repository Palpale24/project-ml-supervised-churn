# Cara Presentasi (8–10 menit, langsung dari VS Code)

Buka folder `project ML Supervised` di VS Code, lalu buka **6 notebook ini sekaligus**
sebagai tab (klik kanan file → Open to the Side kalau perlu). Jangan buka browser,
jangan buka Streamlit.

---

## Urutan tab (susun dari kiri ke kanan)

```
01_data_understanding  →  02_data_cleaning  →  03_feature_engineering
04_eda_churn  →  05_modeling_evaluasi  →  06_storytelling
```

Hafalkan **hanya 4 tab** yang benar-benar dipakai bicara: **01, 04, 05, 06**.
Tab 02 dan 03 cukup disebut sekali ("sudah dibersihkan dan di-encode") sambil
di-scroll cepat.

---

## Alur 9 menit

| Menit | Tab yang dibuka | Yang dikatakan (inti) |
|---|---|---|
| 0:00–0:40 | 01 (bagian 1) | 7.043 pelanggan, 26,5% sudah pergi. Pertanyaannya: bisa tidak kita tahu **sebelum** mereka pergi? |
| 0:40–1:40 | 01 (bagian 3 & 4) | Data hampir bersih — 0 NaN, hanya `TotalCharges` teks berisi 11 kosong (semua `tenure = 0`). Kelas tidak seimbang: 26,5 vs 73,5 → **accuracy menipu**, saya pakai recall. |
| 1:40–2:00 | 02 + 03 (scroll cepat) | "Cleaning ringan: 0 baris dibuang. Split dulu 80–20, baru encoding & scaling di data latih — supaya tidak data leakage." |
| 2:00–4:00 | **04** (bagian 1–3) | Tiga grafik: ① 52,9% churn di 6 bulan pertama ② kontrak bulanan 42,7% tapi **88,6% dari seluruh churn** ③ yang pergi bayar **$74 vs $61** → ini soal komitmen, bukan harga. |
| 4:00–4:40 | **05** (bagian 2) | Bandingkan 3 model + baseline. Tunjukkan baris Dummy: accuracy 0,735 tapi recall 0. "Ini bukti accuracy menipu." |
| 4:40–6:00 | **05** (bagian 5) | Confusion matrix + angka: recall **0,802** → **300 dari 374** pelanggan berisiko tertangkap. Precision 52% → separuhnya alarm palsu, dan itu saya terima dengan alasan biaya. |
| 6:00–7:00 | **05** (bagian 8) | Aturan sederhana tanpa ML cuma menangkap **167**. Model menangkap **300** → selisih **133 orang**. |
| 7:00–8:00 | **05** (bagian 7) | Fitur teratas: `tenure`, `Contract Two year`, `TotalCharges` → semuanya soal komitmen. |
| 8:00–9:00 | **06** (bagian 4) | 3 rekomendasi: onboarding 90 hari, dorong kontrak 1 tahun untuk fiber optic bulanan, pakai skor model sebagai **daftar prioritas**. |
| 9:00–9:40 | **06** (bagian 10 / limitasi di 05) | Limitasi jujur: satu perusahaan, korelasi bukan sebab-akibat, precision 52%. Lalu tutup. |

**Kalimat penutup yang disiapkan:**
> "Model ini bukan peramal nasib — tapi daftar prioritas yang jauh lebih baik
> daripada menebak atau menghubungi semua orang."

---

## 3 kalimat yang harus dihafal (jangan dibaca)

1. "Setengah pelanggan baru pergi dalam 6 bulan pertama."
2. "Kontrak bulanan cuma 55% pelanggan, tapi menyumbang 88,6% churn."
3. "Model menangkap 300 orang; aturan sederhana cuma 167."

Sisanya boleh dibaca dari markdown.

---

## Persiapan teknis 5 menit sebelum maju

- [ ] Semua notebook **sudah ada output** (jangan jalankan ulang di depan kelas —
      notebook 05 butuh ~2 menit).
- [ ] Zoom VS Code ke **110–125%** (`Ctrl` + `=`), font editor cukup besar.
- [ ] Tutup panel output terminal & sidebar, sisakan notebook saja.
- [ ] Matikan mode gelap kalau proyektor kurang terang (opsional).
- [ ] **Cadangan:** folder `notebooks/html/` berisi 6 file HTML. Kalau VS Code
      bermasalah, buka `06_storytelling.html` di browser — grafik & angka tetap ada.
- [ ] Jangan buka Streamlit (`app.py`). Kalau ditanya: "ada dashboard opsional,
      tapi fokus tugas ini di analisis dan model."

---

## Kalau mentor bertanya (jawaban 1–2 kalimat)

| Pertanyaan | Jawaban |
|---|---|
| Kenapa recall, bukan accuracy? | Model yang selalu jawab "tidak churn" sudah dapat akurasi 73,5% dengan recall 0. Melewatkan pelanggan berisiko lebih mahal daripada salah hubungi. |
| Kenapa tidak pakai SMOTE? | Kelas masih 26,5%, tidak ekstrem. `class_weight="balanced"` sudah cukup dan tidak membuat data palsu. |
| Kenapa Random Forest, bukan LogReg? | Unggul tipis di data uji (recall 0,802 vs 0,786) dan punya feature importance. Jujur: LogReg menang di CV dan lebih sederhana. |
| Apa itu data leakage? | Informasi data uji bocor ke training. Dicegah dengan split dulu, baru `fit` encoder & scaler di train saja. |
| Kenapa threshold 0,5? | Diuji 0,20–0,65 pakai prediksi out-of-fold dari data train. 0,50 terbaik dengan syarat precision ≥ 0,50. |
| Precision 52% itu tidak buruk? | Untuk daftar prioritas tidak. Kalau budget ketat, threshold dinaikkan → precision naik, recall turun. |
| Ini sebab-akibat? | Bukan. Ini korelasi; tidak ada A/B test. |
| Kenapa tidak XGBoost? | Bisa dan mungkin lebih akurat, tapi tujuan tugas ini memahami alur. RF sudah cukup dan lebih mudah dijelaskan. |
| 22 duplikat itu apa? | Setelah `customerID` dibuang, ada 22 pelanggan berbeda dengan profil identik. Targetnya konsisten, jadi dibiarkan dan dicatat sebagai limitasi. |

---

## Yang **jangan** dilakukan

- Jangan membaca tabel besar satu per satu — tunjuk angkanya, lanjut.
- Jangan bilang "akurasi 75%" sebagai pencapaian utama.
- Jangan bilang "tech support menyebabkan churn" (itu korelasi).
- Jangan menjalankan sel di depan kelas.
