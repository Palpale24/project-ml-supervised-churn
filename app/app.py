"""
Dashboard Churn Telco — pelengkap project Machine Learning Supervised.

Cara menjalankan (dari folder root project):

    streamlit run app/app.py

Dashboard ini hanya membaca artefak hasil notebook:
    data/clean/churn_clean.csv      -> profil default pelanggan
    data/clean/churn_features.joblib-> data uji (untuk kurva ROC)
    models/churn_model.joblib       -> model, threshold, metrik

Tidak ada training ulang di sini. Kalau model berubah, jalankan ulang notebook 05.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve

# ---------------------------------------------------------------------------
# Konfigurasi halaman
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Churn Telco — Prediksi & Evaluasi", page_icon="📉", layout="wide")

BASE = Path(__file__).resolve().parent.parent
CLEAN = BASE / "data" / "clean" / "churn_clean.csv"
FEAT = BASE / "data" / "clean" / "churn_features.joblib"
MODEL = BASE / "models" / "churn_model.joblib"


@st.cache_resource
def muat_artefak():
    if not MODEL.exists():
        st.error(
            "Model belum ada. Jalankan notebook 01 → 05 dulu "
            "(notebook 05 yang membuat `models/churn_model.joblib`)."
        )
        st.stop()
    return joblib.load(MODEL), joblib.load(FEAT)


@st.cache_data
def muat_data():
    return pd.read_csv(CLEAN)


bundle, feat = muat_artefak()
df = muat_data()

model = bundle["model"]
threshold = bundle["threshold"]
metrik = bundle["metrics"]
conf = bundle["confusion"]
penting = pd.Series(bundle["importance"]).sort_values(ascending=False)

kolom_asli = [c for c in df.columns if c != "Churn"]
default = {}
for c in kolom_asli:
    if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
        default[c] = df[c].mode()[0]
    else:
        default[c] = float(df[c].median())


def prediksi(nilai: dict) -> float:
    """Isi kolom yang tidak diubah dengan nilai default, lalu prediksi peluang churn."""
    baris = dict(default)
    baris.update(nilai)
    X = pd.DataFrame([baris])[kolom_asli]
    X_enc = bundle["encoder"].transform(X)
    X_enc = pd.DataFrame(X_enc, columns=bundle["features"])
    return float(model.predict_proba(X_enc)[0, 1])


def kategori(proba: float) -> tuple[str, str]:
    if proba >= 0.60:
        return "TINGGI", "🔴"
    if proba >= 0.35:
        return "SEDANG", "🟠"
    return "RENDAH", "🟢"


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📉 Churn Telco — Dashboard Model")
st.caption(
    "Model: Random Forest hasil tuning · "
    f"Recall {metrik['recall']:.1%} · Precision {metrik['precision']:.1%} · "
    f"ROC-AUC {metrik['auc']:.3f}"
)

tab1, tab2, tab3 = st.tabs(["🔎 Coba Prediksi", "📊 Evaluasi Model", "📁 Data"])

# ---------------------------------------------------------------------------
# Tab 1 — Prediksi
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Profil pelanggan")
    st.caption("Kolom yang tidak diisi otomatis memakai nilai paling umum di data.")

    c1, c2, c3 = st.columns(3)
    with c1:
        tenure = st.slider("Masa berlangganan (bulan)", 0, 72, 3)
        kontrak = st.selectbox("Tipe kontrak", ["Month-to-month", "One year", "Two year"])
        senior = st.checkbox("Lansia (SeniorCitizen)", value=False)
    with c2:
        internet = st.selectbox("Layanan internet", ["Fiber optic", "DSL", "No"])
        tech = st.selectbox("Dukungan teknis", ["No", "Yes", "No internet service"])
        keamanan = st.selectbox("Keamanan online", ["No", "Yes", "No internet service"])
    with c3:
        bayar = st.selectbox(
            "Metode bayar",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        tagihan = st.number_input("Tagihan bulanan ($)", 10.0, 130.0, 75.0, 1.0)
        paperless = st.checkbox("Tagihan paperless", value=True)

    st.caption(
        f"Perkiraan total tagihan: **${tenure * tagihan:,.0f}** "
        "(dipakai model sebagai gambaran total pembayaran)"
    )

    nilai = {
        "tenure": tenure,
        "Contract": kontrak,
        "SeniorCitizen": int(senior),
        "InternetService": internet,
        "TechSupport": tech,
        "OnlineSecurity": keamanan,
        "PaymentMethod": bayar,
        "MonthlyCharges": tagihan,
        "TotalCharges": float(tenure * tagihan),
        "PaperlessBilling": "Yes" if paperless else "No",
    }

    if st.button("Prediksi risiko churn", type="primary"):
        proba = prediksi(nilai)
        label, emoji = kategori(proba)

        st.divider()
        a, b, c = st.columns([1.2, 1, 1])
        a.metric("Peluang churn", f"{proba:.1%}")
        b.metric("Kategori risiko", f"{emoji} {label}")
        c.metric("Ambang model", f"{threshold:.2f}")

        st.progress(min(proba, 1.0))

        if proba >= threshold:
            st.warning(
                "Pelanggan ini **melewati ambang model** — masuk daftar prioritas kontak."
            )
        else:
            st.success("Di bawah ambang model — belum perlu diprioritaskan bulan ini.")

        st.markdown("**Saran tindakan**")
        if kontrak == "Month-to-month" and tenure <= 12:
            st.write("- Tawarkan perpanjangan kontrak 1 tahun — ini kelompok paling rawan.")
        if internet == "Fiber optic" and tagihan > 70:
            st.write("- Tinjau nilai layanan: pelanggan fiber optic dengan tagihan tinggi paling sering pergi.")
        if tech == "No":
            st.write("- Tawarkan dukungan teknis; tanpa layanan ini risiko churn ~3x lipat.")
        if not any([kontrak == "Month-to-month" and tenure <= 12,
                    internet == "Fiber optic" and tagihan > 70, tech == "No"]):
            st.write("- Risiko relatif rendah; cukup pantau lewat program loyalitas rutin.")

        st.caption(
            "Catatan: model memberi **prioritas**, bukan keputusan otomatis. "
            "Precision model ~52%, jadi sebagian yang ditandai ternyata aman."
        )

# ---------------------------------------------------------------------------
# Tab 2 — Evaluasi
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Performa di data uji (belum pernah dilihat model)")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall", f"{metrik['recall']:.1%}", help="Dari yang benar-benar churn, berapa yang tertangkap")
    m2.metric("Precision", f"{metrik['precision']:.1%}", help="Dari yang ditandai, berapa yang benar churn")
    m3.metric("F1", f"{metrik['f1']:.3f}")
    m4.metric("ROC-AUC", f"{metrik['auc']:.3f}")

    kiri, kanan = st.columns(2)

    with kiri:
        st.markdown("**Confusion matrix**")
        cm = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred: bertahan", "Pred: churn"],
            yticklabels=["Aktual: bertahan", "Aktual: churn"],
        )
        ax.set_title(f"threshold = {threshold}")
        st.pyplot(fig)
        st.caption(
            f"TP {conf['tp']} tertangkap · FN {conf['fn']} terlewat · "
            f"FP {conf['fp']} alarm palsu · TN {conf['tn']} benar aman"
        )

    with kanan:
        st.markdown("**ROC Curve**")
        proba_test = model.predict_proba(feat["X_test"])[:, 1]
        fpr, tpr, _ = roc_curve(feat["y_test"], proba_test)
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
        ax2.plot(fpr, tpr, lw=2, label=f"AUC = {metrik['auc']:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="tebakan acak")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

    st.divider()
    st.markdown("**Fitur paling berpengaruh**")
    fig3, ax3 = plt.subplots(figsize=(7, 3.2))
    penting.head(10).sort_values().plot(kind="barh", ax=ax3, color="#3b82f6")
    ax3.set_xlabel("tingkat kepentingan")
    st.pyplot(fig3)

    st.markdown("**Parameter model**")
    st.json(bundle["best_params"])
    st.caption(
        f"Dilatih pada {bundle['n_train']:,} pelanggan, diuji pada {bundle['n_test']:,} pelanggan."
    )

# ---------------------------------------------------------------------------
# Tab 3 — Data
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Data bersih (hasil notebook 02)")
    st.caption(
        f"{len(df):,} pelanggan · churn rate {df['Churn'].mean():.1%} · "
        "sumber: `data/clean/churn_clean.csv`"
    )

    pilih = st.multiselect(
        "Filter tipe kontrak",
        sorted(df["Contract"].unique()),
        default=sorted(df["Contract"].unique()),
    )
    tampil = df[df["Contract"].isin(pilih)]

    st.dataframe(
        tampil.head(300),
        width="stretch",
        height=420,
    )
    st.download_button(
        "Unduh CSV",
        tampil.to_csv(index=False).encode("utf-8"),
        file_name="churn_terfilter.csv",
        mime="text/csv",
    )
