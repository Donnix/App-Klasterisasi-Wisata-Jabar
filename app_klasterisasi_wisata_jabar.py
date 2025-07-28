import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Klasterisasi Jenis Wisata", layout="wide")
st.title("Klasterisasi Jenis Wisata berdasarkan Kecamatan di Kabupaten Bogor")

# 1. ------------------- Template Dataset dari File Lokal ---------------------
st.subheader("📂 Template Dataset")
st.markdown("Unduh template berikut, lalu isi data jenis wisata per kecamatan:")

with open("template_data_wisata.xlsx", "rb") as f:
    st.download_button(
        label="📥 Download Template Excel",
        data=f,
        file_name="template_data_wisata.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# 2. ------------------- Upload Dataset dari User ---------------------
st.subheader(" Upload File Dataset Anda")
uploaded_file = st.file_uploader("Unggah file Excel (.xlsx) sesuai template", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    except Exception as e:
        st.error(f"❌ Gagal membaca file: {e}")
        st.stop()

    if "nama_kecamatan" not in df.columns:
        st.error("❌ Kolom pertama harus bernama 'nama_kecamatan'. Gunakan template yang diberikan.")
        st.stop()

    df = df.dropna(subset=["nama_kecamatan"]).drop_duplicates()
    if df.drop(columns=["nama_kecamatan"]).isnull().all().all():
        st.warning("⚠️ Data jenis wisata masih kosong. Silakan isi jumlah wisata di kolom yang tersedia.")
        st.dataframe(df)
        st.stop()

    df.fillna(0, inplace=True)
    non_nama_cols = df.drop(columns=["nama_kecamatan"])
    if not all([np.issubdtype(dt, np.number) for dt in non_nama_cols.dtypes]):
        st.error("❌ Semua kolom jenis wisata harus berisi angka. Pastikan Anda tidak memasukkan teks.")
        st.dataframe(non_nama_cols.dtypes)
        st.stop()

    st.success("✅ Data berhasil divalidasi dan dibaca.")
    st.write(f"Jumlah kecamatan: {df.shape[0]}")
    df_tampil = df.copy()
    df_tampil.insert(0, "No", range(1, len(df_tampil) + 1))
    st.dataframe(df_tampil, hide_index=True)

    st.subheader("🎯 Pilih Jenis Wisata untuk Klasterisasi")
    jenis_wisata = st.multiselect("Pilih minimal 2 jenis wisata:", options=df.columns[1:])

    if len(jenis_wisata) >= 2:
        fitur_wisata = df[jenis_wisata]
        scaler = StandardScaler()
        fitur_scaled = scaler.fit_transform(fitur_wisata)

        st.subheader("🔍 Evaluasi Jumlah Klaster (K)")
        range_k = list(range(2, 11))
        inertia = []
        for k in range_k:
            km = KMeans(n_clusters=k, random_state=42).fit(fitur_scaled)
            inertia.append(km.inertia_)

        fig_elbow = px.line(x=range_k, y=inertia, markers=True,
                            labels={'x': 'Jumlah Klaster (K)', 'y': 'Inertia'},
                            title="Metode Elbow")
        st.plotly_chart(fig_elbow)
        elbow_k = range_k[np.argmin(np.diff(np.diff(inertia))) + 1]

        st.markdown(f"""
        **Interpretasi Elbow:**
        Grafik Elbow menunjukkan penurunan total jarak (inertia) antar data ke pusat klaster.
        Setelah **K = {elbow_k}**, penurunan mulai melandai.
        """)

        silhouette_scores = []
        for k in range_k:
            labels = KMeans(n_clusters=k, random_state=42).fit_predict(fitur_scaled)
            silhouette_scores.append(silhouette_score(fitur_scaled, labels))

        fig_silhouette = px.line(x=range_k, y=silhouette_scores, markers=True,
                                 labels={'x': 'Jumlah Klaster (K)', 'y': 'Silhouette Score'},
                                 title="Metode Silhouette")
        st.plotly_chart(fig_silhouette)
        best_k = range_k[np.argmax(silhouette_scores)]

        st.markdown(f"""
        **Interpretasi Silhouette:**
        Skor Silhouette menunjukkan seberapa jelas antar klaster terpisah.
        Nilai tertinggi pada **K = {best_k}** menunjukkan pemisahan terbaik.
        """)

        st.markdown(f"""
        ### ✅ Rekomendasi Jumlah Klaster
        - **Elbow** menyarankan: K = {elbow_k}  
        - **Silhouette** menyarankan: K = {best_k}  
        Anda dapat memilih sesuai analisis.
        """)

        st.subheader("⚙️ Pilih Jumlah Klaster Final")
        jumlah_klaster = st.slider("Jumlah klaster akhir:", min_value=2, max_value=10, value=best_k)

        kmeans = KMeans(n_clusters=jumlah_klaster, random_state=42)
        df['Klaster'] = kmeans.fit_predict(fitur_scaled)

        st.subheader("🗂️ Hasil Klasterisasi")
        df_hasil = df[["nama_kecamatan", "Klaster"] + jenis_wisata].copy()
        df_hasil_tampil = df_hasil.copy()
        df_hasil_tampil.insert(0, "No", range(1, len(df_hasil_tampil) + 1))
        st.dataframe(df_hasil_tampil, hide_index=True)

       # 7. ------------------- Visualisasi 2D ---------------------
        st.subheader("📊 Visualisasi Klaster (2 Jenis Wisata Terpilih)")
        fig_2d = px.scatter(x=fitur_scaled[:, 0], y=fitur_scaled[:, 1],
                            color=df["Klaster"].astype(str),
                            hover_name=df["nama_kecamatan"],
                            labels={"x": jenis_wisata[0], "y": jenis_wisata[1]},
                            title=f"Klasterisasi berdasarkan {jenis_wisata[0]} & {jenis_wisata[1]}",
                            width=900, height=600)
        fig_2d.update_traces(marker=dict(size=16))
        st.plotly_chart(fig_2d)

        st.markdown("""
        ** Interpretasi:**
        Grafik ini menunjukkan kecamatan yang mirip dari sisi dua jenis wisata. Titik berdekatan artinya karakteristiknya serupa.
        """)

        # 8. ------------------- Visualisasi PCA ---------------------
        st.subheader("🗺️ Visualisasi Semua Jenis Wisata Terpilih (2D)")
        pca = PCA(n_components=2)
        hasil_pca = pca.fit_transform(fitur_scaled)
        df["Dimensi X"] = hasil_pca[:, 0]
        df["Dimensi Y"] = hasil_pca[:, 1]

        fig_pca = px.scatter(df, x="Dimensi X", y="Dimensi Y",
                             color=df["Klaster"].astype(str),
                             hover_name="nama_kecamatan",
                             title="Pemetaan Seluruh Jenis Wisata Terpilih (2D)",
                             width=900, height=600)
        fig_pca.update_traces(marker=dict(size=16))
        st.plotly_chart(fig_pca)

        st.markdown("""
        ** Interpretasi:**
        Semua jenis wisata yang dipilih diringkas menjadi dua dimensi untuk mempermudah visualisasi.
        Titik yang berdekatan berarti kecamatan tersebut punya profil wisata yang mirip secara keseluruhan.
        """)

        st.subheader("💾 Unduh Hasil Klasterisasi")
        df_export = df[["nama_kecamatan", "Klaster"] + jenis_wisata + ["Dimensi X", "Dimensi Y"]]
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_export.to_excel(writer, index=False, sheet_name="Hasil Klasterisasi")
            pd.DataFrame({
                "Keterangan": [
                    "Klaster = hasil pengelompokan berdasarkan jenis wisata",
                    "Dimensi X/Y = pemetaan PCA untuk visualisasi",
                    "Gunakan hasil ini untuk kebijakan pariwisata"
                ]
            }).to_excel(writer, sheet_name="Penjelasan", index=False)

        st.download_button(
            label="📥 Download Excel Hasil",
            data=output.getvalue(),
            file_name="hasil_klasterisasi_wisata.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        color: grey;
        text-align: center;
        padding: 10px;
        background-color: #f9f9f9;
        font-size: 14px;
    }
    </style>

    <div class="footer">
       © 2025 Aplikasi Klasterisasi Wisata Kabupaten Bogor. 
    Dibuat oleh Donnix Afrilliando
    </div>
""", unsafe_allow_html=True)
