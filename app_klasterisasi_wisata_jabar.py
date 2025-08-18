import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import plotly.express as px
from io import BytesIO

st.set_page_config(page_title="Aplikasi Klasterisasi Jenis Wisata", layout="wide")
st.title("Aplikasi Klasterisasi Jenis Wisata berdasarkan Kecamatan di Kabupaten Bogor")

# 1. ------------------- Template Dataset dari File  ---------------------
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
        st.warning("⚠️ Data jenis wisata masih kosong. Silahkan isi jumlah wisata di kolom yang tersedia.")
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
    if len(jenis_wisata) < 2:
        st.warning("Anda harus memilih minimal 2 jenis wisata untuk melakukan klasterisasi.")
        st.stop()

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

        Dari hasil analisis, sistem memberikan dua saran jumlah klaster (K) berdasarkan dua metode evaluasi:

        - **Metode Elbow** menyarankan: **K = {elbow_k}**  
        Metode ini bekerja dengan melihat titik "tekukan" (elbow) pada grafik, yaitu titik di mana penambahan jumlah klaster tidak lagi mengurangi error secara signifikan. Ini menandakan jumlah klaster yang cukup optimal untuk mengelompokkan data tanpa overfitting.

        - **Metode Silhouette** menyarankan: **K = {best_k}**  
        Metode ini mengukur seberapa baik objek berada dalam klasternya. Nilai Silhouette yang tinggi berarti objek cocok dengan klasternya dan berbeda dari klaster lain. K dengan skor Silhouette tertinggi dianggap paling optimal secara kualitas pemisahan antar klaster.

        💡 **Tips untuk pengguna:**  
        Jika Anda ingin hasil yang seimbang antara efisiensi dan akurasi, pilih nilai K yang disarankan oleh **Silhouette**. Namun, jika Anda ingin melihat perubahan visual dari distribusi data secara bertahap, Anda bisa mencoba nilai K dari **Elbow** juga.

        """)
    
    st.subheader("⚙️ Pilih Jumlah Klaster Final")
    jumlah_klaster = st.slider("Jumlah klaster akhir:", min_value=2, max_value=10, value=best_k)

    kmeans = KMeans(n_clusters=jumlah_klaster, random_state=42)
    df['Klaster'] = kmeans.fit_predict(fitur_scaled)

    st.subheader("🗂️️ Hasil Klasterisasi")
    df_hasil = df[["nama_kecamatan", "Klaster"] + jenis_wisata].copy()
    df_hasil.insert(0, "No", range(1, len(df_hasil) + 1))
    st.dataframe(df_hasil, hide_index=True)

    if len(jenis_wisata) == 2:
        st.subheader("📊 Visualisasi Klaster (2 Jenis Wisata Terpilih)")
        fig_2d = px.scatter(x=fitur_scaled[:, 0], y=fitur_scaled[:, 1],
                            color=df["Klaster"].astype(str),
                            hover_name=df["nama_kecamatan"],
                            labels={"x": jenis_wisata[0], "y": jenis_wisata[1]},
                            title=f"Klasterisasi berdasarkan {jenis_wisata[0]} & {jenis_wisata[1]}",
                            width=900, height=600)
        fig_2d.update_traces(marker=dict(size=16))
        st.plotly_chart(fig_2d)

        st.markdown(f"""
         **Tentang Grafik Klasterisasi 2D**  
        Grafik ini menunjukkan penyebaran kecamatan berdasarkan dua jenis wisata yang dipilih, yaitu **{jenis_wisata[0]}** dan **{jenis_wisata[1]}**.  
S       etiap titik merepresentasikan satu kecamatan, dan posisinya ditentukan berdasarkan jumlah tempat wisata dari dua kategori tersebut.  
         **Makna Warna pada Grafik**  
        Warna pada setiap titik menunjukkan **hasil klasterisasi**, yaitu kelompok kecamatan dengan karakteristik wisata yang mirip.  
        Semakin dekat dan warnanya sama, semakin mirip pula karakter jenis wisatanya.  
        Dengan grafik ini, pengguna dapat melihat pola pengelompokan secara visual dan sederhana berdasarkan fitur wisata yang dipilih.
        """)
    elif len(jenis_wisata) > 2:
        st.subheader("🗺️️ Visualisasi Klasterisasi dengan PCA")
        pca = PCA(n_components=2)
        hasil_pca = pca.fit_transform(fitur_scaled)
        df["Dimensi X"] = hasil_pca[:, 0]
        df["Dimensi Y"] = hasil_pca[:, 1]

        fig_pca = px.scatter(df, x="Dimensi X", y="Dimensi Y",
                             color=df["Klaster"].astype(str),
                             hover_name="nama_kecamatan",
                             title="Pemetaan Klasterisasi dengan PCA Berdasarkan Semua Jenis Wisata",
                             width=900, height=600)
        fig_pca.update_traces(marker=dict(size=16))
        st.plotly_chart(fig_pca)

        st.markdown("""
         **Tentang Grafik Klasterisasi PCA**  
        Grafik ini menggunakan metode **Principal Component Analysis (PCA)** untuk mereduksi banyak fitur (semua jenis wisata) ke dalam dua sumbu utama (PC1 dan PC2), sehingga memudahkan visualisasi data berdimensi tinggi.  
        Setiap titik tetap mewakili satu kecamatan, tetapi posisi ditentukan oleh kombinasi semua fitur wisata secara matematis.

         **Makna Warna pada Grafik**  
        Warna pada titik-titik menunjukkan klaster yang dihasilkan oleh algoritma K-Means.  
        Meskipun sumbu X dan Y bukan fitur asli, **warna tetap menunjukkan kecamatan yang masuk dalam satu klaster berdasarkan kemiripan jenis wisatanya** secara keseluruhan.
        """)

    st.subheader("💾  Unduh Hasil Klasterisasi")
    export_cols = ["nama_kecamatan", "Klaster"] + jenis_wisata
    if "Dimensi X" in df.columns and "Dimensi Y" in df.columns:
        export_cols += ["Dimensi X", "Dimensi Y"]
    df_export = df[export_cols]

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
        label="📥  Download Excel Hasil",
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
       © 2025 Aplikasi Klasterisasi Wisata Kabupaten Bogor.<br>
       Dibuat oleh Donnix Afrilliando
    </div>
""", unsafe_allow_html=True)