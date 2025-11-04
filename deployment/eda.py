import streamlit as st
import pandas as pd
import plotly.express as px
import os

def run():
    # ===============================
    # Header Section
    # ===============================
    st.title("Delivery Time Prediction — Exploratory Data Analysis (EDA)")

    st.image(
        "https://www.apto.digital/au/wp-content/uploads/2023/09/Swiggy-Banner-1.webp",
        caption="Source: apto.digital (Swiggy Delivery Banner)"
    )

    st.markdown("""
    ## 🧭 Project Background  
    Efisiensi pengiriman merupakan faktor kunci dalam **operasional logistik dan kepuasan pelanggan**.  
    Tujuan EDA ini adalah memahami **pola dan faktor-faktor yang memengaruhi waktu pengantaran** 
    berdasarkan data historis dari berbagai kondisi pengiriman — seperti jarak, cuaca, lalu lintas, 
    waktu pengantaran, dan pengalaman kurir.

    ---
    """)

    # ===============================
    # Dataset Overview
    # ===============================
    st.header("📊 Dataset Overview")

    file_path = os.path.join(os.path.dirname(__file__), "Food_Delivery_Times_Clean.csv")
    df = pd.read_csv(file_path)
    st.dataframe(df.head())
    st.markdown(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ===============================
    # 1. Distribusi waktu pengantaran
    # ===============================
    st.subheader("Distribusi Waktu Pengantaran (`delivery_time_min`)")
    fig = px.histogram(df, x="delivery_time_min", nbins=30, color_discrete_sequence=["#FF7F50"])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    Mayoritas pengiriman selesai dalam waktu **20–40 menit**, dengan beberapa kasus ekstrem di atas 60 menit.
    """)

    # ===============================
    # 2. Distribusi jarak & waktu persiapan
    # ===============================
    st.subheader("Distribusi `distance_km` dan `preparation_time_min`")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x="distance_km", nbins=25, color_discrete_sequence=["#00BFFF"])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.histogram(df, x="preparation_time_min", nbins=25, color_discrete_sequence=["#8A2BE2"])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Insight:**  
    - Sebagian besar pengantaran berjarak **<10 km**.  
    - Waktu persiapan restoran umumnya **10–25 menit**.
    """)

    # ===============================
    # 3. Pengaruh cuaca
    # ===============================
    st.subheader("Pengaruh Cuaca terhadap Waktu Pengantaran")
    fig = px.box(df, x="weather", y="delivery_time_min", color="weather")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    Cuaca **hujan** cenderung meningkatkan waktu pengantaran karena kondisi jalan dan visibilitas menurun.
    """)

    # ===============================
    # 4. Pengaruh tingkat lalu lintas
    # ===============================
    st.subheader("Pengaruh Tingkat Lalu Lintas (`traffic_level`)")
    fig = px.box(df, x="traffic_level", y="delivery_time_min", color="traffic_level")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    Saat **traffic high**, median waktu pengiriman meningkat signifikan dibandingkan kondisi lalu lintas ringan.
    """)

    # ===============================
    # 5. Perbedaan waktu dalam sehari
    # ===============================
    st.subheader("Waktu Pengantaran berdasarkan `time_of_day`")
    avg_tod = df.groupby("time_of_day")["delivery_time_min"].mean().reset_index()
    fig = px.bar(avg_tod, x="time_of_day", y="delivery_time_min", color="time_of_day",
                 title="Rata-rata Waktu Pengantaran per Waktu dalam Sehari")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    Pengiriman saat **Lunch** dan **Evening** cenderung lebih lama — sesuai dengan jam sibuk kendaraan dan pesanan.
    """)

    # ===============================
    # 6. Hubungan pengalaman kurir
    # ===============================
    st.subheader("Pengalaman Kurir (`courier_experience_yrs`) terhadap Efisiensi Pengiriman")
    fig = px.scatter(df, x="courier_experience_yrs", y="delivery_time_min", trendline="ols",
                     color_discrete_sequence=["#32CD32"])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    **Insight:**  
    Kurir dengan pengalaman lebih dari **5 tahun** umumnya menunjukkan waktu pengiriman lebih stabil dan cepat.
    """)

    # ===============================
    # 7. Korelasi antar fitur numerik
    # ===============================
    st.subheader("Korelasi Antar Fitur Numerik")
    num_cols = ["distance_km", "preparation_time_min", "courier_experience_yrs", "delivery_time_min"]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Heatmap Korelasi Fitur Numerik")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    **Insight:**  
    - Jarak dan waktu pengiriman memiliki korelasi positif kuat.  
    - Pengalaman kurir berkorelasi negatif dengan waktu pengiriman (semakin berpengalaman, semakin cepat).
    """)

    # ===============================
    # Summary
    # ===============================
    st.header("📋 Ringkasan Akhir EDA")
    st.markdown("""
    - Sebagian besar pengantaran selesai di bawah 40 menit.  
    - Cuaca dan traffic menjadi faktor eksternal paling signifikan.  
    - Waktu sibuk dan pengalaman kurir juga memengaruhi efisiensi.  
    - Model prediksi nanti akan mempertimbangkan kombinasi variabel **operasional + lingkungan**.
    """)

if __name__ == "__main__":
    run()
