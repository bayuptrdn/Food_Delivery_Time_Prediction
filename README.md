# 🍽️ Food_Delivery_Time_Prediction  
![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## Deskripsi Proyek
**Food Delivery Time Prediction** adalah proyek *machine learning* yang bertujuan untuk mengembangkan model prediksi **durasi pengantaran makanan (ETA)** menggunakan dataset dunia nyata.

Model ini membantu perusahaan *food delivery* dalam:
- Memberikan estimasi waktu pengantaran yang lebih akurat
- Meningkatkan efisiensi operasional
- Meningkatkan kepuasan pelanggan melalui informasi ETA yang realistis

---

## 📁 Struktur Repository

Food_Delivery_Time_Prediction
|
├── README.md
├── FinalProject.ipynb
├── FinalProject_inference.ipynb
├── Food_Delivery_Times.csv
├── airflow_ml_pipeline.py
├── requirements.txt
├── deployment
│ ├── Food_Delivery_Times_Clean.csv
│ ├── app.py
│ ├── best_xgboost_delivery_time_tuned.pkl
│ ├── eda.py
│ └── prediction.py


---

## 🎯 Latar Belakang Masalah
Ketepatan estimasi waktu pengantaran (Estimated Time of Arrival / ETA) merupakan faktor krusial dalam layanan pengantaran makanan.  
Perbedaan antara ETA dan waktu aktual dapat menurunkan kepuasan pelanggan dan efisiensi operasional.

Faktor-faktor seperti:
- Jarak pengantaran  
- Kondisi lalu lintas  
- Cuaca  
- Waktu persiapan makanan  
- Pengalaman kurir  

memiliki pengaruh signifikan terhadap durasi pengantaran.

Proyek ini mengatasi permasalahan tersebut dengan pendekatan **Machine Learning Regression**.

---

## 📊 Data Overview
**Dataset:** `Food_Delivery_Times.csv`

- Jumlah data: **1.000 baris**
- Jumlah fitur: **9 kolom**
- Target: **Delivery_Time_min**

**Fitur utama:**
- Numerik:  
  - `Distance_km`  
  - `Preparation_Time_min`  
  - `Courier_Experience_yrs`
- Kategorikal:  
  - `Weather`  
  - `Traffic_Level`  
  - `Time_of_Day`  
  - `Vehicle_Type`

Beberapa fitur mengandung *missing values* dan telah ditangani pada tahap preprocessing.

---

## ⚙️ Metodologi
Pendekatan yang digunakan adalah **Supervised Learning (Regresi)** dengan tahapan:

1. **Data Cleaning & Exploratory Data Analysis (EDA)**
2. **Feature Engineering & Preprocessing**
   - Handling missing values
   - Encoding data kategorikal
   - Feature scaling
3. **Model Training & Evaluation**
   - Linear Regression
   - KNN
   - SVR
   - Decision Tree
   - Random Forest
   - XGBoost
4. **Model Selection & Hyperparameter Tuning**
5. **Deployment menggunakan Streamlit**

---

## 🧠 Model Terbaik
- **Model:** XGBoost Regressor (tuned)
- **Performa:**
  - MAE: ±7 menit
  - R² Score: ±0.70
- Model menunjukkan keseimbangan yang baik antara akurasi dan generalisasi.

Model tersimpan dalam format `.pkl` dan siap digunakan untuk inferensi.

---

## 💡 Insight Bisnis
- **Jarak pengantaran** merupakan faktor paling dominan.
- **Waktu persiapan makanan** dan **pengalaman kurir** berpengaruh signifikan.
- **Cuaca buruk** dan **lalu lintas padat** meningkatkan durasi pengantaran secara konsisten.

Insight ini dapat dimanfaatkan untuk:
- Estimasi ETA berbasis kondisi nyata
- Optimasi penjadwalan dan alokasi kurir
- Perencanaan operasional yang lebih efisien

---

## 🧩 Tech Stack
**Bahasa & Tools**
- Python 3.9
- Jupyter Notebook
- Streamlit
- Airflow (pipeline)

**Library**
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- pickle

---

## 🌐 Deployment
Aplikasi prediksi interaktif dikembangkan menggunakan **Streamlit**  
dan dapat dijalankan melalui folder `deployment/`.

---

## 🚀 Pengembangan Selanjutnya
- Integrasi API cuaca dan lalu lintas real-time
- Penambahan fitur tracking end-to-end
- Peningkatan performa model dengan data tambahan
- Deployment ke cloud (Streamlit Cloud / Hugging Face)

---

## 📚 Referensi
- Dokumentasi scikit-learn: https://scikit-learn.org/stable/
- Studi ETA Prediction (Grab, Uber Eats, DoorDash)

---

> *Food_Delivery_Time_Prediction*  
> Mengubah data pengantaran menjadi insight yang berdampak.  




