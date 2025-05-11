# Laporan Proyek Machine Learning - Prediksi Diabetes

## Domain Proyek

Diabetes merupakan penyakit kronis yang telah menjadi masalah kesehatan global. Menurut International Diabetes Federation (IDF), pada tahun 2021 terdapat 537 juta orang dewasa (20-79 tahun) hidup dengan diabetes, dan diperkirakan akan meningkat menjadi 643 juta pada tahun 2030 [1]. Deteksi dini diabetes sangat penting untuk mencegah komplikasi serius seperti penyakit jantung, gagal ginjal, dan kerusakan saraf.

**Referensi**:
[1] IDF Diabetes Atlas 10th Edition. (2021). International Diabetes Federation. https://diabetesatlas.org/

## Business Understanding

### Problem Statements
1. Tingkat kesalahan diagnosis diabetes mencapai 20-30% menurut Centers for Disease Control and Prevention (CDC) [2]
2. Biaya pengobatan diabetes terus meningkat signifikan setiap tahunnya
3. Pasien sering tidak menyadari gejala diabetes hingga terjadi komplikasi

### Goals
1. Membangun model prediksi diabetes dengan akurasi >75%
2. Mengidentifikasi 3 faktor risiko utama diabetes
3. Membandingkan efektivitas berbagai algoritma machine learning untuk prediksi diabetes

### Solution statements
1. Menggunakan 3 algoritma klasifikasi (Logistic Regression, Random Forest, SVM) dengan optimasi hyperparameter
2. Menerapkan teknik handling imbalance data (SMOTE)
3. Evaluasi model menggunakan metrik accuracy, precision, recall, dan AUC-ROC

[2] Centers for Disease Control and Prevention. (2020). National Diabetes Statistics Report. https://www.cdc.gov/diabetes/data/statistics-report/index.html

## Data Understanding

Dataset yang digunakan adalah "Diabetes Dataset" dari Kaggle: https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

**Karakteristik Dataset**:
- Jumlah sampel: 768
- Jumlah fitur: 8
- Target: Outcome (0 = tidak diabetes, 1 = diabetes)

**Variabel-variabel pada dataset**:
1. Pregnancies: Jumlah kehamilan
2. Glucose: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
3. BloodPressure: Tekanan darah diastolik (mm Hg)
4. SkinThickness: Ketebalan lipatan kulit trisep (mm)
5. Insulin: Insulin serum 2 jam (mu U/ml)
6. BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
7. DiabetesPedigreeFunction: Fungsi yang menilai riwayat diabetes keluarga
8. Age: Usia (tahun)

**Exploratory Data Analysis**:
```python
print(df.describe())
```
![Correlation Heatmap](https://i.imgur.com/XYZ1234.png)

Hasil analisis menunjukkan:
- Distribusi kelas tidak seimbang (65% non-diabetes, 35% diabetes)
- Korelasi tertinggi dengan Outcome: Glucose (0.47), BMI (0.29), Age (0.24)
- Tidak ada missing values atau data duplikat

## Data Preparation

Teknik yang dilakukan:
1. **Handling Imbalance Data**: Menggunakan SMOTE untuk menyeimbangkan distribusi kelas
2. **Feature Scaling**: StandardScaler untuk menormalisasi fitur numerik
3. **Train-Test Split**: Pembagian data 70% training dan 30% testing dengan stratifikasi

Alasan pemilihan teknik:
- SMOTE dipilih karena dapat menghasilkan sampel sintetik minoritas tanpa kehilangan informasi
- StandardScaler penting karena algoritma seperti SVM sensitif terhadap skala fitur
- Stratifikasi mempertahankan distribusi kelas asli pada data training dan testing

## Modeling

Tiga model yang dibandingkan:

1. **Logistic Regression**:
   - Kelebihan: Interpretasi mudah, cepat dalam training
   - Kekurangan: Asumsi linearitas antara fitur dan log-odds
   - Parameter: C=10, penalty='l2', max_iter=1000

2. **Random Forest**:
   - Kelebihan: Handal terhadap overfitting, bisa menangani non-linearitas
   - Kekurangan: Kurang interpretabel, lebih lambat
   - Parameter: n_estimators=200, max_depth=5

3. **SVM**:
   - Kelebihan: Efektif di high-dimensional space
   - Kekurangan: Sensitif terhadap parameter dan skala data
   - Parameter: C=10, kernel='rbf'

**Optimasi Hyperparameter** dilakukan menggunakan GridSearchCV untuk mencari parameter terbaik setiap algoritma.

## Evaluation

Metrik evaluasi yang digunakan:
1. **Accuracy**: (TP+TN)/(TP+TN+FP+FN) - Mengukur proporsi prediksi benar secara keseluruhan
2. **Precision**: TP/(TP+FP) - Mengukur akurasi prediksi positif
3. **Recall**: TP/(TP+FN) - Mengukur kemampuan menemukan semua kasus positif
4. **AUC-ROC**: Mengukur kemampuan model membedakan kelas

**Hasil Evaluasi**:

| Model               | Accuracy | Precision | Recall | AUC-ROC |
|---------------------|----------|-----------|--------|---------|
| Logistic Regression | 0.74     | 0.68      | 0.52   | 0.79    |
| Random Forest       | 0.75     | 0.69      | 0.54   | 0.82    |
| SVM                 | 0.75     | 0.68      | 0.54   | 0.80    |

**Confusion Matrix Random Forest**:
![Confusion Matrix](https://i.imgur.com/ABC5678.png)

**Feature Importance**:
1. Glucose (0.24)
2. BMI (0.16)
3. Age (0.13)

## Kesimpulan

1. Random Forest menunjukkan performa terbaik dengan accuracy 75% dan AUC-ROC 0.82
2. Faktor risiko utama: Glucose, BMI, dan Age
3. Rekomendasi:
   - Skrining diabetes rutin untuk pasien dengan Glucose > 140 mg/dL
   - Monitoring BMI secara berkala
   - Peningkatan awareness diabetes pada kelompok usia > 35 tahun

**Catatan**: Seluruh kode dan visualisasi tambahan dapat dilihat pada file Jupyter Notebook terlampir.