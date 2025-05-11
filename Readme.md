## TLKM Stock Prices Prediction Using LSTM

**Latar Belakang**:
PT Telekomunikasi Indonesia Tbk (TLKM) merupakan salah satu perusahaan BUMN terbesar di Indonesia di bidang telekomunikasi. Harga saham TLKM menjadi perhatian banyak investor karena mencerminkan stabilitas dan prospek bisnis sektor telekomunikasi nasional. Namun, seperti saham lainnya, harga TLKM berfluktuasi tergantung pada berbagai faktor seperti kondisi pasar, kebijakan pemerintah, dan sentimen investor.

Dengan kemajuan *Machine Learning (ML)*, analisis data historis dapat digunakan untuk mengidentifikasi pola dan tren harga saham. Salah satu metode populer dalam prediksi data time series adalah LSTM (Long Short-Term Memory) yang mampu menangkap pola jangka panjang dan kompleks dalam data historis. Dengan pendekatan ini, investor bisa mendapat wawasan lebih baik tentang prediksi harga saham untuk mendukung pengambilan keputusan.

**Penelitian Terkait**:
*Windha Mega Pradnya Dhuhita* menyatakan bahwa teknik LSTM, yang dikombinasikan dengan optimasi hiperparameter, memungkinkan prediksi harga yang akurat menggunakan data historis, termasuk untuk aset seperti saham maupun logam mulia seperti emas.

## Business Understanding

### Problem Statements

1. Harga saham TLKM berfluktuasi setiap hari dipengaruhi oleh banyak faktor internal dan eksternal. Fluktuasi ini bisa sulit diprediksi tanpa alat bantu statistik atau algoritma pembelajaran mesin.
2. Diperlukan sistem prediksi berbasis ML yang mampu memperkirakan harga saham di masa depan berdasarkan pola historis, dan LSTM cocok digunakan untuk menangani data sekuensial seperti harga saham.

### Goals

Mengembangkan model LSTM untuk memprediksi harga penutupan saham TLKM secara akurat berdasarkan data historis, sehingga dapat memberikan proyeksi harga yang berguna bagi investor dan analis pasar.

### Solution Statements

Melatih dan mengevaluasi model LSTM menggunakan data historis harga saham TLKM agar dapat memprediksi harga penutupan masa depan dengan akurasi tinggi.

## Data Understanding

Dataset: TLKM.JK Stock Historical Data (2019-2024)

### Gambaran Umum Dataset

Berkas CSV yang digunakan memiliki total **1212 baris** dan **6 kolom**, yaitu:

| # | Kolom     | Tipe Data |
| - | --------- | --------- |
| 0 | Adj Close | float64   |
| 1 | Close     | float64   |
| 2 | High      | float64   |
| 3 | Low       | float64   |
| 4 | Open      | float64   |
| 5 | Volume    | float64   |

Tidak terdapat nilai null, dan seluruh data bertipe numerik.

### Korelasi Antar Variabel

![Korelasi Variabel](attachment\:file-Gt2QUXyCvEGNyREHYnbxGV)

Dari grafik di atas, terlihat korelasi tinggi antara variabel harga (Close, Open, High, Low, Adj Close) dan korelasi rendah antara Volume dan variabel harga lainnya. Ini menunjukkan bahwa variabel harga bergerak secara searah, sementara volume relatif independen.

### Visualisasi Harga

1. Grafik Harga Tertinggi dan Terendah TLKM:

![Harga High Low](attachment\:file-4Yxt3DyrGqqKAfyx6iYihN)

2. Grafik Perkembangan Harga Penutupan dan Prediksi:

![Prediksi Harga](attachment\:file-MSjpUGH27vGYjXXfXTLuBZ)

## Data Preparation

### 1. Memilih Kolom yang Diprediksi

Kolom `Close` digunakan sebagai target prediksi, diubah menjadi array NumPy untuk keperluan pelatihan.

### 2. Normalisasi Data

Menggunakan `MinMaxScaler` dari sklearn untuk menskalakan data ke rentang \[0, 1].

### 3. Split Data

* Data latih: 80% (969 data)
* Data uji: 20% (243 data)

### 4. Membuat Windowing (Time Series Format)

Membuat input LSTM dengan window sepanjang 60 langkah waktu sebelumnya untuk memprediksi 1 langkah ke depan.

### 5. Reshape Data

Data diubah menjadi format 3 dimensi (samples, timesteps, features) yang diperlukan oleh model LSTM.

## Modeling

### Arsitektur Model LSTM:

* LSTM(50 units, return\_sequences=True)
* Dropout(0.2)
* LSTM(64 units, return\_sequences=False)
* Dropout(0.2)
* Dense(32) → Dense(16) → Dense(1)

Model dilatih selama 100 epoch. Hasil pelatihan menunjukkan konvergensi yang baik:

![Losses](attachment\:file-XPrcRRZLdC1nP3ayGTHDCy)

Loss (MSE) dan MAE menurun secara signifikan selama pelatihan, menandakan proses belajar yang stabil.

## Evaluation

Model diuji menggunakan data uji (243 observasi), dan hasil evaluasi menunjukkan:

* **Loss**: 0.0016
* **MAE**: 0.0318

Hasil prediksi juga divisualisasikan dan dibandingkan dengan data aktual:

![Prediksi Saham](attachment\:file-MSjpUGH27vGYjXXfXTLuBZ)

Model menunjukkan kemampuan yang baik dalam mengikuti tren harga aktual pada data uji. Garis prediksi (kuning) cukup dekat dengan data aktual (merah), menunjukkan performa prediktif yang akurat.

## Kesimpulan

Model LSTM yang digunakan pada dataset saham TLKM mampu mempelajari pola historis dengan baik dan menghasilkan prediksi harga penutupan yang akurat, dengan MAE sebesar **0.0318** (3.18%). Ini membuktikan efektivitas LSTM dalam menangkap dinamika pasar saham jangka pendek hingga menengah.

Model ini bisa dijadikan dasar untuk pengembangan sistem rekomendasi investasi berbasis AI di masa depan.

---

**Referensi**:

* Data: [Yahoo Finance - TLKM.JK](https://finance.yahoo.com/quote/TLKM.JK)
* Windha Mega Pradnya Dhuhita et al. (2023). *Gold Price Prediction Based On Yahoo Finance Data Using LSTM Algorithm*. IEEE Conference. 10.1109/ICIMCIS60089.2023.10349035.
