# Narasi Augmentasi Data & Arsitektur Model (MLP)

## Augmentasi Data

Augmentasi dilakukan hanya pada data train, sesuai prinsip agar data validasi dan test tetap merepresentasikan distribusi asli tanpa campur tangan buatan. Prosesnya dijalankan lewat fungsi `augment_train_set` di `src/augment.py`, yang menerima kumpulan path gambar train beserta labelnya.

Setiap gambar terlebih dahulu dimuat dan diproses melalui `load_grayscale`, yaitu dikonversi ke grayscale lalu di-resize ke ukuran seragam 224x224 piksel. Setelah itu, satu gambar asli diperluas menjadi enam variasi baru lewat fungsi `augment_image`, dengan lima jenis transformasi:

1. **Flip horizontal** — gambar dicerminkan secara horizontal, mensimulasikan variasi orientasi wajah.
2. **Rotasi kecil ±15 derajat** — gambar diputar sedikit ke kiri dan ke kanan menggunakan `warpAffine`, dengan `BORDER_REPLICATE` agar piksel di tepi hasil rotasi tidak menghasilkan area hitam kosong.
3. **Perubahan brightness (lebih terang)** — kontras dan kecerahan gambar dinaikkan (alpha=1.2, beta=15) untuk mensimulasikan kondisi pencahayaan lebih kuat.
4. **Perubahan brightness (lebih gelap)** — sebaliknya, gambar diredupkan (alpha=0.8, beta=-15) untuk mensimulasikan kondisi minim cahaya.
5. **Translasi kecil** — gambar digeser sejauh 5% dari lebar dan tinggi gambar, mensimulasikan pergeseran posisi wajah dalam frame.

Dari kelima transformasi ini dihasilkan enam variant (rotasi menghasilkan dua variant sekaligus, yaitu -15 dan +15 derajat). Setiap variant tersebut diberi label yang sama dengan gambar aslinya, dan path-nya ditandai dengan sufiks `" (aug)"` agar tetap bisa dibedakan dari data asli saat ditelusuri.

Hasil akhirnya, jumlah data train melonjak signifikan — dari 338 gambar menjadi 2.366 gambar (naik sekitar 7 kali lipat). Tujuannya adalah memperkaya variasi visual yang dilihat model saat training, sehingga model tidak overfit terhadap kondisi pengambilan gambar yang terbatas (satu sudut, satu pencahayaan, satu posisi), dan diharapkan lebih general saat menghadapi data test maupun input baru.

## Encoding Label

Sebelum masuk ke model, label kelas (`damba`, `faiq`, `faqih`) yang masih berupa string perlu diubah ke bentuk numerik. Prosesnya dua tahap:

1. **Label encoding** — `LabelEncoder` dari scikit-learn di-fit ke `y_train_label`, memetakan tiap nama kelas ke sebuah integer (misalnya damba=0, faiq=1, faqih=2). Encoder ini di-fit hanya sekali dari data train, lalu dipakai konsisten (`transform`) untuk train, val, maupun test — supaya pemetaan integer ke label sama persis di ketiga set data.
2. **One-hot encoding** — hasil integer tadi diubah lagi jadi vektor one-hot lewat `to_categorical`, misalnya kelas 1 (faiq) menjadi `[0, 1, 0]`. Bentuk ini dibutuhkan karena output layer model pakai aktivasi softmax dan loss `categorical_crossentropy`, yang mengharapkan target berupa distribusi probabilitas per kelas, bukan angka kelas tunggal.

Di cell yang sama juga dihitung `class_weight` pakai `compute_class_weight(class_weight="balanced", ...)`. Ini perlu karena jumlah data per kelas tidak seimbang (damba, faiq, faqih punya jumlah sampel berbeda) — dengan bobot ini, kesalahan pada kelas yang datanya lebih sedikit (seperti faiq) "dihukum" lebih berat saat training, supaya model tidak bias condong ke kelas mayoritas.

## Arsitektur Model (MLP)

Model yang dibangun adalah Multi-Layer Perceptron (MLP) sederhana menggunakan `Sequential` API dari Keras. Input model berupa citra grayscale 224x224 yang sudah di-flatten menjadi vektor 1 dimensi berukuran 50.176 fitur (224 × 224 × 1), lalu dinormalisasi ke rentang 0–1.

Struktur jaringannya terdiri dari tiga hidden layer yang mengecil secara bertahap (funnel-shaped), diselingi dropout untuk mencegah overfitting:

| Layer | Tipe | Output Shape | Parameter | Keterangan |
|---|---|---|---|---|
| 1 | Dense | 128 | 6.422.656 | Aktivasi ReLU |
| — | Dropout | 128 | 0 | Rate 0.4 |
| 2 | Dense | 64 | 8.256 | Aktivasi ReLU |
| — | Dropout | 64 | 0 | Rate 0.3 |
| 3 | Dense | 32 | 2.080 | Aktivasi ReLU |
| — | Dropout | 32 | 0 | Rate 0.2 |
| 4 | Dense (output) | 3 | 99 | Aktivasi Softmax |

Total parameter yang dilatih mencapai 6.433.091 (~24.5 MB), dan hampir seluruhnya (6,4 juta) berada di layer pertama — konsekuensi dari input yang sangat besar (50.176 fitur) langsung terhubung penuh (fully-connected) ke 128 neuron.

Tiga poin desain yang penting untuk dijelaskan:

- **Funnel architecture (128 → 64 → 32 → 3)**: jumlah neuron mengecil bertahap agar model bisa mengekstrak representasi fitur secara hierarkis — dari fitur kasar (128 neuron) ke representasi yang semakin ringkas dan abstrak, sebelum akhirnya diklasifikasikan ke 3 kelas output.
- **Dropout bertingkat (0.4, 0.3, 0.2)**: dropout rate yang lebih tinggi ditempatkan di layer-layer awal (yang parameternya jauh lebih besar dan rawan overfitting), lalu diturunkan bertahap di layer-layer berikutnya yang parameternya makin kecil dan mendekati output.
- **Output layer softmax dengan 3 neuron**: sesuai jumlah kelas target (3 anggota kelompok: damba, faiq, faqih), menghasilkan distribusi probabilitas untuk klasifikasi multi-kelas.

Model dikompilasi dengan optimizer Adam, loss `categorical_crossentropy` (karena label berbentuk one-hot), dan metrik F1 Score macro-average — pemilihan F1 macro relevan karena metrik ini memberi bobot setara ke tiap kelas, cocok untuk memantau performa yang seimbang antar anggota kelompok meski ada sedikit ketimpangan jumlah data antar kelas (yang juga sudah diatasi lewat `class_weight` saat training).

## Hasil Training & Evaluasi

Model dilatih selama 50 epoch dengan batch size 16. Pada evaluasi terhadap data test (195 gambar), model mencapai akurasi keseluruhan 89%, dengan rincian per kelas sebagai berikut:

| Kelas | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| damba | 0.94 | 1.00 | 0.97 | 96 |
| faiq | 0.83 | 0.60 | 0.69 | 42 |
| faqih | 0.83 | 0.91 | 0.87 | 57 |

Macro avg F1-score berada di angka 0.84. Kelas **faiq** menunjukkan recall paling rendah (0.60), artinya model masih cukup sering salah mengklasifikasikan gambar faiq sebagai kelas lain — kemungkinan karena jumlah datanya yang lebih sedikit (support 42, paling kecil di antara tiga kelas) meski sudah dibantu `class_weight` saat training. Sebaliknya, kelas **damba** paling mudah dikenali model dengan recall sempurna (1.00).
