# Tugas Kelompok - Praktikum Kecerdasan Komputasional

Proyek klasifikasi wajah anggota kelompok menggunakan model MLP (Multi Layer Perceptron) dan CNN (Convolutional Neural Network). Dataset berupa video wajah setiap anggota yang diekstraksi menjadi frame gambar untuk keperluan training dan testing.

## Struktur Folder

```
tugas-kelompok/
├── data/
│   ├── videos/
│   │   ├── train/
│   │   │   └── <nama_anggota>/    video mentah untuk data latih
│   │   └── test/
│   │       └── <nama_anggota>/    video mentah untuk data uji
│   └── images/
│       ├── train/
│       │   └── <nama_anggota>/    frame hasil ekstraksi video train
│       └── test/
│           └── <nama_anggota>/    frame hasil ekstraksi video test
├── notebooks/
│   ├── mlp_classic.ipynb          eksperimen model MLP
│   └── cnn_model.ipynb            eksperimen model CNN
├── src/
│   ├── video_to_img.py            ekstraksi frame dari video ke gambar
│   └── post_video_to_hugging_face.py   upload video mentah ke Hugging Face Hub
└── README.md
```

### `data/videos`

Berisi video mentah wajah tiap anggota, dipisah berdasarkan split `train` dan `test`. Setiap anggota punya folder sendiri berisi satu atau lebih file video (`.mov` atau `.mp4`).

### `data/images`

Berisi frame gambar hasil ekstraksi dari `data/videos`, dengan struktur folder yang sama (`train`/`test` per anggota). Folder ini yang dipakai langsung oleh notebook untuk training dan evaluasi model, sehingga tidak perlu membaca file video secara langsung.

### `notebooks`

Berisi notebook eksperimen model:

- `mlp_classic.ipynb`: pipeline lengkap mulai dari load data, augmentasi, training MLP, hingga evaluasi dan inferensi.
- `cnn_model.ipynb`: pipeline setara menggunakan arsitektur CNN.

### `src`

Berisi script pendukung di luar notebook:

- `video_to_img.py`: membaca semua video di `data/videos/{train,test}/<nama_anggota>`, lalu mengekstrak frame ke `data/images/{train,test}/<nama_anggota>`.
- `post_video_to_hugging_face.py`: menyalin video lokal ke `data/videos/<split>/<nama_anggota>` sekaligus mengunggahnya ke dataset Hugging Face `raihanfaiq72/my_face`.

## Alur Kerja

1. Rekam video wajah, lalu simpan atau upload lewat `post_video_to_hugging_face.py` sehingga tersimpan di `data/videos/<split>/<nama_anggota>`.
2. Jalankan `video_to_img.py` untuk mengekstrak frame dari seluruh video menjadi gambar di `data/images/<split>/<nama_anggota>`.
3. Buka notebook di folder `notebooks` untuk melatih dan mengevaluasi model klasifikasi wajah.

## Dataset

Video mentah disimpan di Hugging Face Hub pada repo `raihanfaiq72/my_face` (.mov atau .mp4), sebagai cadangan dan sumber data bersama antar anggota kelompok.
