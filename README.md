# Computer Vision Web App (Streamlit)

Aplikasi web berbasis Python + Streamlit untuk demonstrasi tahapan computer vision:

- Contrast Enhancement (CLAHE)
- Thresholding
- Harris Corner
- Convolution: Gaussian Blur, Sobel, Prewitt
- Morphology: Erosi, Dilasi, Filling Holes
- Feature Matching: ORB
- Evaluasi: Precision, Recall, F1-score, IoU

## Struktur Folder

```text
computer-vision-pra-uts/
├── app.py
├── requirements.txt
├── dataset/
│   ├── images/
│   └── masks/
├── modules/
└── utils/
```

## Jalankan Aplikasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Format Dataset

- Simpan gambar pada `dataset/images`
- Subfolder di dalam `dataset/images` didukung (scan rekursif)
- Simpan ground-truth mask pada `dataset/masks`
- Nama file mask harus sama dengan gambar (contoh: `img1.jpg` dan `img1.png`)
- Disarankan mirror struktur folder image ke mask (contoh: `images/kelas1/a.jpg` -> `masks/kelas1/a.png`)

## Tab Aplikasi

- `Galeri Dataset`: preview gambar lokal dan pilih gambar untuk analisis
- `Analisis Gambar`: tampilkan before/after semua tahap pemrosesan
- `Feature Matching`: pilih 2 gambar lalu jalankan ORB matching
- `Evaluasi`: hitung metrik segmentasi dari mask prediksi vs ground truth
