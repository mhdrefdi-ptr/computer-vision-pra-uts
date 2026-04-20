# CV Workbench - Analisis Citra Pra-UTS

Aplikasi Streamlit berbasis Computer Vision klasik untuk kebutuhan eksperimen akademik:

- Dataset manager (download Kaggle + validasi struktur split)
- Preprocessing (min-max, contrast enhancement, sharpening)
- Teknik CV (thresholding + Harris corner)
- Konvolusi (Gaussian, Sobel, Prewitt)
- Morfologi (opening, closing, filling holes)
- Feature matching (ORB + BFMatcher)
- Evaluasi IoU
- Manual calculation lab patch 15x15
- Export hasil gambar, metrik, dan matriks

## Struktur

```text
cv_workbench/
├── streamlit_app.py
├── pages/
├── core/
├── exports/
├── datasets/
└── requirements.txt
```

## Jalankan

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Catatan Dataset

Secara default aplikasi mengunduh dataset:

- `wanderdust/coin-images`

dan menyiapkannya pada folder lokal:

- `datasets/coin-images`

Ground truth mask disimpan terpisah dan persisten:

- `datasets/coin-images-masks`

Skema path mask:

- Image: `datasets/coin-images/data/<split>/<class>/abc.jpg`
- Mask: `datasets/coin-images-masks/<split>/<class>/abc.png`

Dengan skema ini, saat dataset di-download ulang (termasuk force download/cache reset), folder mask tetap aman karena tidak berada di root dataset hasil download.
Saat `Force download` dipilih di Dataset Manager, data image akan di-refresh dengan mode merge non-destruktif pada `datasets/coin-images`.

Folder `datasets/` dan `exports/` tidak di-commit.
