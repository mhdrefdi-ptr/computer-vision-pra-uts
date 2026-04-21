# CV Workbench - Analisis Citra Pra-UTS

Aplikasi Streamlit berbasis Computer Vision klasik untuk kebutuhan eksperimen akademik:

- Dataset manager (download Kaggle + validasi struktur split)
- Preprocessing (min-max, contrast enhancement, sharpening)
- Teknik CV (thresholding + Harris corner)
- Konvolusi (Gaussian, Sobel, Prewitt)
- Morfologi (opening, closing, filling holes)
- Feature matching (ORB + BFMatcher)
- Evaluasi IoU
- Manual calculation lab patch 15x15 + manual Hamming untuk feature matching
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

## Cara Setup & Jalankan

### 1) Ambil source code

#### Opsi A - Clone dengan Git

```bash
git clone https://github.com/mhdrefdi-ptr/computer-vision-pra-uts.git
cd computer-vision-pra-uts
```

#### Opsi B - Download ZIP dari GitHub

1. Buka repo GitHub
2. Klik `Code` -> `Download ZIP`
3. Extract ZIP
4. Buka terminal di folder hasil extract

### 2) Jalankan di Windows (PowerShell)

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 3) Jalankan di macOS (Terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 4) Jalankan di Linux (Terminal)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### 5) Buka aplikasi

Setelah command jalan, buka URL Streamlit yang tampil di terminal (biasanya `http://localhost:8501`).

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
