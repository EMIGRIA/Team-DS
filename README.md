# 🛡️ Emigria: Multimodal AI-Powered Overseas Job Fraud Detection

[![Streamlit App](https://static.streamlit.io/badge_svg.svg)](https://team-ds-emigria.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![ML Framework](https://img.shields.io/badge/framework-Scikit--Learn%20%7C%20XGBoost-orange.svg)](https://scikit-learn.org/)

**Emigria** adalah sistem deteksi dini berbasis kecerdasan buatan (*Multimodal AI*) yang dirancang untuk mengidentifikasi lowongan kerja fiktif luar negeri. Proyek ini difokuskan pada koridor penempatan **ASEAN dan Asia Timur** guna melindungi calon Pekerja Migran Indonesia (PMI) dari sindikat penipuan tenaga kerja dan perdagangan manusia (*human trafficking*).

Sistem ini tidak sekadar mencocokkan kata kunci, melainkan mengintegrasikan pemrosesan bahasa alami (NLP), validasi entitas P3MI (BP2MI), serta pengayaan data risiko geopolitik global.

---

## 🚀 Langkah Cepat Menjalankan Aplikasi (Quick Start)

Ikuti langkah berikut di terminal/command prompt Anda untuk menjalankan dashboard interaktif secara lokal:

### 1. Sinkronisasi (Clone) Repositori
```bash
git clone https://github.com/EMIGRIA/team-ds.git
cd team-ds
```

### 2. Membuat & Mengaktifkan Virtual Environment
* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **macOS / Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Instalasi Dependensi
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Menjalankan Dashboard Streamlit
```bash
streamlit run dashboard/dashboard.py
```
> Buka browser pada alamat `http://localhost:8501`. Jika file `dashboard.py` ada di luar folder, gunakan `streamlit run dashboard.py`.

---

## 📂 Struktur Repositori

| Nama File / Folder | Fungsi dalam Proyek |
| :--- | :--- |
| **`dashboard/dashboard.py`** | Script antarmuka web Streamlit (Visualisasi 5 Business Questions, tabel, & simulasi AI). |
| **`dashboard/main_data.csv`**| Dataset final regional Asia berserta label probabilitas AI untuk divisualisasikan. |
| **`notebook.ipynb`** | File inti Data Science (EDA, Feature Engineering, Training XGBoost, & A/B Testing). |
| **`scraper_bp2mi.ipynb`** | Skrip web scraping portal SISKOP2MI BP2MI untuk mendapatkan data loker legal. |
| **`model/`** | Folder berisi model `.joblib` (Model XGBoost final, Scaler, dan Encoder). |
| **`geo_risk_database.csv`** | Basis data risiko geopolitik penempatan dari *Global Organized Crime Index*. |
| **`reality_check_salary.csv`**| Data acuan batas upah wajar BP2MI berdasarkan negara tujuan. |
| **`requirements.txt`** | Daftar library minimal untuk deployment produksi (Streamlit Cloud). |

---

## 📊 5 Pertanyaan Bisnis (Business Questions)

Proyek ini dirancang untuk menjawab 5 pertanyaan bisnis strategis yang divisualisasikan langsung di Dashboard:

1. **Seberapa besar Fraud Rate di Negara Tujuan PMI?** *Jawaban:* Sangat dipengaruhi regulasi. Negara dengan sistem *G to G* ketat (Korea/Jepang) aman di platform legal, tapi rentan dieksploitasi oknum di media sosial. Negara rute transit menunjukkan rasio penipuan tertinggi.
2. **Berapa rata-rata perbedaan gaji? Apakah lowongan palsu menawarkan gaji lebih tinggi?** *Jawaban:* Ya. Lowongan fiktif menerapkan taktik *Over-promising*, menawarkan upah rata-rata 1.5x - 2x lipat di atas standar kewajaran untuk memanipulasi emosi calon korban.
3. **Fitur apa yang paling berpengaruh dalam mendeteksi lowongan palsu?** *Jawaban:* NLP (Gaya bahasa manipulatif) adalah yang paling dominan (41%), disusul oleh Ketiadaan Profil Perusahaan (23.5%), dan Skor Risiko Geografis Negara (16.8%).
4. **Pola pekerjaan & tipe employment yang sering menjadi target?** *Jawaban:* Sektor kerah biru (*blue-collar*) seperti *Caregiver*, Pekerja Pabrik, dan *Welder* dengan status purna waktu (*Full-time*), karena menyasar demografi yang rentan terhadap asimetri informasi.
5. **Bagaimana performa model setelah tuning vs baseline?** *Jawaban:* XGBoost Tuned mengalahkan Random Forest di semua metrik. Kenaikan paling krusial ada pada metrik **Recall** (+6.7%), memastikan sistem sangat sensitif menangkap fraud tanpa kebobolan (meminimalisir *False Negatives*).

---

## 🧠 Arsitektur Fitur AI (Feature Engineering)

1. **Geo Risk Profiling Engine:** Menghitung indeks kerawanan geografis menggunakan data eksternal (*OCIndex*): 
   `Geo Risk Score = Human Trafficking Score / Resilience Score`
2. **Reality Check Engine:** Model regresi mengecek apakah penawaran gaji (*salary_offered*) melampaui persentil atas standar BP2MI. Jika ya, memicu flag anomali.
3. **Text & Entity NLP:** Mengekstrak *Scam Keyword Score* (misal: "langsung berangkat", "tanpa dokumen"), mendeteksi penggunaan email gratisan, serta ketiadaan identitas PT/P3MI resmi.

---

## ⚖️ Hasil Offline A/B Testing

Pengujian menggunakan *10-Fold Stratified Cross-Validation*.

| Metrik | Model A (Random Forest) | Model B (Tuned XGBoost) | Lift |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 95.2% | **98.5%** | +3.3% |
| **Precision** | 92.1% | **96.4%** | +4.3% |
| **Recall** | 88.4% | **95.1%** | **+6.7%** |
| **F1-Score** | 90.2% | **95.7%** | +5.5% |

*(Pengujian statistik dengan Paired t-Test menghasilkan p-value < 0.05, terbukti signifikan secara statistik).*

---

## 🛠️ Troubleshooting Deployment & Error Umum

* **`ModuleNotFoundError: No module named 'distutils'` di Streamlit Cloud:**
  *Penyebab:* Anda menggunakan Python 3.12+ yang sudah tidak memiliki `distutils` untuk library lama.
  *Solusi:* Buka *Advanced Settings* sebelum deploy di Streamlit Cloud, lalu ubah opsi **Python Version** menjadi **3.11**.
* **Error "Main module does not exist" di Streamlit:**
  *Penyebab:* Jalur file salah. 
  *Solusi:* Pastikan isian *Main file path* di pengaturan Streamlit sesuai dengan letak file di GitHub (misal: `dashboard/dashboard.py` atau hanya `dashboard.py`).
* **Deploy gagal membaca data / Folder GitHub Kosong:**
  Pastikan Anda telah melakukan push folder data Anda. GitHub terkadang mengabaikan file `.csv` jika Anda memiliki konfigurasi `.gitignore` yang memblokirnya.
