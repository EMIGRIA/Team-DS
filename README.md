# Team-DS

Beberapa Hasil Review Data Science

1. Kesenjangan Linguistik (Bahasa)
Kekurangan: Fitur scam_keyword_count saat ini sepenuhnya menggunakan kata kunci bahasa Inggris seperti earn money, work from home, dan no degree required.
Masalah: Penipuan loker yang menargetkan pekerja migran Indonesia menggunakan istilah lokal yang tidak akan terdeteksi oleh daftar kata kunci tersebut.
Solusi:
Tim DS harus mengganti atau menambah daftar scam_kw dengan istilah "Red Flag" Indonesia seperti: "tanpa potong gaji", "biaya administrasi", "visa turis/ziarah", "proses cepat", dan "P3MI".
Gunakan Gemini API untuk melakukan Sentiment/Trust Analysis pada kolom deskripsi guna mendeteksi nada bahasa yang terlalu menekan atau manipulatif.
2. Bias Geografis & Kontekstual
Kekurangan: Meskipun ada pemetaan negara tujuan TKW pada bagian EDA, data latih utama tetap didominasi oleh negara-negara korporat barat seperti US, GB, dan NZ.
Masalah: Pola penipuan loker kantor di Amerika Serikat sangat berbeda dengan pola eksploitasi pekerja migran di Malaysia atau Timur Tengah.
Solusi:
Lakukan Data Augmentation dengan membuat data sintetis (menggunakan LLM) yang mensimulasikan teks lowongan kerja penipuan dalam bahasa Indonesia dengan konteks negara tujuan migran.
Tambahkan fitur is_free_email untuk mendeteksi apakah penyedia kerja menggunakan domain email gratisan (@gmail, @yahoo) yang merupakan indikator umum penipuan lokal.
3. Preprocessing yang Tidak Cocok untuk MLP
Kekurangan: Tim DS menggunakan LabelEncoder untuk fitur kategorikal seperti country, employment_type, dan industry.
Masalah: MLP (Neural Network) akan menganggap label angka (misal: label 10 > label 1) sebagai tingkatan nilai (ordinal), padahal kategori tersebut bersifat nominal.
Solusi:
Ubah teknik encoding menjadi One-Hot Encoding untuk kategori dengan jumlah sedikit atau gunakan Embedding Layer di dalam arsitektur MLP Anda untuk kategori besar seperti industry.
4. Absensi Scaling (Standarisasi)
Kekurangan: Fitur numerik seperti salary_mid, title_length, dan desc_length dibiarkan dalam skala aslinya.
Masalah: MLP sangat sensitif terhadap perbedaan skala; fitur dengan angka besar (seperti gaji ribuan) akan mendominasi fitur biner (0 atau 1), yang menyebabkan proses training menjadi tidak stabil atau lambat konvergen.
Solusi:
Terapkan StandardScaler atau MinMaxScaler pada seluruh fitur numerik sebelum dimasukkan ke dalam input layer MLP untuk memastikan semua fitur memiliki kontribusi yang seimbang.
5. Penanganan Data Kosong (Imputasi)
Kekurangan: Pengisian data kosong pada fitur gaji (salary_mid) menggunakan median global dari dataset.
Masalah: Gaji sangat bergantung pada negara dan jenis industri; menggunakan satu nilai median untuk semua negara dapat mengaburkan deteksi anomali pada model MLP Anda.

Solusi:
Gunakan Group-based Imputation (mengisi nilai kosong berdasarkan rata-rata per negara atau per kategori pekerjaan) agar estimasi gaji lebih realistis sebelum dihitung deviasinya.

NOTE:
Untuk fitur kata kunci (scam_kw), tolong jangan cuma copy-paste dari saran gemini kayak istilah 'P3MI' atau 'Visa Turis'. Ambil logika kategorinya. Misal, kalau loker bilang 'Visa Turis bisa kerja' itu bobot risikonya harus lebih besar daripada sekadar loker tanpa logo perusahaan pokoknya kalau ada kata kunci yang harus di ubah buat kata kata nya yang relevan sama penipuan di indonesia. Sama, tolong siapkan satu versi dataset yang sudah di-StandardScale dan One-Hot Encoded agar saya bisa langsung inject ke model MLP tanpa perlu re-processing lagi
