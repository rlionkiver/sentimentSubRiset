# Sentiment Analysis — Steam Reviews

Analisis sentimen 3 kelas (**positive / neutral / negative**) untuk review
game di platform Steam, end-to-end mulai dari data mentah hasil scraping
sampai model siap pakai dan insight bisnis.

---

## Daftar Isi

1. [Tujuan Proyek](#tujuan-proyek)
2. [Dataset](#dataset)
3. [Struktur Folder](#struktur-folder)
4. [Cara Menjalankan](#cara-menjalankan)
5. [Alur Pipeline](#alur-pipeline)
6. [Penjelasan Tiap Tahap](#penjelasan-tiap-tahap)
7. [Hasil Akhir](#hasil-akhir)
8. [Glosarium Istilah](#glosarium-istilah)
9. [Praktik Aman yang Diterapkan](#praktik-aman-yang-diterapkan)
10. [Roadmap Lanjutan](#roadmap-lanjutan)

---

## Tujuan Proyek

Membangun pipeline analisis sentimen yang dapat:

- Mengubah review teks menjadi label sentimen 3 kelas.
- Memberi gambaran kondisi opini pemain (apa yang dipuji, dikeluhkan, dan
  bagaimana opini berubah seiring waktu).
- Menjadi model yang **dapat dipakai ulang** untuk review baru (monitoring
  berkelanjutan).

---

## Dataset

- **Sumber**: Steam Reviews API publik (`store.steampowered.com/appreviews`).
- **Jumlah**: ~3.000 review (setelah filter bahasa & duplikat).
- **Kolom utama**:
  - `recommendationid` — ID unik review.
  - `review` — teks review.
  - `voted_up` — jempol up/down dari user (label proxy 2 kelas Steam).
  - `language` — bahasa review.
  - `playtime_forever_min`, `playtime_at_review_min` — jam main pemain.
  - `votes_up`, `votes_funny`, `weighted_vote_score` — interaksi review.
  - `timestamp_created`, `timestamp_updated` — kapan review dibuat/diedit.
  - `steam_purchase`, `received_for_free` — info pembelian.

---

## Struktur Folder

```
.
├── steamScript/
│   ├── steam_reviews_<appid>.csv          # data mentah hasil scraping
│   └── steam_reviews_<appid>_clean.csv    # data setelah preprocessing
├── model-sentiment/                        # cache model transformer (HF)
├── model_<id>.ipynb                        # notebook utama
├── sentiment_model_3class.joblib           # model akhir siap pakai
└── README.md
```

---

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install pandas numpy nltk emoji ftfy tqdm \
            vaderSentiment scikit-learn imbalanced-learn \
            matplotlib seaborn joblib transformers torch wordcloud
```

> Kalau hanya pakai VADER (tanpa transformer), `transformers` & `torch`
> bisa dilewati.

### 2. Buka notebook

```bash
jupyter notebook model_<id>.ipynb
```

### 3. Jalankan dari atas ke bawah

Eksekusi sel sesuai urutan: **Import Library → Load Data → Preprocessing →
Feature Engineering → Labelling → EDA → Split Data → Modelling**.

> Setelah restart kernel, **selalu jalankan ulang dari atas** karena
> variabel `df` perlu dibangun kembali.

---

## Alur Pipeline

```
Scraping (selesai)
   ↓
1. Import Library
   ↓
2. Load Data (CSV mentah)
   ↓
3. Preprocessing teks (3 level pembersihan)
   ↓
4. Feature Engineering (panjang, playtime, kredibilitas, dsb.)
   ↓
5. Labelling sentimen 3 kelas (VADER atau Transformer)
   ↓
6. EDA + visualisasi             ← pemahaman data
   ↓
7. Train-Test Split (BATAS LEAKAGE)
   ↓
8. Pipeline Modelling (TF-IDF + SMOTE/class_weight + Classifier)
   ↓
9. Cross-Validation + Tuning (TRAIN saja)
   ↓
10. Evaluasi Final di TEST (sekali saja)
   ↓
11. Insight + simpan model
   ↓
12. Pertanyaan bisnis
```

---

## Penjelasan Tiap Tahap

### 1. Import Library

Memuat semua tools yang dibutuhkan: `pandas` & `numpy` untuk data, `nltk`
untuk bahasa alami, `vaderSentiment` & `transformers` untuk labeling,
`scikit-learn` & `imbalanced-learn` untuk modelling, `matplotlib`,
`seaborn`, `wordcloud` untuk visualisasi.

Sumber NLTK yang diunduh:

- **`punkt` / `punkt_tab`** — model untuk **tokenisasi** (memecah teks jadi
  kata/kalimat).
- **`stopwords`** — daftar kata fungsional umum (the, is, a, …) yang sering
  dibuang.
- **`wordnet` + `omw-1.4`** — kamus untuk **lemmatisasi** (mengubah kata ke
  bentuk dasarnya, mis. *running → run*).
- **`averaged_perceptron_tagger`** — model **POS tagging** (mengenali kata
  sebagai noun/verb/adj/adv).

### 2. Load Data

Memuat CSV mentah hasil scraping. Data mentah masih punya noise, duplikat,
dan kolom yang belum siap untuk model.

### 3. Preprocessing Teks

Pembersihan dilakukan dalam 3 level:

| Level | Fungsi | Dipakai untuk |
|---|---|---|
| Ringan (`clean_light`) | Pertahankan kapital, tanda baca, emoji | VADER & Transformer |
| Menengah (`clean_no_emoji`, `emoji_to_text`) | Buang/translate emoji | Model klasik |
| Berat (`tokenize_lemmatize`) | Tokenisasi + lemmatisasi + stopword removal | TF-IDF & topic modeling |

Filter awal: hanya review berbahasa Inggris, tidak kosong, dan unik
berdasarkan `recommendationid`.

### 4. Feature Engineering

Membuat fitur tambahan dari kolom yang sudah ada:

- `review_char_len`, `review_word_len`, `review_token_len` — ukuran teks.
- `created_at`, `updated_at`, `was_edited` — informasi waktu & edit.
- `playtime_forever_hr`, `playtime_at_review_hr` — playtime dalam jam.
- `is_credible` — flag review yang kredibel (playtime ≥ 60 menit, dibeli
  langsung di Steam, tidak gratis).
- `helpfulness_ratio` — rasio votes_up vs total interaksi.
- `is_too_short` — flag review yang terlalu pendek (<3 token).
- `label_voted` — label proxy 2 kelas dari `voted_up`.

### 5. Labelling

Karena `voted_up` hanya 2 kelas, kita generate label 3 kelas via salah satu:

- **Opsi A — VADER**: lexicon rule-based, cepat, ringan, tanpa GPU.
  Skor `compound` ≥ +0.05 → positive, ≤ −0.05 → negative, sisanya neutral.
- **Opsi B — Transformer pre-trained** `cardiffnlp/twitter-roberta-base-sentiment-latest`:
  asli 3 kelas, akurasi lebih tinggi terutama untuk teks pendek/informal.

> Operasi labeling **per-baris dan independen**, jadi aman dilakukan
> sebelum train-test split tanpa menyebabkan data leakage.

Hasil disimpan ke `steam_reviews_<appid>_clean.csv`.

### 6. EDA + Visualisasi

Pemahaman data sebelum modelling. Yang divisualisasikan:

1. Distribusi label sentimen 3 kelas vs `voted_up`.
2. Mismatch model vs jempol Steam → kandidat **sarkasme**.
3. Distribusi panjang review per kelas.
4. Tren mingguan: volume + komposisi sentimen.
5. Playtime vs sentimen.
6. Persentase review yang diedit.
7. Top kata per kelas + WordCloud.
8. Korelasi metrik numerik.

### 7. Train-Test Split

Data dibagi 80% train + 20% test dengan **stratifikasi** untuk menjaga
proporsi 3 kelas tetap sama di kedua split. Mulai dari sini test set
**tidak boleh disentuh** sampai evaluasi akhir.

### 8. Pipeline Modelling

Tiga kandidat model (semua di dalam `imblearn.Pipeline`):

- **`logreg_balanced`** — Logistic Regression + `class_weight="balanced"`.
- **`linsvc_balanced`** — LinearSVC + `class_weight="balanced"`.
- **`nb_smote`** — Multinomial Naive Bayes + SMOTE.

Vektorisasi: TF-IDF unigram + bigram, `min_df=3`, `max_df=0.95`,
`sublinear_tf=True`.

### 9. Cross-Validation + Tuning

`StratifiedKFold(n_splits=5)` dengan tiga metrik:

- `f1_macro` — rata-rata F1 per kelas (bobot sama).
- `balanced_accuracy` — rata-rata recall per kelas.
- `roc_auc_ovr_weighted` — kemampuan ranking (hanya untuk model dengan
  `predict_proba`).

Model terbaik di-tune lebih lanjut dengan `GridSearchCV`
(grid pada `tfidf__ngram_range`, `tfidf__min_df`, `clf__C`).

### 10. Evaluasi Final

Test set dievaluasi **satu kali saja** dengan:

- `classification_report` per kelas.
- Confusion matrix.
- ROC-AUC OvR weighted.
- Balanced accuracy & F1 macro.

### 11. Insight + Simpan Model

- Top 15 kata indikator per kelas (dari koefisien Logistic Regression).
- Model disimpan ke `sentiment_model_3class.joblib`.

### 12. Pertanyaan Bisnis

Notebook menjawab 8 pertanyaan: persepsi keseluruhan, apa yang dikeluhkan
(ABSA), apa yang dipuji, perubahan opini sepanjang waktu, perbedaan antar
segmen pemain, opini tersembunyi (sarkasme), risiko churn, dan rencana
monitoring berkelanjutan.

---

## Hasil Akhir

### Hasil cross-validation pada train

| Model | F1 macro | Std | Balanced Acc | ROC-AUC OvR |
|---|---|---|---|---|
| **logreg_balanced** | **0.6603** | 0.0228 | **0.7128** | **0.8799** |
| linsvc_balanced | 0.6564 | 0.0127 | 0.6558 | – |
| nb_smote | 0.5657 | 0.0323 | 0.5907 | 0.8703 |

**Pemenang**: `logreg_balanced` — menang di semua metrik, stabil antar
fold (std rendah), dan punya `predict_proba` sehingga bisa dipakai untuk
threshold tuning di analisis bisnis.

### Artefak

- `sentiment_model_3class.joblib` — pipeline lengkap (TF-IDF + LogReg)
  siap dipakai untuk review baru:
  ```python
  import joblib
  model = joblib.load("sentiment_model_3class.joblib")
  model.predict(["Best game I've played this year!", "Crashes every 5 minutes"])
  ```
- `steam_reviews_<appid>_clean.csv` — dataset bersih + label sentimen.

---

## Glosarium Istilah

| Istilah | Arti singkat |
|---|---|
| **Tokenisasi** | Memecah teks menjadi unit lebih kecil (kata/kalimat) yang disebut **token**. |
| **Stopwords** | Kata umum tanpa makna konten (the, is, a). Sering dibuang sebelum modelling. |
| **Lemmatisasi** | Mengubah kata ke bentuk dasarnya (running → run). |
| **POS tagging** | Menandai kata sebagai noun/verb/adj/adv. |
| **TF-IDF** | Mengubah teks jadi vektor angka berdasar frekuensi & kekhasan kata. |
| **n-gram** | Urutan n kata berurutan. Unigram = 1 kata, bigram = 2 kata berurutan. |
| **Imbalanced data** | Distribusi kelas yang sangat tidak seimbang (mis. 80% positif). |
| **SMOTE** | *Synthetic Minority Over-sampling Technique* — membuat sampel sintetis untuk kelas minoritas. |
| **class_weight="balanced"** | Memberi bobot lebih besar ke kelas minoritas saat training. |
| **Stratified split / KFold** | Pembagian data yang menjaga proporsi kelas tetap sama di setiap fold/split. |
| **Cross-validation** | Pengukuran performa model dengan membagi data jadi beberapa fold. |
| **Data leakage** | Kebocoran informasi test ke proses training (TF-IDF/SMOTE yang salah dipasang sering jadi penyebab). |
| **F1 macro** | Rata-rata F1 dari semua kelas dengan bobot sama — adil untuk kelas minoritas. |
| **Balanced accuracy** | Rata-rata recall per kelas — tidak bisa "ditipu" oleh kelas mayoritas. |
| **ROC-AUC OvR** | Mengukur kemampuan model memberi peringkat (ranking) prediksi, dirata-rata one-vs-rest. |
| **Confusion matrix** | Tabel prediksi vs aktual. Diagonal = benar, sel lainnya = salah. |
| **VADER** | Lexicon rule-based untuk sentimen (skor compound −1..+1). |
| **Transformer / BERT** | Model neural network besar yang paham konteks bahasa. |
| **Pipeline (imblearn)** | Rangkaian tahap (TF-IDF → SMOTE → classifier) yang aman dari leakage saat CV. |
| **GridSearchCV** | Pencarian otomatis kombinasi hyperparameter terbaik. |
| **ABSA** | *Aspect-Based Sentiment Analysis* — sentimen per aspek (graphics, performance, story, dsb.). |

---

## Praktik Aman yang Diterapkan

| Risiko | Tindakan |
|---|---|
| TF-IDF / SMOTE bocor ke validasi | Dimasukkan ke `imblearn.Pipeline` → fit hanya pada fold train |
| TEST dipakai berulang saat tuning | TEST dievaluasi sekali saja di tahap akhir |
| Proporsi kelas miring di train/test | `stratify=y` + `StratifiedKFold` |
| Akurasi menyesatkan saat imbalance | Pakai `f1_macro`, `balanced_accuracy`, `roc_auc_ovr_weighted` |
| Label hanya 2 kelas (jempol Steam) | Generate ulang via VADER / Transformer untuk dapat 3 kelas |
| Labeling menyebabkan leakage? | Tidak — operasi independen per baris, aman di-run sebelum split |

---

## Roadmap Lanjutan

| Tahap | Hasil yang diharapkan | Effort |
|---|---|---|
| ABSA berbasis transformer (DeBERTa-ABSA) | Sentimen per aspek lebih akurat | Sedang |
| Topic modeling negatif (BERTopic) | Klaster keluhan yang muncul organik | Sedang |
| Sarcasm detector | Mengurangi mismatch positif-tapi-negatif | Tinggi |
| Anomaly detection time-series | Deteksi otomatis pergeseran opini | Sedang |
| Dashboard live (Streamlit) | Monitoring berkelanjutan untuk stakeholder | Sedang |
| Multilingual labeling | Cakupan komunitas non-Inggris | Tinggi |

---

## Lisensi & Atribusi

- Data review: Steam Web API (publik).
- Model labeling: `cardiffnlp/twitter-roberta-base-sentiment-latest` (HuggingFace).
- VADER: Hutto, C.J. & Gilbert, E.E. (2014).
- scikit-learn, imbalanced-learn, NLTK — open source.
