# Collaborative Filtering Recommender System

This project implements a modular, script-based movie recommender system using the MovieLens 100k dataset.  
It includes:

- ✅ User-Based Collaborative Filtering  
- ✅ Item-Based Collaborative Filtering  
- ✅ Matrix Factorization (SVD)  
- ✅ Metadata enrichment (titles + genres)  
- ✅ Streamlit web app interface  

---

## 🔍 How the Recommendation Methods Work

| Algorithm | Description | Example |
|----------|-------------|---------|
| **User-Based CF** | Recommends movies liked by users similar to you | *"Users like you rated these highly"* |
| **Item-Based CF** | Recommends movies similar to those you've rated highly | *"You liked The Matrix? Try Inception."* |
| **Matrix Factorization (SVD)** | Learns hidden preferences using matrix decomposition | *"You seem to enjoy sci-fi thrillers with strong leads"* |

---

## 🗂️ Project Structure

```
CollaborativeFiltering_Recommender/
│
├── main.py                      # Runs full pipeline
├── streamlit_app.py             # Streamlit web UI
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
├── README.md                    # Project docs
│
├── data/
│   ├── raw/                     # Place 'u.data' and 'u.item' here
│   └── processed/               # Preprocessed rating CSV
│
├── outputs/                     # Recommended CSVs and predictions
│
└── src/
    ├── data_loader.py
    ├── recommender.py
    ├── matrix_factorization.py
    ├── metadata_loader.py
    └── evaluation.py
```

---

## 📥 Dataset Setup

1. Download the dataset:  
   [https://grouplens.org/datasets/movielens/100k/](https://grouplens.org/datasets/movielens/100k/)

2. Place the following files in `data/raw/`:
   ```
   u.data
   u.item
   ```

---

## 🚀 Run the Pipeline (Terminal)

```bash
pip install -r requirements.txt
python main.py
```

This will:
- Load, clean, and preprocess data
- Generate user/item/MF recommendations
- Evaluate each using RMSE
- Save recommendations to `outputs/`

---

## 🌐 Launch Streamlit Web App

```bash
streamlit run streamlit_app.py
```

### Features:
- Select a user ID
- Choose algorithm (User CF, Item CF, MF)
- View top-5 recommendations
- Movie titles and genres included
- Educational explanation for new users

---

## 👨‍💻 Author

**Balaji Koneti**  
Graduate student in Computer Science  
Passionate about AI, data engineering, and real-world machine learning solutions.  
[LinkedIn](https://www.linkedin.com/in/balajikoneti)

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE).
