# Amazon Sentiment Analysis: Scaling to 1 Million Reviews

## 📊 Project Vision
This project demonstrates an end-to-end Machine Learning pipeline for sentiment classification. While the original dataset contains 3.6 million reviews, this project intentionally scales the analysis to **1,000,000 samples** to balance high-performance modeling with the computational constraints of a standard 16GB RAM development environment.

## 🧠 What is Sentiment Classification?

Sentiment Classification is a branch of **Natural Language Processing (NLP)** that uses algorithms to determine the emotional tone behind a body of text. 

### How it works in this project:
1. **Binary Classification:** We treat sentiment as a "Yes/No" problem. Label **1** represents Positive (4-5 star reviews), and Label **0** represents Negative (1-2 star reviews).
2. **Feature Extraction (TF-IDF):** This process identifies "signature words." For example, the word "waste" appears frequently in negative reviews but rarely in positive ones. TF-IDF gives "waste" a high mathematical weight for the Negative class.
3. **The Prediction:** When we give the model a new review, it looks at the weights of all the words present. If the "Negative weights" outweigh the "Positive weights," the model predicts a `0`.

### Why this matters:
In a real-world business context, this allows companies to:
* **Monitor Brand Health:** Instantly track customer satisfaction across millions of data points.
* **Identify Product Flaws:** Automatically flag reviews mentioned "broken" or "defective" for the quality control team.
* **Competitive Analysis:** Compare the sentiment of your products against competitors at scale.

---

## 🛠️ Technical Decisions & Methodology

### 1. Data Ingestion: The "Streaming" Approach
* **Decision:** Used `bz2.open` with text-mode streaming instead of pre-extracting the files.
* **Why:** The raw training data is ~450MB compressed but expands significantly when uncompressed. Streaming allows us to parse the data line-by-line, saving gigabytes of disk space and preventing memory spikes.

### 2. Sample Scaling (1,000,000 Rows)
* **Decision:** Scaled from an initial 500,000 rows to 1,000,000 rows.
* **Why:** In NLP, more data often leads to better generalization. 1 Million rows was chosen as the "sweet spot"—large enough to be a "big data" portfolio piece, but small enough to allow for iterative training and testing without crashing the 16GB RAM limit during vectorization.

### 3. Preprocessing & Data Cleaning
* **Decision:** Applied aggressive regex cleaning (removing special characters/numbers) and lowercasing.
* **Why:** Sentiment is largely carried by descriptive adjectives and verbs. Removing numbers and punctuation reduces the "vocabulary noise," which makes the TF-IDF matrix more efficient and the model more accurate.
* **Sanity Check:** Identified and removed 2 empty records post-cleaning to prevent mathematical errors during the vectorization phase.

### 4. Feature Engineering: TF-IDF Vectorization
* **Decision:** Used `TfidfVectorizer` with `max_features=50000` and English stop-word removal.
* **Why:** TF-IDF (Term Frequency-Inverse Document Frequency) was chosen over simple word counts because it penalizes common words (like "the" or "product") and rewards sentiment-rich, unique words. Limiting to 50,000 features ensures the resulting sparse matrix remains manageable in memory.

### 5. Model Selection: Logistic Regression (SAGA Solver)
* **Decision:** Chose Logistic Regression with the `saga` optimization solver.
* **Why:** * **Efficiency:** Unlike the default solver, `saga` is specifically designed for very large datasets and handles sparse data exceptionally well.
    * **Interpretability:** Logistic Regression provides clear probability scores, allowing us to see how "confident" the model is in its sentiment prediction.
    * **Speed:** The model trained on 800,000 samples in just ~22.4 seconds.

---

## 📈 Performance & Results

* **Final Accuracy:** 89.40% on a test set of 200,000 unseen reviews.
* **Class Balance:** Nearly identical Precision and Recall for both Negative (0) and Positive (1) classes, confirming the model has no bias.
* **Memory Optimization:** The final serialized model artifacts are under 2.5MB total, making them lightweight and ready for web-deployment (e.g., Streamlit or Flask).



---

## 📂 Repository Contents
* `notebooks/`: Comprehensive Jupyter Notebook documenting every phase from ingestion to evaluation.
* `models/`: Pre-trained `sentiment_model.pkl` and `tfidf_vectorizer.pkl` for immediate use.
* `README.md`: Project documentation and technical breakdown.

---

## 🚀 Future Improvements
* Implementation of a Deep Learning approach (LSTM or Transformers) to compare performance against this baseline.
* Deployment of a web-based UI for real-time sentiment prediction.
