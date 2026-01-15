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

In this section, we break down exactly how well our model performed and what those numbers actually mean in a real-world business context.

### 📈 How the Model Performed
After training on 800,000 reviews, we tested the model on 200,000 "blind" reviews (data the model had never seen before). 

* **Overall Accuracy: 89.40%**
    * **What this means:** If you give our model 100 random reviews, it will correctly guess whether they are positive or negative about 89 times. For a computer program reading human language—which is full of slang and sarcasm—this is a very high score.

* **Balanced Understanding (Precision & Recall)**
    * **The Problem:** Sometimes a model is "lazy" and just guesses "Positive" for everything. 
    * **Our Solution:** Our model achieved a balance of ~90% for both positive and negative reviews. This means it is just as good at spotting a happy customer as it is at spotting an angry one.

### 🔍 Deep Dive: The "Confusion Matrix"
A "Confusion Matrix" is a table that shows exactly where the model got confused. Out of our 200,000 test cases:



1.  **True Negatives (87,868):** The model correctly identified these as negative reviews.
2.  **True Positives (90,997):** The model correctly identified these as positive reviews.
3.  **The "Confusion" (Errors):**
    * **False Positives (~11,000):** The model thought these were positive, but they were actually negative.
    * **False Negatives (~10,000):** The model thought these were negative, but they were actually positive.

**Why do errors happen?** Language is tricky. A review like *"This was not a bad product"* contains the word "bad," which might confuse a basic model into thinking it's negative, even though the overall sentiment is decent.

### 💡 Real-World Impact
Why did we go through all this effort? Imagine you are a manager at Amazon:
* **Speed:** This model processed 1,000,000 data points and learned to read them in just **22.4 seconds**. A human would take years to read that many reviews.
* **Automation:** You can now automatically flag the most "Negative" reviews for your customer service team to investigate immediately, ensuring unhappy customers get help faster.



---

## 📂 Repository Contents
* `notebooks/`: Comprehensive Jupyter Notebook documenting every phase from ingestion to evaluation.
* `models/`: Pre-trained `sentiment_model.pkl` and `tfidf_vectorizer.pkl` for immediate use.
* `README.md`: Project documentation and technical breakdown.

---

## 🚀 Future Improvements
* Implementation of a Deep Learning approach (LSTM or Transformers) to compare performance against this baseline.
* Deployment of a web-based UI for real-time sentiment prediction.
