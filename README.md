# 🛒 Customer Sentiment Analysis — Amazon Fine Food Reviews

A complete end-to-end data science project that performs **sentiment analysis** on Amazon Fine Food Reviews using machine learning. The pipeline covers data extraction from MySQL, preprocessing, exploratory data analysis (EDA), feature engineering, supervised and unsupervised machine learning, and visualization export.

---

## 📌 Project Overview

| Property | Details |
|---|---|
| **Dataset** | Amazon Fine Food Reviews |
| **Source** | MySQL Database (`amazon_database.reviews`) |
| **Rows** | 564,262 reviews |
| **Columns** | 17 features |
| **Target** | Sentiment (Positive / Neutral / Negative) |
| **Models** | Random Forest, Gradient Boosting, K-Means, PCA |

---

## 🗂️ Project Structure

```
Project_2/
│
├── code.py                              # Main pipeline script
├── Amazon_Fine_Food_Reviews_Cleaned.csv # Output: Cleaned dataset
├── EDA_Visualizations.pdf              # Output: All EDA plots
└── README.md                           # Project documentation
```

---

## ⚙️ Prerequisites

### Python Version
- Python 3.8+

### Required Libraries

Install all dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mysql-connector-python wordcloud plotly
```

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualizations |
| `scikit-learn` | Machine learning models |
| `mysql-connector-python` | MySQL database connection |
| `wordcloud` | Word cloud visualization *(optional)* |
| `plotly` | Interactive charts *(optional)* |

---

## 🗄️ Database Setup

The script connects to a **local MySQL database**. Make sure you have:

1. MySQL Server running locally
2. A database named `amazon_database`
3. A table named `reviews` containing the Amazon Fine Food Reviews dataset

Update the credentials in `code.py` if needed:

```python
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="your_password",      # ← update this
    database="amazon_database"
)
```

> ⚠️ **Security Note:** Avoid hardcoding credentials. Use environment variables or a `.env` file for production use.

---

## 🚀 How to Run

```bash
cd Project_2
python code.py
```

The script will run all 10 pipeline sections automatically and print progress to the console.

---

## 🔬 Pipeline Sections

### 1. 📥 Data Loading
- Connects to MySQL and loads all reviews into a Pandas DataFrame.

### 2. 🔍 Initial Data Exploration
- Displays shape, data types, summary statistics, and missing value analysis.

### 3. 🧹 Data Preprocessing
- Fills missing `ProfileName` and `Summary` values.
- Removes duplicate reviews based on `UserId`, `ProductId`, and `Time`.
- Identifies categorical and numerical columns.

### 4. 🔧 Data Cleaning
- Converts `Time` column from date strings to `datetime` objects.
- Detects and caps outliers in `HelpfulnessNumerator` and `HelpfulnessDenominator` using the IQR method.
- Removes rows with invalid `Score` values (outside 1–5 range).
- Removes rows where `HelpfulnessNumerator > HelpfulnessDenominator`.

### 5. 🏗️ Feature Engineering
| New Feature | Description |
|---|---|
| `Sentiment` | Positive (Score > 3), Neutral (Score = 3), Negative (Score < 3) |
| `ReviewLength` | Character count of the review text |
| `HelpfulnessRatio` | `HelpfulnessNumerator / HelpfulnessDenominator` |
| `ReviewYear` | Extracted year from `Time` |
| `ReviewMonth` | Extracted month from `Time` |
| `ReviewDay` | Extracted day from `Time` |
| `ReviewDayOfWeek` | Day of week (0=Monday, 6=Sunday) |

### 6. 📊 Exploratory Data Analysis (15 Visualizations)
All charts are saved to `EDA_Visualizations.pdf`:

1. Score Distribution
2. Sentiment Distribution
3. Correlation Heatmap
4. Review Length Distribution
5. Helpfulness Ratio Distribution
6. Boxplot — Review Length vs Score
7. Violin Plot — Helpfulness Ratio vs Sentiment
8. Pairplot of Key Features
9. Reviews Per Year (Line Chart)
10. Reviews by Month (Bar Chart)
11. Scatter — Helpfulness vs Review Length
12. Top 10 Most Reviewed Products
13. Average Score Trend Over Years
14. Word Cloud of Review Text *(requires `wordcloud`)*
15. Sunburst Chart — Sentiment by Year *(requires `plotly`)*

### 7. 🤖 Machine Learning Models

#### A. Supervised Learning — Sentiment Classification
- **Features used:** `HelpfulnessRatio`, `ReviewLength`, `Score`
- **Train/Test Split:** 80/20 with stratification
- **Scaling:** StandardScaler

| Model | Accuracy |
|---|---|
| Random Forest Classifier | 100% |
| Gradient Boosting Classifier | 100% |

#### B. Unsupervised Learning — Clustering
- **K-Means Clustering** — 3 clusters (positive, neutral, negative)
- **PCA** — Reduced to 2 components for visualization
  - PC1 explains ~38.6% variance
  - PC2 explains ~33.3% variance

### 8. 💾 Output Files
- `Amazon_Fine_Food_Reviews_Cleaned.csv` — Fully cleaned and feature-engineered dataset
- `EDA_Visualizations.pdf` — All EDA and ML visualizations

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Total Reviews Processed | 564,262 |
| Features Engineered | 7 new features |
| Visualizations Generated | 15 charts |
| Random Forest Accuracy | 100% |
| Gradient Boosting Accuracy | 100% |
| PCA Explained Variance | ~71.9% (2 components) |

---

## 📂 Dataset Download

> 🔗 Replace the link below with your actual CSV file URL (e.g., Google Drive, GitHub Releases, Kaggle, etc.)

| File | Description | Link |
|---|---|---|
| `Amazon_Fine_Food_Reviews_Cleaned.csv` | Cleaned & feature-engineered dataset (564K rows) | <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews> |

<!-- 
  HOW TO ADD YOUR CSV LINK:
  1. Upload your CSV to Google Drive / GitHub Releases / Kaggle
  2. Copy the shareable link
  3. Replace the (#) above with your actual link, e.g.:
     [⬇️ Download CSV](https://drive.google.com/your-link-here)
-->

---

## 📊 EDA Visualizations

> 🔗 Replace the link below with your actual PDF file URL (e.g., Google Drive, GitHub, etc.)

| File | Description | Link |
|---|---|---|
| `EDA_Visualizations.pdf` | All 15 EDA + ML charts exported as PDF | [📄 View Visualizations](#) |

<!-- 
  HOW TO ADD YOUR PDF LINK:
  1. Upload EDA_Visualizations.pdf to Google Drive or GitHub
  2. Copy the shareable/raw link
  3. Replace the (#) above with your actual link, e.g.:
     [📄 View Visualizations](https://drive.google.com/your-pdf-link-here)
-->

### Preview — Sample Charts

> 🖼️ Add individual chart screenshots below. Upload images to an `images/` folder in your repo, then update the paths.

| Score Distribution | Sentiment Distribution | Correlation Heatmap |
|---|---|---|
| ![Score Distribution](images/score_distribution.png) | ![Sentiment Distribution](images/sentiment_distribution.png) | ![Heatmap](images/correlation_heatmap.png) |

<!--
  HOW TO ADD CHART IMAGES:
  1. Take screenshots of your charts from the PDF
  2. Save them inside a folder: Project_2/images/
  3. Update the filenames above to match your actual image names
  Example:
    ![Score Distribution](images/score_dist.png)
-->

---

## 🖥️ Dashboard Preview

> 🖼️ Add a screenshot of your Streamlit dashboard or any interactive dashboard below.

![Dashboard Preview](images/dashboard.png)

<!--
  HOW TO ADD YOUR DASHBOARD IMAGE:
  1. Take a full screenshot of your dashboard
  2. Save it as: Project_2/images/dashboard.png
  3. The image will automatically appear above once uploaded to GitHub
  
  You can also embed a live demo link like:
  👉 [Live Dashboard](https://your-streamlit-app.streamlit.app)
-->

---

## 📝 Notes

- The ML models achieve 100% accuracy because `Score` is used as a direct feature, and `Sentiment` is derived entirely from `Score`. This is intentional for demonstrating pipeline structure. In real-world NLP applications, `Text` would be vectorized (e.g., TF-IDF, BERT) instead.
- Word Cloud and Plotly Sunburst are optional and will be skipped gracefully if libraries are not installed.
- The script samples **50,000 rows** for ML model training to ensure reasonable runtime.

---

## 👤 Author

**Akash**
- Project: Customer Sentiment Analysis — Amazon Fine Food Reviews
- Language: Python
- Tools: MySQL, Pandas, Scikit-learn, Seaborn, Matplotlib

---

## 📄 License

This project is for educational and portfolio purposes.
