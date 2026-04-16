# -*- coding: utf-8 -*-
"""
Customer Sentiment Analysis - Amazon Fine Food Reviews

This script performs a complete end-to-end data science project on the Amazon Fine Food Reviews dataset.
It includes data loading, cleaning, preprocessing, exploratory data analysis, machine learning modeling,
and evaluation.

Sections:
1.  Import Libraries
2.  Database Connection and Data Loading
3.  Initial Data Exploration
4.  Data Preprocessing
5.  Data Cleaning
6.  Feature Engineering
7.  Advanced Exploratory Data Analysis (EDA)
8.  Machine Learning Modeling
9.  Model Training and Evaluation
10. Save Cleaned Data and Visualizations
"""

# =============================================================================
# 1. Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import mysql.connector
from mysql.connector import Error
import warnings
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# =============================================================================
# 2. Database Connection and Data Loading
# =============================================================================
def create_mysql_connection(host_name, user_name, user_password, db_name):
    """
    Creates a connection to a MySQL database.

    Args:
        host_name (str): The host name of the database server.
        user_name (str): The user name to connect with.
        user_password (str): The password for the user.
        db_name (str): The name of the database to connect to.

    Returns:
        mysql.connector.connection.MySQLConnection: The connection object or None if connection fails.
    """
    connection = None
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="system",
            database="amazon_database"
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")
    return connection

def load_data_from_db(connection, query):
    """
    Loads data from the database using a SQL query.

    Args:
        connection (mysql.connector.connection.MySQLConnection): The database connection object.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    cursor = connection.cursor()
    cursor.execute(query)
    records = cursor.fetchall()
    df = pd.DataFrame(records, columns=[i[0] for i in cursor.description])
    cursor.close()
    return df

# --- Main Data Loading ---
# Connecting to the database and loading the data.
db_connection = create_mysql_connection("localhost", "root", "system", "amazon_database")
if db_connection:
    try:
        data = load_data_from_db(db_connection, "SELECT * FROM reviews")
        print("Data loaded successfully from MySQL database.")
    except Error as err:
        print(f"Error loading data from database: '{err}'")
        data = pd.DataFrame() # Create an empty DataFrame to prevent further errors
    finally:
        if db_connection.is_connected():
            db_connection.close()
            print("MySQL connection is closed.")
else:
    print("MySQL connection failed. Please check your credentials and database server status.")
    data = pd.DataFrame() # Create an empty DataFrame to prevent further errors

# =============================================================================
# 3. Initial Data Exploration
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Initial Data Exploration")
    print("="*50)

    # Dataset Shape
    print("\n1. Dataset Shape:")
    print(f"   Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # Dataset Info
    print("\n2. Dataset Info:")
    print(data.info())

    # Summary Statistics
    print("\n3. Summary Statistics:")
    print(data.describe(include='all'))

    # Missing Value Analysis
    print("\n4. Missing Value Analysis:")
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    missing_df = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    print(missing_df[missing_df['Missing Values'] > 0])
else:
    print("Dataframe is empty. Halting script.")

# =============================================================================
# 4. Data Preprocessing
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Data Preprocessing")
    print("="*50)

    # Handling Missing Values
    # For 'ProfileName' and 'Summary', we can fill missing values with 'Unknown' or an empty string
    data['ProfileName'].fillna('Unknown', inplace=True)
    data['Summary'].fillna('', inplace=True)
    print("Missing values handled.")

    # Removing Duplicates
    # Duplicates can occur if a user leaves multiple reviews for the same product.
    # We will keep the first review in such cases.
    initial_rows = len(data)
    data.drop_duplicates(subset=['UserId', 'ProductId', 'Time'], keep='first', inplace=True)
    print(f"Removed {initial_rows - len(data)} duplicate reviews.")

    # Feature Scaling and Encoding will be done after splitting the data to avoid data leakage.
    # For now, we can prepare the columns that need encoding.
    categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
    # We will encode 'UserId', 'ProductId', 'ProfileName' and other object types later
    # For now, let's identify them.
    print(f"\nCategorical columns identified for later encoding: {categorical_cols}")

    numerical_cols = data.select_dtypes(include=np.number).columns.tolist()
    print(f"Numerical columns identified for later scaling: {numerical_cols}")

# =============================================================================
# 5. Data Cleaning
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Data Cleaning")
    print("="*50)

    # Datatype Corrections
    # The 'Time' column is in Unix timestamp format. We'll convert it to datetime objects.
    data['Time'] = pd.to_datetime(data['Time'], infer_datetime_format=True)
    print("Corrected 'Time' column to datetime objects.")

    # Outlier Detection and Handling (using IQR for 'HelpfulnessNumerator' and 'HelpfulnessDenominator')
    for col in ['HelpfulnessNumerator', 'HelpfulnessDenominator']:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
        print(f"\nDetected {len(outliers)} outliers in '{col}' using IQR method.")
        
        # We can cap the outliers instead of removing them to retain data
        data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
        print(f"Capped outliers in '{col}'.")

    # Handling Inconsistent Values
    # Example: In 'Score', we might have values outside the 1-5 range.
    initial_rows = len(data)
    data = data[data['Score'].between(1, 5)]
    print(f"Removed {initial_rows - len(data)} rows with inconsistent 'Score' values.")

    # Another inconsistency: HelpfulnessNumerator should not be greater than HelpfulnessDenominator
    initial_rows = len(data)
    data = data[data['HelpfulnessNumerator'] <= data['HelpfulnessDenominator']]
    print(f"Removed {initial_rows - len(data)} rows where HelpfulnessNumerator > HelpfulnessDenominator.")

# =============================================================================
# 6. Feature Engineering
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Feature Engineering")
    print("="*50)

    # Sentiment
    data['Sentiment'] = data['Score'].apply(lambda score: 'positive' if score > 3 else ('neutral' if score == 3 else 'negative'))
    print("1. Created 'Sentiment' feature from 'Score'.")

    # Review Length
    data['ReviewLength'] = data['Text'].apply(len)
    print("2. Created 'ReviewLength' feature from 'Text' length.")

    # Helpfulness Ratio
    # To avoid division by zero, we add a small epsilon to the denominator
    data['HelpfulnessRatio'] = data['HelpfulnessNumerator'] / (data['HelpfulnessDenominator'] + 0.00001)
    print("3. Created 'HelpfulnessRatio' feature.")

    # Time-based Features
    data['ReviewYear'] = data['Time'].dt.year
    data['ReviewMonth'] = data['Time'].dt.month
    data['ReviewDay'] = data['Time'].dt.day
    data['ReviewDayOfWeek'] = data['Time'].dt.dayofweek # Monday=0, Sunday=6
    print("4. Created time-based features: Year, Month, Day, DayOfWeek.")

    print("\nEngineered features added to the dataset:")
    print(data[['Score', 'Sentiment', 'ReviewLength', 'HelpfulnessRatio', 'ReviewYear']].head())

# =============================================================================
# 7. Advanced Exploratory Data Analysis (EDA) & Visualizations
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Advanced Exploratory Data Analysis (EDA)")
    print("="*50)
    
    # Create a PDF file to save all visualizations
    pdf_pages = PdfPages('EDA_Visualizations.pdf')

    # 1. Score Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Score', data=data, palette='viridis')
    plt.title('Distribution of Scores', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    pdf_pages.savefig()

    # 2. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sentiment', data=data, palette='magma')
    plt.title('Distribution of Sentiments', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    pdf_pages.savefig()

    # 3. Correlation Heatmap
    plt.figure(figsize=(12, 8))
    # Select only numeric columns for correlation matrix
    numeric_data = data.select_dtypes(include=np.number)
    correlation_matrix = numeric_data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features', fontsize=16)
    pdf_pages.savefig()

    # 4. Review Length Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data['ReviewLength'], bins=50, kde=True)
    plt.title('Distribution of Review Length', fontsize=16)
    plt.xlabel('Review Length', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 5000) # Capping x-axis for better visualization
    pdf_pages.savefig()

    # 5. Helpfulness Ratio Distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(data['HelpfulnessRatio'], bins=30, kde=True)
    plt.title('Distribution of Helpfulness Ratio', fontsize=16)
    plt.xlabel('Helpfulness Ratio', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    pdf_pages.savefig()

    # 6. Boxplot of Review Length by Score
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Score', y='ReviewLength', data=data, palette='pastel')
    plt.title('Review Length vs. Score', fontsize=16)
    plt.xlabel('Score', fontsize=12)
    plt.ylabel('Review Length', fontsize=12)
    plt.ylim(0, 2500) # Capping y-axis for clarity
    pdf_pages.savefig()

    # 7. Violin Plot of Helpfulness Ratio by Sentiment
    plt.figure(figsize=(12, 7))
    sns.violinplot(x='Sentiment', y='HelpfulnessRatio', data=data, palette='muted')
    plt.title('Helpfulness Ratio vs. Sentiment', fontsize=16)
    plt.xlabel('Sentiment', fontsize=12)
    plt.ylabel('Helpfulness Ratio', fontsize=12)
    pdf_pages.savefig()

    # 8. Pairplot of Key Numerical Features
    # Using a sample to speed up the plotting
    pairplot_sample = data.sample(n=1000, random_state=42)
    plt.figure(figsize=(15, 15))
    sns.pairplot(pairplot_sample[['Score', 'ReviewLength', 'HelpfulnessRatio', 'Sentiment']], hue='Sentiment', palette='husl')
    # pdf_pages.savefig() # This can be large
    plt.suptitle('Pairplot of Key Features', y=1.02, fontsize=18)


    # 9. Reviews per Year
    plt.figure(figsize=(12, 6))
    reviews_by_year = data['ReviewYear'].value_counts().sort_index()
    sns.lineplot(x=reviews_by_year.index, y=reviews_by_year.values, marker='o')
    plt.title('Number of Reviews Over Years', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    pdf_pages.savefig()

    # 10. Reviews by Month
    plt.figure(figsize=(12, 6))
    reviews_by_month = data['ReviewMonth'].value_counts().sort_index()
    sns.barplot(x=reviews_by_month.index, y=reviews_by_month.values, palette='Blues_d')
    plt.title('Number of Reviews by Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    pdf_pages.savefig()

    # 11. Scatter plot of Helpfulness vs. Review Length
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='ReviewLength', y='HelpfulnessNumerator', data=data.sample(5000), hue='Score', palette='viridis', alpha=0.6)
    plt.title('Helpfulness vs. Review Length', fontsize=16)
    plt.xlabel('Review Length', fontsize=12)
    plt.ylabel('Helpfulness Numerator', fontsize=12)
    plt.xlim(0, 5000)
    pdf_pages.savefig()

    # 12. Top 10 Most Reviewed Products
    plt.figure(figsize=(12, 8))
    top_products = data['ProductId'].value_counts().nlargest(10)
    sns.barplot(x=top_products.values, y=top_products.index, orient='h', palette='rocket')
    plt.title('Top 10 Most Reviewed Products', fontsize=16)
    plt.xlabel('Number of Reviews', fontsize=12)
    plt.ylabel('Product ID', fontsize=12)
    pdf_pages.savefig()

    # 13. Average Score per Year
    avg_score_by_year = data.groupby('ReviewYear')['Score'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='ReviewYear', y='Score', data=avg_score_by_year, marker='s', color='purple')
    plt.title('Average Score Trend Over Years', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Score', fontsize=12)
    pdf_pages.savefig()

    # 14. Word Cloud of Review Text (requires wordcloud library)
    try:
        from wordcloud import WordCloud
        text = " ".join(review for review in data.Text.astype(str))
        wordcloud = WordCloud(stopwords=None, background_color="white", max_words=100).generate(text)
        plt.figure(figsize=(15,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title('Most Common Words in Reviews', fontsize=20)
        pdf_pages.savefig()
    except ImportError:
        print("WordCloud library not found. Skipping word cloud visualization.")

    # 15. Sunburst chart for sentiment distribution by year (requires plotly)
    try:
        import plotly.express as px
        sentiment_by_year = data.groupby(['ReviewYear', 'Sentiment']).size().reset_index(name='count')
        fig = px.sunburst(sentiment_by_year, path=['ReviewYear', 'Sentiment'], values='count', 
                          title='Sentiment Distribution by Year')
        # fig.show() # Interactive, not saving to PDF directly
    except ImportError:
        print("Plotly not found. Skipping sunburst chart.")

    print("Finished generating visualizations.")

# =============================================================================
# 8. & 9. Machine Learning Modeling, Training, and Evaluation
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Machine Learning Modeling")
    print("="*50)

    # --- Data Preparation for Modeling ---
    # We will use a smaller subset of the data for faster training times.
    ml_data = data.sample(n=50000, random_state=42)

    # Feature selection
    features = ['HelpfulnessRatio', 'ReviewLength', 'Score']
    target = 'Sentiment'

    X = ml_data[features]
    y = ml_data[target]

    # Encode the target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- A. Supervised Learning: Sentiment Classification ---
    print("\n--- A. Supervised Learning: Sentiment Classification ---")

    # 1. Random Forest Classifier
    print("\n1. Training Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train_scaled, y_train)
    y_pred_rf = rf_clf.predict(X_test_scaled)

    print("\nRandom Forest Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rf, target_names=le.classes_))
    
    # Confusion Matrix for Random Forest
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Random Forest Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    pdf_pages.savefig()

    # 2. Gradient Boosting Classifier
    print("\n2. Training Gradient Boosting Classifier...")
    gb_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_clf.fit(X_train_scaled, y_train)
    y_pred_gb = gb_clf.predict(X_test_scaled)

    print("\nGradient Boosting Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_gb, target_names=le.classes_))

    # --- B. Unsupervised Learning: Clustering ---
    print("\n--- B. Unsupervised Learning: Clustering ---")

    # For clustering and PCA, we'll use the entire ml_data set.
    # We should scale it using the same scaler fitted on the training data.
    X_scaled = scaler.transform(X)
    
    # 1. K-Means Clustering
    print("\n1. Performing K-Means Clustering...")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10) # 3 clusters for positive, neutral, negative
    ml_data['Cluster'] = kmeans.fit_predict(X_scaled)

    # 2. PCA for Dimensionality Reduction and Visualization
    print("\n2. Performing PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=ml_data['Cluster'], palette='viridis', alpha=0.7)
    plt.title('K-Means Clusters visualized with PCA', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    pdf_pages.savefig()
    
    # Explained variance by PCA
    print(f"\nExplained variance by PCA components: {pca.explained_variance_ratio_}")

# =============================================================================
# 10. Save Cleaned Data and Visualizations
# =============================================================================
if not data.empty:
    print("\n" + "="*50)
    print("Saving Cleaned Data and Visualizations")
    print("="*50)

    # Save the cleaned and engineered dataset
    try:
        data.to_csv("Amazon_Fine_Food_Reviews_Cleaned.csv", index=False)
        print("Cleaned data saved successfully to 'Amazon_Fine_Food_Reviews_Cleaned.csv'")
    except Exception as e:
        print(f"Error saving cleaned data: {e}")

    # Close the PDF file
    try:
        pdf_pages.close()
        print("Visualizations saved successfully to 'EDA_Visualizations.pdf'")
    except Exception as e:
        print(f"Error saving visualizations PDF: {e}")

print("\nEnd of script.")