import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import re
import sqlite3

# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS for styling
# =============================================================================
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file '{file_name}' not found. Using default styles.")

local_css("style.css")

# =============================================================================
# Lottie Animation
# =============================================================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_q5jw2w4s.json")

# =============================================================================
# Data Loading and Caching
# =============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv('Amazon_Fine_Food_Reviews_Cleaned.csv')
    df['Time'] = pd.to_datetime(df['Time']) # Correctly parse datetime strings
    
    # Feature Engineering from code.py
    df['Sentiment'] = df['Score'].apply(lambda score: 'positive' if score > 3 else ('neutral' if score == 3 else 'negative'))
    df['ReviewLength'] = df['Text'].str.len()
    df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 0.00001)
    df['ReviewYear'] = df['Time'].dt.year
    df['ReviewMonth'] = df['Time'].dt.month
    
    # Using a larger sample for more detailed analysis, consider adjusting if performance is an issue
    df_sample = df.sample(frac=0.3, random_state=42)
    return df_sample

data = load_data()

# =============================================================================
# Sidebar Navigation
# =============================================================================
with st.sidebar:
    st.title("Sentiment Navigator")
    if lottie_animation:
        st_lottie(lottie_animation, height=180)
    
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Explorer", "Sentiment Deep Dive", "Review Insights", "Product Deep Dive", "Analyze Your Text", "Database Chatbox"],
        icons=["house-door-fill", "clipboard-data-fill", "emoji-sunglasses-fill", "search-heart-fill", "box-seam-fill", "pen-fill", "chat-dots-fill"],
        menu_icon="compass-fill",
        default_index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard created for analyzing Amazon Fine Food Reviews, powered by Streamlit.")

# =============================================================================
# Home Page
# =============================================================================
if selected == "Home":
    st.title("Customer Sentiment Analysis Dashboard")
    st.markdown("### Welcome to the analysis of Amazon Fine Food Reviews!")
    st.write(
        "This interactive dashboard provides a comprehensive analysis of customer sentiment from over half a million reviews. "
        "The project transforms raw text data into actionable insights, helping to understand customer feedback, identify trends, and pinpoint key discussion topics."
    )
    
    st.header("Project Highlights")
    st.markdown("""
    - **Data Processing:** Cleaned and preprocessed a large dataset to ensure data quality.
    - **Feature Engineering:** Created new features like `Sentiment`, `ReviewLength`, and `HelpfulnessRatio` to enrich the analysis.
    - **In-depth EDA:** Visualized data distributions, trends over time, and relationships between features.
    - **Interactive Dashboard:** Built with Streamlit and Plotly for a dynamic and user-friendly experience.
    
    **👈 Use the sidebar to navigate through the different sections of the analysis.**
    """)

# =============================================================================
# Data Explorer Page
# =============================================================================
if selected == "Data Explorer":
    st.header("Data Explorer")
    st.write("Take a closer look at the dataset. Here's a random sample and some key metrics.")
    st.dataframe(data.head(10))

    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Reviews (Sampled)", f"{data.shape[0]:,}")
    with col2:
        st.metric("Unique Products", f"{data['ProductId'].nunique():,}")
    with col3:
        st.metric("Unique Users", f"{data['UserId'].nunique():,}")
    with col4:
        st.metric("Average Score", f"{data['Score'].mean():.2f} ★")

    st.subheader("Visual Analysis of the Data")
    
    col1, col2 = st.columns(2)
    with col1:
        score_dist_fig = px.bar(data['Score'].value_counts().sort_index(), 
                                title="Distribution of Review Scores (1-5)",
                                labels={'value':'Count', 'index':'Score'},
                                color=data['Score'].value_counts().sort_index().index,
                                color_continuous_scale='viridis')
        st.plotly_chart(score_dist_fig, use_container_width=True)
    with col2:
        review_len_fig = px.histogram(data, x='ReviewLength', nbins=50,
                                      title="Distribution of Review Length",
                                      labels={'ReviewLength': 'Length of Review Text'},
                                      log_y=True)
        review_len_fig.update_layout(showlegend=False)
        st.plotly_chart(review_len_fig, use_container_width=True)
        
    st.subheader("Temporal Analysis")
    reviews_by_year = data.groupby('ReviewYear').size().reset_index(name='count')
    reviews_by_year_fig = px.line(reviews_by_year, x='ReviewYear', y='count', 
                                  title="Number of Reviews Over the Years", 
                                  markers=True)
    st.plotly_chart(reviews_by_year_fig, use_container_width=True)


# =============================================================================
# Sentiment Deep Dive Page
# =============================================================================
if selected == "Sentiment Deep Dive":
    st.header("Sentiment Deep Dive")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Overall Sentiment")
        sentiment_counts = data['Sentiment'].value_counts()
        sentiment_pie_fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                                 title="Sentiment Proportions", hole=0.4,
                                 color=sentiment_counts.index,
                                 color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'})
        st.plotly_chart(sentiment_pie_fig, use_container_width=True)

    with col2:
        st.subheader("Sentiment Trends Over Time")
        sentiment_over_time = data.groupby(['ReviewYear', 'Sentiment']).size().reset_index(name='count')
        sentiment_line_fig = px.line(sentiment_over_time, x='ReviewYear', y='count', color='Sentiment', 
                                    title="Monthly Sentiment Trends",
                                    color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'})
        st.plotly_chart(sentiment_line_fig, use_container_width=True)
    
    st.subheader("How Features Relate to Sentiment")
    col1, col2 = st.columns(2)
    with col1:
        helpfulness_sentiment_fig = px.box(data, x='Sentiment', y='HelpfulnessRatio', 
                                           color='Sentiment',
                                           title="Helpfulness Ratio vs. Sentiment",
                                           labels={'HelpfulnessRatio': 'Helpfulness Ratio'},
                                           notched=True)
        st.plotly_chart(helpfulness_sentiment_fig, use_container_width=True)
    with col2:
        review_length_fig = px.box(data, x='Sentiment', y='ReviewLength',
                                  color='Sentiment',
                                  title="Review Length vs. Sentiment",
                                  labels={'ReviewLength': 'Review Length'},
                                  notched=True)
        review_length_fig.update_yaxes(range=[0, 2000]) # Zoom in for better readability
        st.plotly_chart(review_length_fig, use_container_width=True)

# =============================================================================
# Review Insights Page
# =============================================================================
if selected == "Review Insights":
    st.header("Deeper Insights from Reviews")

    st.subheader("Most Common Words in Reviews")
    sentiment_for_wc = st.selectbox("Select Sentiment for Word Cloud", ['positive', 'negative'])
    
    text_data = ' '.join(data[data['Sentiment'] == sentiment_for_wc]['Text'].dropna().head(10000))
    
    if text_data:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.write(f"Not enough data for '{sentiment_for_wc}' word cloud.")

    st.subheader("Top Products and Users")
    col1, col2 = st.columns(2)
    with col1:
        top_products = data['ProductId'].value_counts().nlargest(10).reset_index()
        top_products.columns = ['ProductId', 'Count']
        top_products_fig = px.bar(top_products, x='Count', y='ProductId', orientation='h', 
                                  title="Top 10 Products by Review Count",
                                  labels={'Count':'Number of Reviews', 'ProductId':'Product ID'})
        top_products_fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(top_products_fig, use_container_width=True)

    with col2:
        top_users = data['UserId'].value_counts().nlargest(10).reset_index()
        top_users.columns = ['UserId', 'Count']
        top_users_fig = px.bar(top_users, x='Count', y='UserId', orientation='h', 
                               title="Top 10 Users by Review Count",
                               labels={'Count':'Number of Reviews', 'UserId':'User ID'})
        top_users_fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(top_users_fig, use_container_width=True)

# =============================================================================
# Product Deep Dive Page
# =============================================================================
if selected == "Product Deep Dive":
    st.header("🔍 Product Deep Dive")
    st.write("Select a product to see its detailed review analysis.")

    # Get list of top 100 products for the select box
    top_100_products = data['ProductId'].value_counts().nlargest(100).index.tolist()
    selected_product = st.selectbox("Select a Product ID", top_100_products)

    if selected_product:
        product_data = data[data['ProductId'] == selected_product]

        st.subheader(f"Analysis for Product: {selected_product}")

        # Metrics for the selected product
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", f"{product_data.shape[0]:,}")
        with col2:
            st.metric("Average Score", f"{product_data['Score'].mean():.2f} ★")
        with col3:
            st.metric("Helpfulness Ratio", f"{product_data['HelpfulnessRatio'].mean():.2%}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            sentiment_counts = product_data['Sentiment'].value_counts()
            product_pie_fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index, 
                                     title="Sentiment Distribution", hole=0.4,
                                     color=sentiment_counts.index,
                                     color_discrete_map={'positive':'green', 'negative':'red', 'neutral':'blue'})
            st.plotly_chart(product_pie_fig, use_container_width=True)
        with col2:
            score_over_time = product_data.groupby('ReviewYear')['Score'].mean().reset_index()
            product_line_fig = px.line(score_over_time, x='ReviewYear', y='Score', 
                                       title="Average Score Over Time", markers=True)
            st.plotly_chart(product_line_fig, use_container_width=True)

        # Display recent reviews
        st.subheader("Recent Reviews")
        recent_reviews = product_data.sort_values(by='Time', ascending=False).head(5)
        for _, row in recent_reviews.iterrows():
            st.markdown(f"**Score: {row['Score']} ★** | **Sentiment: {row['Sentiment']}**")
            st.markdown(f"> {row['Text']}")
            st.markdown("---")

# =============================================================================
# Analyze Your Text Page
# =============================================================================
if selected == "Analyze Your Text":
    st.header("Analyze Your Own Review Text")
    st.write("Enter any text below to get an instant sentiment analysis.")

    user_text = st.text_area("Your text here:", "I absolutely loved this product! It exceeded all my expectations.", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_text:
            blob = TextBlob(user_text)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.1:
                st.success(f"**Positive Sentiment** (Polarity Score: {sentiment_score:.2f})")
                st.balloons()
            elif sentiment_score < -0.1:
                st.error(f"**Negative Sentiment** (Polarity Score: {sentiment_score:.2f})")
            else:
                st.warning(f"**Neutral Sentiment** (Polarity Score: {sentiment_score:.2f})")
        else:
            st.info("Please enter some text to analyze.")

# =============================================================================
# Database Chatbox Page
# =============================================================================
if selected == "Database Chatbox":
    st.header("💬 AI Database Chatbox")
    st.write("Chat with your Amazon Fine Food Reviews data! Ask questions in plain English and get instant SQL-powered answers.")

    # Sidebar config
    st.sidebar.markdown("---")
    st.sidebar.subheader("API Configuration")
    api_key = st.sidebar.text_input("Groq API Key", type="password", value="")
    base_url = st.sidebar.text_input("API Base URL", value="https://api.groq.com/openai/v1")
    model_name = st.sidebar.text_input("Model Name", value="llama-3.3-70b-versatile")

    if not api_key:
        st.warning("Please enter your Groq API Key in the sidebar.")
    else:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key=api_key)

        @st.cache_resource
        def get_sqlite_connection():
            conn = sqlite3.connect(':memory:', check_same_thread=False)
            data.to_sql('reviews', conn, index=False, if_exists='replace')
            return conn

        conn = get_sqlite_connection()

        # Build schema string once
        schema_info = "Table: reviews\nColumns:\n"
        for col, dtype in data.dtypes.items():
            schema_info += f"  - {col} ({dtype})\n"

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sql" in msg:
                    st.code(msg["sql"], language="sql")
                if "dataframe" in msg:
                    st.dataframe(msg["dataframe"])

        # Custom MySQL developer system prompt
        master_system_prompt = f"""You are an expert MySQL developer, data analyst, and intelligent assistant.

Your job is to understand questions written in natural language and generate accurate, efficient, and executable MySQL queries.

Core Responsibilities:
1. Understand questions written in any human language, including informal or typo-filled text.
2. Decide whether the user question requires SQL or a general explanation.
3. If the question relates to data analysis or databases, generate MySQL queries.
4. If the question is general knowledge unrelated to the dataset, answer normally.

RESPONSE FORMAT — always use exactly one of these two formats:

For data/database questions:
MODE: DATA
SQL: <your complete MySQL query here>

For general knowledge, greetings, or non-data questions:
MODE: CHAT
ANSWER: <your clear explanation>

MySQL Knowledge — use proper MySQL syntax including:
SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, DISTINCT
JOIN (INNER JOIN, LEFT JOIN, RIGHT JOIN), SUBQUERIES, WINDOW FUNCTIONS
CASE WHEN, AGGREGATE FUNCTIONS (COUNT, SUM, AVG, MIN, MAX)

SQL Writing Rules:
1. Always produce syntactically correct MySQL queries.
2. ONLY use this table: reviews
3. ONLY use these exact column names: {', '.join(data.columns.tolist())}
4. Use GROUP BY properly when aggregation functions are used.
5. Use ORDER BY when sorting results.
6. Use LIMIT only when the user asks for top/bottom N results.
7. Ensure the query can run directly without modification.
8. NEVER combine SELECT DISTINCT with GROUP BY on the same column.
9. NEVER invent column names not listed above.

Full SQL Script Rule — if user asks for "full SQL code", "complete script", "start from database creation":
Generate a complete MySQL script including:
  CREATE DATABASE sentiment_db;
  USE sentiment_db;
  CREATE TABLE reviews (...all columns...);
  INSERT INTO reviews VALUES (...sample rows...);
  <the required SELECT query>

Dataset: Amazon Fine Food Reviews
Table name: reviews
Columns: {', '.join(data.columns.tolist())}
Total rows: {len(data):,}
Column context:
- Score: 1-5 stars (5=best)
- Sentiment: 'positive', 'neutral', or 'negative'
- ReviewYear, ReviewMonth, ReviewDay: integer date parts
- HelpfulnessRatio: fraction of helpful votes (float)
- ReviewLength: character length of the review text

Window Function Examples — use these when user asks for "unique per product", "first review per product", "one row per user", "latest review", "highest rated per product", etc:

User: "unique ProductId with all columns" / "first review per product":
SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY ProductId ORDER BY Id) AS rn FROM reviews) t WHERE rn = 1

User: "latest review per user":
SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY UserId ORDER BY Id DESC) AS rn FROM reviews) t WHERE rn = 1

User: "highest rated review per product":
SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY ProductId ORDER BY Score DESC, Id) AS rn FROM reviews) t WHERE rn = 1

User: "running total of reviews per year":
SELECT ReviewYear, COUNT(*) AS reviews_count, SUM(COUNT(*)) OVER (ORDER BY ReviewYear) AS running_total FROM reviews GROUP BY ReviewYear ORDER BY ReviewYear
"""

        # Chat input
        if prompt := st.chat_input("Ask me anything! e.g. 'top 5 products', 'what is pandas', 'hi'"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": master_system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=600,
                            temperature=0.0
                        )
                        raw = response.choices[0].message.content.strip()
                    except Exception as e:
                        st.error(f"❌ API Error: {e}")
                        st.session_state.messages.append({"role": "assistant", "content": f"API Error: {e}"})
                        st.stop()

                # Parse response - resilient to model variations
                is_data_mode = "MODE: DATA" in raw or ("SQL:" in raw and "SELECT" in raw.upper()) or raw.upper().lstrip().startswith("SELECT")

                if is_data_mode:
                    # Extract SQL — try line-by-line first, then split
                    sql_query = ""
                    lines = raw.splitlines()
                    for i, line in enumerate(lines):
                        stripped = line.strip()
                        if stripped.upper().startswith("SQL:"):
                            # Could be single line or multiline
                            remainder = stripped[4:].strip()
                            if remainder:
                                sql_query = remainder
                            else:
                                # Grab remaining lines
                                sql_query = " ".join(l.strip() for l in lines[i+1:] if l.strip())
                            break
                    if not sql_query and "SQL:" in raw:
                        sql_query = raw.split("SQL:", 1)[1].strip()

                    # Strategy 3: Directly find the SELECT statement in raw text
                    if not sql_query:
                        for line in lines:
                            if line.strip().upper().startswith("SELECT"):
                                sql_query = line.strip()
                                break

                    # Clean any stray markdown
                    for tag in ["```sql", "```SQL", "```"]:
                        sql_query = sql_query.replace(tag, "").strip()
                    # Only keep up to first blank line
                    if "\n\n" in sql_query:
                        sql_query = sql_query.split("\n\n")[0].strip()

                    # MySQL Sanitizer — strip any SQLite-only syntax so queries work in MySQL Workbench
                    def sanitize_for_mysql(q):
                        import re
                        # Remove SQLite-only PRAGMA statements
                        q = re.sub(r'PRAGMA\s+\w+(\s*=\s*\S+)?;?', '', q, flags=re.IGNORECASE).strip()
                        # Replace SQLite TYPEOF() with MySQL-compatible CAST workaround
                        q = re.sub(r'TYPEOF\(([^)]+)\)', r'(\1)', q, flags=re.IGNORECASE)
                        # Remove SQLite date('now') and replace with NOW()
                        q = re.sub(r"date\('now'\)", 'NOW()', q, flags=re.IGNORECASE)
                        # Strip stray semicolons at end (clean)
                        q = q.rstrip(';').strip()
                        return q

                    sql_query = sanitize_for_mysql(sql_query)

                    # Detect full SQL script (CREATE DATABASE / CREATE TABLE / INSERT)
                    is_full_script = any(kw in sql_query.upper() for kw in ["CREATE DATABASE", "CREATE TABLE", "INSERT INTO"])

                    if is_full_script:
                        # Show full script as MySQL-ready code
                        st.markdown("**🟢 Full MySQL Script** — copy and run this in MySQL Workbench:")
                        st.code(sql_query, language="sql")
                        summary_text = "Here's a complete MySQL script you can copy and run directly in MySQL Workbench. It includes the database creation, table setup, sample data, and the query."
                        st.info(summary_text)
                        # Try to extract and run just the SELECT part for preview
                        select_part = ""
                        for stmt in sql_query.split(";"):
                            if stmt.strip().upper().startswith("SELECT"):
                                select_part = stmt.strip()
                                break
                        if select_part:
                            try:
                                result_df = pd.read_sql_query(select_part, conn)
                                st.markdown("**📊 Preview from your actual dataset:**")
                                st.dataframe(result_df.head(20))
                            except Exception:
                                pass
                        st.session_state.messages.append({"role": "assistant", "content": summary_text})
                        st.stop()

                    # Safety guard — if no valid SQL found, show helpful message
                    if not sql_query or not sql_query.upper().strip().startswith("SELECT"):
                        msg = f"⚠️ Couldn't extract a valid SQL query. AI said:\n\n{raw}\n\nPlease rephrase your question."
                        st.warning(msg)
                        st.session_state.messages.append({"role": "assistant", "content": msg})
                        st.stop()

                    # Execute SQL (using SQLite internally for the app)
                    try:
                        result_df = pd.read_sql_query(sql_query, conn)
                    except Exception as db_e:
                        err_msg = f"⚠️ I understood your question but had trouble running the query.\n\n**Error:** {db_e}\n\n**SQL attempted:**\n```sql\n{sql_query}\n```\n\nTry rephrasing, e.g. *'show top 5 products by review count'*"
                        st.error("Could not run SQL query.")
                        st.markdown(err_msg)
                        st.session_state.messages.append({"role": "assistant", "content": err_msg})
                        st.stop()

                    # Human-friendly summary via a second call
                    with st.spinner("💬 Writing summary..."):
                        try:
                            summary_response = client.chat.completions.create(
                                model=model_name,
                                messages=[{
                                    "role": "user",
                                    "content": f"""The user asked (possibly informally): "{prompt}"
We ran this SQL query: {sql_query}
Got {len(result_df)} rows. First 10 rows:
{result_df.head(10).to_string(index=False)}

Now give a warm, human, conversational answer explaining what the data shows. 
- Speak naturally, like a helpful colleague explaining results
- Highlight the most interesting finding
- Use 2-4 sentences max
- If 0 rows returned, explain that no matching data was found
- Do NOT just list the rows — interpret them"""
                                }],
                                max_tokens=200,
                                temperature=0.4
                            )
                            summary = summary_response.choices[0].message.content.strip()
                        except Exception:
                            summary = f"Here are the results — {len(result_df)} rows found."

                    st.markdown(f"💡 {summary}")
                    with st.expander("📊 View Data Table"):
                        st.dataframe(result_df)
                    st.markdown("**🟢 MySQL-Ready Query** — copy and run this directly in MySQL Workbench:")
                    st.code(sql_query, language="sql")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"💡 {summary}",
                        "sql": sql_query,
                        "dataframe": result_df
                    })

                else:
                    # CHAT MODE
                    answer = raw
                    if "ANSWER:" in raw:
                        answer = raw.split("ANSWER:", 1)[1].strip()
                    elif "MODE: CHAT" in raw:
                        answer = raw.replace("MODE: CHAT", "").strip()

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})



