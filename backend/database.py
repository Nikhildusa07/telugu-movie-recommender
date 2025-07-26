# backend/database.py
import pandas as pd

def load_movies(filepath="C:\\movie_recommender\\data\\Telugu Movies List.csv"):
    df = pd.read_csv(filepath)
    keep_cols = [col for col in ['Title', 'Year', 'Genre', 'Description'] if col in df.columns]
    df = df[keep_cols].dropna()

    # Ensure numeric year
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)

    # Fallback description if missing
    if 'Description' not in df.columns:
        df['Description'] = df['Title'] + " (" + df['Year'].astype(str) + ") - " + df['Genre'].fillna("")

    df['Description'] = df['Description'].astype(str)
    return df
