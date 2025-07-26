# backend/recommender.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

class MovieRecommender:
    def __init__(self, movies_df):
        self.movies = movies_df
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(self.movies['Description'].tolist(), convert_to_tensor=True)

    def recommend(self, query, top_n=10):
        if not query.strip():
            return self.movies.head(top_n).to_dict(orient='records')

        # Extract year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)
        year_filter = int(year_match.group()) if year_match else None
        cleaned_query = re.sub(r'\b(19\d{2}|20\d{2})\b', '', query).strip()

        # Filter by year
        filtered_movies = self.movies
        if year_filter:
            filtered_movies = filtered_movies[filtered_movies['Year'] == year_filter]
            if filtered_movies.empty:
                filtered_movies = self.movies[(self.movies['Year'] >= year_filter-5) & (self.movies['Year'] <= year_filter+5)]
        if filtered_movies.empty:
            filtered_movies = self.movies

        # Encode only filtered set
        subset_embeddings = self.model.encode(filtered_movies['Description'].tolist(), convert_to_tensor=True)
        query_embedding = self.model.encode([cleaned_query if cleaned_query else query], convert_to_tensor=True)

        scores = cosine_similarity(query_embedding, subset_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_n]
        results = filtered_movies.iloc[top_indices].copy()
        results['Score'] = scores[top_indices]
        return results[['Title', 'Year', 'Genre', 'Description', 'Score']].to_dict(orient='records')
