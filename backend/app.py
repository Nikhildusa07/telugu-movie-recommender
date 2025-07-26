# backend/app.py
from flask import Flask, request, jsonify, render_template
from database import load_movies
from recommender import MovieRecommender

app = Flask(__name__)

movies_df = load_movies()
recommender = MovieRecommender(movies_df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_movies():
    data = request.get_json()
    query = data.get('query', '')
    recommendations = recommender.recommend(query)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
