import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

def setup_data():
    movie_file = 'movies.csv'
    ratings_file = 'ratings.csv'
    
    # 1. Load movies
    if not os.path.exists(movie_file):
        print(f"Error: Could not find {movie_file}")
        return None, None
    
    movies = pd.read_csv(movie_file)
    
    # 2. Check/Create Ratings File
    create_new = False
    if not os.path.exists(ratings_file):
        create_new = True
    else:
        temp = pd.read_csv(ratings_file, nrows=1)
        if 'userId' not in temp.columns:
            create_new = True
            
    if create_new:
        print("Creating a new ratings.csv with correct columns...")
        np.random.seed(42)
        movie_ids = movies['movieId'].head(500).values
        data = []
        for u_id in range(1, 101): # 100 users
            num_rated = np.random.randint(15, 40)
            chosen = np.random.choice(movie_ids, num_rated, replace=False)
            for m_id in chosen:
                data.append([u_id, m_id, np.random.randint(1, 6)])
        
        ratings = pd.DataFrame(data, columns=['userId', 'movieId', 'rating'])
        ratings.to_csv(ratings_file, index=False)
    else:
        ratings = pd.read_csv(ratings_file)
        
    return movies, ratings

def get_recommendations(movie_title, movies_df, sim_df, num_recs=5):
    matches = movies_df[movies_df['title'].str.contains(movie_title, case=False, na=False)]
    
    if matches.empty:
        return None, "Movie not found. Try something else'."
    
    target_movie = matches.iloc[0]
    movie_id = target_movie['movieId']
    
    if movie_id not in sim_df.columns:
        return None, f"Not enough data for '{target_movie['title']}'."
    
    # Calculate similarity
    scores = sim_df[movie_id].sort_values(ascending=False)
    top_ids = scores.iloc[1:num_recs+1].index
    recs = movies_df[movies_df['movieId'].isin(top_ids)][['title', 'genres']]
    
    return target_movie['title'], recs

def main():
    print("--- Loading Movie Recommendation System ---")
    movies, ratings = setup_data()
    
    if movies is None: return

    # Pivot Data: Rows=Users, Columns=Movies
    print("Building similarity matrix (this may take a moment)...")
    user_item = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Calculate Cosine Similarity
    sim_matrix = cosine_similarity(user_item.T)
    sim_df = pd.DataFrame(sim_matrix, index=user_item.columns, columns=user_item.columns)
    
    print("System Ready!")
    
    while True:
        print("\n" + "="*50)
        query = input("Enter movie name (or 'quit'): ")
        if query.lower() == 'quit': break
        
        title, result = get_recommendations(query, movies, sim_df)
        if title is None:
            print(result)
        else:
            print(f"Recommendations for '{title}':")
            print(result.to_string(index=False))

if __name__ == "__main__":
    main()
