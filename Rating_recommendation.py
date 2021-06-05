import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import boto3
from flask import Flask, render_template, request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors



###############################################################################################
#s3 = boto3.resource(service_name = 's3',
#                     region_name = 'us-east-2',
#                    aws_access_key_id= 'AKIA2EZZKSXVU7FABPQG',
#                    aws_secret_access_key='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')
#movie_obj = s3.Bucket('movierecommendationsystemapp').Object('movies.csv').get()
#ratings_obj = s3.Bucket('movierecommendationsystemapp').Object('ratings.csv').get()

movies = pd.read_csv('movies.csv')           #movie_obj['Body'])
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=movies['title'], columns=movies['title'])
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function that get movie recommendations based on the cosine similarity score of movie genres

def genre_recommendations(title):
    i = indices[title]
    similarity_score = list(enumerate(cosine_sim[i]))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:11]
    movie_index = [i[0] for i in similarity_score]
    t = titles.iloc[movie_index]
    return t
## create Tdif Vectorizer

# Function that get movie recommendations based on the cosine similarity score of movie genres

########################################################################################################
#Ratings based Recommendation Code starts here
########################################################################################################
movies_csv_url = 'movies.csv' #'https://gt-parrothunters-finalproject.s3.us-east-2.amazonaws.com/movies.csv'
df_movies = pd.read_csv(movies_csv_url)

ratings_csv_url = 'ratings.csv' #'https://gt-parrothunters-finalproject.s3.us-east-2.amazonaws.com/ratings.csv'
df_ratings = pd.read_csv(ratings_csv_url)

#Filtering Movies and Ratings (won't need if csv is filtered)
df_movies_cnt = pd.DataFrame(
            df_ratings.groupby('movieId').size(),
            columns=['count'])
popular_movies = list(set(df_movies_cnt.query('count >= 50').index))  #filtering out movies with fewer than 50 ratings
movies_filter = df_ratings.movieId.isin(popular_movies).values

df_users_cnt = pd.DataFrame(
            df_ratings.groupby('userId').size(),
            columns=['count'])
active_users = list(set(df_users_cnt.query('count >= 5').index))  # filtering out users with fewer than 10 ratings
users_filter = df_ratings.userId.isin(active_users).values

df_ratings_filtered = df_ratings[movies_filter & users_filter]


movie_user_mat = df_ratings_filtered.pivot(
    index='movieId', columns='userId', values='rating').fillna(0)
# create mapper from movie title to index
hashmap = {
    movie: i for i, movie in
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title)) 
}
# transform matrix to scipy sparse matrix
movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
#return movie_user_mat_sparse, hashmap

#Create Model
model = NearestNeighbors()
model.fit(movie_user_mat_sparse)

def rating_recommendations(title):
    movie_name = title 
    idx = hashmap[movie_name]    

    distances, indices = model.kneighbors(
        movie_user_mat_sparse[idx],
        n_neighbors=11)                      #will only return 10

    raw_recommends = \
    sorted(
        list(
            zip(
                indices.squeeze().tolist(),
                distances.squeeze().tolist()
            )
        ),
        key=lambda x: x[1]
    )[:0:-1]

    reverse_hashmap = {v: k for k, v in hashmap.items()}

    movie_list = []
    for i, (idx, dist) in enumerate(raw_recommends):
        movie_list.append('{1}'.format(i+1, reverse_hashmap[idx]))

    return movie_list
########################################################################################################
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/recommendations")
def recommend():
    movie = request.args.get('movie')
    r = genre_recommendations(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='t')


@app.route("/rating_recommendations")
def rating_recommend():
    movie = request.args.get('movie')
    r = rating_recommendations(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('rating_recommend.html',movie=movie,r=r,t='s')               
    else:
        return render_template('rating_recommend.html',movie=movie,r=r,t='t')


if __name__ == '__main__':
    app.run()
