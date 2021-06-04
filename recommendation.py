import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import boto3
from flask import Flask, render_template, request
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

s3 = boto3.resource(service_name = 's3',
                     region_name = 'us-east-2',
                    aws_access_key_id= 'AKIA2EZZKSXVU7FABPQG',
                    aws_secret_access_key='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')
movie_obj = s3.Bucket('movierecommendationsystemapp').Object('movies.csv').get()
movies = pd.read_csv(movie_obj['Body'])
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




app = Flask(__name__)

@app.route("/home")
def home():
    return render_template("index.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/recommender")
def recommender():
    return render_template("home.html")

@app.route("/results")
def results():
    movie = request.args.get('movie')
    r = genre_recommendations(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='t')



if __name__ == '__main__':
    app.run()
