from django.shortcuts import render

import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create your views here.


def index(request):
        
    # """Data pre processing"""

    movies_data = pd.read_csv('C:\projects\movie_recommender\movie_r\data_set\movies.csv')

    selected_features = ['genres','keywords','tagline','cast','director']
    # replacing the missing valuess with null string
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # combining all the 5 selected features


    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # getting the similarity scores using cosine similarity

    similarity = cosine_similarity(feature_vectors)

    # """Movies recommending system"""

    if request.method == "POST" :

        movie_name = request.POST["mv"]

        list_of_all_titles = movies_data['title'].tolist()

        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

        if len(find_close_match)== 0:
            return render(request, "home.html", {"results":["please enter movie name correctly"]})

        close_match = find_close_match[0]

        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 



        result = []
        i = 0

        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index==index]['title'].values[0]
            result.append(title_from_index)
            i+=1
            if i == 30:
                break
        return render(request, "home.html", {"results": result, "input" : movie_name })
    return render(request, "home.html", {"results":[""]})


